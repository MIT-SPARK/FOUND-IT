import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _robust_median(ratios: np.ndarray) -> float | None:
    """Median of per-step scale ratios after a MAD-based outlier trim.

    Shared by :class:`OdomScaleEstimator` and :class:`PcdScaleWindow` so both
    reduce a set of displacement ratios to a scale the same way.  Returns None if
    fewer than two ratios are given.
    """
    ratios = np.asarray(ratios, dtype=np.float64)
    if ratios.size < 2:
        return None
    med = np.median(ratios)
    mad = np.median(np.abs(ratios - med))
    if mad > 0:
        keep = np.abs(ratios - med) <= 3.0 * 1.4826 * mad
        if keep.sum() >= 2:
            ratios = ratios[keep]
    return float(np.median(ratios))


class PcdScaleWindow:
    """Sliding window of the most recent ~``max_travel_m`` metres of motion, used
    to metric-scale an object point cloud at publish time.

    Unlike :class:`OdomScaleEstimator` (a one-shot calibration that rescales the
    whole factor graph), this keeps a *rolling* record of per-frame camera poses
    (from the VGGT-SLAM map) paired with the wheel-odometry pose at the same
    instant, capped by wheel-odom arc length.  On demand it returns the robust
    median displacement-ratio scale over that window — applied ONLY to an outgoing
    point cloud, never to the map.

    It is meant to run *alongside* the global ``--odom_scale`` calibration, not
    instead of it: because the window uses the current (already globally-scaled)
    map poses, its scale is the *residual* correction for however far VGGT-SLAM's
    local scale has since drifted from metric — near 1.0 when the map is still
    metric, larger/smaller where it has drifted.  Feeding it current map poses is
    what makes this compose correctly with the global scale.

    The camera/odom pairing and per-step ratio maths mirror
    ``OdomScaleEstimator`` (same SE(3)-invariant displacement-ratio idea); the
    difference is purely that this is a bounded rolling buffer estimated on query.
    """

    def __init__(self, max_travel_m: float = 3.0,
                 min_cam_step: float = 0.05, min_odom_step: float = 0.02):
        self._max_travel = max_travel_m
        self._min_cam_step = min_cam_step
        self._min_odom_step = min_odom_step
        # Each entry: (cam_pos (3,), odom_pos (3,)) at one frame, oldest first.
        self._samples: list[tuple[np.ndarray, np.ndarray]] = []

    def clear(self) -> None:
        """Drop all buffered samples.

        Called when the map is rescaled in one discrete step (the one-shot
        ``--odom_scale`` calibration), so the window never mixes pre- and
        post-rescale camera poses — every retained sample stays in one frame.
        """
        self._samples.clear()

    def _odom_travel(self) -> float:
        """Total wheel-odom arc length currently spanned by the window (metres)."""
        if len(self._samples) < 2:
            return 0.0
        odom = np.stack([o for _c, o in self._samples])
        return float(np.linalg.norm(np.diff(odom, axis=0), axis=1).sum())

    def add_submap_samples(self, cam_poses: np.ndarray,
                           frame_timestamps: list[float], odom_buf) -> None:
        """Append one processed submap's frames, then evict oldest to stay within
        ``max_travel_m`` of wheel-odom travel.

        Args mirror :meth:`OdomScaleEstimator.add_submap_samples`:
            cam_poses: (N,4,4) cam-to-world matrices from the factor graph.
            frame_timestamps: N frame timestamps in seconds.
            odom_buf: OdomBuffer (provides get_at_time).
        """
        N = len(cam_poses)
        if N == 0 or len(frame_timestamps) != N:
            return
        for j in range(N):
            odom = odom_buf.get_at_time(frame_timestamps[j])
            if odom is None:
                continue
            self._samples.append(
                (np.asarray(cam_poses[j])[:3, 3].copy(),
                 np.asarray(odom)[:3, 3].copy()))
        # Keep the *smallest* suffix of samples whose wheel-odom arc length still
        # covers max_travel — i.e. evict the oldest only while doing so leaves the
        # window still >= max_travel.  Evicting down to <= max_travel (the naive
        # cap) would leave the window a hair short of the threshold, so
        # estimate_scale (which requires >= max_travel) would essentially never
        # fire even after driving far.  Always keep at least two for a ratio.
        while len(self._samples) > 2:
            first_seg = float(np.linalg.norm(self._samples[1][1] - self._samples[0][1]))
            if self._odom_travel() - first_seg >= self._max_travel:
                self._samples.pop(0)
            else:
                break

    def estimate_scale(self, require_full_window: bool = True) -> float | None:
        """Robust median metric scale (metric = scale * VGGT units) over the window.

        Returns None if the window has not yet accumulated a full
        ``max_travel_m`` of wheel-odom travel (when ``require_full_window``), or if
        too few steps had real motion from both sensors to form a stable median.
        """
        if require_full_window and self._odom_travel() < self._max_travel:
            return None
        ratios: list[float] = []
        for i in range(1, len(self._samples)):
            cam0, odom0 = self._samples[i - 1]
            cam1, odom1 = self._samples[i]
            cam_disp = float(np.linalg.norm(cam1 - cam0))
            odom_disp = float(np.linalg.norm(odom1 - odom0))
            # Require real motion from BOTH sensors (same guard as the calibrator):
            # rejects divide-by-tiny and stale/held odom samples that bias to ~0.
            if cam_disp < self._min_cam_step or odom_disp < self._min_odom_step:
                continue
            ratios.append(odom_disp / cam_disp)
        return _robust_median(np.asarray(ratios))


class OdomScaleEstimator:
    """Accumulates per-step displacement ratios and returns the median scale."""

    def __init__(self, min_odom_travel_m: float = 5.0,
                 min_cam_step: float = 0.05, min_odom_step: float = 0.02,
                 log_path: str | None = None):
        self._min_travel = min_odom_travel_m
        # A step contributes a ratio only if the camera moved > min_cam_step (VGGT
        # units, guards against divide-by-tiny) AND the robot moved > min_odom_step
        # metres (rejects stale/held odom samples, the dominant noise source).
        self._min_cam_step = min_cam_step
        self._min_odom_step = min_odom_step
        # When set, every submap's camera/odom poses are appended to this file for
        # offline hand-eye calibration.  None disables logging entirely.
        self._log_path = log_path
        self._ratios: list[float] = []
        self._odom_travel: float = 0.0
        self._last_odom_pos: np.ndarray | None = None
        self._scale_applied = False

    @property
    def scale_applied(self) -> bool:
        return self._scale_applied

    def odom_travel_m(self) -> float:
        return self._odom_travel

    def add_submap_samples(
        self,
        cam_poses: np.ndarray,
        frame_timestamps: list[float],
        odom_buf,
    ) -> None:
        """Feed one processed submap's worth of camera poses and frame timestamps.

        Args:
            cam_poses: (N, 4, 4) cam-to-world matrices from the factor graph.
            frame_timestamps: N timestamps in seconds, one per frame.
            odom_buf: OdomBuffer instance (provides get_at_time).
        """
        if self._scale_applied:
            return
        N = len(cam_poses)
        if N < 2 or len(frame_timestamps) != N:
            return

        # Optionally append this submap's full poses to a running log file for
        # offline hand-eye calibration.  Each line is the flattened (row-major, 16
        # values) camera-to-world matrix, a comma, then the flattened odometry pose
        # matrix at the same frame timestamp.  Disabled unless log_path is set.
        if self._log_path is not None:
            with open(self._log_path, "a") as log_file:
                for j in range(N):
                    odom = odom_buf.get_at_time(frame_timestamps[j])
                    if odom is None:
                        continue
                    cam_flat  = " ".join(f"{v:.6f}" for v in np.asarray(cam_poses[j]).flatten())
                    odom_flat = " ".join(f"{v:.6f}" for v in np.asarray(odom).flatten())
                    log_file.write(f"{cam_flat}, {odom_flat}\n")

        for i in range(1, N):
            t0 = frame_timestamps[i - 1]
            t1 = frame_timestamps[i]

            odom0 = odom_buf.get_at_time(t0)
            odom1 = odom_buf.get_at_time(t1)
            if odom0 is None or odom1 is None:
                continue

            cam_disp  = float(np.linalg.norm(cam_poses[i][:3, 3] - cam_poses[i - 1][:3, 3]))
            odom_disp = float(np.linalg.norm(odom1[:3, 3] - odom0[:3, 3]))

            # Require real motion from BOTH sensors.  Skipping low-odom steps is
            # what keeps stale/held odom samples from biasing the scale to ~0.
            if cam_disp < self._min_cam_step or odom_disp < self._min_odom_step:
                continue

            self._ratios.append(odom_disp / cam_disp)

            # Accumulate odom travel for the min_travel trigger.
            if self._last_odom_pos is None:
                self._last_odom_pos = odom0[:3, 3].copy()
            self._odom_travel += float(np.linalg.norm(odom1[:3, 3] - self._last_odom_pos))
            self._last_odom_pos = odom1[:3, 3].copy()

    def try_estimate(self) -> float | None:
        """Return the robust median scale if the travel threshold is reached, else None.

        A MAD-based trim drops residual outliers (e.g. a bad VGGT step) before the
        final median.
        """
        if self._scale_applied:
            return None
        if self._odom_travel < self._min_travel or len(self._ratios) < 2:
            return None
        return _robust_median(np.asarray(self._ratios, dtype=np.float64))

    def mark_applied(self) -> None:
        self._scale_applied = True
