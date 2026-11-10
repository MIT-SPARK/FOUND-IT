"""
Regulated Pure Pursuit follower.
https://publications.ri.cmu.edu/storage/publications/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf

INSTRUMENTED BUILD.  Control behaviour is byte-for-byte identical to the
previous version; every change below is diagnostic:

  * every phase logs a row (facing and backing wrote nothing before, so the
    tick counter advanced while the CSV stood still)
  * every branch logs pose (pos/yaw were NaN on the stop_and_spin rows)
  * look_reason distinguishes "hairpin" from "path_end"
  * pose staleness is measured and logged
  * each run gets its own timestamped file plus a waypoint sidecar
"""

import csv
import datetime
import json
import math
import threading
import time

import numpy as np
import os

LINEAR_VEL        = 0.2
BACKUP_LINEAR_VEL = 0.15  # reverse speed for the leave-object back-up maneuver
MAX_ANGULAR_VEL   = 0.15
ACCEPTANCE_RADIUS = 0.1   # per-waypoint (not currently used by pure pursuit)
GOAL_ACCEPTANCE_RADIUS = 0.05  # tighter bound for the final goal waypoint
LOOKAHEAD_MIN     = 0.35
LOOKAHEAD_MAX     = 0.90
LOOKAHEAD_GAIN    = 1.5
CURVATURE_MIN_RADIUS = LINEAR_VEL / MAX_ANGULAR_VEL
APPROACH_DIST        = 0.50
MIN_APPROACH_VEL     = 0.05
MAX_LINEAR_ACCEL  = 0.3
MAX_ANGULAR_ACCEL = 0.75
MIN_CURVE_VEL     = 0.04
LOOKAHEAD_MARGIN = 0.10   # how far L_d must exceed cross_track to guarantee a real crossing
STOP_AND_SPIN_THRESHOLD  = math.radians(110)  # rotate in place when lookahead is behind by this much
FACE_ALIGNMENT_THRESHOLD = math.radians(5)    # heading error below which "facing done" is declared
HEADING_OFFSET = math.radians(10)

SHARP_TURN_THRESHOLD = math.radians(75)   # vertex turn angle above which we stop-and-spin instead of arcing

LOG_DIR       = '~/tmp'
POSE_EPS      = 1e-9      # translation/heading change below this counts as "no new pose"


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


class TickLogger:
    """Writes one CSV row per tick, one file per process."""

    FIELDS = [
        # --- identity / timing -------------------------------------------
        "wall_time", "mission", "tick", "dt", "phase",
        # --- pose --------------------------------------------------------
        "pos_x", "pos_z", "yaw_deg", "fwd_xz", "pose_stale", "pose_age_ticks",
        # --- goal --------------------------------------------------------
        "goal_x", "goal_z", "dist_to_goal",
        # --- path tracking -----------------------------------------------
        "wp_idx", "wp_count", "seg_idx", "t_closest", "cross_track", "cx", "cz",
        # --- lookahead ----------------------------------------------------
        "L_d", "look_dist", "look_found", "look_reason", "look_vertex", "lx", "lz",
        # --- heading -------------------------------------------------------
        "heading_err_deg", "heading_raw_deg", "sin_err", "cos_err",
        # --- steering / speed ----------------------------------------------
        "curvature", "turn_radius", "speed_limiter",
        "prev_v", "prev_w", "v_target", "w_target", "w_saturated", "w_spin",
        "v_cmd", "w_cmd",
        # --- back-up maneuver ------------------------------------------------
        "back_traveled", "back_target",
    ]

    def __init__(self, directory=LOG_DIR, stem="wp_tracker"):
        directory = os.path.expanduser(directory)
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(directory, f"{stem}_{stamp}.csv")
        self.stem_path = os.path.join(directory, f"{stem}_{stamp}")
        self._fh = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._w.writeheader()
        self._n = 0
        self._warned = set()

        # Stable name so tooling can always find the newest run.
        latest = os.path.join(directory, f"{stem}_latest.csv")
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(self.path, latest)
        except OSError:
            pass
        print(f"[TickLogger] {self.path}")

    def log(self, **row):
        # Loudly reject unknown keys.  The old logger dropped them silently,
        # which is how fwd_xz=... sat in three call sites doing nothing.
        for k in row:
            if k not in self.FIELDS and k not in self._warned:
                self._warned.add(k)
                print(f"[TickLogger] WARNING: '{k}' is not a logged field — dropping.")
        row.setdefault("wall_time", f"{time.time():.3f}")
        self._w.writerow({k: row.get(k) for k in self.FIELDS})
        self._n += 1
        if self._n % 20 == 0:
            self._fh.flush()

    def dump_waypoints(self, mission, waypoints):
        """Sidecar file: the actual path, plus the turn angle at every vertex.

        Without this the CSV can't be interpreted — there is no way to tell a
        hairpin from a gentle bend, or to know where the robot was relative to
        the route.
        """
        path = f"{self.stem_path}_waypoints.jsonl"
        rows = []
        for i, wp in enumerate(waypoints):
            entry = {"i": i, "x": float(wp[0]), "y": float(wp[1]), "z": float(wp[2])}
            if 0 < i < len(waypoints) - 1:
                ang = WaypointFollower._turn_angle_at_vertex(waypoints, i)
                entry["turn_deg"] = math.degrees(ang)
                entry["is_hairpin"] = bool(ang > SHARP_TURN_THRESHOLD)
            rows.append(entry)
        try:
            with open(path, "a") as fh:
                fh.write(json.dumps({"mission": mission,
                                     "t": time.time(),
                                     "waypoints": rows}) + "\n")
        except OSError as exc:
            print(f"[TickLogger] could not write waypoint sidecar: {exc}")
        hairpins = [r["i"] for r in rows if r.get("is_hairpin")]
        if hairpins:
            print(f"[TickLogger] mission {mission}: hairpin vertices at {hairpins}")

    def close(self):
        self._fh.close()


class WaypointFollower:
    def __init__(self):
        self._lock      = threading.Lock()
        self._waypoints: list[np.ndarray] = []
        self._idx       = 0
        self._active    = False
        self._on_waypoint_reached = None
        self._prev_linear  = 0.0
        self._prev_angular = 0.0
        self._last_time    = None
        self._face_target: "np.ndarray | None" = None
        self._facing       = False
        # Leave-object back-up maneuver (reverse straight out; see back_up()).
        self._backing            = False
        self._backing_recentered = False
        self._back_distance      = 0.0
        self._back_recenter      = True   # rotate to face target squarely before reversing
        self._back_vel_sign      = -1.0   # -1 reverse (back_up), +1 forward (drive_forward)
        self._back_start: "tuple | None" = None
        self._log = TickLogger()
        self._tick = 0

        self._spin_sign = 0.0
        self._spin_phase = None

        # --- diagnostics ---------------------------------------------------
        self._mission       = 0
        self._prev_pose_fp  = None
        self._pose_age      = 0

    @property
    def active(self):
        with self._lock:
            return ((self._active and self._idx < len(self._waypoints))
                    or self._facing or self._backing)

    def set_face_target(self, pos: "np.ndarray"):
        """Set the 3D position the robot should face after reaching the goal."""
        with self._lock:
            self._face_target = np.asarray(pos, dtype=np.float64)

    def set_waypoints(self, waypoints):
        with self._lock:
            self._waypoints = [np.asarray(wp, dtype=np.float64) for wp in waypoints]
            self._idx = 0

    def start(self):
        with self._lock:
            if not self._waypoints:
                print("[Follower] No waypoints loaded — ignoring start.")
                return
            was_active = self._active
            was_facing = self._facing
            was_backing = self._backing
            old_idx = self._idx
            self._idx = 0
            self._active = True
            self._prev_linear = 0.0
            self._prev_angular = 0.0
            self._last_time = None
            self._mission += 1
            mission = self._mission
            waypoints = list(self._waypoints)
            self._spin_sign = 0.0
            self._spin_phase = None
        print(
            f"[FollowerTransition] start requested: was_active={was_active} "
            f"was_facing={was_facing} was_backing={was_backing} "
            f"old_idx={old_idx} new_waypoint_count={len(waypoints)}"
        )
        if self._log:
            self._log.dump_waypoints(mission, waypoints)
        print(f"[Follower] Started — mission {mission}, {len(waypoints)} waypoints.")

    def stop(self):
        with self._lock:
            self._active = False
            self._facing = False
            self._backing = False
            self._prev_linear = 0.0
            self._prev_angular = 0.0
        print("[Follower] Stopped.")

    def back_up(self, distance: float = 0.4, recenter: bool = True):
        """Reverse straight back ``distance`` metres to leave an object.

        Used instead of the normal rotate-and-drive waypoint following, which
        would swing the robot around next to the obstacle and risk grazing it.
        When ``recenter`` is True the maneuver first rotates to face the target
        squarely (removing HEADING_OFFSET) so the reverse runs along the approach
        normal; when False it reverses immediately with no rotation.  It then
        drives backward until ``distance`` is covered (wheel-odom displacement).
        active() stays True until it completes.
        """
        with self._lock:
            self._backing = True
            self._backing_recentered = False
            self._back_distance = float(distance)
            self._back_recenter = bool(recenter)
            self._back_vel_sign = -1.0        # reverse
            self._back_start = None
            self._active = False
            self._facing = False
            self._prev_linear = 0.0
            self._prev_angular = 0.0
            self._last_time = None
        print(f"[Follower] Backing up {distance:.2f} m.")

    def drive_forward(self, distance: float = 0.2):
        """Drive straight forward ``distance`` metres along the current heading.

        The forward twin of :meth:`back_up`: it reuses the same straight-line
        drive (no rotation, wheel-odom distance gate) but moves forward instead of
        reversing.  Used to close a small standoff gap to an object after a
        loop-closure re-estimates the distance.  active() stays True until done.
        """
        with self._lock:
            self._backing = True
            self._backing_recentered = False   # no recenter — drive straight ahead
            self._back_distance = float(abs(distance))
            self._back_recenter = False
            self._back_vel_sign = 1.0          # forward
            self._back_start = None
            self._active = False
            self._facing = False
            self._prev_linear = 0.0
            self._prev_angular = 0.0
            self._last_time = None
        print(f"[Follower] Driving forward {abs(distance):.2f} m.")

    def set_on_waypoint_reached(self, callback):
        with self._lock:
            self._on_waypoint_reached = callback

    # ------------------------------------------------------------------ #
    # diagnostics helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pose_fingerprint(T_prop):
        """Position + forward axis, enough to detect a pose that never updated."""
        return (float(T_prop[0, 3]), float(T_prop[2, 3]),
                float(T_prop[0, 2]), float(T_prop[2, 2]))

    def _note_pose(self, T_prop):
        """Returns (stale, age_in_ticks) for this tick's pose."""
        fp = self._pose_fingerprint(T_prop)
        prev = self._prev_pose_fp
        stale = prev is not None and all(abs(a - b) < POSE_EPS for a, b in zip(fp, prev))
        self._pose_age = self._pose_age + 1 if stale else 0
        self._prev_pose_fp = fp
        return stale, self._pose_age

    def _base_row(self, dt, phase, T_prop, stale, age, prev_v, prev_w):
        """Fields every row should carry, regardless of which branch we took."""
        fwd = T_prop[:3, 2]
        fwd_xz = math.hypot(float(fwd[0]), float(fwd[2]))
        yaw = math.degrees(math.atan2(float(fwd[0]), float(fwd[2]))) if fwd_xz > 1e-9 else None
        with self._lock:
            mission = self._mission
            wp_idx = self._idx
            wp_count = len(self._waypoints)
            goal = self._waypoints[-1] if self._waypoints else None
        return dict(
            mission=mission, tick=self._tick, dt=dt, phase=phase,
            pos_x=float(T_prop[0, 3]), pos_z=float(T_prop[2, 3]),
            yaw_deg=yaw, fwd_xz=fwd_xz,
            pose_stale=stale, pose_age_ticks=age,
            goal_x=(float(goal[0]) if goal is not None else None),
            goal_z=(float(goal[2]) if goal is not None else None),
            wp_idx=wp_idx, wp_count=wp_count,
            prev_v=prev_v, prev_w=prev_w,
        )

    # ------------------------------------------------------------------ #
    # geometry                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _closest_point_on_path(px, pz, start_idx, waypoints):
        """
        Project the current position onto line segments defined by the waypoints, and find the closest point on the path.

        :param px: current position x
        :param pz: current position z
        :param start_idx: already completed waypoints index
        :param waypoints: list of waypoints
        """
        best = None
        for i in range(start_idx, len(waypoints) - 1):
            ax, az = waypoints[i][0],     waypoints[i][2]
            bx, bz = waypoints[i + 1][0], waypoints[i + 1][2]
            ex, ez = bx - ax, bz - az
            seg2 = ex * ex + ez * ez
            t = 0.0 if seg2 < 1e-12 else _clamp(((px - ax) * ex + (pz - az) * ez) / seg2, 0.0, 1.0)
            cx, cz = ax + t * ex, az + t * ez
            d2 = (px - cx) ** 2 + (pz - cz) ** 2
            if best is None or d2 < best[0]:
                best = (d2, i, t, cx, cz)
        if best is None:
            return 0, 0.0, waypoints[0][0], waypoints[0][2], 0
        return best[1], best[2], best[3], best[4], math.sqrt(best[0])

    @staticmethod
    def _turn_angle_at_vertex(waypoints, i):
        """Absolute heading change at vertex ``i`` (between segment i-1→i and i→next).

        Skips zero-length outgoing segments so duplicate waypoints don't produce
        a spurious angle.  Returns 0.0 if there is no well-defined turn.
        """
        ax, az = waypoints[i - 1][0], waypoints[i - 1][2]
        bx, bz = waypoints[i][0],     waypoints[i][2]
        in_x, in_z = bx - ax, bz - az
        in_len = math.hypot(in_x, in_z)
        if in_len < 1e-9:
            return 0.0
        # find next distinct vertex for the outgoing direction
        for j in range(i + 1, len(waypoints)):
            out_x, out_z = waypoints[j][0] - bx, waypoints[j][2] - bz
            out_len = math.hypot(out_x, out_z)
            if out_len > 1e-9:
                cross = in_x * out_z - in_z * out_x
                dot   = in_x * out_x + in_z * out_z
                return abs(math.atan2(cross, dot))
        return 0.0

    @staticmethod
    def _lookahead_along_path(cx, cz, seg_idx, L_d, waypoints):
        """Lookahead point ``L_d`` of ARC LENGTH ahead of the robot along the path.

        Starts at the robot's projection on the path (``cx, cz`` on segment
        ``seg_idx``) and walks forward across the waypoints, accumulating segment
        lengths until ``L_d`` metres of path have been travelled — so the target
        stays on the route around corners rather than a straight-line (circle)
        distance that can jump across them.

        Returns ``(lx, lz, found, reason, vertex)``.  ``found`` is unchanged.
        ``reason`` separates the two very different ways found can be False:

          "along"     target sits L_d along the route (normal)
          "hairpin"   target is PINNED to a sharp vertex; it no longer moves as
                      the robot advances, so the lookahead distance decays to
                      zero and curvature (2*left_y/L**2) diverges
          "path_end"  route ran out; target is the final waypoint

        ``vertex`` is the waypoint index the target is pinned to, or None.
        """
        remaining = L_d
        sx, sz = cx, cz                       # start at the projection on the current segment
        for i in range(seg_idx, len(waypoints) - 1):
            bx, bz = waypoints[i + 1][0], waypoints[i + 1][2]
            seg_dx, seg_dz = bx - sx, bz - sz
            seg_len = math.hypot(seg_dx, seg_dz)
            if seg_len > 1e-9:
                if remaining <= seg_len:
                    f = remaining / seg_len
                    return sx + f * seg_dx, sz + f * seg_dz, True, "along", None
                remaining -= seg_len
            if i + 1 < len(waypoints) - 1:
                turn_angle = WaypointFollower._turn_angle_at_vertex(waypoints, i + 1)
                if turn_angle > SHARP_TURN_THRESHOLD and seg_len > ACCEPTANCE_RADIUS:
                    return bx, bz, False, "hairpin", i + 1
            sx, sz = bx, bz                   # advance to the next vertex
        return (waypoints[-1][0], waypoints[-1][2], False, "path_end", len(waypoints) - 1)

    # ------------------------------------------------------------------ #
    # control                                                             #
    # ------------------------------------------------------------------ #

    def update(self, T_prop, dt=None):
        now = time.monotonic()
        self._tick += 1

        with self._lock:
            facing     = self._facing
            backing    = self._backing
            back_recentered = self._backing_recentered
            back_recenter   = self._back_recenter
            back_start = self._back_start
            back_dist  = self._back_distance
            back_vel_sign = self._back_vel_sign
            face_tgt   = self._face_target
            prev_v     = self._prev_linear
            prev_w     = self._prev_angular
            if dt is None:
                dt = (now - self._last_time) if self._last_time is not None else 0.0
            self._last_time = now

        pose_stale, pose_age = self._note_pose(T_prop)
        if pose_age == 10:
            phase = "backing" if backing else "facing" if facing else "pursuit"
            print(
                f"[FollowerWarning] pose unchanged for 10 ticks: phase={phase} "
                f"prev_v={prev_v:.3f} prev_w={prev_w:.3f}"
            )

        if backing:
            pos_x, pos_z = T_prop[0, 3], T_prop[2, 3]
            # Phase 1: recenter — rotate to face the target squarely (WITHOUT the
            # HEADING_OFFSET) so backing straight reverses along the approach normal.
            # Skipped when recenter=False (reverse immediately, no rotation).
            if back_recenter and not back_recentered and face_tgt is not None:
                fwd = T_prop[:3, 2]
                fwd_x, fwd_z = fwd[0], fwd[2]
                fwd_xz = math.hypot(fwd_x, fwd_z)
                if fwd_xz > 1e-6:
                    fwd_x /= fwd_xz
                    fwd_z /= fwd_xz
                tx, tz = face_tgt[0] - pos_x, face_tgt[2] - pos_z
                t_norm = math.hypot(tx, tz)
                if t_norm > 1e-6:
                    tx /= t_norm
                    tz /= t_norm
                sin_err = fwd_x * tz - fwd_z * tx
                cos_err = fwd_x * tx + fwd_z * tz
                heading_err = math.atan2(sin_err, cos_err)   # no HEADING_OFFSET → square on
                if abs(heading_err) >= FACE_ALIGNMENT_THRESHOLD:
                    w_spin = math.copysign(MAX_ANGULAR_VEL, heading_err)
                    v_cmd, w_cmd = self._apply_slew(prev_v, prev_w, 0.0, w_spin, dt)
                    if self._log:
                        row = self._base_row(dt, "backing_recenter", T_prop,
                                             pose_stale, pose_age, prev_v, prev_w)
                        row.update(heading_err_deg=math.degrees(heading_err),
                                   heading_raw_deg=math.degrees(heading_err),
                                   sin_err=sin_err, cos_err=cos_err, w_spin=w_spin,
                                   speed_limiter="backing_recenter",
                                   back_target=back_dist,
                                   v_cmd=v_cmd, w_cmd=w_cmd)
                        self._log.log(**row)
                    return v_cmd, w_cmd
                # Recentered — begin the straight reverse from here.
                back_start = (pos_x, pos_z)
                with self._lock:
                    self._backing_recentered = True
                    self._back_start = back_start
            # Phase 2: reverse straight (no steering) until backed up back_dist metres.
            if back_start is None:
                back_start = (pos_x, pos_z)
                with self._lock:
                    self._backing_recentered = True
                    self._back_start = back_start
            traveled = math.hypot(pos_x - back_start[0], pos_z - back_start[1])
            if traveled >= back_dist:
                with self._lock:
                    self._backing = False
                    self._prev_linear = 0.0
                    self._prev_angular = 0.0
                print(f"[Follower] Backed up {traveled:.2f} m — done.")
                if self._log:
                    row = self._base_row(dt, "backing_done", T_prop,
                                         pose_stale, pose_age, prev_v, prev_w)
                    row.update(speed_limiter="backing_done",
                               back_traveled=traveled, back_target=back_dist,
                               v_cmd=0.0, w_cmd=0.0)
                    self._log.log(**row)
                return 0.0, 0.0
            v_target = back_vel_sign * BACKUP_LINEAR_VEL   # +fwd / -reverse
            v_cmd, w_cmd = self._apply_slew(prev_v, prev_w, v_target, 0.0, dt)
            if self._log:
                row = self._base_row(dt, "backing", T_prop,
                                     pose_stale, pose_age, prev_v, prev_w)
                row.update(speed_limiter="backing",
                           back_traveled=traveled, back_target=back_dist,
                           v_target=v_target, w_target=0.0,
                           v_cmd=v_cmd, w_cmd=w_cmd)
                self._log.log(**row)
            return v_cmd, w_cmd

        if facing and face_tgt is not None:
            pos_x, pos_z = T_prop[0, 3], T_prop[2, 3]
            fwd = T_prop[:3, 2]
            fwd_x, fwd_z = fwd[0], fwd[2]
            fwd_xz = math.hypot(fwd_x, fwd_z)
            if fwd_xz > 1e-6:
                fwd_x /= fwd_xz
                fwd_z /= fwd_xz
            tx, tz = face_tgt[0] - pos_x, face_tgt[2] - pos_z
            t_norm = math.hypot(tx, tz)
            if t_norm > 1e-6:
                tx /= t_norm
                tz /= t_norm
            sin_err = fwd_x * tz - fwd_z * tx
            cos_err = fwd_x * tx + fwd_z * tz
            heading_raw = math.atan2(sin_err, cos_err)
            heading_err = heading_raw + HEADING_OFFSET   # NOTE: not re-wrapped to (-pi, pi]
            if abs(heading_err) < FACE_ALIGNMENT_THRESHOLD:
                with self._lock:
                    self._facing = False
                    self._prev_linear = 0.0
                    self._prev_angular = 0.0
                    self._spin_sign = 0.0
                    self._spin_phase = None
                print("[Follower] Facing target — done.")
                if self._log:
                    row = self._base_row(dt, "facing_done", T_prop,
                                         pose_stale, pose_age, prev_v, prev_w)
                    row.update(heading_err_deg=math.degrees(heading_err),
                               heading_raw_deg=math.degrees(heading_raw),
                               sin_err=sin_err, cos_err=cos_err,
                               speed_limiter="facing_done", v_cmd=0.0, w_cmd=0.0)
                    self._log.log(**row)
                return 0.0, 0.0
            w_spin = math.copysign(MAX_ANGULAR_VEL, heading_err)
            spin_sign = math.copysign(1.0, heading_err)
            if self._spin_phase != "facing" or spin_sign != self._spin_sign:
                print(
                    f"[FacingDirectionFlip] raw_deg={math.degrees(heading_raw):.2f} "
                    f"offset_deg={math.degrees(HEADING_OFFSET):.2f} "
                    f"command_error_deg={math.degrees(heading_err):.2f} "
                    f"previous_sign={self._spin_sign:+.0f} new_sign={spin_sign:+.0f} "
                    f"pose_stale={pose_stale} pose_age={pose_age}"
                )
                self._spin_sign = spin_sign
                self._spin_phase = "facing"
            v_cmd, w_cmd = self._apply_slew(prev_v, prev_w, 0.0, w_spin, dt)
            if self._log:
                row = self._base_row(dt, "facing", T_prop,
                                     pose_stale, pose_age, prev_v, prev_w)
                row.update(heading_err_deg=math.degrees(heading_err),
                           heading_raw_deg=math.degrees(heading_raw),
                           sin_err=sin_err, cos_err=cos_err, w_spin=w_spin,
                           speed_limiter="facing", v_cmd=v_cmd, w_cmd=w_cmd)
                self._log.log(**row)
            return v_cmd, w_cmd

        with self._lock:
            if not self._active or not self._waypoints:
                return 0.0, 0.0
            idx       = self._idx
            waypoints = self._waypoints
            cb        = self._on_waypoint_reached

        pos_x, pos_z = T_prop[0, 3], T_prop[2, 3]

        goal = waypoints[-1]
        dist_to_goal = math.hypot(goal[0] - pos_x, goal[2] - pos_z)
        if dist_to_goal < GOAL_ACCEPTANCE_RADIUS:
            with self._lock:
                self._active = False
                self._prev_linear = 0.0
                self._prev_angular = 0.0
                fired = list(range(self._idx, len(waypoints)))
                self._idx = len(waypoints)
                if self._face_target is not None:
                    self._facing = True
                    self._spin_sign = 0.0
                    self._spin_phase = None
                    print(
                        "[FollowerTransition] pursuit -> facing: "
                        f"robot=({pos_x:.3f}, {pos_z:.3f}) "
                        f"goal=({goal[0]:.3f}, {goal[2]:.3f}) "
                        f"dist_to_goal={dist_to_goal:.3f}"
                    )
                else:
                    print("[Follower] Reached goal — stopping. Distance to Goal:", dist_to_goal)
            if cb is not None:
                for i in fired:
                    cb(i)
            if self._log:
                row = self._base_row(dt, "goal", T_prop,
                                     pose_stale, pose_age, prev_v, prev_w)
                row.update(dist_to_goal=dist_to_goal, speed_limiter="goal",
                           v_cmd=0.0, w_cmd=0.0)
                self._log.log(**row)
            return 0.0, 0.0

        seg_idx, t_closest, cx, cz, cross_track = self._closest_point_on_path(pos_x, pos_z, idx, waypoints)
        if seg_idx > idx:
            with self._lock:
                self._idx = seg_idx
            if cb is not None:
                for i in range(idx, seg_idx):
                    cb(i)

        fwd = T_prop[:3, 2]
        fwd_x, fwd_z = fwd[0], fwd[2]
        fwd_xz = math.hypot(fwd_x, fwd_z)
        if fwd_xz < 1e-6:
            v_cmd, w_cmd = self._apply_slew(prev_v, prev_w, 0.0, MAX_ANGULAR_VEL, dt)
            if self._log:
                row = self._base_row(dt, "bad_angle", T_prop,
                                     pose_stale, pose_age, prev_v, prev_w)
                row.update(dist_to_goal=dist_to_goal, cross_track=cross_track,
                           seg_idx=seg_idx, t_closest=t_closest, cx=cx, cz=cz,
                           speed_limiter="bad_angle", v_cmd=v_cmd, w_cmd=w_cmd)
                self._log.log(**row)
            return v_cmd, w_cmd
        fwd_x /= fwd_xz
        fwd_z /= fwd_xz

        L_d = _clamp(LOOKAHEAD_GAIN * max(prev_v, MIN_APPROACH_VEL), LOOKAHEAD_MIN, LOOKAHEAD_MAX)
        L_d = _clamp(max(L_d, cross_track + LOOKAHEAD_MARGIN), LOOKAHEAD_MIN, LOOKAHEAD_MAX)
        # Lookahead measured as ARC LENGTH along the path from the robot's projection
        # (cx, cz), not a straight-line distance — the target follows the route around
        # corners instead of the circle jumping across them.
        lx, lz, look_found, look_reason, look_vertex = self._lookahead_along_path(
            cx, cz, seg_idx, L_d, waypoints)
        dx, dz = lx - pos_x, lz - pos_z
        L2 = dx * dx + dz * dz
        L_norm = math.sqrt(L2) if L2 > 1e-9 else 0.0

        # Shared prefix for whichever branch we take below.
        row = self._base_row(dt, "pursuit", T_prop, pose_stale, pose_age, prev_v, prev_w)
        row.update(dist_to_goal=dist_to_goal, cross_track=cross_track,
                   seg_idx=seg_idx, t_closest=t_closest, cx=cx, cz=cz,
                   L_d=L_d, look_dist=L_norm, look_found=look_found,
                   look_reason=look_reason, look_vertex=look_vertex, lx=lx, lz=lz)

        # If the lookahead target is more than STOP_AND_SPIN_THRESHOLD behind the
        # robot, stop and rotate in place toward it before resuming pure pursuit.
        heading_err = 0.0
        sin_err = cos_err = 0.0
        if L_norm > 1e-6:
            sin_err = fwd_x * (dz / L_norm) - fwd_z * (dx / L_norm)
            cos_err = fwd_x * (dx / L_norm) + fwd_z * (dz / L_norm)
            heading_err = math.atan2(sin_err, cos_err)
            if abs(heading_err) > STOP_AND_SPIN_THRESHOLD and look_reason != "path_end": # TODO Dominic, hack to keep from spinning back and forth at the end
                w_spin = math.copysign(MAX_ANGULAR_VEL, heading_err)
                spin_sign = math.copysign(1.0, heading_err)
                if self._spin_phase != "stop_and_spin" or spin_sign != self._spin_sign:
                    print(
                        f"[StopAndSpin] heading_deg={math.degrees(heading_err):.2f} "
                        f"reason={look_reason} waypoint_idx={idx}/{len(waypoints)} "
                        f"segment={seg_idx} cross_track={cross_track:.3f} "
                        f"look_distance={L_norm:.3f} "
                        f"robot=({pos_x:.3f}, {pos_z:.3f}) "
                        f"lookahead=({lx:.3f}, {lz:.3f}) "
                        f"previous_sign={self._spin_sign:+.0f} "
                        f"new_sign={spin_sign:+.0f}"
                    )
                    self._spin_sign = spin_sign
                    self._spin_phase = "stop_and_spin"
                v_cmd, w_cmd = self._apply_slew(prev_v, prev_w, 0.0, w_spin, dt)
                if self._log:
                    row.update(heading_err_deg=math.degrees(heading_err),
                               sin_err=sin_err, cos_err=cos_err, w_spin=w_spin,
                               speed_limiter="stop_and_spin",
                               v_cmd=v_cmd, w_cmd=w_cmd)
                    self._log.log(**row)
                return v_cmd, w_cmd

        # Make a later stop-and-spin episode print its entry even if it turns in
        # the same direction as the previous episode.
        if self._spin_phase == "stop_and_spin":
            self._spin_sign = 0.0
            self._spin_phase = None

        left_y = fwd_x * dz - fwd_z * dx
        curvature = 0.0 if L2 < 1e-9 else (2.0 * left_y / L2)

        candidates = {"cruise": LINEAR_VEL}
        if abs(curvature) > 1e-6:
            r = 1.0 / abs(curvature)
            if r < CURVATURE_MIN_RADIUS:
                candidates["curvature"] = max(MIN_CURVE_VEL, LINEAR_VEL * (r / CURVATURE_MIN_RADIUS))
        if dist_to_goal < APPROACH_DIST:
            candidates["approach"] = max(MIN_APPROACH_VEL, LINEAR_VEL * (dist_to_goal / APPROACH_DIST))
        limiter = min(candidates, key=candidates.get)
        v_target = candidates[limiter]
        w_raw = curvature * v_target
        w_target = _clamp(w_raw, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        w_saturated = abs(w_raw) > MAX_ANGULAR_VEL

        v_cmd, w_cmd = self._apply_slew(prev_v, prev_w, v_target, w_target, dt)

        if self._log:
            row.update(heading_err_deg=math.degrees(heading_err),
                       sin_err=sin_err, cos_err=cos_err,
                       curvature=curvature,
                       turn_radius=(1.0 / abs(curvature) if abs(curvature) > 1e-9 else None),
                       speed_limiter=limiter,
                       v_target=v_target, w_target=w_target, w_saturated=w_saturated,
                       v_cmd=v_cmd, w_cmd=w_cmd)
            self._log.log(**row)
        return v_cmd, w_cmd

    def _apply_slew(self, prev_v, prev_w, v, w, dt):
        """
        Apply slew rate limiting to the proposed velocities.

        :param prev_v: previous linear velocity
        :param prev_w: previous angular velocity
        :param v: proposed linear velocity
        :param w: proposed angular velocity
        :param dt: time delta since last update
        """
        if dt > 0.0:
            dv = _clamp(v - prev_v, -MAX_LINEAR_ACCEL * dt, MAX_LINEAR_ACCEL * dt)
            dw = _clamp(w - prev_w, -MAX_ANGULAR_ACCEL * dt, MAX_ANGULAR_ACCEL * dt)
            v = prev_v + dv
            w = prev_w + dw
        with self._lock:
            self._prev_linear = v
            self._prev_angular = w
        return v, w
