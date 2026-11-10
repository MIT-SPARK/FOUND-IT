#!/usr/bin/env python3
"""ROS 2 wrapper for VGGT-SLAM.

Subscribes to a ROS 2 image topic (sensor_msgs/Image) and feeds frames into
the VGGT-SLAM pipeline, replacing the direct camera backends used by
main_realtime.py.  When running under this wrapper, the ROS topic is the only
image source.

Usage:
    python main_ros2.py --image_topic /camera/image_raw [other SLAM flags]

The node expects BGR or RGB images; encoding is handled by cv_bridge.
"""

import argparse
import os
import queue
import sys
import threading
import time

# ---------------------------------------------------------------------------
# ROS / conda PYTHONPATH fix — must happen before any rclpy import.
#
# When /opt/ros/<distro>/setup.bash is sourced it prepends the ROS distro's
# Python 3.12 site-packages to PYTHONPATH.  If this script runs under a
# different Python (e.g. a conda env on 3.11) the rclpy .so is incompatible
# and the import fails.  We fix this by ensuring the site-packages that
# belongs to *this* Python binary comes first, so the correctly-compiled
# rclpy (installed via robostack) is always found before the system ROS one.
# ---------------------------------------------------------------------------
import sysconfig as _sysconfig
_own_sp = _sysconfig.get_path("purelib")
if _own_sp:
    # Force this Python's own site-packages to the front regardless of where
    # it already sits — a sourced ROS setup.bash puts its Python 3.12 packages
    # at position 1, shadowing the conda-installed (Python 3.11) rclpy.
    sys.path = [_own_sp] + [p for p in sys.path if p != _own_sp]

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import viser.transforms as viser_tf

from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Bool

from foundit.odom_scale_estimator import OdomScaleEstimator, PcdScaleWindow
from foundit.waypoint_follower import WaypointFollower
from foundit.reset import ResetMission
from foundit.session import save_session, load_session
from foundit.app_setup import (
    add_common_slam_args,
    validate_common_args,
    build_objects_module,
    build_solver,
    build_model,
    run_interactive_loop,
)

# Height (m) the Place Object button lifts the published centroid along the true
# vertical (floor normal), so a manipulator targets just above the object.
PLACE_OBJ_RAISE_M = 0.20

# Minimum distance (m) the followed object's bbox centre must move before the
# path is replanned while following (avoids churn on tiny loop-closure nudges).
OBJECT_REPLAN_DIST = 0.10

# Thread safety.
solver_lock = threading.Lock()
data_lock = threading.Lock()

# Set by the viser "Get Object PCD in Cam" button; fulfilled by the capture loop.
# pcd_done is set by the loop once the publish completes, so the Reset mission can
# wait for it.
pcd_request = threading.Event()
pcd_done = threading.Event()
# Reset-mission capture retry support:
#   cached_pcd_request : publish the CACHED object cloud (used as the last-resort
#                        fallback when SAM3 fails to re-detect the object live).
#   pcd_publish_ok     : set by the capture handler when it actually published a
#                        cloud this pass (cleared before each attempt) so the
#                        mission can tell success from a failed live detection.
#   mission_capturing  : set while a Reset-mission capture is in progress; gates the
#                        pre-publish SLAM pause inside the handler to the mission
#                        (the manual button must not pause SLAM).
cached_pcd_request = threading.Event()
pcd_publish_ok = threading.Event()
mission_capturing = threading.Event()

# Set by the Reset mission to force a fresh submap at the robot's current pose
# (so check_missing_object sees the current view); the loop sets force_submap_done
# when the submap has been processed.
force_submap_request = threading.Event()
force_submap_done = threading.Event()
# Optional (submap_id, frame_index) loop-closure target for the next forced
# submap: when set, that submap forces a loop back to the object's query frame so
# the current camera is re-aligned to where the object was first seen.  The loop
# handler reads and clears it.  A plain dict so the mission thread can set it.
force_submap_loop = {"target": None}

# Set by the Stop Following button to abort an in-progress Reset mission (in
# addition to Cancel Session's stop_event). Cleared when a new mission starts.
reset_stop = threading.Event()

# Set by ConfirmationNode from a single std_msgs/Bool topic (--pickup_topic): a
# True message (pickup complete) sets pickup_event, a False message (drop-off
# complete) sets dropoff_event.  The Reset mission clears the relevant one before
# the corresponding publish, then waits on it.
pickup_event = threading.Event()
dropoff_event = threading.Event()


class GraspSuccess:
    """Thread-safe latest value of /grasp_success (True = the grasp worked).

    Published by the manipulator together with the grasp-attempted ack.  Reset to
    None before each grasp so a stale value can't satisfy the next one; ``get()``
    returns None until a fresh value arrives.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._value: "bool | None" = None

    def reset(self):
        with self._lock:
            self._value = None

    def set(self, value: bool):
        with self._lock:
            self._value = bool(value)

    def get(self) -> "bool | None":
        with self._lock:
            return self._value


grasp_success = GraspSuccess()
GRASP_SUCCESS_TIMEOUT = 2.0   # s to wait for /grasp_success after a grasp attempt

# When re-aligning to an object before grasp/place, don't bother driving for a
# standoff error smaller than this (m); clamp any single correction to at most
# ALIGN_MAX_ADJUST (m) as a guard against a bad distance estimate.
ALIGN_STANDOFF_DEADBAND = 0.03
ALIGN_MAX_ADJUST = 1.0

# Set while the manipulator arm is executing a pick-up / drop-off in front of the
# robot.  The image subscriber drops frames while this is set, so no keyframes or
# submaps are created (the SLAM loop must not run with the arm in view); it is
# cleared once the arm has finished/retracted (the pickup/drop-off ack).
slam_paused = threading.Event()


# ---------------------------------------------------------------------------
# Odometry helpers
# ---------------------------------------------------------------------------

def _tf_to_mat4(tf_transform) -> np.ndarray:
    """geometry_msgs/Transform → 4×4 float64 homogeneous matrix."""
    tx = tf_transform.translation.x
    ty = tf_transform.translation.y
    tz = tf_transform.translation.z
    qx = tf_transform.rotation.x
    qy = tf_transform.rotation.y
    qz = tf_transform.rotation.z
    qw = tf_transform.rotation.w
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-10:
        R = np.eye(3, dtype=np.float64)
    else:
        s = 2.0 / n
        R = np.array([
            [1 - s*(qy*qy + qz*qz),     s*(qx*qy - qz*qw),     s*(qx*qz + qy*qw)],
            [    s*(qx*qy + qz*qw), 1 - s*(qx*qx + qz*qz),     s*(qy*qz - qx*qw)],
            [    s*(qx*qz - qy*qw),     s*(qy*qz + qx*qw), 1 - s*(qx*qx + qy*qy)],
        ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = [tx, ty, tz]
    return T


def _load_T_OC_from_yaml(path: str) -> "np.ndarray | None":
    """Load the full 4x4 camera_from_base transform (T_OC) from a calibration YAML.

    T_OC (X) is the full 4x4 transform from the odom/base frame to the VGGT
    camera frame.  An axis-aligned mount with zero lever arm maps
    odom (x fwd, y left, z up — ROS REP 103 base_link) → camera (x right, y down,
    z fwd); the matrix produced offline by scripts/calibrate_test.py additionally
    captures the measured mount rotation and translation (lever arm).

    Returns the 4x4 matrix, or None if the file is missing / malformed.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        M = np.asarray(data["camera_from_base"], dtype=np.float64)
        if M.shape != (4, 4):
            raise ValueError(f"expected a 4x4 matrix, got shape {M.shape}")
        return M.copy()
    except Exception as e:  # noqa: BLE001 — any parse error is reported and returns None
        print(f"[Calib] Failed to load camera_from_base from '{path}': {e}.")
        return None


def _odom_delta_to_cam(delta: np.ndarray, T_OC: np.ndarray) -> np.ndarray:
    """Re-express a 4×4 relative motion measured in the odom/base frame into the
    camera frame by conjugating with the full camera_from_base transform:

        delta_cam = X · delta · X⁻¹        (X = T_OC = camera_from_base)

    Using the full 4×4 (not just its rotation) accounts for the mount lever arm,
    so a base rotation about a point offset from the camera produces the correct
    camera translation.
    """
    return T_OC @ delta @ np.linalg.inv(T_OC)


def _propagated_camera_pose(odom_anchor, odom_anchor_lock, odom_buf, T_OC):
    """Current camera pose in the VGGT world frame, forward-propagated from the
    last submap anchor using the latest odometry.

    Returns a 4×4 cam-to-world matrix, or None when no anchor / odom is available.
    This is the same pose drawn as ``odom_propagated_pose`` in viser and is used
    as the path-planning start, so paths begin where the robot is *now* rather
    than at the last submap's (stale) camera pose.
    """
    with odom_anchor_lock:
        slam_pose = odom_anchor["slam_pose"]
        odom_T    = odom_anchor["odom_T"]
    if slam_pose is None or odom_T is None:
        return None
    latest = odom_buf.get_latest()
    if latest is None:
        return None
    _, T_odom_curr = latest
    delta = np.linalg.inv(odom_T) @ T_odom_curr
    return slam_pose @ _odom_delta_to_cam(delta, T_OC)


def _tilt_correction_from_T_OC(T_OC: np.ndarray) -> np.ndarray:
    """Return a 3×3 rotation R_corr that compensates for camera tilt.

    The VGGT world frame is anchored to the first camera pose, so a downward-
    tilted camera makes the world Y-axis non-vertical and the XZ plane non-level.
    R_corr is the minimal rotation (about the cross-product axis) that maps the
    actual "up" direction (derived from T_OC) to [0, -1, 0] — the expected
    "up" in VGGT camera convention (Y is down, so -Y is up).

    Pre-multiplying any world-frame pose by R_corr brings it into a
    gravity-aligned frame where XZ is the actual floor.

    Returns the identity for a perfectly level camera.
    """
    up_world = T_OC[:3, :3] @ np.array([0., 0., 1.])  # base z-up in VGGT world
    target   = np.array([0., -1., 0.])                  # -Y = up in VGGT convention

    dot   = float(np.dot(up_world, target))
    cross = np.cross(up_world, target)
    s     = float(np.linalg.norm(cross))

    if s < 1e-8 and dot > 0.0:   # already aligned
        return np.eye(3, dtype=np.float64)
    if s < 1e-8:                  # 180° anti-parallel — pick any perpendicular axis
        perp = np.array([1., 0., 0.]) if abs(up_world[0]) < 0.9 else np.array([0., 0., 1.])
        axis = np.cross(up_world, perp)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    # Rodrigues' formula: rotate up_world → target about cross/|cross|
    K = np.array([[0., -cross[2], cross[1]],
                  [cross[2], 0., -cross[0]],
                  [-cross[1], cross[0], 0.]])
    return np.eye(3, dtype=np.float64) + K + (K @ K) / (1.0 + dot)


class OdomBuffer:
    """Thread-safe circular buffer of (timestamp_sec, 4×4 matrix) odom readings."""
    def __init__(self, maxlen: int = 500):
        from collections import deque
        self._lock = threading.Lock()
        self._buf = deque(maxlen=maxlen)

    def push(self, t: float, T: np.ndarray) -> None:
        with self._lock:
            self._buf.append((t, T.copy()))

    def get_latest(self):
        with self._lock:
            if not self._buf:
                return None
            ts, T = self._buf[-1]
            return ts, T.copy()

    def get_at_time(self, t: float, max_gap: float = 0.5) -> np.ndarray | None:
        """Return odom pose at time t via linear interpolation of translation.

        Returns None if the buffer is empty or the closest bracketing entry is
        more than max_gap seconds away from t.
        """
        with self._lock:
            if not self._buf:
                return None
            buf = list(self._buf)

        # Find lo (last entry with ts <= t) and hi (first entry with ts > t).
        lo_idx: int | None = None
        hi_idx: int | None = None
        for k, (ts, _) in enumerate(buf):
            if ts <= t:
                lo_idx = k
            elif hi_idx is None:
                hi_idx = k
                break

        if lo_idx is None and hi_idx is None:
            return None

        if lo_idx is None:
            ts_hi, T_hi = buf[hi_idx]
            return T_hi.copy() if ts_hi - t <= max_gap else None

        if hi_idx is None:
            ts_lo, T_lo = buf[lo_idx]
            return T_lo.copy() if t - ts_lo <= max_gap else None

        ts_lo, T_lo = buf[lo_idx]
        ts_hi, T_hi = buf[hi_idx]
        alpha = (t - ts_lo) / (ts_hi - ts_lo)
        T_interp = np.eye(4, dtype=np.float64)
        # Only translation is used by the scale estimator; nearest-neighbour rotation is fine.
        T_interp[:3, :3] = T_lo[:3, :3] if alpha < 0.5 else T_hi[:3, :3]
        T_interp[:3, 3] = T_lo[:3, 3] + alpha * (T_hi[:3, 3] - T_lo[:3, 3])
        return T_interp


def _apply_odom_metric_scale(solver, scale: float) -> None:
    """Apply metric scale to the first graph anchor, then re-optimize.

    Mirrors what solver.add_edge does when --metric is active:
      H[0:3, 0:3] *= scale_factor
    then re-pins the prior factor so the optimiser keeps the anchor fixed.

    Must be called with data_lock held (modifies graph values and factors).
    """
    first_key = 0  # submap_id=0, frame_id=0 → key 0 for the very first submap
    H = solver.graph.get_homography(first_key)
    if H is None:
        print("[OdomScale] First anchor node not yet in graph — skipping scale application.")
        return

    H_scaled = H.copy()
    H_scaled[0:3, 0:3] *= scale

    # Remove the old tight prior and re-add with the scaled value.
    solver.graph.remove_factor(solver.first_anchor_factor_idx)
    solver.graph.update_homography(first_key, H_scaled)
    solver.first_anchor_factor_idx = solver.graph.add_prior_factor(first_key, H_scaled)

    solver.graph.optimize()
    print(f"[OdomScale] Applied metric scale {scale:.4f} to first anchor; graph re-optimised.")


def _run_controller(follower: "WaypointFollower",
                    cmd_vel_node: "CmdVelNode",
                    odom_buf: "OdomBuffer",
                    odom_anchor: dict,
                    odom_anchor_lock: "threading.Lock",
                    stop_event: "threading.Event",
                    T_OC: "np.ndarray",
                    T_world_corr: "np.ndarray | None" = None) -> None:
    """Background thread: runs the waypoint follower at 10 Hz.

    Uses the odom-propagated, submap-corrected pose (same formula as the viser
    visualization) so the controller always benefits from the latest SLAM correction.
    Publishes zero velocity immediately when the follower becomes inactive.

    T_OC: the 4×4 camera_from_base transform used to conjugate odom motion into
    the camera frame (see _odom_delta_to_cam).

    T_world_corr: optional 4×4 rotation (no translation) that maps the VGGT world
    frame to a gravity-aligned frame where XZ is the actual floor.  Derived from
    T_OC via _tilt_correction_from_T_OC; is the identity for a level camera.
    """
    was_active = False
    while not stop_event.is_set():
        if follower.active:
            was_active = True
            with odom_anchor_lock:
                slam_pose = odom_anchor["slam_pose"]
                odom_T    = odom_anchor["odom_T"]

            if slam_pose is not None and odom_T is not None:
                latest = odom_buf.get_latest()
                if latest is not None:
                    _, T_odom_curr = latest
                    delta  = np.linalg.inv(odom_T) @ T_odom_curr
                    T_prop = slam_pose @ _odom_delta_to_cam(delta, T_OC)
                    if T_world_corr is not None:
                        T_prop = T_world_corr @ T_prop
                    linear_x, angular_z = follower.update(T_prop)
                    cmd_vel_node.publish_vel(linear_x, angular_z)
        elif was_active:
            # Follower just became inactive — send one final stop.
            cmd_vel_node.stop()
            was_active = False

        time.sleep(0.1)  # 10 Hz control loop


def _trim_path_stop_short(waypoints: np.ndarray, stop_short: float) -> np.ndarray:
    """Return waypoints with the last stop_short metres removed.

    Walks backward from the final waypoint, accumulating 3-D segment lengths,
    and replaces the trimmed tail with a single interpolated cut point.
    Returns the original array if stop_short <= 0 or the path is shorter than
    stop_short (the robot will just stop at the nearest reachable waypoint).
    """
    if stop_short <= 0.0 or len(waypoints) < 2:
        return waypoints
    remaining = stop_short
    for i in range(len(waypoints) - 1, 0, -1):
        seg = waypoints[i] - waypoints[i - 1]
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            continue
        if remaining <= seg_len:
            alpha = remaining / seg_len
            cut_point = waypoints[i] - alpha * seg
            return np.vstack([waypoints[:i], cut_point[np.newaxis]])
        remaining -= seg_len
    return waypoints  # path shorter than stop_short — keep as-is


def _update_odom_anchor(solver, odom_buf, odom_anchor, odom_anchor_lock,
                        last_frame_ts=None, debug=False) -> None:
    """After a submap is optimized, sync the last SLAM pose with the odom pose at
    that same instant.

    IMPORTANT: the odom baseline must be sampled at the timestamp of the submap's
    last camera keyframe (``last_frame_ts``), NOT at wall-clock "now".  Submap
    processing (VGGT + graph optimize) takes hundreds of ms to seconds, during
    which the robot keeps moving; anchoring to get_latest() would offset the
    forward-propagation by exactly that much and produce a jump each submap.
    """
    latest_sm = solver.map.get_latest_submap(ignore_loop_closure_submaps=True)
    if latest_sm is None:
        return
    poses = latest_sm.get_all_poses_world(solver.graph)
    if len(poses) == 0:
        return
    slam_pose = poses[-1].astype(np.float64)  # cam-to-world 4×4 at the last keyframe

    # Odom at the last keyframe's timestamp (time-consistent with slam_pose).
    odom_T = None
    if last_frame_ts is not None:
        odom_T = odom_buf.get_at_time(last_frame_ts)
    if odom_T is None:
        # Fallback: no timestamp / outside buffer window — use latest (old behaviour).
        result = odom_buf.get_latest()
        if result is None:
            return
        _, odom_T = result

    if debug and last_frame_ts is not None:
        latest = odom_buf.get_latest()
        if latest is not None:
            lts, T_now = latest
            moved = float(np.linalg.norm(T_now[:3, 3] - odom_T[:3, 3]))
            print(f"[OdomDebug] anchor latency dt={lts - last_frame_ts:+.3f}s; "
                  f"robot moved {moved:.3f} m during submap processing "
                  f"(this is the jump avoided by time-consistent anchoring).")

    with odom_anchor_lock:
        odom_anchor["slam_pose"] = slam_pose
        odom_anchor["odom_T"]   = odom_T.astype(np.float64)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="VGGT-SLAM ROS 2 image subscriber")
add_common_slam_args(parser)
# --- Frame-source-specific arguments (ROS 2 image topic + odometry) ---
parser.add_argument("--image_topic", type=str, default="/camera/image_raw",
                    help="ROS 2 topic to subscribe to (sensor_msgs/Image or CompressedImage)")
parser.add_argument("--compressed", action="store_true",
                    help="Subscribe to a sensor_msgs/CompressedImage topic instead of a raw Image")
parser.add_argument("--image_queue_size", type=int, default=4,
                    help="Max frames buffered between ROS subscriber and SLAM loop; "
                         "excess frames are dropped to prevent lag build-up")
parser.add_argument("--keyframe_folder", type=str, default="keyframes")
parser.add_argument("--use_odom_topic", action="store_true",
                    help="Subscribe to /tf and forward-propagate the latest SLAM pose using "
                         "odometry; displayed in viser at ≤10 Hz")
parser.add_argument("--odom_frame", type=str, default="odom",
                    help="TF parent frame ID for odometry (default: odom)")
parser.add_argument("--base_frame", type=str, default="base_link",
                    help="TF child frame ID for the robot base (default: base_link)")
parser.add_argument("--odom_scale", action="store_true",
                    help="Estimate metric scale from odometry path length vs camera path length. "
                         "Requires --use_odom_topic. Once the robot travels 5 m the scale is "
                         "applied to the first graph anchor and estimation stops.")
parser.add_argument("--odom_scale_min_travel", type=float, default=5.0,
                    help="Minimum odometry travel (metres) before scale is applied (default: 5.0)")
parser.add_argument("--pcd_scale_window_m", type=float, default=3.0,
                    help="Wheel-odom arc length (metres) of the rolling window used to "
                         "metric-scale each grasp point cloud at publish time. The scale is "
                         "applied ONLY to the outgoing pcd, never to the map. Requires "
                         "--use_odom_topic; ignored without it (default: 3.0).")
parser.add_argument("--no_pcd_odom_scale", action="store_true",
                    help="Disable the per-pcd wheel-odom metric scale correction; publish "
                         "grasp point clouds in raw VGGT units.")
parser.add_argument("--collect_calibration", action="store_true",
                    help="Log per-submap camera + odom poses to a text file for offline "
                         "hand-eye calibration (scripts/calibrate_test.py). Requires "
                         "--use_odom_topic. Logs only; does not require or apply --odom_scale.")
parser.add_argument("--calibration_log_path", type=str, default="pose_log.txt",
                    help="Output file for --collect_calibration pose log (default: pose_log.txt)")
parser.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel",
                    help="ROS 2 topic for velocity commands published by the waypoint follower")
parser.add_argument("--pickup_topic", type=str, default="/grasp_attempted",
                    help="ROS 2 topic (std_msgs/Bool) an external manipulator publishes on "
                         "for both pickup and drop-off confirmation: data=True once it has "
                         "picked up the object the Reset mission published, data=False once it "
                         "has set the object back down. The mission waits for the matching "
                         "value before moving on.")
parser.add_argument("--grasp_success_topic", type=str, default="/grasp_success",
                    help="ROS 2 topic (std_msgs/Bool) published together with --pickup_topic: "
                         "True if the grasp worked, False if it failed. On a failure the Reset "
                         "mission re-detects the object and republishes its cloud to retry.")
parser.add_argument("--object_pcd_topic", type=str, default="/object_pcd",
                    help="ROS 2 topic on which the 'Get Object PCD in Cam' button publishes "
                         "the object point cloud (sensor_msgs/PointCloud2, XYZRGB)")
parser.add_argument("--object_pcd_frame", type=str, default="camera_f_color_optical_frame",
                    help="frame_id stamped on the published object PointCloud2 (the camera "
                         "optical frame the cloud is expressed in)")
parser.add_argument("--place_obj_topic", type=str, default="place_obj_location",
                    help="ROS 2 topic (geometry_msgs/PointStamped) on which the viser 'Place "
                         "Object' button publishes the last queried object's bbox centroid, "
                         "expressed in the current camera frame")
parser.add_argument("--stop_short", type=float, default=0.45,
                    help="Stop this many metres before the end of the planned path (default: 0.45 m)")
parser.add_argument("--debug_odom", action="store_true",
                    help="Print odom/SLAM anchor timing diagnostics (submap processing "
                         "latency and how far the robot moved during it).")
parser.add_argument("--camera_from_base_yaml", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "camera_from_base.yaml"),
                    help="YAML with a hand-eye 'camera_from_base' matrix (from "
                         "scripts/calibrate_test.py). The full 4x4 transform is used as T_OC "
                         "to convert odom motion into the camera frame.")
parser.add_argument("--robot_urdf", type=str,
                    default="/home/dominator/Documents/mobile_aloha_sim/aloha_new_description/urdf/aloha_new.urdf",
                    help="Path to a robot URDF to visualize in viser at the robot's live pose. "
                         "package:// mesh refs resolve under the workspace root two levels above "
                         "the URDF. Set to '' to disable.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_keyframe(img: np.ndarray, folder: str, idx: int) -> str:
    filename = os.path.join(folder, f"frame_{idx:06d}.png")
    cv2.imwrite(filename, img)
    return filename


def run_predictions(image_names_subset, solver, model, objects_module, args,
                             odom_buf=None, odom_anchor=None, odom_anchor_lock=None,
                             odom_scale_estimator=None, frame_timestamps=None,
                             pcd_scale_window=None, forced_loop=None, max_loops=None):
    """Background thread: run model inference + graph optimisation for one submap.

    forced_loop, when given as (detected_submap_id, detected_frame), forces a loop
    closure from this submap's newest frame back to that frame (object capture).

    max_loops overrides args.max_loops for this submap (e.g. 0 to disable loop
    closure); defaults to args.max_loops when None.
    """
    max_loops = args.max_loops if max_loops is None else max_loops
    try:
        print(f"[SLAM] Processing submap ({len(image_names_subset)} frames)...")
        if args.use_da:
            predictions = solver.run_predictions_da(
                image_names_subset, model, max_loops=max_loops,
                clip_model=objects_module.clip_model,
                clip_preprocess=objects_module.clip_preprocess,
                forced_loop_detection=forced_loop,
            )
        else:
            predictions = solver.run_predictions(
                image_names_subset, model, max_loops,
                objects_module.clip_model, objects_module.clip_preprocess,
                forced_loop_detection=forced_loop,
            )
        _cam_poses_for_scale = None
        with data_lock:
            solver.add_points(predictions)
            solver.graph.optimize()
            if args.vis_map:
                solver.update_latest_submap_vis()
                if len(predictions.get("detected_loops", [])) > 0:
                    solver.update_all_submap_vis()
            _need_world_poses = (
                (odom_scale_estimator is not None and not odom_scale_estimator.scale_applied)
                or pcd_scale_window is not None)
            if _need_world_poses:
                _sm = solver.map.get_latest_submap(ignore_loop_closure_submaps=True)
                if _sm is not None:
                    _cam_poses_for_scale = _sm.get_all_poses_world(solver.graph)
        print("[SLAM] Submap done.")
        _last_frame_ts = (frame_timestamps[-1]
                          if frame_timestamps is not None and len(frame_timestamps) > 0
                          else None)
        if odom_buf is not None:
            _update_odom_anchor(solver, odom_buf, odom_anchor, odom_anchor_lock,
                                last_frame_ts=_last_frame_ts,
                                debug=getattr(args, "debug_odom", False))
        if (odom_scale_estimator is not None
                and not odom_scale_estimator.scale_applied
                and _cam_poses_for_scale is not None
                and len(_cam_poses_for_scale) > 1
                and frame_timestamps is not None
                and odom_buf is not None):
            odom_scale_estimator.add_submap_samples(
                _cam_poses_for_scale, frame_timestamps, odom_buf)
            # Only auto-apply the scale when --odom_scale is set; a collect-only
            # run (--collect_calibration alone) just logs and leaves the map intact.
            if args.odom_scale:
                scale = odom_scale_estimator.try_estimate()
                if scale is not None:
                    n = len(odom_scale_estimator._ratios)
                    print(f"[OdomScale] Estimated scale = {scale:.4f}  "
                          f"(odom {odom_scale_estimator.odom_travel_m():.2f} m, "
                          f"n={n} measurements)")
                    with data_lock:
                        _apply_odom_metric_scale(solver, scale)
                        if args.vis_map:
                            solver.update_all_submap_vis()
                    odom_scale_estimator.mark_applied()
                    # The map just jumped to a new (metric) scale.  Drop the rolling
                    # per-pcd window so it never mixes pre- and post-rescale camera
                    # poses; it refills from post-scale poses and thereafter yields
                    # only the residual local-drift correction.
                    if pcd_scale_window is not None:
                        pcd_scale_window.clear()
                    # The map was just rescaled — refresh the odom anchor so the
                    # forward-propagated pose uses the new metric slam_pose instead
                    # of the pre-scale one (otherwise a one-time jump appears right
                    # after calibration).
                    if odom_buf is not None:
                        _update_odom_anchor(solver, odom_buf, odom_anchor,
                                            odom_anchor_lock,
                                            last_frame_ts=_last_frame_ts,
                                            debug=getattr(args, "debug_odom", False))
        # Feed the rolling per-pcd scale window with this submap's frames (current
        # map poses paired with wheel odom).  Independent of the one-shot
        # calibrator above — this is only read when a grasp cloud is published.
        if (pcd_scale_window is not None
                and _cam_poses_for_scale is not None
                and len(_cam_poses_for_scale) > 0
                and frame_timestamps is not None
                and odom_buf is not None):
            pcd_scale_window.add_submap_samples(
                _cam_poses_for_scale, frame_timestamps, odom_buf)
        if solver.places is not None and solver.viewer is not None:
            with data_lock:
                classification = solver.compute_frontier()
                graph = solver.places.graph
                rotations = dict(solver.places.submap_plane_rotations)
                place_size = solver.places.place_size
            solver.viewer.show_frontier_tiles(classification, graph, rotations, place_size)
    except Exception as e:
        import traceback
        print(f"[SLAM ERROR] {e}")
        traceback.print_exc()
    finally:
        if solver_lock.locked():
            solver_lock.release()


def _submap_odom_kwargs(odom_buf, odom_anchor, odom_anchor_lock,
                        odom_scale_estimator, pcd_scale_window, timestamps_subset):
    """Assemble the odom-related kwargs for run_predictions.

    Empty when odometry is disabled, so run_predictions falls back to its
    non-odom defaults.
    """
    if odom_buf is None:
        return {}
    return {
        "odom_buf": odom_buf,
        "odom_anchor": odom_anchor,
        "odom_anchor_lock": odom_anchor_lock,
        "odom_scale_estimator": odom_scale_estimator,
        "pcd_scale_window": pcd_scale_window,
        "frame_timestamps": list(timestamps_subset),
    }


def force_submap_from_buffer(image_names_subset, timestamps_subset, solver, model,
                             objects_module, args, odom_buf, odom_anchor,
                             odom_anchor_lock, odom_scale_estimator,
                             pcd_scale_window=None, forced_loop=None):
    """Force the current keyframe buffer into a submap so a current camera pose
    exists, then return the trimmed (image_names_subset, timestamps_subset).

    Runs synchronously (a user-triggered action).  ``forced_loop``, when given as
    (submap_id, frame_index), forces a loop closure back to that frame — used by
    object capture to cut drift between when the object was seen and now.
    """
    if len(image_names_subset) < 2:
        print("[ForceSubmap] Not enough keyframes yet to form a submap.")
        return image_names_subset, timestamps_subset

    print(f"[ForceSubmap] Forcing a submap from {len(image_names_subset)} keyframes...")
    run_kwargs = _submap_odom_kwargs(odom_buf, odom_anchor, odom_anchor_lock,
                                     odom_scale_estimator, pcd_scale_window,
                                     timestamps_subset)
    solver_lock.acquire()  # blocking: serialise with any in-flight submap thread
    # run_predictions releases solver_lock in its own finally block.
    # max_loops=0 → this forced submap never triggers a loop closure.
    run_predictions(list(image_names_subset), solver, model, objects_module, args,
                    forced_loop=forced_loop, max_loops=0, **run_kwargs)

    return image_names_subset[-1:], timestamps_subset[-1:]


def _scale_object_cloud(pts_cam, pcd_scale_window):
    """Return ``pts_cam`` metric-scaled from the rolling wheel-odom window.

    ``pts_cam`` is (N,3) in the camera optical frame (camera-relative), so a single
    scalar multiply about the origin maps the current map units → metres for both
    the object's distance and its physical size.  The scale comes from
    :meth:`PcdScaleWindow.estimate_scale`, which returns None until a full window
    of wheel-odom travel is buffered — in that case the cloud is published
    unscaled with a warning.  Never touches the factor graph.

    This composes with the global ``--odom_scale`` calibration: the window reads
    the current (already globally-scaled) map poses, so its scale is the *residual*
    correction for however far VGGT-SLAM's local scale has drifted from metric
    since — ~1.0 when still metric.  Skipped only when the window is disabled.
    """
    if pcd_scale_window is None:
        return pts_cam
    scale = pcd_scale_window.estimate_scale()
    if scale is None:
        print(f"[PcdScale] < {pcd_scale_window._max_travel:.1f} m of wheel odom "
              f"buffered — publishing grasp cloud in raw VGGT units (uncorrected).")
        return pts_cam
    print(f"[PcdScale] Applied metric scale {scale:.4f} to grasp cloud "
          f"(from last {pcd_scale_window._max_travel:.1f} m of wheel odom).")
    return np.asarray(pts_cam, dtype=np.float64) * scale


def handle_object_pcd_request(image_names_subset, timestamps_subset, solver, model,
                              objects_module, args, odom_buf, odom_anchor,
                              odom_anchor_lock, odom_scale_estimator, object_pcd_node,
                              pcd_scale_window=None, use_cache=False):
    """Force the current keyframe buffer into a submap, then publish the last
    queried object's point cloud (in that submap's newest camera frame) over ROS.

    Runs synchronously (a user-triggered action) so the new submap — and hence
    the current camera pose — exists before the object cloud is transformed and
    published.  Returns the trimmed (image_names_subset, timestamps_subset) to
    continue capture from.

    When ``use_cache`` is False the object is freshly re-detected with SAM3 in the
    current frame (may fail if not in view); when True the CACHED object cloud is
    re-projected into the current camera frame instead (the Reset mission's
    last-resort fallback when live detection keeps failing).

    Sets ``pcd_publish_ok`` iff a cloud was actually published this call, so the
    caller can distinguish success from a failed live detection.
    """
    pcd_publish_ok.clear()
    if len(image_names_subset) < 2:
        print("[ObjectPCD] Not enough keyframes yet to form a submap.")
        return image_names_subset, timestamps_subset

    image_names_subset, timestamps_subset = force_submap_from_buffer(
        image_names_subset, timestamps_subset, solver, model, objects_module, args,
        odom_buf, odom_anchor, odom_anchor_lock, odom_scale_estimator,
        pcd_scale_window=pcd_scale_window)

    # Express the object's cloud in the current camera frame and publish it over
    # ROS.  Runs under data_lock since it reads the graph / submap geometry.
    #   use_cache=False : freshly detect in the current frame (NOT cached).
    #   use_cache=True  : re-project the cached cloud (no live detection).
    with data_lock:
        latest_sm = solver.map.get_latest_submap(ignore_loop_closure_submaps=True)
        if latest_sm is None:
            result = None
        elif use_cache:
            result = objects_module.last_object_in_camera_frame(solver, latest_sm)
        else:
            result = objects_module.detect_last_object_in_current_frame(solver, latest_sm)
    if result is not None:
        if use_cache:
            pts_cam, colors, label = result
        else:
            pts_cam, colors, label, _ = result
        # Metric-scale the outgoing cloud from the last window of wheel odometry —
        # a residual correction for local VGGT scale drift on top of the global
        # --odom_scale calibration.  Scales ONLY the published pcd, about the
        # camera origin (pts_cam are camera-relative), so both the object's
        # distance and its size are corrected.  The map is never touched.
        pts_cam = _scale_object_cloud(pts_cam, pcd_scale_window)
        # About to publish → the manipulator will reach in.  Pause SLAM first, but
        # only for a Reset-mission capture (the manual button must not pause SLAM).
        if mission_capturing.is_set():
            slam_paused.set()
        object_pcd_node.publish(pts_cam, colors)
        pcd_publish_ok.set()
        src = "cached" if use_cache else "live"
        print(f"[ObjectPCD] Published {len(pts_cam)} object points ('{label}', {src}) on "
              f"'{args.object_pcd_topic}' (frame '{args.object_pcd_frame}').")

    return image_names_subset, timestamps_subset


# ---------------------------------------------------------------------------
# ROS 2 nodes
# ---------------------------------------------------------------------------

class OdomNode(Node):
    """Subscribes to /tf and buffers the odom→base_link transform."""

    def __init__(self, odom_frame: str, base_frame: str, buf: OdomBuffer):
        super().__init__("vggt_slam_odom")
        self._odom_frame = odom_frame
        self._base_frame = base_frame
        self._buf = buf
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(TFMessage, "/tf", self._cb, qos)
        self.get_logger().info(f"Listening for {odom_frame}→{base_frame} on /tf")

    def _cb(self, msg: TFMessage) -> None:
        for tf in msg.transforms:
            if (tf.header.frame_id == self._odom_frame and
                    tf.child_frame_id == self._base_frame):
                t = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
                self._buf.push(t, _tf_to_mat4(tf.transform))


class CmdVelNode(Node):
    """Minimal ROS 2 node that publishes geometry_msgs/Twist velocity commands.

    Publishing does not require spinning — the DDS layer handles transport
    independently.  Call publish_vel() from any thread.
    """

    def __init__(self, topic: str = "/cmd_vel"):
        super().__init__("vggt_slam_controller")
        self._pub = self.create_publisher(Twist, topic, 10)
        self.get_logger().info(f"Velocity publisher ready on {topic}")

    def publish_vel(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x  = float(linear_x)
        msg.angular.z = float(angular_z)
        self._pub.publish(msg)

    def stop(self) -> None:
        self.publish_vel(0.0, 0.0)


class ConfirmationNode(Node):
    """Subscribes to one std_msgs/Bool topic carrying both manipulator acks.

    An external manipulation node publishes ``data=True`` once it has picked up
    the object the Reset mission published, and ``data=False`` once it has set the
    object back down.  Each routes to the matching shared event that unblocks the
    mission's wait; the value is logged.  Reliable QoS so the ack is never dropped.
    """

    def __init__(self, topic: str, pickup_event: "threading.Event",
                 dropoff_event: "threading.Event"):
        super().__init__("vggt_slam_confirm")
        self._pickup_event = pickup_event
        self._dropoff_event = dropoff_event
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Bool, topic, self._cb, qos)
        self.get_logger().info(f"Listening for pickup/drop-off confirmation on {topic}")

    def _cb(self, msg: Bool) -> None:
        action = "pickup" if msg.data else "drop-off"
        self.get_logger().info(f"{action} confirmation received (data={msg.data}).")
        (self._pickup_event if msg.data else self._dropoff_event).set()


class GraspSuccessNode(Node):
    """Tracks /grasp_success (std_msgs/Bool): True = the grasp worked.

    Published by the manipulator at the same time as the grasp-attempted ack, so
    the Reset mission can tell a successful grasp from one that needs a retry.
    """

    def __init__(self, topic: str, holder: "GraspSuccess"):
        super().__init__("vggt_slam_grasp_success")
        self._holder = holder
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Bool, topic, self._cb, qos)
        self.get_logger().info(f"Listening for grasp success on {topic}")

    def _cb(self, msg: Bool) -> None:
        self.get_logger().info(f"grasp success = {msg.data}")
        self._holder.set(msg.data)


class ObjectPcdNode(Node):
    """Publishes the captured object point cloud as a sensor_msgs/PointCloud2.

    Points are in the current camera's optical frame (X right, Y down, Z fwd).
    Publishing does not require spinning; call publish() from any thread.
    """

    def __init__(self, topic: str, frame_id: str):
        super().__init__("vggt_slam_object_pcd")
        self._pub = self.create_publisher(PointCloud2, topic, QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1))
        self._frame_id = frame_id
        self.get_logger().info(f"Object point-cloud publisher ready on {topic} "
                               f"(frame '{frame_id}')")

    def publish(self, points_xyz: np.ndarray, colors_rgb=None) -> None:
        """Publish (N,3) float points, optionally with (N,3) uint8 RGB colors."""
        n = len(points_xyz)
        xyz = np.asarray(points_xyz, dtype=np.float32)

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        if colors_rgb is not None and len(colors_rgb) == n:
            rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
            rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            c = np.asarray(colors_rgb, dtype=np.uint32)
            packed = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]     # 0x00RRGGBB
            rec["rgb"] = packed.view(np.float32)                    # reinterpret bits as float32
            fields.append(PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1))
            point_step = 16
        else:
            rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            point_step = 12

        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.height = 1
        msg.width = n
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * n
        msg.is_dense = True
        msg.data = rec.tobytes()
        self._pub.publish(msg)


class PlaceObjNode(Node):
    """Publishes a single 3-D point (geometry_msgs/PointStamped) for the viser
    'Place Object' button — the last queried object's bbox centroid expressed in
    the current camera frame.

    Latched (TRANSIENT_LOCAL, depth 1) so a manipulator that subscribes later
    still receives the most recent target.  Publishing needs no spinning; call
    publish() from any thread.
    """

    def __init__(self, topic: str, frame_id: str):
        super().__init__("vggt_slam_place_obj")
        self._pub = self.create_publisher(
            PointStamped, topic,
            QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1),
        )
        self._frame_id = frame_id
        self.get_logger().info(f"Place-object point publisher ready on {topic} "
                               f"(frame '{frame_id}')")

    def publish(self, point_xyz) -> None:
        """Publish an (x, y, z) point in the configured frame."""
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.point.x = float(point_xyz[0])
        msg.point.y = float(point_xyz[1])
        msg.point.z = float(point_xyz[2])
        self._pub.publish(msg)


class ImageSubscriberNode(Node):
    """Thin ROS 2 node: receives images and enqueues them for processing.

    The callbacks are kept as lightweight as possible — no decoding happens
    here.  For CompressedImage the raw JPEG/PNG bytes are copied (small) and
    the expensive cv2.imdecode is deferred to the main loop, so the executor
    thread is free to handle the next incoming message immediately.
    """

    def __init__(self, topic: str, frame_queue: "queue.Queue", compressed: bool = False):
        super().__init__("vggt_slam")
        self._bridge = CvBridge()
        self._queue = frame_queue
        self._compressed = compressed
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        if compressed:
            self.create_subscription(CompressedImage, topic, self._compressed_callback, qos)
            self.get_logger().info(f"Subscribed to compressed topic {topic} (best-effort QoS)")
        else:
            self.create_subscription(Image, topic, self._image_callback, qos)
            self.get_logger().info(f"Subscribed to {topic} (best-effort QoS)")

    def _enqueue(self, data, ts: float) -> None:
        # Drop frames while the arm is executing in front of the robot, so the SLAM
        # loop makes no keyframes/submaps until the arm has retracted (see slam_paused).
        if slam_paused.is_set():
            return
        try:
            self._queue.put_nowait((data, ts))
        except queue.Full:
            pass  # drop frame; SLAM is slower than the topic rate

    def _image_callback(self, msg: Image) -> None:
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge conversion failed: {e}")
            return
        self._enqueue(bgr, ts)

    def _compressed_callback(self, msg: CompressedImage) -> None:
        # Copy only the compact compressed bytes — no JPEG decode here.
        # cv2.imdecode runs in the main loop instead so this callback returns
        # in microseconds and never delays the next message from being received.
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._enqueue(bytes(msg.data), ts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Strip ROS-specific args (--ros-args, remappings, etc.) before argparse.
    rclpy.init()
    clean_argv = rclpy.utilities.remove_ros_args(sys.argv[1:])
    args = parser.parse_args(clean_argv)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

    # Check for valid arguments.
    if not validate_common_args(args):
        return
    if args.odom_scale and not args.use_odom_topic:
        print("Error: --odom_scale requires --use_odom_topic.")
        return
    if args.collect_calibration and not args.use_odom_topic:
        print("Error: --collect_calibration requires --use_odom_topic.")
        return

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.keyframe_folder, exist_ok=True)

    # Bounded queue between ROS subscriber callback and SLAM processing loop.
    # Frames that arrive while the queue is full are silently dropped so the
    # pipeline never lags behind the live feed.
    frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=args.image_queue_size)

    def _resume_slam_after_arm() -> None:
        # Re-enable SLAM after an arm action: clear the pause and drain any frames
        # left in the queue from before the pause (now stale — the robot sat still
        # through the whole grasp) so the loop restarts on fresh, post-retract frames.
        slam_paused.clear()
        try:
            while True:
                frame_queue.get_nowait()
        except queue.Empty:
            pass

    # Each node gets its own SingleThreadedExecutor on a dedicated thread, so
    # high-frequency odom/confirmation callbacks never compete with image reception.
    node = ImageSubscriberNode(args.image_topic, frame_queue, compressed=args.compressed)
    print(f"ROS 2 node spinning — waiting for images on '{args.image_topic}'...")

    odom_buf              = None
    odom_anchor           = {"slam_pose": None, "odom_T": None}
    odom_anchor_lock      = threading.Lock()
    last_odom_vis         = 0.0
    last_replan_check     = 0.0
    odom_scale_estimator  = None
    pcd_scale_window      = None

    # Image node gets its own executor so high-frequency /tf callbacks from the
    # odom node never compete with image reception.
    img_executor = SingleThreadedExecutor()
    img_executor.add_node(node)
    ros_thread = threading.Thread(target=img_executor.spin, daemon=True)
    ros_thread.start()

    if args.use_odom_topic:
        odom_buf = OdomBuffer()
        odom_node = OdomNode(args.odom_frame, args.base_frame, odom_buf)
        odom_executor = SingleThreadedExecutor()
        odom_executor.add_node(odom_node)
        odom_thread = threading.Thread(target=odom_executor.spin, daemon=True)
        odom_thread.start()
        print(f"[Odom] Spinning — waiting for /tf ({args.odom_frame}→{args.base_frame})")
        if not args.no_pcd_odom_scale:
            pcd_scale_window = PcdScaleWindow(max_travel_m=args.pcd_scale_window_m)
            print(f"[PcdScale] Grasp clouds will be metric-scaled from the last "
                  f"{args.pcd_scale_window_m:.1f} m of wheel odom (map left untouched).")
        if args.odom_scale or args.collect_calibration:
            odom_scale_estimator = OdomScaleEstimator(
                min_odom_travel_m=args.odom_scale_min_travel,
                log_path=(args.calibration_log_path if args.collect_calibration else None),
            )
            if args.odom_scale:
                print(f"[OdomScale] Scale estimator active — will apply once "
                      f"{args.odom_scale_min_travel:.1f} m of odom travel accumulated.")
            if args.collect_calibration:
                print(f"[Calib] Collecting calibration poses to "
                      f"'{args.calibration_log_path}'.")

    # Load models
    objects_module = build_objects_module(args)
    solver = build_solver(args)
    solver.register_object_cache(objects_module)
    model = build_model(args, device)
    print("All models loaded. Starting SLAM loop.")

    from foundit.measure_memory import rss_bytes, report_memory
    mem_baseline = rss_bytes() if args.report_memory else None

    if args.load_session:
        load_session(args.load_session, solver, objects_module)
        solver.mark_prior_session()
        solver.update_all_submap_vis()

    # from odom/base frame to camera
    T_OC = _load_T_OC_from_yaml(args.camera_from_base_yaml)

    # Compute world-frame tilt correction from T_OC.
    # If the camera is tilted, the VGGT world XZ plane is also tilted.
    # R_world_corr pre-multiplies world-frame poses so that XZ becomes the
    # actual floor — used by the waypoint follower and path preparation.
    R_world_corr = _tilt_correction_from_T_OC(T_OC)
    T_world_corr = np.eye(4, dtype=np.float64)
    T_world_corr[:3, :3] = R_world_corr

    # ---------------------------------------------------------------------------
    # Waypoint follower + /cmd_vel publisher
    # ---------------------------------------------------------------------------
    follower     = None
    cmd_vel_node = None
    if args.use_odom_topic:
        follower     = WaypointFollower()
        cmd_vel_node = CmdVelNode(args.cmd_vel_topic)
        # Single pickup/drop-off confirmation subscriber (used by the Reset
        # mission): True routes to pickup_event, False to dropoff_event.  Own
        # executor + thread, mirroring the odom node, so its callback never
        # competes with image reception.
        confirm_node = ConfirmationNode(args.pickup_topic, pickup_event, dropoff_event)
        grasp_success_node = GraspSuccessNode(args.grasp_success_topic, grasp_success)
        confirm_executor = SingleThreadedExecutor()
        confirm_executor.add_node(confirm_node)
        confirm_executor.add_node(grasp_success_node)
        threading.Thread(target=confirm_executor.spin, daemon=True).start()
        print(f"[Confirm] Listening for pickup/drop-off confirmation on '{args.pickup_topic}', "
              f"grasp success on '{args.grasp_success_topic}'")

    # Publisher for the object point cloud captured via the "Get Object PCD in Cam" button.
    object_pcd_node = ObjectPcdNode(args.object_pcd_topic, args.object_pcd_frame)

    # Publisher for the object centroid published via the "Place Object" button
    # (same current-camera frame as the object point cloud).
    place_obj_node = PlaceObjNode(args.place_obj_topic, args.object_pcd_frame)

    stop_event = threading.Event()
    if solver.viewer is not None:
        solver.viewer.add_cancel_button(stop_event)
        # Optional robot URDF visualization, repositioned to the live robot pose
        # in the main loop below.
        if args.robot_urdf and os.path.exists(args.robot_urdf):
            solver.viewer.add_robot_urdf(args.robot_urdf, base_frame_link=args.base_frame)
        elif args.robot_urdf:
            print(f"[Robot] URDF not found at '{args.robot_urdf}' — skipping robot visualization.")
        solver.viewer.add_object_query_gui(objects_module, solver, data_lock)
        solver.viewer.add_frontier_gui(solver, data_lock)
        solver.viewer.add_get_object_pcd_button(pcd_request.set)
        if odom_buf is not None:
            # Plan paths from the odom-forward-propagated current pose, not the
            # last submap's pose.
            def _planning_start():
                T = _propagated_camera_pose(odom_anchor, odom_anchor_lock, odom_buf, T_OC)
                return None if T is None else T[:3, 3]
            solver.viewer.set_planning_start_provider(_planning_start)

            def _place_detection(det):
                # Publish det's bbox centroid as a drop-off point in the current
                # camera frame. Reuses the propagated current camera pose (same one
                # the object-PCD path expresses points relative to).  Returns
                # (label, p_cam) on success, or (None, reason) on failure.
                with data_lock:
                    center_world = (np.asarray(det.bbox.center, dtype=np.float64)
                                    if det is not None and det.bbox is not None else None)
                    label = det.bbox.label if (det is not None and det.bbox is not None) else None
                if center_world is None:
                    return None, "no object / bbox available"
                T_cam_to_world = _propagated_camera_pose(odom_anchor, odom_anchor_lock,
                                                         odom_buf, T_OC)
                if T_cam_to_world is None:
                    return None, "no current robot pose yet (need odometry)"
                # Raise the centroid PLACE_OBJ_RAISE_M along the true vertical. The
                # camera is tilted, so use R_world_corr (= _tilt_correction_from_T_OC)
                # to enter the gravity-aligned frame where up is -Y, lift there, then
                # map back — a genuinely vertical raise despite the tilt.
                center_corr = R_world_corr @ center_world
                center_corr[1] -= PLACE_OBJ_RAISE_M            # up is -Y
                center_world = R_world_corr.T @ center_corr
                # World centroid → current camera frame: p_cam = inv(cam_to_world) @ p_world.
                p_h = np.linalg.inv(T_cam_to_world) @ np.append(center_world, 1.0)
                p_cam = p_h[:3] / p_h[3]
                place_obj_node.publish(p_cam)
                label = label or "object"
                print(f"[PlaceObj] Published '{label}' centroid at camera-frame "
                      f"({p_cam[0]:.3f}, {p_cam[1]:.3f}, {p_cam[2]:.3f}) on "
                      f"'{args.place_obj_topic}'.")
                return label, p_cam

            def _place_object(set_status) -> None:
                # Viser button: place the last queried object.
                label, info = _place_detection(objects_module.last_detected_object)
                if label is None:
                    set_status(f"*{info}.*")
                    return
                p_cam = info
                set_status(f"*Published '{label}' at cam "
                           f"({p_cam[0]:.2f}, {p_cam[1]:.2f}, {p_cam[2]:.2f}) m.*")
            solver.viewer.add_place_object_button(_place_object)
        if args.places:
            solver.viewer.add_room_query_gui(objects_module, solver, data_lock,
                                             top_region=args.top_region)
        if follower is not None:
            _viewer = solver.viewer
            # State for auto-replan: the object being followed and the bbox centre
            # the current path was planned to. Updated on (re)plan, cleared on stop.
            follow_state = {"det": None, "center": None}
            def _start_following(wps, det=None, track_object=True, _follower=follower,
                                 _stop_short=args.stop_short, _viewer=_viewer,
                                 _R_corr=R_world_corr):
                # Path from the viewer is object→camera order; reverse to camera→object.
                ordered = wps[::-1]
                # Trim in VGGT world frame so the viser spheres match the path length.
                trimmed = _trim_path_stop_short(ordered, _stop_short)
                path_length = (float(np.sum(np.linalg.norm(np.diff(trimmed, axis=0), axis=1)))
                               if len(trimmed) > 1 else 0.0)
                print(f"[FollowPath] {len(trimmed)} waypoints, length {path_length:.2f} m, "
                      f"goal {np.round(trimmed[-1], 3)}.")
                if _viewer is not None:
                    # Show waypoints in the VGGT world frame so they align with the
                    # places graph; the controller uses a corrected frame separately.
                    _viewer.set_waypoint_spheres(trimmed)
                    # Blue sphere at the actual stop point (path end after the
                    # stop-short trim) — where the robot will halt short of the object.
                    _viewer.set_stop_point_sphere(trimmed[-1])
                    _follower.set_on_waypoint_reached(
                        lambda idx, v=_viewer: v.mark_waypoint_reached(idx))
                # Follower receives tilt-corrected waypoints matching the corrected
                # T_prop it gets from _run_controller (same gravity-aligned frame).
                _follower.set_waypoints((_R_corr @ trimmed.T).T)
                # ordered[0] = camera end, ordered[-1] = object floor projection.
                # Tilt-correct and pass as the face target so the robot turns toward
                # the object after arriving.
                _follower.set_face_target(_R_corr @ ordered[-1])
                _follower.start()
                # Remember the target object + the centre this path was planned to so
                # the capture loop can replan if a loop closure later moves the object.
                # On the initial follow, det is the latest detection; on a replan the
                # caller passes the object already being followed so it stays fixed.
                # track_object=False (e.g. Go to Room) disables loop-closure replan —
                # the target is a place, not a movable object.
                if not track_object:
                    follow_state["det"] = None
                    follow_state["center"] = None
                    return
                if det is None:
                    det = objects_module.last_detected_object
                follow_state["det"] = det
                follow_state["center"] = (np.asarray(det.bbox.center, dtype=float).copy()
                                          if det is not None and det.bbox is not None
                                          else None)
            def _stop_following(_follower=follower, _node=cmd_vel_node, _viewer=_viewer):
                # Also abort a running Reset mission — Stop Following halts every
                # form of autonomous driving, not just a manual follow.
                reset_stop.set()
                _follower.stop()
                _node.stop()
                follow_state["det"] = None
                follow_state["center"] = None
                if _viewer is not None:
                    _viewer.clear_waypoint_spheres()
            def _replan_follow(_follower=follower):
                # If the followed object moved (loop closure), replan from the robot's
                # current pose to the object's new location and resume following.
                det = follow_state["det"]
                if det is None or det.bbox is None or not _follower.active:
                    return
                cur_center = np.asarray(det.bbox.center, dtype=float)
                planned = follow_state["center"]
                if planned is not None and \
                        np.linalg.norm(cur_center - planned) < OBJECT_REPLAN_DIST:
                    return  # object hasn't moved enough to warrant replanning
                start = _propagated_camera_pose(odom_anchor, odom_anchor_lock,
                                                odom_buf, T_OC)
                if start is None:
                    return

                with data_lock:
                    prev_wps = solver.viewer.get_planned_path_points()
                    if prev_wps is not None:
                        prev_wps = prev_wps.copy()
                    objects_module.plan_path(start[:3, 3], cur_center, solver)
                    wps = solver.viewer.get_planned_path_points()

                # An equivalent or invalid path means nothing to resume; record the
                # centre so we don't retry every tick until the object moves again.
                same_path = (
                    prev_wps is not None and wps is not None
                    and prev_wps.shape == wps.shape
                    and np.allclose(prev_wps, wps, rtol=0.0, atol=1e-4)
                )
                if same_path or wps is None or len(wps) < 2:
                    follow_state["center"] = cur_center.copy()
                    return

                moved = (float(np.linalg.norm(cur_center - planned))
                         if planned is not None else float("nan"))
                # Keep following the same object, not whatever the latest detection is.
                _start_following(wps, det=det)  # re-trims/tilt-corrects, restarts, records centre
                print(f"[Follow] Object moved {moved:.2f} m (loop closure) — replanned path.")
            solver.viewer.add_follow_path_button(
                start_cb=_start_following,
                stop_cb=_stop_following,
            )

            if args.places:
                def _go_to_room(set_status) -> None:
                    # Navigate to the last queried region: route to the nearest safe
                    # node to the region centre (find_shortest_path snaps the end),
                    # then follow. Not object-tracked (a room is a place, not movable).
                    center = solver.viewer.get_last_room_center()
                    if center is None:
                        set_status("*Query a room first (Room Query panel).*")
                        return
                    start = _propagated_camera_pose(odom_anchor, odom_anchor_lock,
                                                    odom_buf, T_OC)
                    if start is None:
                        set_status("*No current robot pose yet (need odometry).*")
                        return
                    with data_lock:
                        planned = objects_module.plan_path_to_point(start[:3, 3], center, solver)
                        wps = solver.viewer.get_planned_path_points() if planned else None
                    if not planned or wps is None or len(wps) < 2:
                        set_status("*No path to the room found.*")
                        return
                    _start_following(wps, track_object=False)
                    set_status("*Navigating to room.*")
                solver.viewer.add_go_to_room_button(_go_to_room)

            # --- Reset Scene mission primitives (injected into ResetMission) ---
            # Last object handed to _mission_capture, so a failed grasp can re-capture it.
            capture_state = {"det": None}

            def _mission_navigate_to(det) -> bool:
                # Plan from the robot's current pose to the object, follow, and
                # block until the follower finishes (arrived + turned to face).
                start = _propagated_camera_pose(odom_anchor, odom_anchor_lock,
                                                odom_buf, T_OC)
                if start is None or det.bbox is None:
                    return False
                with data_lock:
                    objects_module.plan_path(start[:3, 3], det.bbox.center, solver)
                    wps = solver.viewer.get_planned_path_points()
                if wps is None or len(wps) < 2:
                    return False
                _start_following(wps, det=det)
                # Loop closures moving the target replan the path here, exactly as
                # in manual following (main loop's _replan_follow keeps running).
                while (follower.active and not stop_event.is_set()
                       and not reset_stop.is_set()):
                    time.sleep(0.2)
                # follower.active also drops when Stop Following is pressed, so
                # treat an abort signal as "did not arrive".
                return not (stop_event.is_set() or reset_stop.is_set())

            def _mission_force_submap(loop_target=None) -> None:
                # Ask the capture loop to force a fresh submap at the current pose.
                # loop_target=(submap_id, frame_index) additionally forces a loop
                # closure back to that frame (e.g. the object's query frame).
                force_submap_loop["target"] = loop_target
                force_submap_done.clear()
                force_submap_request.set()
                force_submap_done.wait(timeout=60.0)

            def _mission_align_to_object(det) -> bool:
                # Run once on reaching an object (grasp or place) before acting on
                # it.  (1) Force a loop closure between the current frame and the
                # frame this object was first seen in (its query frame from the
                # cache) to correct local scale/drift; (2) re-measure the horizontal
                # distance to the object in that re-aligned map; (3) drive straight
                # forward/back by the difference so we sit at the target standoff
                # (--stop_short).  Returns False only on abort.
                aborted = lambda: stop_event.is_set() or reset_stop.is_set()
                if det is None or det.bbox is None:
                    return True

                # (1) Loop-close to the object's query frame, if we know it.
                loop_target = None
                if det.submap_id is not None and det.frame_index is not None:
                    loop_target = (det.submap_id, det.frame_index)
                _mission_force_submap(loop_target)
                if aborted():
                    return False

                # (2) Re-measure our distance to the object.  Refresh the cached
                # object positions with the just-updated graph, then take the
                # horizontal (XZ) gap from the current camera to the bbox center.
                with data_lock:
                    objects_module.move_objects(solver.graph)
                    sm = solver.map.get_latest_submap(ignore_loop_closure_submaps=True)
                    if sm is None:
                        return True
                    poses = sm.get_all_poses_world(solver.graph)
                    fi = sm.get_last_non_loop_frame_index()
                    if len(poses) == 0 or fi >= len(poses):
                        return True
                    cam_pos = np.asarray(poses[fi], dtype=np.float64)[:3, 3]
                    obj_c = np.asarray(det.bbox.center, dtype=np.float64)
                dist = float(np.hypot(obj_c[0] - cam_pos[0], obj_c[2] - cam_pos[2]))
                delta = dist - args.stop_short   # >0 too far → fwd; <0 too close → back
                label = det.bbox.label or "object"
                print(f"[Align] '{label}': distance {dist:.2f} m, target "
                      f"{args.stop_short:.2f} m → adjust {delta:+.2f} m.")

                # (3) Drive straight (no rotation) by the signed correction, clamped
                # for safety.  Below the deadband we're already close enough.
                if abs(delta) < ALIGN_STANDOFF_DEADBAND:
                    return True
                if abs(delta) > ALIGN_MAX_ADJUST:
                    print(f"[Align] correction {delta:+.2f} m exceeds "
                          f"{ALIGN_MAX_ADJUST:.2f} m cap — clamping.")
                    delta = ALIGN_MAX_ADJUST if delta > 0 else -ALIGN_MAX_ADJUST
                if delta > 0:
                    follower.drive_forward(delta)
                else:
                    follower.back_up(-delta, recenter=False)
                while follower.active and not aborted():
                    time.sleep(0.2)
                return not aborted()

            def _mission_capture(det, align=True) -> None:
                # Publish this object's cloud for the manipulator.  If SAM3 cannot
                # re-detect the object live in the current frame, back up 5 cm and
                # retry up to 3 times; if every live attempt fails, drive back to the
                # object and publish the CACHED cloud instead.  SLAM is paused (inside
                # the capture handler) right before any actual publish, since the arm
                # then reaches into the robot's view.
                #
                # align=True (on first reaching the object) force-closes a loop to
                # the object's query frame and corrects the standoff distance before
                # detecting; the grasp-retry passes align=False (already in place).
                aborted = lambda: stop_event.is_set() or reset_stop.is_set()
                capture_state["det"] = det   # remember, so a failed grasp can re-capture
                if align and not _mission_align_to_object(det):
                    return
                mission_capturing.set()
                try:
                    for attempt in range(4):   # 1 initial + 3 back-up-and-retry
                        if aborted():
                            return
                        objects_module.last_detected_object = det
                        pickup_event.clear()     # fresh grasp-ack window for this object
                        grasp_success.reset()    # ... and a fresh grasp-success value
                        pcd_publish_ok.clear()
                        pcd_done.clear()
                        pcd_request.set()
                        pcd_done.wait(timeout=60.0)
                        if pcd_publish_ok.is_set():
                            return             # live cloud published (SLAM already paused)
                        if attempt < 3 and not aborted():
                            print(f"[ObjectPCD] object not detected (attempt {attempt + 1}/4) — "
                                  f"backing up and retrying.")
                            _mission_back_up(0.02, recenter=False)   # straight back, no rotation

                    # Every live attempt failed → return to the object, publish cache.
                    if aborted():
                        return
                    print("[ObjectPCD] live detection failed after retries — returning to "
                          "the object and publishing the cached cloud.")
                    if not _mission_navigate_to(det):
                        return
                    objects_module.last_detected_object = det
                    pickup_event.clear()
                    grasp_success.reset()
                    pcd_publish_ok.clear()
                    pcd_done.clear()
                    cached_pcd_request.set()
                    pcd_done.wait(timeout=60.0)
                finally:
                    mission_capturing.clear()
            def _grasp_succeeded() -> bool:
                # /grasp_success is published together with the grasp-attempted ack;
                # give it a brief window to arrive.  If none ever comes (topic not
                # published), fall back to treating the pickup as a success — the
                # pre-grasp-feedback behaviour.
                deadline = time.monotonic() + GRASP_SUCCESS_TIMEOUT
                while time.monotonic() < deadline:
                    v = grasp_success.get()
                    if v is not None:
                        return v
                    time.sleep(0.02)
                print("[Grasp] no /grasp_success received — assuming success.")
                return True

            def _mission_wait_for_pickup() -> bool:
                # Wait for the arm to report a grasp attempt (pickup_event), staying
                # abort-responsive.  Once the arm has attempted (and retracted) resume
                # SLAM; if /grasp_success is True the mission continues, otherwise the
                # grasp failed — re-observe the object, republish its cloud, and give
                # it exactly ONE more try before moving on.
                for attempt in range(2):                  # initial grasp + one retry
                    got_attempt = False
                    while not (stop_event.is_set() or reset_stop.is_set()):
                        if pickup_event.wait(timeout=0.2):
                            got_attempt = True
                            break
                    if not got_attempt:
                        _resume_slam_after_arm()
                        return False                      # aborted while waiting
                    # Arm attempted and retracted → resume SLAM so we can re-observe.
                    _resume_slam_after_arm()
                    if _grasp_succeeded():
                        return True                       # grasp worked — continue
                    if stop_event.is_set() or reset_stop.is_set():
                        return False
                    if attempt == 0:
                        print("[Grasp] grasp reported failure — re-detecting the object "
                              "and republishing its cloud to retry once.")
                        # align=False: already at the object, no need to re-close a loop.
                        _mission_capture(capture_state["det"], align=False)
                # Retry also failed — give up on this object and continue the mission.
                print("[Grasp] retry also failed — moving on.")
                return True

            def _mission_place(det) -> None:
                # Re-align to the (original) location before placing, same as grasp:
                # force-close a loop to its query frame and correct the standoff.
                if not _mission_align_to_object(det):
                    return
                # Publish the drop-off point for det in the current camera frame.
                # Pause SLAM first (arm about to reach in), then open a fresh dropoff
                # window so a stale ack can't satisfy it.
                slam_paused.set()
                dropoff_event.clear()
                _place_detection(det)

            def _mission_wait_for_dropoff() -> bool:
                # Block until the drop-off ack arrives (same pattern as pickup),
                # resuming SLAM on every exit path.
                try:
                    while not (stop_event.is_set() or reset_stop.is_set()):
                        if dropoff_event.wait(timeout=0.2):
                            return True
                    return False
                finally:
                    _resume_slam_after_arm()

            def _mission_back_up(dist=0.4, recenter=True) -> bool:
                # Reverse straight out of the object and block until the follower
                # finishes.  recenter=True first squares up on the object (leave-object
                # maneuver); recenter=False reverses immediately with no rotation
                # (used for the capture retry, which must not turn).
                follower.back_up(dist, recenter=recenter)
                while (follower.active and not stop_event.is_set()
                       and not reset_stop.is_set()):
                    time.sleep(0.2)
                return not (stop_event.is_set() or reset_stop.is_set())

            def _launch_reset(set_status) -> None:
                if solver.places is None:
                    set_status("*Reset needs the places layer — run with --places.*")
                    return

                def _run() -> None:
                    reset_stop.clear()  # discard any stale abort from a prior run
                    try:
                        ResetMission(
                            objects_module, solver,
                            navigate_to=_mission_navigate_to,
                            refresh_view=_mission_force_submap,
                            capture_object=_mission_capture,
                            place_object=_mission_place,
                            back_up=_mission_back_up,
                            with_data_lock=data_lock,
                            wait_for_pickup=_mission_wait_for_pickup,
                            wait_for_dropoff=_mission_wait_for_dropoff,
                            set_status=set_status,
                            should_abort=lambda: stop_event.is_set() or reset_stop.is_set(),
                        ).run()
                    except Exception as e:  # noqa: BLE001 — report and keep the UI alive
                        import traceback
                        traceback.print_exc()
                        set_status(f"*Reset error: {e}*")
                    finally:
                        _stop_following()  # clear waypoint spheres, halt the robot

                threading.Thread(target=_run, daemon=True).start()

            solver.viewer.add_reset_scene_button(_launch_reset)

    # Start controller thread now that stop_event exists.
    if follower is not None:
        ctrl_thread = threading.Thread(
            target=_run_controller,
            args=(follower, cmd_vel_node, odom_buf, odom_anchor,
                  odom_anchor_lock, stop_event, T_OC),
            kwargs={"T_world_corr": T_world_corr},
            daemon=True,
        )
        ctrl_thread.start()

    frame_count = 0
    image_names_subset: list[str] = []
    timestamps_subset: list[float] = []
    target_size = args.submap_size + 1 # plus one since there is one overlapping frame
    submap_count = 0
    last_status_frame = 0
    _loop_start = time.time()
    _no_frame_warned = False

    def _append_current_keyframe() -> None:
        # Save the live frame to disk and add it to the current submap buffer.
        image_names_subset.append(save_keyframe(img, args.keyframe_folder, frame_count))
        timestamps_subset.append(ts)

    try:
        while True:
            if stop_event.is_set():
                print("\n[Main] Session cancelled via viser.")
                break

            try:
                data, ts = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if not _no_frame_warned and frame_count == 0 and time.time() - _loop_start > 8.0:
                    _no_frame_warned = True
                    rmw = os.environ.get("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp (default)")
                    print(
                        f"\n[WARN] No frames received after 8 s on '{args.image_topic}'.\n"
                        f"  Active RMW : {rmw}\n"
                        f"  If the camera publishes via FastDDS, set:\n"
                        f"    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp\n"
                        f"  CycloneDDS cross-vendor CDR errors cause silent message drops.\n"
                        f"  Continuing to wait...\n"
                    )
                continue  # no frame yet; keep waiting

            # Decode JPEG here (not in the callback) so the ROS executor thread
            # is free to receive the next message as soon as possible.
            if args.compressed:
                buf = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
            else:
                img = data

            frame_count += 1

            added_keyframe = False
            if solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow):
                _append_current_keyframe()
                added_keyframe = True

            # User asked (via viser) for the last object's cloud in the current
            # camera frame: make the current frame the newest keyframe (if it was
            # not already added this iteration), then force a submap and save.
            if pcd_request.is_set() or cached_pcd_request.is_set():
                use_cache = cached_pcd_request.is_set()   # cached-cloud fallback
                pcd_request.clear()
                cached_pcd_request.clear()
                if not added_keyframe:
                    _append_current_keyframe()
                    added_keyframe = True
                image_names_subset, timestamps_subset = handle_object_pcd_request(
                    image_names_subset, timestamps_subset, solver, model,
                    objects_module, args, odom_buf, odom_anchor,
                    odom_anchor_lock, odom_scale_estimator, object_pcd_node,
                    pcd_scale_window=pcd_scale_window, use_cache=use_cache)
                pcd_done.set()

            # Reset mission asked for a fresh submap at the robot's current pose so
            # check_missing_object sees the current view.
            if force_submap_request.is_set():
                force_submap_request.clear()
                _loop = force_submap_loop["target"]   # optional loop-closure target
                force_submap_loop["target"] = None
                if not added_keyframe:
                    _append_current_keyframe()
                    added_keyframe = True
                image_names_subset, timestamps_subset = force_submap_from_buffer(
                    image_names_subset, timestamps_subset, solver, model,
                    objects_module, args, odom_buf, odom_anchor,
                    odom_anchor_lock, odom_scale_estimator,
                    pcd_scale_window=pcd_scale_window, forced_loop=_loop)
                force_submap_done.set()

            if len(image_names_subset) >= target_size:
                if solver_lock.acquire(blocking=False):
                    submap_count += 1
                    print(f"[Main] Launching submap {submap_count} (frame {frame_count})...")
                    thread_kwargs = _submap_odom_kwargs(
                        odom_buf, odom_anchor, odom_anchor_lock,
                        odom_scale_estimator, pcd_scale_window, timestamps_subset)
                    t = threading.Thread(
                        target=run_predictions,
                        args=(list(image_names_subset), solver, model, objects_module, args),
                        kwargs=thread_kwargs,
                        daemon=True,
                    )
                    t.start()
                    image_names_subset = image_names_subset[-1:]
                    timestamps_subset  = timestamps_subset[-1:]
                else:
                    if len(image_names_subset) > target_size * 2:
                        n_dropped = len(image_names_subset) - target_size
                        print(f"[Main] SLAM busy — dropping {n_dropped} keyframe(s) "
                              f"(buffer {len(image_names_subset)} > {target_size * 2}); "
                              f"keeping newest {target_size}.")
                        image_names_subset = image_names_subset[-target_size:]
                        timestamps_subset  = timestamps_subset[-target_size:]

            if frame_count - last_status_frame >= 30:
                busy_str = "  [SLAM running]" if solver_lock.locked() else ""
                print(f"[ROS] frame={frame_count}  KFs={len(image_names_subset)}/{target_size}"
                      f"  submaps={submap_count}{busy_str}")
                last_status_frame = frame_count

            # Forward-propagate SLAM pose with latest odom and display in viser (≤10 Hz)
            if odom_buf is not None and solver.viewer is not None:
                _now = time.time()
                if _now - last_odom_vis >= 0.1:
                    _T_prop = _propagated_camera_pose(odom_anchor, odom_anchor_lock, odom_buf, T_OC)
                    if _T_prop is not None:
                        solver.viewer.server.scene.add_frame(
                            "odom_propagated_pose",
                            wxyz=viser_tf.SO3.from_matrix(_T_prop[:3, :3]).wxyz,
                            position=_T_prop[:3, 3],
                            axes_length=0.15,
                            axes_radius=0.008,
                            origin_radius=0.01,
                        )
                        # Place the robot URDF at the base pose: base→world is the
                        # camera→world pose composed with T_OC (camera_from_base).
                        if T_OC is not None:
                            solver.viewer.set_robot_pose(_T_prop @ T_OC)
                    last_odom_vis = _now

            # Replan the followed path if a loop closure moved the target object.
            # Runs for both manual following and the Reset mission (which drives via
            # the same _start_following/follow_state), so the path and target stay
            # updated across loop closures in either mode.
            if follower is not None and time.time() - last_replan_check >= 0.5:
                last_replan_check = time.time()
                _replan_follow()

    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
    finally:
        rclpy.shutdown()

    if not args.vis_map:
        solver.update_all_submap_vis()

    if args.save_session:
        save_session(solver, args.save_session, objects_module)

    if args.rooms:
        from foundit.places import FiltrationClusterer, FiltrationClusterConfig
        config = FiltrationClusterConfig()
        result = FiltrationClusterer(config).cluster(solver.places.graph)
        solver.viewer.visualize_clusters(result, solver.places)
        def recluster(cfg: FiltrationClusterConfig):
            return FiltrationClusterer(cfg).cluster(solver.places.graph)
        solver.viewer.add_clustering_gui(config, recluster, solver.places)

    if args.report_memory:
        report_memory(solver, objects_module, mem_baseline)

    run_interactive_loop(args, objects_module, solver)

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)
        if not args.skip_dense_log:
            solver.map.write_points_to_file(
                solver.graph, args.log_path.replace(".txt", "_points.pcd")
            )


if __name__ == "__main__":
    main()
