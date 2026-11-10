import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Default output shared with main_ros2.py (repo root / camera_from_base.yaml).
CAMERA_FROM_BASE_YAML = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "camera_from_base.yaml"))

# ---------------------------------------------------------
# 1. Math and Transformation Utilities
# ---------------------------------------------------------

def orthonormalize(R):
    """Project a near-rotation matrix onto SO(3) (VGGT/decomposed R can drift)."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


# Camera height above the robot base (odom Z) — locked, since planar motion makes
# this vertical offset unobservable from the data.
LOCK_CAM_HEIGHT = 1.0


def build_X_from_mount(theta):
    """Build the hand-eye matrix X (= camera_from_base) from the state vector.

    We parametrize the *mount* M = base_from_camera (the camera's pose in the odom
    base frame) so its vertical component can be locked, then invert to get X.

    theta = [rx, ry, rz, px, py, scale]
      (rx, ry, rz) : rotation of M as a rotation vector (full SO(3), 3 DOF)
      (px, py)     : horizontal camera position in the base frame (metres)
                     (pz is fixed to LOCK_CAM_HEIGHT)
      scale        : camera-trajectory scale (handled separately in the residual)
    """
    M = np.eye(4)
    M[:3, :3] = Rotation.from_rotvec(theta[0:3]).as_matrix()
    M[:3, 3] = [theta[3], theta[4], LOCK_CAM_HEIGHT]
    return np.linalg.inv(M)


def compute_relative_motions(absolute_poses):
    """Convert absolute world poses into relative (frame-to-frame) motions."""
    relative_motions = []
    for i in range(len(absolute_poses) - 1):
        T_1 = absolute_poses[i]
        T_2 = absolute_poses[i + 1]
        relative_motions.append(np.linalg.inv(T_1) @ T_2)
    return relative_motions


def dominant_rotation_axis(motions):
    """Angle-weighted mean rotation axis of a list of relative motions."""
    acc = np.zeros(3)
    for M in motions:
        acc += Rotation.from_matrix(orthonormalize(M[:3, :3])).as_rotvec()  # |rotvec| == angle
    n = np.linalg.norm(acc)
    return acc / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def rotation_between(a, b):
    """Rotation matrix that maps unit vector a onto unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = np.cross(a, perp)
        return Rotation.from_rotvec(np.pi * axis / np.linalg.norm(axis)).as_matrix()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def umeyama(P, Q):
    """Similarity transform (scale s, rotation R, translation t) with Q ~= s*R@P + t.

    P, Q are (N, 3) point sets in correspondence.  Returns (s, R, t).
    """
    muP, muQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - muP, Q - muQ
    C = (Qc.T @ Pc) / len(P)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    varP = (Pc ** 2).sum() / len(P)
    s = float(np.trace(np.diag(D) @ S) / varP)
    t = muQ - s * R @ muP
    return s, R, t


# ---------------------------------------------------------
# 2. The Optimizer Core  (solve  s*A X = X B)
# ---------------------------------------------------------

def hand_eye_residuals_with_scale(theta, A_motions, B_motions, W_rot=50.0):
    """Residuals for hand-eye AX = XB with a camera scale factor.

    A = camera relative motion, B = odometry relative motion, X = camera_from_base.
    Translation part (scaled): (R_A - I) t_X + s * t_A - R_X t_B = 0
    Rotation part:             R_A R_X = R_X R_B
    """
    X = build_X_from_mount(theta)
    R_X, t_X = X[:3, :3], X[:3, 3]
    s = theta[5]

    residuals = []
    for A, B in zip(A_motions, B_motions):
        R_A, t_A = A[:3, :3], A[:3, 3]
        R_B, t_B = B[:3, :3], B[:3, 3]

        e_t = (R_A - np.eye(3)) @ t_X + (s * t_A) - (R_X @ t_B)

        R_err = (R_X @ R_B).T @ (R_A @ R_X)
        e_R = Rotation.from_matrix(orthonormalize(R_err)).as_rotvec()

        residuals.extend(e_t)
        residuals.extend(W_rot * e_R)
    return np.array(residuals)


def calibrate_hand_eye_with_scale(camera_absolute_poses, odom_absolute_poses,
                                  min_cam_step=0.003, min_odom_step=0.02):
    A_all = compute_relative_motions(camera_absolute_poses)
    B_all = compute_relative_motions(odom_absolute_poses)

    # --- Drop degenerate pairs --------------------------------------------
    # The odometry is logged at a lower rate than the camera keyframes, so many
    # consecutive frames share the same (held) odom pose -> zero odom motion while
    # the camera moves.  These pairs carry no hand-eye information and would drag
    # the scale toward zero, so keep only pairs where BOTH sensors actually moved.
    dA = np.array([np.linalg.norm(A[:3, 3]) for A in A_all])
    dB = np.array([np.linalg.norm(B[:3, 3]) for B in B_all])
    keep = (dA > min_cam_step) & (dB > min_odom_step)
    A_motions = [A_all[i] for i in range(len(A_all)) if keep[i]]
    B_motions = [B_all[i] for i in range(len(B_all)) if keep[i]]
    print(f"Using {len(A_motions)} / {len(A_all)} motion pairs "
          f"(dropped {int((~keep).sum())} low-motion / stale-odom pairs).")
    if len(A_motions) < 3:
        raise RuntimeError("Not enough non-degenerate motion pairs after filtering.")

    # --- Robust initialisation --------------------------------------------
    # Scale from the median per-step displacement ratio (odom / camera).
    ratios = dB[keep] / dA[keep]
    s0 = float(np.clip(np.median(ratios), 1e-3, None))

    # Rotation from aligning the dominant rotation axes: R_A axis = R_X (R_B axis),
    # so R_X maps the odom axis onto the camera axis.  The mount M = X^-1 uses R_X^T.
    axis_A = dominant_rotation_axis(A_motions)
    axis_B = dominant_rotation_axis(B_motions)
    R_X0 = rotation_between(axis_B, axis_A)
    rot0 = Rotation.from_matrix(R_X0.T).as_rotvec()   # rotvec of the mount M

    # theta = [rx, ry, rz, px, py, scale]   (mount Z locked to LOCK_CAM_HEIGHT)
    theta_guess = np.array([rot0[0], rot0[1], rot0[2], 0.0, 0.0, s0])

    lower = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 1e-4]
    upper = [np.inf] * 6

    print(f"Running hand-eye optimization (SO(3) + planar lever arm + scale, "
          f"camera height locked to {LOCK_CAM_HEIGHT} m)...")
    result = least_squares(
        fun=hand_eye_residuals_with_scale,
        x0=theta_guess,
        bounds=(lower, upper),
        args=(A_motions, B_motions, 50.0),
        method='trf',
        loss='soft_l1',
    )

    X = build_X_from_mount(result.x)
    s_opt = float(result.x[5])
    r = result.fun
    rms = float(np.sqrt(np.mean(r ** 2)))

    rpy = Rotation.from_matrix(X[:3, :3]).as_euler('xyz', degrees=True)
    print("\n=========================================")
    print("         CALIBRATION RESULTS             ")
    print("=========================================")
    print(f"Camera scale:        {s_opt:.4f}  (multiply camera translation to get metres)")
    print(f"X = camera_from_base rotation (roll,pitch,yaw): "
          f"{rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f} deg")
    print(f"X translation (m):   {np.round(X[:3, 3], 4)}")
    print(f"Final residual RMS:  {rms:.5f}")
    print("\nX (camera_from_base):")
    print(np.round(X, 5))
    print("\nX^-1 (base_from_camera == camera->odom transform):")
    print(np.round(np.linalg.inv(X), 5))
    print("=========================================\n")

    return X, s_opt, A_motions, B_motions


# ---------------------------------------------------------
# 3. Parsing and Visualization
# ---------------------------------------------------------

def save_camera_from_base(X, scale, path=CAMERA_FROM_BASE_YAML):
    """Write the hand-eye result (camera_from_base = X) to a YAML file.

    main_ros2.py loads the full 4x4 `camera_from_base` as _T_OC (the transform
    mapping the odom/base frame into the VGGT camera frame).
    """
    data = {
        "camera_from_base": [[float(v) for v in row] for row in np.asarray(X)],
        "scale": float(scale),
    }
    header = (
        "# Hand-eye calibration output (generated by scripts/calibrate_test.py).\n"
        "# camera_from_base (X): 4x4 homogeneous transform mapping the odom/base\n"
        "# frame into the VGGT camera frame; main_ros2.py loads the full 4x4 as\n"
        "# _T_OC. 'scale' converts VGGT translation to metres.\n"
    )
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)
    print(f"Saved camera_from_base to {path}")


def load_poses_from_txt(filepath):
    """Parse the log: each line is <16 camera floats> , <16 odom floats>."""
    cam_poses, odom_poses = [], []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                print(f"Skipping line {line_num}: expected 1 comma separator.")
                continue
            cam = np.fromstring(parts[0], sep=' ').reshape(4, 4)
            odom = np.fromstring(parts[1], sep=' ').reshape(4, 4)
            cam[:3, :3] = orthonormalize(cam[:3, :3])
            odom[:3, :3] = orthonormalize(odom[:3, :3])
            cam_poses.append(cam)
            odom_poses.append(odom)
    print(f"Loaded {len(cam_poses)} pose pairs from {filepath}.")
    return cam_poses, odom_poses


def _draw_frame(ax, R, t, length=0.15, alpha=1.0, labels=None, lw=2.5):
    """Draw an XYZ coordinate triad (x=red, y=green, z=blue) at pose (R, t)."""
    colors = ['r', 'g', 'b']
    names = ['x', 'y', 'z']
    for k in range(3):
        ax.quiver(t[0], t[1], t[2], R[0, k], R[1, k], R[2, k],
                  length=length, color=colors[k], alpha=alpha, linewidth=lw)
        if labels:
            tip = t + R[:, k] * length
            ax.text(tip[0], tip[1], tip[2], f"{labels}_{names[k]}",
                    color=colors[k], fontsize=8)


def visualize(camera_poses, odom_poses, A_motions, B_motions, X, scale):
    cam_pos = np.array([P[:3, 3] for P in camera_poses])
    odom_pos = np.array([P[:3, 3] for P in odom_poses])

    # --- (a) Absolute trajectories lined up via similarity (Umeyama) fit ----
    # Directly maps the camera position track into the odom world frame; this is
    # what "lined up" should look like and is independent of the hand-eye lever arm.
    s_u, R_u, t_u = umeyama(cam_pos, odom_pos)
    cam_aligned = (s_u * (R_u @ cam_pos.T).T) + t_u
    rmse = float(np.sqrt(np.mean(np.sum((cam_aligned - odom_pos) ** 2, axis=1))))
    print(f"[Trajectory alignment] Umeyama scale={s_u:.4f}  "
          f"(hand-eye scale={scale:.4f})  position RMSE={rmse:.4f} m")

    # --- (b) Hand-eye consistency: integrate relative motions from identity ---
    # If X is correct, X^-1 A_scaled X reproduces the odom relative motion B.
    X_inv = np.linalg.inv(X)
    odom_int = [np.eye(4)]
    cam_int = [np.eye(4)]
    for A, B in zip(A_motions, B_motions):
        odom_int.append(odom_int[-1] @ B)
        A_scaled = A.copy()
        A_scaled[:3, 3] *= scale
        cam_int.append(cam_int[-1] @ (X_inv @ A_scaled @ X))
    odom_int = np.array([T[:3, 3] for T in odom_int])
    cam_int = np.array([T[:3, 3] for T in cam_int])

    fig = plt.figure(figsize=(16, 9))

    # (a) top-down aligned absolute trajectories
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title(f"Absolute trajectories aligned (Umeyama)\n"
                  f"scale={s_u:.3f}, RMSE={rmse:.3f} m", fontsize=11)
    ax1.plot(odom_pos[:, 0], odom_pos[:, 1], color="blue", lw=2.5, label="Odometry")
    ax1.plot(cam_aligned[:, 0], cam_aligned[:, 1], color="orange", ls="--", lw=2.5,
             label="Camera → odom frame")
    ax1.scatter(*odom_pos[0, :2], c="green", marker="o", s=90, zorder=5, label="Start")
    ax1.scatter(*odom_pos[-1, :2], c="red", marker="X", s=90, zorder=5, label="End")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.axis("equal"); ax1.grid(True, ls=":", alpha=0.7); ax1.legend()

    # (b) hand-eye relative-motion consistency check
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Hand-eye check: integrated relative motion\n"
                  "(camera mapped through X vs odometry)", fontsize=11)
    ax2.plot(odom_int[:, 0], odom_int[:, 1], color="blue", lw=2.5, label="Odometry")
    ax2.plot(cam_int[:, 0], cam_int[:, 1], color="orange", ls="--", lw=2.5,
             label="X$^{-1}$ · (s·A) · X")
    ax2.scatter(odom_int[0, 0], odom_int[0, 1], c="green", marker="o", s=90, zorder=5)
    ax2.scatter(odom_int[-1, 0], odom_int[-1, 1], c="red", marker="X", s=90, zorder=5)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.axis("equal"); ax2.grid(True, ls=":", alpha=0.7); ax2.legend()

    # (c) the camera -> odometry (base) transform as coordinate triads
    ax3 = fig.add_subplot(2, 1, 2, projection='3d')
    ax3.set_title("Camera → odometry transform  X$^{-1}$ (camera pose in the odom/base frame)",
                  fontsize=11)
    # Base/odom frame at the origin (faded), camera frame placed by X^-1.
    _draw_frame(ax3, np.eye(3), np.zeros(3), length=0.18, alpha=0.35, labels="odom", lw=2.0)
    R_cam = X_inv[:3, :3]
    t_cam = X_inv[:3, 3]
    _draw_frame(ax3, R_cam, t_cam, length=0.18, alpha=1.0, labels="cam", lw=3.0)
    rpy = Rotation.from_matrix(R_cam).as_euler('xyz', degrees=True)
    ax3.text2D(0.02, 0.95,
               f"camera-in-base  rpy = ({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f})°\n"
               f"offset t = ({t_cam[0]:.3f}, {t_cam[1]:.3f}, {t_cam[2]:.3f}) m\n"
               f"scale = {scale:.3f}",
               transform=ax3.transAxes, fontsize=9, va="top",
               bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    lim = 0.3
    ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim); ax3.set_zlim(-lim, lim)
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    try:
        ax3.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    fig.tight_layout()
    out = "calibration_alignment.png"
    fig.savefig(out, dpi=120)
    print(f"Saved figure to {out}")
    plt.show()


# ---------------------------------------------------------
# 4. Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    data_file = "/home/dominator/Documents/vgs/VGGT-SLAM/pose_log.txt"

    try:
        camera_poses, odom_poses = load_poses_from_txt(data_file)
        if len(camera_poses) < 3:
            print("Not enough data to run optimization. Need at least 3 pairs.")
        else:
            X, scale, A_motions, B_motions = calibrate_hand_eye_with_scale(
                camera_poses, odom_poses)
            save_camera_from_base(X, scale)
            visualize(camera_poses, odom_poses, A_motions, B_motions, X, scale)
    except FileNotFoundError:
        print(f"Error: Could not find '{data_file}'. Check the path.")
