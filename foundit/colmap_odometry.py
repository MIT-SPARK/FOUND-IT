"""
ColmapOdometry: reads a COLMAP sparse reconstruction and provides per-submap
relative odometry factors that are compatible with the H_inner convention used
by add_edge:

    relative_odom[k] = T_world_to_cam[k] @ inv(T_world_to_cam[k+1])

COLMAP images.bin stores (qvec, tvec) as the world-to-camera SE(3):
    p_cam = R @ p_world + t

T_body_cam (camera-to-odometry-body-frame transform) is provided for
coordinate-frame reconciliation. Set to identity when the COLMAP camera axes
match the VGGT-SLAM camera axes exactly.
"""

import os
import struct
import numpy as np


def _qvec_to_rotmat(qvec):
    """Quaternion (qw, qx, qy, qz) → 3×3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qz*qw,  2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,      1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,  1 - 2*qx**2 - 2*qy**2],
    ])


def _read_images_binary(path):
    """
    Parse COLMAP images.bin.
    Returns:
        poses      : {image_filename: 4×4 world-to-cam SE(3) np.ndarray}
        camera_ids : {image_filename: camera_id}
    """
    poses = {}
    camera_ids = {}
    with open(path, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            _image_id = struct.unpack('<I', f.read(4))[0]
            qvec = struct.unpack('<4d', f.read(32))   # qw, qx, qy, qz
            tvec = struct.unpack('<3d', f.read(24))   # tx, ty, tz
            camera_id = struct.unpack('<I', f.read(4))[0]

            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name += c
            name = name.decode('utf-8')

            num_points2d = struct.unpack('<Q', f.read(8))[0]
            f.read(num_points2d * 24)   # skip 2D observations (x, y, point3D_id)

            T = np.eye(4)
            T[:3, :3] = _qvec_to_rotmat(qvec)
            T[:3, 3] = np.array(tvec)
            poses[name] = T
            camera_ids[name] = camera_id

    return poses, camera_ids


# Number of intrinsic parameters per COLMAP camera model.
_COLMAP_MODEL_NPARAMS = {
    0: 3,   # SIMPLE_PINHOLE: f, cx, cy
    1: 4,   # PINHOLE:        fx, fy, cx, cy
    2: 4,   # SIMPLE_RADIAL:  f, cx, cy, k
    3: 5,   # RADIAL:         f, cx, cy, k1, k2
    4: 8,   # OPENCV:         fx, fy, cx, cy, k1, k2, p1, p2
    5: 8,   # OPENCV_FISHEYE: fx, fy, cx, cy, k1..k4
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12, # THIN_PRISM_FISHEYE
}


def _read_cameras_binary(path):
    """
    Parse COLMAP cameras.bin.
    Returns dict {camera_id: {'model': int, 'width': int, 'height': int,
                               'fx': float, 'fy': float, 'cx': float, 'cy': float}}.
    Distortion params are ignored — only the pinhole K is extracted.
    """
    cameras = {}
    with open(path, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_cameras):
            cam_id  = struct.unpack('<I', f.read(4))[0]
            model   = struct.unpack('<i', f.read(4))[0]
            width   = struct.unpack('<Q', f.read(8))[0]
            height  = struct.unpack('<Q', f.read(8))[0]
            nparams = _COLMAP_MODEL_NPARAMS.get(model, 4)
            params  = struct.unpack(f'<{nparams}d', f.read(nparams * 8))

            # Extract fx, fy, cx, cy — first param is f (single) or fx,fy (two focal lengths)
            if model in (0, 2, 3, 8, 9):  # single focal length
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:  # PINHOLE (1), OPENCV (4), OPENCV_FISHEYE (5), FULL_OPENCV (6), THIN_PRISM (10)
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]

            cameras[cam_id] = {
                'model': model, 'width': width, 'height': height,
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            }
    return cameras


class ColmapOdometry:
    """
    Provides relative odometry measurements derived from a COLMAP sparse reconstruction.

    Usage:
        odom = ColmapOdometry("/path/to/sparse/0")
        relative_odom = odom.get_relative_odom(image_paths_for_this_submap)
        solver.add_points(predictions, relative_odom=relative_odom)

    Args:
        colmap_sparse_path: Directory containing images.bin (standard COLMAP sparse output).
        T_body_cam: 4×4 transform from COLMAP camera frame to the VGGT-SLAM camera frame.
                    Defaults to identity — assumes the two camera conventions are identical.
    """

    def __init__(self, colmap_sparse_path: str, T_body_cam: np.ndarray = None):
        images_bin = os.path.join(colmap_sparse_path, 'images.bin')
        cameras_bin = os.path.join(colmap_sparse_path, 'cameras.bin')
        if not os.path.exists(images_bin):
            raise FileNotFoundError(f"COLMAP images.bin not found: {images_bin}")

        self._poses, self._image_camera_ids = _read_images_binary(images_bin)
        self._cameras = _read_cameras_binary(cameras_bin) if os.path.exists(cameras_bin) else {}
        # T_body_cam: colmap-cam → vggt-cam  (identity = same frame)
        self.T_body_cam = T_body_cam if T_body_cam is not None else np.eye(4)
        self.T_cam_body = np.linalg.inv(self.T_body_cam)

        print(f"[ColmapOdometry] Loaded {len(self._poses)} poses from {images_bin}")

    def has_pose(self, image_path):
        """Return True if COLMAP has a pose for the given image path."""
        return os.path.basename(image_path) in self._poses

    def get_intrinsics(self, image_paths, target_h, target_w):
        """Return (S, 3, 3) array of COLMAP K matrices scaled to (target_h, target_w).

        Images are rescaled for VGGT, so COLMAP intrinsics (stored at original
        resolution) are scaled proportionally.  Falls back to None if cameras.bin
        is missing or a camera_id is not found.
        """
        if not self._cameras:
            return None
        Ks = []
        for path in image_paths:
            name = os.path.basename(path)
            cam_id = self._image_camera_ids.get(name)
            cam = self._cameras.get(cam_id) if cam_id is not None else None
            if cam is None:
                return None  # Missing entry — caller falls back to VGGT intrinsics
            sx = target_w / cam['width']
            sy = target_h / cam['height']
            K = np.array([
                [cam['fx'] * sx,            0.0, cam['cx'] * sx],
                [           0.0, cam['fy'] * sy, cam['cy'] * sy],
                [           0.0,            0.0,            1.0],
            ], dtype=np.float32)
            Ks.append(K)
        return np.stack(Ks, axis=0)  # (S, 3, 3)

    def get_relative_odom(self, image_paths, colmap_scale=1.0):
        relative_odom = [np.eye(4)]
        T_k = None
        for k in range(len(image_paths) - 1):
            name_k  = os.path.basename(image_paths[k])
            name_k1 = os.path.basename(image_paths[k + 1])

            if T_k is None:
                T_k  = self._poses.get(name_k)
            T_k1 = self._poses.get(name_k1)

            if T_k is None or T_k1 is None:
                missing = name_k if T_k is None else name_k1
                print(f"[ColmapOdometry] No pose for '{missing}', skipping pair ({k}, {k+1})")
                relative_odom.append(None)
                continue

            # Convert each world-to-cam pose from COLMAP cam frame to VGGT cam frame.
            # With identity T_body_cam this is a no-op.
            T_k_vggt  = self.T_body_cam @ T_k  @ self.T_cam_body
            T_k1_vggt = self.T_body_cam @ T_k1 @ self.T_cam_body

            # H_inner convention: T[k] @ inv(T[k+1])
            T_rel = T_k_vggt @ np.linalg.inv(T_k1_vggt)
            T_rel[:3, 3] *= colmap_scale
            relative_odom.append(T_rel)

        return [np.linalg.inv(T) if T is not None else None for T in relative_odom]