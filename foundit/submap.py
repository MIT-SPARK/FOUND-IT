import re
import os
import sys
import cv2
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

from vggt.utils.geometry import depth_to_cam_coords_points
from foundit.slam_utils import decompose_camera, voxel_downsample

class Submap:
    def __init__(self, submap_id):
        self.submap_id = submap_id
        self.R_world_map = None
        self.poses = None
        self.proj_mats = None
        self.retrieval_vectors = None
        self.last_non_loop_frame_index = None
        self.frame_ids = None
        self.is_lc_submap = False
        self.img_names = []
        self.semantic_vectors = []
        self.ground_seg = None
        self.time_created = None

        # Full-resolution confidence data (kept for scale estimation in the solver).
        self.conf_masks = None      # (S, H, W) bool

        # Depth maps stored as float16 for on-demand dense reconstruction
        # (used in get_points_in_mask for object-detection visualization).
        self.depth_maps = None      # (S, H, W) float16

        # Voxel-sparsified point cloud in VGGT-local-world space.
        # All arrays are length N_sparse and sorted by frame index.
        self.sparse_points = None          # (N, 3) float32
        self.sparse_frame_indices = None   # (N,)   int8
        self.sparse_pixel_row = None       # (N,)   uint16 — source pixel row
        self.sparse_pixel_col = None       # (N,)   uint16 — source pixel col
        self.sparse_colors = None          # (N, 3) uint8
        # (N,) bool — True where the sparse point is NON-ground (wall / obstacle),
        # set by GroundExtractor during places extraction.  Stored as a mask (not
        # world points) so it stays valid after loop closures move the submap.
        self.non_ground_mask = None

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_lc_status(self, is_lc_submap):
        self.is_lc_submap = is_lc_submap

    def add_all_poses(self, poses):
        self.poses = poses

    def add_all_points(self, points, colors, conf, conf_threshold_percentile,
                       intrinsics_inv, depth_maps, voxel_size,
                       depth_min=None, depth_max=None,
                       skip_frame_indices=None, voxel_min_points=1):
        """
        Store point-cloud data for all S frames.

        points       : (S, H, W, 3) float32 — 3-D points in each frame's camera space
        colors       : (S, H, W, 3) uint8
        conf         : (S, H, W)    float32 — per-pixel depth confidence
        depth_maps   : (S, H, W, 1) or (S, H, W) — raw depth, stored as float16
        depth_min    : float or None — if set, use depth range [depth_min, depth_max]
                       for masking instead of confidence percentile thresholding
        depth_max    : float or None — upper bound for depth range masking
        skip_frame_indices      : set of int or None — frames excluded from the sparse
                                  cloud but still stored for scale estimation.
        voxel_min_points        : int — a voxel must contain at least this many points
                                  to survive downsampling; >1 drops sparse/noisy voxels.
        """
        self.time_created = time.time()

        S, H, W = points.shape[:3]
        skip_set = set(skip_frame_indices) if skip_frame_indices else set()

        # --- Confidence / depth masks (kept at H×W for scale estimation) ---
        conf_thresholds = (
            np.percentile(conf, conf_threshold_percentile, axis=(1, 2), keepdims=True) + 1e-6
        )
        if depth_min is not None and depth_max is not None:
            d = depth_maps
            if isinstance(d, torch.Tensor):
                d = d.cpu().numpy()
            d = d.squeeze(-1).astype(np.float32)
            self.conf_masks = (d >= depth_min) & (d <= depth_max)
        else:
            if conf_threshold_percentile == 0:
                self.conf_masks = conf > -1
            else:
                self.conf_masks = conf > conf_thresholds
        self.proj_mats = intrinsics_inv

        # --- Depth maps as float16 for on-demand dense reconstruction ---
        if isinstance(depth_maps, torch.Tensor):
            depth_maps = depth_maps.cpu().numpy()
        self.depth_maps = depth_maps.squeeze(-1)

        # --- Build world-space point maps for all frames ---
        cam_to_world = np.linalg.inv(self.poses)  # (S, 4, 4)
        R_c2w = cam_to_world[:, :3, :3]           # (S, 3, 3)
        t_c2w = cam_to_world[:, :3, 3]            # (S, 3)
        pts_world_maps = (
            np.einsum('sij,shwj->shwi', R_c2w, points)
            + t_c2w[:, np.newaxis, np.newaxis, :]
        )  # (S, H, W, 3)

        # --- Sequential per-frame fusion loop ---
        all_world_pts = []
        all_colors    = []
        all_frames    = []
        all_rows      = []
        all_cols      = []

        for i in range(S):
            if i in skip_set:
                continue

            new_rs, new_cs = np.where(self.conf_masks[i])
            if len(new_rs) == 0:
                continue

            all_world_pts.append(pts_world_maps[i, new_rs, new_cs])
            all_colors.append(colors[i, new_rs, new_cs])
            all_frames.append(np.full(len(new_rs), i, dtype=np.int32))
            all_rows.append(new_rs.astype(np.uint16))
            all_cols.append(new_cs.astype(np.uint16))

        # --- Assemble & voxelise the deduplicated cloud ---
        if all_world_pts:
            sp   = np.concatenate(all_world_pts, axis=0).astype(np.float32)
            scl  = np.concatenate(all_colors,    axis=0)
            si   = np.concatenate(all_frames,    axis=0).astype(np.int8)
            sr   = np.concatenate(all_rows,      axis=0)
            sc   = np.concatenate(all_cols,      axis=0)

            if voxel_size > 0:
                sp, si, sr, sc, scl = voxel_downsample(
                    sp, si, sr, sc, scl, voxel_size,
                    min_points_per_voxel=voxel_min_points,
                )

            order = np.argsort(si, kind='stable')
            self.sparse_points        = sp[order]
            self.sparse_frame_indices = si[order]
            self.sparse_pixel_row     = sr[order]
            self.sparse_pixel_col     = sc[order]
            self.sparse_colors        = scl[order]
        else:
            self.sparse_points        = np.empty((0, 3),  dtype=np.float32)
            self.sparse_frame_indices = np.empty(0,       dtype=np.int8)
            self.sparse_pixel_row     = np.empty(0,       dtype=np.uint16)
            self.sparse_pixel_col     = np.empty(0,       dtype=np.uint16)
            self.sparse_colors        = np.empty((0, 3),  dtype=np.uint8)

    def set_img_names(self, img_names):
        self.img_names = img_names

    def add_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors

    def set_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors

    def set_all_semantic_vectors(self, semantic_vectors):
        self.semantic_vectors = semantic_vectors

    def set_ground_seg(self, ground_seg):
        self.ground_seg = ground_seg

    def set_last_non_loop_frame_index(self, last_non_loop_frame_index):
        self.last_non_loop_frame_index = last_non_loop_frame_index

    def set_frame_ids(self, file_paths):
        """Extract the numeric frame ID from each file path."""
        frame_ids = []
        for path in file_paths:
            filename = os.path.basename(path)
            match = re.search(r'\d+(?:\.\d+)?', filename)
            if match:
                frame_ids.append(float(match.group()))
            else:
                raise ValueError(f"No number found in image name: {filename}")
        self.frame_ids = frame_ids

    # ------------------------------------------------------------------
    # Simple getters
    # ------------------------------------------------------------------

    def get_lc_status(self):
        return self.is_lc_submap

    def get_id(self):
        return self.submap_id
    
    def get_time_created(self):
        return self.time_created

    def get_conf_masks_frame(self, index):
        """Returns the full-resolution (H, W) confidence mask for frame `index`."""
        return self.conf_masks[index]

    def get_frame_at_index(self, index):
        return cv2.cvtColor(cv2.imread(self.get_img_names_at_index(index)), cv2.COLOR_BGR2RGB)

    def get_last_non_loop_frame_index(self):
        return self.last_non_loop_frame_index

    def get_img_names_at_index(self, index):
        return self.img_names[index]

    def get_all_frames(self):
        if self.get_lc_status():
            return None

        n_frames = len(self.img_names)
        if n_frames == 0:
            return np.array([])

        # Pre-fetch all image paths to avoid sequential function calls
        img_names = [self.get_img_names_at_index(i) for i in range(n_frames)]
        
        # Read the first image to get dimensions (H, W, C)
        first_img = cv2.imread(img_names[0])
        if first_img is None:
            raise FileNotFoundError(f"Could not read image at {img_names[0]}")
        h, w, c = first_img.shape

        # 1. Preallocate the final array with the exact shape (N, C, H, W) and float32 dtype
        frames = np.empty((n_frames, c, h, w), dtype=np.float32)

        def load_and_process(index):
            img = cv2.imread(img_names[index])
            if img is not None:
                # Convert color
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 2. Cast to float32 BEFORE division to save memory, transpose to (C, H, W)
                processed_img = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
                
                # 3. Insert directly into preallocated memory
                frames[index] = processed_img

        # 4. Multithread the I/O operations
        # You can tune max_workers based on your disk speed (usually 8-16 is a sweet spot for SSDs)
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(load_and_process, range(n_frames))

        return frames

    def get_all_retrieval_vectors(self):
        return self.retrieval_vectors

    def get_all_semantic_vectors(self):
        return self.semantic_vectors

    def get_frame_ids(self):
        return self.frame_ids

    def get_all_poses(self):
        return self.poses

    def get_ground_segmentation_mask(self):
        return self.ground_seg

    def get_pose_subframe(self, pose_index):
        return np.linalg.inv(self.poses[pose_index])

    # ------------------------------------------------------------------
    # Camera / projection helpers
    # ------------------------------------------------------------------

    def get_first_homography_world(self, graph):
        homography = graph.get_homography(self.get_id())
        if homography is None:
            return None
        return homography / homography[-1, -1]

    def get_last_homography_world(self, graph):
        homography = graph.get_homography(self.get_id() + self.get_last_non_loop_frame_index())
        if homography is None:
            return None
        return homography / homography[-1, -1]

    def get_first_pose_world(self, graph):
        h = self.get_first_homography_world(graph)
        return None if h is None else np.linalg.inv(h)

    def get_last_pose_world(self, graph):
        projection_mat = graph.get_projection_matrix(
            self.get_id() + self.get_last_non_loop_frame_index()
        )
        if projection_mat is None:
            return None
        return projection_mat / projection_mat[-1, -1]

    def get_project_mat(self, index, graph):
        homography_world = graph.get_homography(index + self.get_id())
        if homography_world is None:
            return None
        return self.proj_mats[index] @ np.linalg.inv(homography_world)

    def get_all_poses_world(self, graph, give_camera_mat=False):
        homography_list = [
            graph.get_homography(i + self.get_id()) for i in range(len(self.poses))
        ]
        if any(h is None for h in homography_list):
            return np.zeros((0, 4, 4))
        poses = []
        for index, homography_world in enumerate(homography_list):
            projection_mat = self.proj_mats[index] @ np.linalg.inv(homography_world)
            projection_mat = projection_mat / projection_mat[-1, -1]
            if give_camera_mat:
                poses.append(projection_mat)
            else:
                _, rot, trans, _ = decompose_camera(projection_mat[0:3, :])
                pose = np.eye(4)
                pose[0:3, 0:3] = rot
                pose[0:3, 3] = trans
                poses.append(pose)
        return np.stack(poses, axis=0)

    def get_local_pose(self, index):
        return self.poses[index]

    # ------------------------------------------------------------------
    # Point-cloud accessors
    # ------------------------------------------------------------------

    def get_frame_pointcloud(self, pose_index):
        """Returns the (H, W, 3) camera-space point cloud for any frame.

        Unprojected on demand from the stored float16 depth map so that no
        dense point arrays need to be kept in memory between calls.
        """
        depth = self.depth_maps[pose_index].astype(np.float32)   # (H, W)
        K     = self.proj_mats[pose_index][:3, :3]                # (3, 3)
        return depth_to_cam_coords_points(depth, K)               # (H, W, 3)

    def get_world_points_and_frame_indices(self, graph):
        """Return (world_pts, frame_indices) from the sparse representation.

        Sparse points are stored in VGGT-local-world space and are converted to
        global world space by applying the per-frame graph correction:
            global_world = graph.H(frame_i) @ poses[frame_i] @ local_world_pt

        Both output arrays are ordered frame-by-frame (0, 1, …, S-1) and are
        aligned with the submap's sparse_* arrays.
        """
        S = len(self.poses)
        all_pts, all_idx = [], []

        for frame_idx in range(S):
            sel = self.sparse_frame_indices == frame_idx
            if not sel.any():
                continue

            pts = self.sparse_points[sel]  # (N_f, 3) in VGGT-local-world
            H_world = graph.get_homography(frame_idx + self.get_id())
            if H_world is None:
                continue
            correction = H_world @ self.poses[frame_idx]

            pts_h = np.hstack([pts, np.ones((len(pts), 1))])
            pts_w = (correction @ pts_h.T).T
            pts_w = pts_w[:, :3] / pts_w[:, 3:]

            all_pts.append(pts_w)
            all_idx.append(np.full(len(pts_w), frame_idx, dtype=np.int32))

        if not all_pts:
            return np.empty((0, 3), dtype=np.float32), np.array([], dtype=np.int32)
        return np.vstack(all_pts), np.concatenate(all_idx)

    def get_points_in_world_frame(self, graph):
        """Return all sparse world-frame points as (N, 3).

        give_all_points is kept for API compatibility but has no effect — sparse
        points are always confidence-filtered at storage time.
        """
        return self.get_world_points_and_frame_indices(graph)[0]

    def set_non_ground_mask(self, mask):
        """Store the boolean non-ground mask (length N_sparse, aligned with
        sparse_points) computed during places extraction."""
        self.non_ground_mask = None if mask is None else np.asarray(mask, dtype=bool)

    def get_non_ground_points_in_world_frame(self, graph):
        """World-frame sparse points flagged NON-ground (wall / obstacle), or None
        if no mask has been computed for this submap.

        Applies the stored non-ground mask per frame while transforming to world
        with the current graph (same convention as
        get_world_points_and_frame_indices), so the points track loop closures.
        """
        if self.non_ground_mask is None or self.sparse_points is None:
            return None
        S = len(self.poses)
        all_pts = []
        for frame_idx in range(S):
            sel = (self.sparse_frame_indices == frame_idx) & self.non_ground_mask
            if not sel.any():
                continue
            pts = self.sparse_points[sel]
            H_world = graph.get_homography(frame_idx + self.get_id())
            if H_world is None:
                continue
            correction = H_world @ self.poses[frame_idx]
            pts_h = np.hstack([pts, np.ones((len(pts), 1))])
            pts_w = (correction @ pts_h.T).T
            all_pts.append(pts_w[:, :3] / pts_w[:, 3:])
        return np.vstack(all_pts) if all_pts else np.empty((0, 3), dtype=np.float32)

    def get_vis_world_points(self, graph, vis_voxel_size: float):
        """Return (world_pts, colors) for visualization, with optional downsampling
        performed in the stable submap-local coordinate frame.

        Unlike world-space voxelisation, local-space voxelisation produces the
        same representative pixel set regardless of loop-closure pose updates,
        so the displayed cloud never becomes sparser after a loop closure.
        """

        sp = self.sparse_points         # (N, 3) submap-local world
        sf = self.sparse_frame_indices  # (N,) int8
        sc = self.sparse_colors         # (N, 3) uint8

        if len(sp) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

        if vis_voxel_size > 0:
            sp, sf, _, _, sc = voxel_downsample(
                sp, sf, self.sparse_pixel_row, self.sparse_pixel_col, sc, vis_voxel_size
            )

        S = len(self.poses)
        all_pts, all_cols = [], []
        for frame_idx in range(S):
            sel = sf == frame_idx
            if not sel.any():
                continue
            pts = sp[sel]
            H_world = graph.get_homography(frame_idx + self.get_id())
            if H_world is None:
                continue
            correction = H_world @ self.poses[frame_idx]
            pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
            pts_w = (correction @ pts_h.T).T
            all_pts.append((pts_w[:, :3] / pts_w[:, 3:]).astype(np.float32))
            all_cols.append(sc[sel])

        if not all_pts:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        return np.vstack(all_pts), np.vstack(all_cols)
    
    def get_img_size_at_index(self, frame_index):
        # returns height,width of the size of the image processed by the GFM
        return self.depth_maps[frame_index].shape

    def get_points_in_mask(self, frame_index, mask, graph, use_conf_mask=False):
        """Reconstruct the dense point cloud for a frame and filter by a 2-D mask.

        Uses the stored float16 depth map to recreate full (HxW) camera-space 3-D
        points on demand, then transforms them to global world space and applies
        the mask. Returns dense points suitable for object visualisation.
        """

        depth = self.depth_maps[frame_index].astype(np.float32)   # (H, W)
        K = self.proj_mats[frame_index][:3, :3] # (3, 3) intrinsics

        # Unproject depth to camera-space 3-D (H, W, 3)
        cam_pts = depth_to_cam_coords_points(depth, K)

        # graph.H maps directly from camera-i space to global world
        H_world  = graph.get_homography(frame_index + self.get_id())
        if H_world is None:
            return np.empty((0, 3), dtype=np.float32)
        pts_flat = cam_pts.reshape(-1, 3)
        pts_h    = np.hstack([pts_flat, np.ones((len(pts_flat), 1))])
        world_pts = (H_world @ pts_h.T).T
        world_pts = world_pts[:, :3] / world_pts[:, 3:]

        if use_conf_mask:
            return world_pts[mask.reshape(-1) * self.conf_masks[frame_index].reshape(-1)], self.conf_masks[frame_index]
        return world_pts[mask.reshape(-1)]

    def get_all_ground_points_in_world_frame(self, graph):
        """Unproject all ground-segmented pixels from stored depth maps, bypassing
        the confidence threshold.  Used with --places_full so that RANSAC plane
        fitting sees the complete ground surface, not just the confident subset.

        Returns (world_pts, frame_indices) where world_pts is (N, 3) float32.
        """
        if self.ground_seg is None:
            return np.empty((0, 3), dtype=np.float32), np.array([], dtype=np.int32)

        ground_seg_np = self.ground_seg.cpu().numpy()   # (S, H, W) bool
        S = len(self.poses)
        all_pts, all_idx = [], []

        for frame_idx in range(S):
            ground_mask = ground_seg_np[frame_idx]   # (H, W) bool
            if not ground_mask.any():
                continue
            pts, _ = self.get_points_in_mask(frame_idx, ground_mask, graph, use_conf_mask=True)
            if len(pts) == 0:
                continue
            all_pts.append(pts)
            all_idx.append(np.full(len(pts), frame_idx, dtype=np.int32))

        if not all_pts:
            return np.empty((0, 3), dtype=np.float32), np.array([], dtype=np.int32)
        return np.vstack(all_pts), np.concatenate(all_idx)

    def remove_ground_seg(self):
        """Remove ground segmentation mask to save memory

        Should be called after places computation can be completed for this submap if we
        do not intend to visualize the ground seg in viser. To visualize it in viser, we have to 
        store the ground segmentation masks.
        """
        self.ground_seg = None

    def get_points_colors(self):
        """Return RGB colors for all sparse points as (N, 3) uint8."""
        return self.sparse_colors
