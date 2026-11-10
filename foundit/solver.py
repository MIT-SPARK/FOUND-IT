import numpy as np
import cv2
import torch
import open3d as o3d
from termcolor import colored

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

from foundit.slam_utils import compute_image_embeddings, Accumulator
from foundit.loop_closure import ImageRetrieval, LoopMatch
from foundit.frame_overlap import FrameTracker
from foundit.map import GraphMap
from foundit.submap import Submap
from foundit.graph import PoseGraph
from foundit.scale_solver import estimate_scale_pairwise
from foundit.viewer import Viewer
from foundit.places import Places

def _get_da3_backbone(model):
    """Return the DinoV2 backbone from a DepthAnything3 wrapper."""
    inner = model.model
    if hasattr(inner, "da3"):   # NestedDepthAnything3Net
        return inner.da3.backbone
    return inner.backbone       # DepthAnything3Net

def debug_visualize(pcd1_points, pcd2_points):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_points)
    pcd1.paint_uniform_color([1, 0, 0])  # red

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_points)
    pcd2.paint_uniform_color([0, 0, 1])  # blue

    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Pairwise Point Clouds")


class Solver:
    def __init__(self,
        init_conf_threshold: float,  # represents percentage (e.g., 50 means filter lowest 50%)
        lc_thres: float = 0.80,
        use_places_layer: bool = False,
        use_all_points_for_places: bool = False,
        vis_voxel_size: float = None,
        use_metric_scale: bool = False,
        voxel_size: float = 0.05,
        voxel_min_points: int = 1,
        vis_imgs: bool = False,
        use_omega: bool = False,
        omega_resolution: int = 512):

        self.init_conf_threshold = init_conf_threshold
        self.vis_voxel_size = vis_voxel_size
        self.voxel_size = voxel_size
        self.voxel_min_points = voxel_min_points
        self.vis_imgs = vis_imgs

        self.viewer = Viewer()

        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.graph = PoseGraph()

        self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None

        self.lc_thres = lc_thres

        self.temp_count = 0
        self.vggt_timer = Accumulator()
        self.loop_closure_timer = Accumulator()
        self.clip_timer = Accumulator()
        self.places_timer = Accumulator()

        self.places = None
        self.use_all_points_for_places = use_all_points_for_places
        self.floor_conf_threshold = 0.85
        self.use_places_layer = use_places_layer
        self.use_all_points_for_places = use_all_points_for_places
        if use_places_layer:
            self.places = Places()

        # Graph node indices already classified as interior (non-frontier).
        # Classification is one-way: once a node is interior it is never
        # reclassified as a frontier again (enforced in compute_frontier).
        self._confirmed_interior: set[int] = set()

        self._object_cache = None
        
        # --- Multi-session support ---
        # Submap IDs belonging to a prior session loaded from disk.  The current
        # session runs in its own coordinate frame until a loop closure links it
        # to one of these submaps, at which point it is aligned to the prior frame.
        self.prior_session_submap_ids: set = set()
        self.awaiting_session_anchor = False
        # Graph factor index of the new session's temporary anchor (removed on align).
        self.new_session_anchor_factor_idx = None
        # Graph factor index of the very first session anchor (node key 0).
        self.first_anchor_factor_idx = None
        # Whether the current session has already been aligned to the prior one.
        self.session_aligned = False
        # Last loop-closure measurement (lc_node_key, detected_node_key, H_overlap),
        # stashed by add_edge so cross-session alignment can be computed.
        self._last_loop_closure = None
        # Set transiently by add_points so add_edge knows the first-submap branch is
        # anchoring a new session (and should record the anchor factor index).
        self._anchoring_new_session = False

        self.use_metric_scale = use_metric_scale
        self.use_omega = use_omega
        self.omega_resolution = omega_resolution
        self.first_depth_img = None # Store the first depth image of the first submap to use when computing metric depth.
        if use_metric_scale:
            from foundit.metric_scale_estimator import MetricScaleEstimator
            self.metric_scale_estimator = MetricScaleEstimator()

    def register_object_cache(self, objects_module) -> None:
        """Register the open-set object cache so its detections are kept in sync
        with the pose graph (moved on loop-closure re-optimization)."""
        self._object_cache = objects_module

    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.vis_voxel_size is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_in_world_frame.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(points_colors.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(self.vis_voxel_size)
            points_in_world_frame = np.asarray(pcd.points, dtype=np.float32)
            points_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        self.viewer.add_point_cloud("pcd_" + name, points_in_world_frame, points_colors, point_size)
    
    def add_places_layer(self, submap):
        name = str(submap.get_id())
        print(f"adding places layer for submap id {name}")
        assert name not in self.places.submap_ref_homographies, "Trying to add a places layer to a submap that already has a places layer"
        tiles = self.places.compute_places(submap, self.graph, self.use_all_points_for_places, can_remove_ground_seg = not self.vis_imgs)
        self._render_tiles(tiles)

    def move_and_merge_all_places_layers(self):
        for submap in self.map.get_submaps():
            tiles = self.places.move_submap_places(submap, self.graph)
        for submap in self.map.get_submaps():
            tiles = self.places.merge_submap_places(submap)
            if tiles is not None:
                self._render_tiles(tiles)

    def set_submap_point_cloud(self, submap):
        """
        Visualizes the submap point cloud.
        If ground_mask is provided, it also computes and visualizes a 2D 'places' layer
        consisting of h x h squares (tiles) that cover the walkable area.
        """
        # Ground points are intentionally kept in sparse_points so the point cloud
        # stays dense and consistent across renders.  The tile overlay is a separate
        # visual layer added by compute_places / move_submap_places.
        name = str(submap.get_id())

        if self.vis_voxel_size is not None:
            points_in_world_frame, points_colors = submap.get_vis_world_points(
                self.graph, self.vis_voxel_size
            )
            self.viewer.add_point_cloud("pcd_" + name, points_in_world_frame, points_colors, 0.005)
        else:
            points_in_world_frame = submap.get_points_in_world_frame(self.graph)
            points_colors = submap.get_points_colors()
            self.set_point_cloud(points_in_world_frame, points_colors, name, 0.005)

    def compute_frontier(self) -> "FrontierClassification":
        """Classify place-graph nodes relative to the current map boundary.

        Boundary (frontier) nodes are split into blocked/unexplored using the
        SLAM sparse points, then interior nodes hugging the boundary are flagged
        near-boundary.  Returns a FrontierClassification whose three index sets
        are mutually disjoint.  Requires the places layer to be active.
        """
        from foundit.places.frontier import (
            FrontierClassification,
            classify_frontier_nodes,
            find_near_boundary_nodes,
        )
        world_points = self._gather_world_points()

        blocked, unexplored = classify_frontier_nodes(
            self.places.graph,
            world_points,
            self.places.place_size,
            known_interior=frozenset(self._confirmed_interior),
            submap_plane_rotations=self.places.submap_plane_rotations,
        )
        near_boundary = find_near_boundary_nodes(
            self.places.graph,
            blocked + unexplored,
            self.places.near_boundary_threshold,
        )
        # Latch every currently non-frontier node as interior so it can never be
        # reclassified as a frontier on a later call (one-way transition).
        frontier = set(blocked) | set(unexplored)
        self._confirmed_interior.update(
            idx for idx in self.places.graph.active_indices if idx not in frontier
        )
        return FrontierClassification(
            blocked=frozenset(blocked),
            unexplored=frozenset(unexplored),
            near_boundary=frozenset(near_boundary),
        )

    def _gather_world_points(self) -> np.ndarray:
        """Concatenate all world-space sparse points across every submap."""
        pts_list = []
        for submap in self.map.get_submaps():
            pts = submap.get_points_in_world_frame(self.graph)
            if pts is not None and len(pts) > 0:
                pts_list.append(pts)
        return np.concatenate(pts_list, axis=0) if pts_list else np.empty((0, 3), dtype=np.float32)

    def _gather_non_ground_world_points(self) -> np.ndarray:
        """Concatenate the NON-ground world-space sparse points across every submap.

        Uses each submap's stored non-ground mask (set during places extraction);
        submaps without a mask contribute nothing.  These are the wall/obstacle
        points used to flag blocked place nodes, so the floor itself is never
        mistaken for an obstacle.
        """
        pts_list = []
        for submap in self.map.get_submaps():
            pts = submap.get_non_ground_points_in_world_frame(self.graph)
            if pts is not None and len(pts) > 0:
                pts_list.append(pts)
        return np.concatenate(pts_list, axis=0) if pts_list else np.empty((0, 3), dtype=np.float32)

    def compute_blocked_nodes(self) -> frozenset:
        """Project all submap points onto the places layer and return the set of
        node indices now blocked by an obstacle.

        Whole-graph collision check for path planning: catches obstacles sitting
        on previously-clear INTERIOR tiles (e.g. an object moved into the path),
        which compute_frontier's leaf-only pass does not cover.  Returns an empty
        set when the places layer is inactive.
        """
        if self.places is None:
            return frozenset()
        from foundit.places.frontier import find_obstacle_blocked_nodes
        blocked = find_obstacle_blocked_nodes(
            self.places.graph,
            self._gather_non_ground_world_points(),   # obstacles only — floor excluded
            self.places.place_size,
            submap_plane_rotations=self.places.submap_plane_rotations,
            obstacle_max_height=self.places.obstacle_height_limit,
        )
        return frozenset(blocked)

    def refresh_places_and_frontier(self) -> "FrontierClassification":
        """Prepare the places layer for a fresh path plan.

        Detects nodes now covered by an obstacle, removes them from the graph,
        rebuilds the affected tile meshes + all cross edges, recomputes the
        frontier, and pushes every change to the viewer.  Returns the fresh
        FrontierClassification so the caller can plan against the updated graph.

        Removed nodes leave the active graph, so the pathfinder never routes
        through them and the returned keep_out set already reflects the new
        boundary.
        """
        from foundit.places.frontier import FrontierClassification
        if self.places is None:
            return FrontierClassification(frozenset(), frozenset(), frozenset())

        blocked = self.compute_blocked_nodes()
        removed = bool(blocked)
        if removed:
            tiles, cross_edges = self.places.remove_nodes(blocked)
            if self.viewer is not None:
                for t in tiles:
                    self._render_tiles(t)
                for sid, edges in cross_edges:
                    self.viewer.update_places_cross_edges(sid, edges)
            # Topology changed — drop the interior latch so nodes now bordering the
            # freed hole are re-evaluated (frontier is recomputed from scratch).
            self._confirmed_interior.clear()
            print(f"[Places] Removed {len(blocked)} obstacle-blocked node(s) before planning.")

        classification = self.compute_frontier()
        # Only re-render the frontier overlay when a removal actually changed the
        # graph — otherwise a no-op plan would needlessly rebuild every tile mesh.
        if removed and self.viewer is not None:
            self.viewer.show_frontier_tiles(
                classification, self.places.graph,
                dict(self.places.submap_plane_rotations), self.places.place_size,
            )
        return classification

    def _render_tiles(self, tiles) -> None:
        """Render SubmapTiles geometry via viser."""
        self.viewer.add_places_tiles(
            sid=tiles.submap_id,
            tile_vertices=tiles.tile_vertices if (tiles.tile_vertices is not None and tiles.tile_faces is not None) else None,
            tile_faces=tiles.tile_faces if (tiles.tile_vertices is not None and tiles.tile_faces is not None) else None,
            internal_edge_points=tiles.internal_edge_points,
            cross_edge_points=tiles.cross_edge_points,
        )

    def set_submap_poses(self, submap):
        # Add the camera poses and images to the visualization.
        extrinsics = submap.get_all_poses_world(self.graph)
        images = None
        if self.vis_imgs:
            images = submap.get_all_frames()
        # In a multi-session (--load_session) run, colour poses by session so it's
        # clear which session each belongs to: prior session orange, current blue.
        # Otherwise (single session) keep the per-submap distinct colours.
        color_override = None
        if self.prior_session_submap_ids:
            color_override = ((255, 165, 0) if submap.get_id() in self.prior_session_submap_ids
                              else (0, 128, 255))
        self.viewer.visualize_frames(extrinsics, submap.get_id(), images,
                                     is_loop_closure=submap.get_lc_status(),
                                     color_override=color_override)

        # Visualize the floor segmentation
        if self.vis_imgs and self.use_places_layer:
            ground_mask = submap.get_ground_segmentation_mask()
            if ground_mask is not None:
                self.viewer.visualize_floor_seg(extrinsics, images, ground_mask, submap.get_id())

    def update_all_submap_vis(self):
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap)
            self.set_submap_poses(submap)
        if self.use_places_layer:
            with self.places_timer:
                self.move_and_merge_all_places_layers()
                rebuilt = self.places.rebuild_all_cross_edges()
            for sid, edges in rebuilt:
                self.viewer.update_places_cross_edges(sid, edges)
        # Keep cached objects in sync with the re-optimized graph (delta-H move),
        # the same way places nodes are moved above, then refresh their rendered
        # bboxes / point clouds so the viewer follows the correction.
        if self._object_cache is not None:
            self._object_cache.move_objects(self.graph)
            if self.viewer is not None:
                self.viewer.refresh_object_geometry()

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap(ignore_loop_closure_submaps=True)
        self.set_submap_point_cloud(submap)
        if self.use_places_layer:
            with self.places_timer:
                self.add_places_layer(submap)
        self.set_submap_poses(submap)

    def add_edge(self, submap_id_curr, frame_id_curr, submap_id_prev=None, frame_id_prev=None, is_loop_closure=False, relative_odom=None, use_external_depth=False):
        assert not (is_loop_closure and submap_id_prev is None), "Loop closure must have a previous submap"
        scale_factor = 1.0
        current_submap = self.map.get_submap(submap_id_curr)
        H_w_submap = np.eye(4)
        is_first_submap = submap_id_prev is None

        if not is_first_submap:
            prior_submap = self.map.get_submap(submap_id_prev)

            # Estimate the scale factor and affine calibration correction from the
            # single overlapping frame shared by the two submaps.
            current_conf = current_submap.get_conf_masks_frame(frame_id_curr)
            prior_conf   = prior_submap.get_conf_masks_frame(frame_id_prev)
            good_mask = prior_conf * current_conf
            if np.sum(good_mask) < 100:
                print(colored("Not enough overlapping points to estimate scale factor, using a less restrictive mask", 'red'))
                good_mask = prior_conf
            if np.sum(good_mask) < 100:
                print(colored("Still not enough overlapping points to estimate scale factor, using all points", 'red'))
                good_mask = np.ones_like(prior_conf)
            good_mask = good_mask.reshape(-1)

            affine = np.linalg.inv(prior_submap.proj_mats[frame_id_prev]) @ current_submap.proj_mats[frame_id_curr]

            t1 = (affine[0:3, 0:3] @ current_submap.get_frame_pointcloud(frame_id_curr).reshape(-1, 3)[good_mask].T).T
            t2 = prior_submap.get_frame_pointcloud(frame_id_prev).reshape(-1, 3)[good_mask]
            scale_factor = float(estimate_scale_pairwise(t1, t2)[0])

            H_scale  = np.diag((scale_factor, scale_factor, scale_factor, 1.0))
            H_overlap = affine @ H_scale  # affine correction + scale

            # Anchor the overlapping node of the new submap in world space.
            overlapping_node_id_prev = submap_id_prev + frame_id_prev
            H_w_submap = self.graph.get_homography(overlapping_node_id_prev) @ H_overlap

            if not is_loop_closure:
                self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)

            # Cross-submap between factor tying the overlapping frame pair.
            if relative_odom is not None:
                self.graph.add_between_factor(
                    submap_id_prev + frame_id_prev,
                    submap_id_curr + frame_id_curr,
                    np.eye(4), self.graph.odom_noise)
            else:
                self.graph.add_between_factor(
                    submap_id_prev + frame_id_prev,
                    submap_id_curr + frame_id_curr,
                    H_overlap, self.graph.intra_submap_noise)

        else:
            # First submap of a session.  For the very first session this is
            # (0, 0); for a session started after loading a prior one the submap
            # id continues the prior numbering, but it is still its own anchor.
            assert frame_id_curr == 0, "First added node of a submap must be frame 0"
            # Only apply DepthAnything3 metric scale when NOT using external depth.
            # External depth is already in COLMAP metric via depth_scale, so applying a
            # second metric estimate here would double-scale the anchor and break the
            # depth-COLMAP scale consistency.
            if self.use_metric_scale and not use_external_depth:
                assert self.first_depth_img is not None, "Did not store the first VGGT depth image of the first submap to estimate metric scale factor"

                metric_scale_factor = self.metric_scale_estimator.get_scale_factor(current_submap.get_img_names_at_index(frame_id_curr),
                                                                self.first_depth_img,
                                                                current_submap.get_conf_masks_frame(frame_id_curr))
                H_w_submap[0:3, 0:3] *= metric_scale_factor
                self.first_depth_img = None # No longer need to store this since we have metric scale
                self.metric_scale_estimator.unload()

            self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)
            anchor_idx = self.graph.add_prior_factor(submap_id_curr + frame_id_curr, H_w_submap)
            # Remember this anchor only when it pins a *new* session on top of a
            # loaded prior one, so it can be released once the two are linked.
            if self._anchoring_new_session:
                self.new_session_anchor_factor_idx = anchor_idx
            else:
                # Normal first submap: store so an odom scale estimator can
                # remove and re-add the prior after applying metric scale.
                self.first_anchor_factor_idx = anchor_idx

        # Loop closure only gets intra submap constraints.
        if is_loop_closure:
            self._last_loop_closure = (
                submap_id_prev + frame_id_prev,
                submap_id_curr + frame_id_curr,
                H_overlap.copy(),
            )
            return

        # Add nodes and edges for the inner submap constraints.
        world_to_cam = current_submap.get_all_poses()
        for index, pose in enumerate(world_to_cam):
            if index == 0:
                continue

            H_inner = world_to_cam[index-1] @ np.linalg.inv(pose)
            current_node = self.graph.get_homography(submap_id_curr + index - 1) @ H_inner

            # Add node and edge to graph
            self.graph.add_homography(submap_id_curr + index, current_node)

            # Optionally add odometry factor when COLMAP poses are available for
            # both this frame and the previous one.
            odom_available = (
                relative_odom is not None
                and index - 1 < len(relative_odom)
                and index < len(relative_odom)
                and relative_odom[index - 1] is not None
                and relative_odom[index] is not None
            )
            if odom_available:
                T_odom = relative_odom[index - 1] @ np.linalg.inv(relative_odom[index])
                self.graph.add_between_factor(submap_id_curr + index - 1,
                                              submap_id_curr + index,
                                              T_odom,
                                              self.graph.odom_noise)
            # Visual factor always added — odometry supplements but does not replace it.
            self.graph.add_between_factor(submap_id_curr + index - 1, submap_id_curr + index, H_inner, self.graph.inner_submap_noise)


    def add_points(self, pred_dict, relative_odom_orig=None, depth_images=None,
                   depth_scale=None, depth_min=0.20, depth_max=3.0,
                   colmap_intrinsics=None):
        # TODO Dominic, some fields are missing from docstring
        """
        Args:
            pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
                "ground_seg": (S, H, W) (optional, only if use_places_layer is True)
            }
            relative_odom: Optional[List[Optional[np.ndarray]]]
                Length S-1 list aligned with the S frames in the submap.
                relative_odom[i] is the SE(3) 4x4 cam-to-cam odometry transform
                from frame i to frame i+1, expressed at **metric scale**, or None
                if odometry is not available for that pair.
                The pair at index 0 (overlap frame -> first new frame) is skipped
                automatically for non-first submaps.
            depth_images: Optional[List[Optional[np.ndarray]]]
                Length S list of raw (H, W) float32 depth arrays or None per frame.
                When provided (along with depth_scale), replaces VGGT depth estimates.
            depth_scale: float — multiplied by raw depth values to convert to metric scale.
            depth_min: float — minimum valid depth in metric units (default 0.25 m).
            depth_max: float — maximum valid depth in metric units (default 4.0 m).
        """
        # Unpack prediction dict
        images = pred_dict["images"]  # (S, 3, H, W)
        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

        detected_loops = pred_dict["detected_loops"]

        depth_map = pred_dict["depth"]  # (S, H, W, 1)
        conf = pred_dict["depth_conf"]  # (S, H, W)

        # Replace VGGT depth with external depth images when provided.
        use_external_depth = depth_images is not None and depth_scale is not None
        if use_external_depth:
            H, W = depth_map.shape[1], depth_map.shape[2]
            scaled = []
            for d in depth_images:
                if d is None:
                    # No depth for this frame — fill with zeros (masked out by depth_min).
                    scaled.append(np.zeros((H, W), dtype=np.float32))
                else:
                    d_scaled = d * depth_scale
                    if d_scaled.shape[0] != H or d_scaled.shape[1] != W:
                        d_scaled = cv2.resize(d_scaled, (W, H), interpolation=cv2.INTER_NEAREST)
                    scaled.append(d_scaled)
            depth_map = np.stack(scaled, axis=0)[..., np.newaxis]  # (S, H, W, 1)

        # Replace VGGT intrinsics with COLMAP intrinsics when external depth is used.
        if use_external_depth and colmap_intrinsics is not None:
            intrinsics_cam = colmap_intrinsics

        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # now (S, H, W, 3)
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4)

        # Create projection matrices
        N = cam_to_world.shape[0]
        K_4x4 = np.tile(np.eye(4), (N, 1, 1))
        K_4x4[:, :3, :3] = intrinsics_cam
        world_to_cam = np.linalg.inv(cam_to_world)


        submap_id_prev = self.map.get_largest_key(ignore_loop_closure_submaps=True)
        submap_id_curr = self.current_working_submap.get_id()
        frame_id_curr = 0
        frame_id_prev = None

        # When a prior session has been loaded, the first submap of the new session
        # must start its own coordinate frame (its own anchor) rather than chaining
        # onto the prior session's last submap.  It only links to the prior session
        # later, via a loop closure.
        self._anchoring_new_session = self.awaiting_session_anchor
        if self.awaiting_session_anchor:
            submap_id_prev = None
            self.awaiting_session_anchor = False

        first_edge = submap_id_prev is None

        if not first_edge:
            last_frame = self.map.get_latest_submap(ignore_loop_closure_submaps=True).get_last_non_loop_frame_index()
            # frame_id_prev points to the single overlapping frame in the prior
            # submap (its last frame is shared with the first frame of this submap).
            frame_id_prev = last_frame
        else:
            # Store VGGT depth (not external depth) for metric_scale_estimator, which
            # expects VGGT-scale depth to compute the VGGT→metric scale factor.
            # When external depth is used, metric_scale_factor is skipped entirely (see add_edge).
            if not use_external_depth:
                self.first_depth_img = depth_map[0]

        # The single overlapping frame (index 0) is already present in the prior
        # submap, so skip re-adding its points for a non-first submap.
        skip_frames = {0} if not first_edge else set()

        # Add attributes to submap and add submap to map.
        self.current_working_submap.add_all_poses(world_to_cam)
        self.current_working_submap.add_all_points(
            world_points, colors, conf, self.init_conf_threshold, K_4x4, depth_map, self.voxel_size,
            depth_min=depth_min if use_external_depth else None,
            depth_max=depth_max if use_external_depth else None,
            skip_frame_indices=skip_frames,
            voxel_min_points=self.voxel_min_points,
        )
        if self.use_places_layer:
            self.current_working_submap.set_ground_seg(pred_dict["ground_seg"])
        self.map.add_submap(self.current_working_submap)

        # Add all constraints for the new submap.
        self.add_edge(submap_id_curr, frame_id_curr, submap_id_prev, frame_id_prev,
                      is_loop_closure=False, relative_odom=relative_odom_orig)
        self._anchoring_new_session = False

        # Add in loop closures if any were detected.
        for loop in detected_loops:
            assert loop.query_submap_id == self.current_working_submap.get_id()

            cam_to_world_lc = closed_form_inverse_se3(pred_dict["extrinsic_lc"])
            n_lc = pred_dict["frames_lc"].shape[0]
            K_4x4_lc = np.tile(np.eye(4), (n_lc, 1, 1))
            K_4x4_lc[:, :3, :3] = pred_dict["intrinsic_lc"]
            world_to_cam_lc = np.linalg.inv(cam_to_world_lc)
            depth_map_lc = pred_dict["depth_lc"]  # (S, H, W, 1)
            conf_lc = pred_dict["depth_conf_lc"]  # (S, H, W)

            intrinsics_cam = pred_dict["intrinsic_lc"]
            

            world_points_lc = unproject_depth_map_to_point_map(depth_map_lc, pred_dict["extrinsic_lc"], intrinsics_cam)

            lc_submap_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1
            print(f"Creating new Loop closure submap with id {lc_submap_num}")
            lc_submap = Submap(lc_submap_num)
            lc_submap.set_lc_status(True)
            lc_submap.set_frame_ids(pred_dict["frames_lc_names"])
            lc_submap.set_last_non_loop_frame_index(n_lc - 1)

            lc_submap.add_all_poses(world_to_cam_lc)
            lc_colors = (np.transpose(pred_dict["frames_lc"].cpu().numpy(), (0, 2, 3, 1)) * 255).astype(np.uint8)
            lc_submap.add_all_points(world_points_lc, lc_colors, conf_lc, self.init_conf_threshold, K_4x4_lc, depth_map_lc, self.voxel_size, voxel_min_points=self.voxel_min_points)
            print("Loop closure conf", conf_lc.shape)
            self.map.add_submap(lc_submap)

            self.add_edge(lc_submap_num, 0, loop.query_submap_id, loop.query_submap_frame, is_loop_closure=False)
            self.add_edge(loop.detected_submap_id, loop.detected_submap_frame, lc_submap_num, n_lc // 2, is_loop_closure=True)

            # Multi-session: the first loop closure that links the current session
            # to a loaded prior session snaps this session into the prior frame.
            if (self.prior_session_submap_ids
                    and not self.session_aligned
                    and loop.detected_submap_id in self.prior_session_submap_ids):
                self._align_to_prior_session()

    def mark_prior_session(self):
        """Flag every submap currently in the map as belonging to a prior session.

        Call right after load_session(). The next submap added starts a fresh
        coordinate frame; it is aligned to this prior session on the first loop
        closure that links the two.
        """
        self.prior_session_submap_ids = set(self.map.submaps.keys())
        self.awaiting_session_anchor = True
        self.session_aligned = False
        self.new_session_anchor_factor_idx = None
        # Keep the prior session's place nodes from merging / cross-edging with the
        # current session until a loop closure aligns the two frames (otherwise the
        # proximity-based links would be nonsense across unaligned coordinates).
        if self.places is not None:
            self.places.set_session_isolation(self.prior_session_submap_ids)
        print(f"[Session] Marked {len(self.prior_session_submap_ids)} prior-session submaps; "
              f"current session will align on first cross-session loop closure.")

    def _align_to_prior_session(self):
        """Snap the current session into the prior session's frame via a gauge transform.

        Uses the most recent loop-closure measurement to compute a single SL4
        transform T that brings the current session's connected component into the
        prior frame, applies it to every current-session node (which leaves all of
        the component's internal between-factor errors unchanged), then releases the
        current session's temporary anchor so the prior session — whose nodes are
        pinned on load — becomes the single gauge reference.
        """
        if self._last_loop_closure is None:
            print("[Session] No loop-closure measurement available; skipping alignment.")
            return

        lc_key, det_key, H_overlap = self._last_loop_closure
        H_lc = self.graph.get_homography(lc_key)
        H_det = self.graph.get_homography(det_key)
        if H_lc is None or H_det is None:
            print("[Session] Loop-closure nodes missing; skipping alignment.")
            return

        # Loop closure constraint: H_det = T @ H_lc @ H_overlap
        #   →  T = H_det @ inv(H_overlap) @ inv(H_lc)
        T = H_det @ np.linalg.inv(H_overlap) @ np.linalg.inv(H_lc)

        # Left-multiply every current-session node by T (gauge transform).
        moved = 0
        for submap in self.map.get_submaps():
            sid = submap.get_id()
            if sid in self.prior_session_submap_ids:
                continue
            poses = submap.get_all_poses()
            n_nodes = len(poses) if poses is not None else submap.get_last_non_loop_frame_index() + 1
            for i in range(n_nodes):
                key = sid + i
                H = self.graph.get_homography(key)
                if H is not None:
                    self.graph.update_homography(key, T @ H)
                    moved += 1

        # Release the current session's temporary anchor so the loop-closure factor
        # (and the pinned prior-session nodes) fully determine the merged graph.
        self.graph.remove_factor(self.new_session_anchor_factor_idx)
        self.new_session_anchor_factor_idx = None
        self.session_aligned = True
        # The two sessions now share a frame: drop the places isolation so the next
        # move/merge/cross-edge rebuild (update_all_submap_vis after this loop) can
        # stitch them together normally.
        if self.places is not None:
            self.places.set_session_isolation(set())
        print(f"[Session] Aligned current session to prior session "
              f"({moved} nodes transformed) and released temporary anchor.")

    def sample_pixel_coordinates(self, H, W, n):
        # Sample n random row indices (y-coordinates)
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        # Sample n random column indices (x-coordinates)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        # Stack to create an (n,2) tensor
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords
    
    def load_imgs_from_names(self, image_names):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.use_omega:
            from vggt_omega.utils.load_fn import load_and_preprocess_images as omega_load
            images = omega_load(image_names, image_resolution=self.omega_resolution).to(device)
        else:
            images = load_and_preprocess_images(image_names).to(device)
        return images

    def _resolve_detected_loops(self, new_submap, max_loops, forced_loop_detection):
        """Return the loop matches to act on for ``new_submap``.

        Normally loops are found by image retrieval.  When
        ``forced_loop_detection = (detected_submap_id, detected_frame)`` is given,
        image retrieval is bypassed and a single LoopMatch is returned that ties
        new_submap's newest frame (the current camera) to that frame — used to
        deliberately close a loop back to where an object was observed, cutting
        drift accumulated since.  Returns [] if the target submap is not in the map.
        """
        if forced_loop_detection is None:
            return self.image_retrieval.find_loop_closures(
                self.map, new_submap,
                max_loop_closures=max_loops, max_similarity_thres=self.lc_thres,
            )

        detected_submap_id, detected_frame = forced_loop_detection
        if self.map.get_submap(detected_submap_id) is None:
            print(colored(f"[ForcedLC] detected submap {detected_submap_id} not in map — "
                          f"skipping forced loop closure.", "red"))
            return []
        query_frame = new_submap.get_last_non_loop_frame_index()  # newest = current camera
        print(colored(f"[ForcedLC] forcing loop closure: submap {new_submap.get_id()} "
                      f"frame {query_frame} <-> submap {detected_submap_id} frame {detected_frame}",
                      "yellow"))
        return [LoopMatch(
            similarity_score=1.0,
            query_submap_id=new_submap.get_id(),
            query_submap_frame=query_frame,
            detected_submap_id=detected_submap_id,
            detected_submap_frame=detected_frame,
        )]

    def run_predictions(self, image_names, model, max_loops, clip_model, clip_preprocess,
                        forced_loop_detection=None):
        with self.vggt_timer:
            images = self.load_imgs_from_names(image_names)

        # First submap so set new pcd num to 0
        if self.map.get_largest_key() is None:
            new_pcd_num = 0
        else:
            new_pcd_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1

        print(f"Creating new submap with id {new_pcd_num}")
        new_submap = Submap(new_pcd_num)
        new_submap.set_frame_ids(image_names)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        new_submap.set_all_retrieval_vectors(self.image_retrieval.get_all_submap_embeddings(images))
        new_submap.set_img_names(image_names)

        with self.clip_timer:
            if clip_model is not None and clip_preprocess is not None:
                image_embs = compute_image_embeddings(clip_model, clip_preprocess, image_names)
                new_submap.set_all_semantic_vectors(image_embs)

        self.current_working_submap = new_submap

        with torch.no_grad():
            with self.vggt_timer:
                predictions = model(images)

            if self.use_places_layer:
                with self.places_timer:
                    floor_logits = predictions.pop("floor_mask")
                    predictions["ground_seg"] = (floor_logits.sigmoid() > self.floor_conf_threshold).squeeze(0).cpu()

        # Check for loop closures and add retrieval vectors from new submap to the database
        predictions_lc = None
        with self.loop_closure_timer:
            detected_loops = self._resolve_detected_loops(new_submap, max_loops, forced_loop_detection)
        loop_closure_frame_names = []
        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
            loop = detected_loops[0]
            det_submap = self.map.get_submap(loop.detected_submap_id)

            def _adjacent(idx: int, last: int) -> int:
                """Next frame index, or previous if idx is the last frame."""
                if idx < last:
                    return idx + 1
                return idx - 1 if idx > 0 else 0

            query_idx     = loop.query_submap_frame
            query_adj_idx = _adjacent(query_idx, new_submap.get_last_non_loop_frame_index())
            det_idx       = loop.detected_submap_frame
            det_adj_idx   = _adjacent(det_idx, det_submap.get_last_non_loop_frame_index())

            loop_closure_frame_names = [
                new_submap.get_img_names_at_index(query_idx),
                new_submap.get_img_names_at_index(query_adj_idx),
                det_submap.get_img_names_at_index(det_idx),
                det_submap.get_img_names_at_index(det_adj_idx),
            ]
            lc_frames = self.load_imgs_from_names(loop_closure_frame_names)
            # Verify the loop closure 
            with torch.no_grad():
                if self.use_omega:
                    lc_verify_threshold = -1.0 # disabling for omega
                    lc_verify_score = 1.0 # disabling for omega
                    predictions_lc = model(lc_frames)
                    print(colored(f"Loop closure register cosine sim: {lc_verify_score:.4f}", "yellow"))
                else:
                    predictions_lc = model(lc_frames, compute_similarity=True, query_idx=0, det_idx=2, num_imgs=len(loop_closure_frame_names))
                    lc_verify_score = float(predictions_lc["image_match_ratio"])
                    lc_verify_threshold = 0.95
                    print(colored(f"Loop closure image match ratio: {lc_verify_score:.4f}", "yellow"))

            if lc_verify_score < lc_verify_threshold and forced_loop_detection is None:
                print(colored("Loop closure verification score too low, skipping loop closure", "red"))
                detected_loops = []
                predictions_lc = None
            elif lc_verify_score < lc_verify_threshold:
                print(colored("Loop closure verification score low but forced — keeping it.", "yellow"))

        if self.use_omega:
            from vggt_omega.utils.pose_enc import encoding_to_camera
            extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], images.shape[-2:])
            extrinsic = extrinsic.float().cpu().numpy().squeeze(0)
            intrinsic = intrinsic.float().cpu().numpy().squeeze(0)
        else:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        predictions["detected_loops"] = detected_loops

        if predictions_lc is not None:
            if self.use_omega:
                from vggt_omega.utils.pose_enc import encoding_to_camera
                extrinsic_lc, intrinsic_lc = encoding_to_camera(predictions_lc["pose_enc"], images.shape[-2:])
                extrinsic_lc = extrinsic_lc.float().cpu().numpy().squeeze(0)
                intrinsic_lc = intrinsic_lc.float().cpu().numpy().squeeze(0)
                predictions["depth_lc"] = predictions_lc["depth"].float().cpu().numpy().squeeze(0)
                predictions["depth_conf_lc"] = predictions_lc["depth_conf"].float().cpu().numpy().squeeze(0)
            else:
                extrinsic_lc, intrinsic_lc = pose_encoding_to_extri_intri(predictions_lc["pose_enc"], images.shape[-2:])
                predictions["depth_lc"] = predictions_lc["depth"]
                predictions["depth_conf_lc"] = predictions_lc["depth_conf"]
            self.graph.increment_loop_closure()
            predictions["extrinsic_lc"] = extrinsic_lc
            predictions["intrinsic_lc"] = intrinsic_lc

            
        for key in predictions.keys():
            # No need to squeeze target tokens and ground seg
            if isinstance(predictions[key], torch.Tensor) and key not in ("target_tokens", "ground_seg"):
                predictions[key] = predictions[key].float().cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

        if predictions_lc is not None:
            predictions["frames_lc"] = lc_frames  # TODO Dominic, we should ideally just get these from the stored frames and not store them again.
            print(loop_closure_frame_names)
            predictions["frames_lc_names"] = loop_closure_frame_names

        return predictions

    def run_predictions_da(self, image_names, model, max_loops=1, clip_model=None, clip_preprocess=None, forced_loop_detection=None):
        """Run Depth Anything 3 inference to populate a new submap.

        Replaces run_predictions() when using Depth Anything 3 instead of VGGT.
        Loop closure detection uses DINOv2/SALAD retrieval (same as the VGGT path)
        but skips VGGT-specific loop closure verification.  Places are not supported.

        Args:
            image_names: List of image file paths for this submap window.
            model: DepthAnything3 instance (must support pose estimation, e.g. the
                   nested-giant-large variant).
            max_loops: Maximum number of loop closures to detect per submap.
        Returns:
            pred_dict compatible with add_points().
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Assign submap ID
        if self.map.get_largest_key() is None:
            new_pcd_num = 0
        else:
            new_pcd_num = (self.map.get_largest_key()
                           + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1)

        print(f"Creating new submap with id {new_pcd_num}")
        new_submap = Submap(new_pcd_num)
        new_submap.set_frame_ids(image_names)
        new_submap.set_last_non_loop_frame_index(len(image_names) - 1)
        new_submap.set_img_names(image_names)
        self.current_working_submap = new_submap

        # Shuffle so the middle frame is first (DA's reference/anchor frame),
        # then unshuffle all per-frame outputs back to temporal order.
        S = len(image_names)
        unshuffle_perm = None
        shuffle_perm = None
        if S > 2:
            mid = S // 2
            shuffle_perm = [mid] + list(range(0, mid)) + list(range(mid + 1, S))
            unshuffle_perm = [0] * S
            for new_idx, old_idx in enumerate(shuffle_perm):
                unshuffle_perm[old_idx] = new_idx
            names_for_da = [image_names[i] for i in shuffle_perm]
        else:
            names_for_da = image_names

        # We then normalise all poses so the first *temporal* frame is the world origin,
        # and shuffle to match the DA3 input order (middle frame first).
        da_extrinsics = None
        da_intrinsics = None

        # Run Depth Anything 3 on the main submap frames.
        # If floor seg is needed, request the backbone layer tokens in the same pass.
        _seg_layer = getattr(model, "seg_layer_idx", None)
        _feat_layers = [_seg_layer] if (self.use_places_layer and getattr(model, "seg_head", None) is not None and _seg_layer is not None) else []
        with self.vggt_timer:
            prediction = model.inference(
                names_for_da,
                extrinsics=da_extrinsics,
                intrinsics=da_intrinsics,
                infer_gs=False,
                export_feat_layers=_feat_layers if _feat_layers else None,
                ref_view_strategy="first"
            )

        assert prediction.extrinsics is not None, (
            "Depth Anything 3 did not return camera poses. "
            "Use a model variant that includes pose estimation (e.g. da3nested-giant-large)."
        )
        assert prediction.intrinsics is not None, (
            "Depth Anything 3 did not return camera intrinsics."
        )

        # Restore temporal order in all per-frame numpy arrays.
        if unshuffle_perm is not None:
            prediction.depth            = prediction.depth[unshuffle_perm]
            prediction.extrinsics       = prediction.extrinsics[unshuffle_perm]
            prediction.intrinsics       = prediction.intrinsics[unshuffle_perm]
            prediction.processed_images = prediction.processed_images[unshuffle_perm]
            if prediction.conf is not None:
                prediction.conf = prediction.conf[unshuffle_perm]
            if prediction.aux and _feat_layers:
                for k in prediction.aux:
                    if isinstance(prediction.aux[k], np.ndarray) and prediction.aux[k].shape[0] == S:
                        prediction.aux[k] = prediction.aux[k][unshuffle_perm]

        # depth: (N, H, W) → (S, H, W, 1)
        depth = prediction.depth.astype(np.float32)[:, :, :, np.newaxis]

        # confidence: (N, H, W) or uniform 1s
        if prediction.conf is not None:
            conf = prediction.conf.astype(np.float32)
        else:
            conf = np.ones(depth.shape[:3], dtype=np.float32)

        # extrinsics: (N, 4, 4) world-to-cam → (S, 3, 4) [R|t], matching VGGT format
        extrinsics_3x4 = prediction.extrinsics.astype(np.float32)[:, :3, :]

        # intrinsics: (N, 3, 3) — same format as VGGT
        intrinsics = prediction.intrinsics.astype(np.float32)

        # images: (N, H, W, 3) uint8 → (S, 3, H, W) float32 in [0, 1]
        images_chw = (prediction.processed_images.transpose(0, 3, 1, 2)
                      .astype(np.float32) / 255.0)
        images_tensor = torch.from_numpy(images_chw).to(device)

        # Compute retrieval embeddings for loop closure detection
        new_submap.set_all_retrieval_vectors(
            self.image_retrieval.get_all_submap_embeddings(images_tensor)
        )

        with self.clip_timer:
            if clip_model is not None and clip_preprocess is not None:
                image_embs = compute_image_embeddings(clip_model, clip_preprocess, image_names)
                new_submap.set_all_semantic_vectors(image_embs)

        # Loop closure detection via DINOv2/SALAD (no VGGT verification)
        pred_dict_lc = {}
        with self.loop_closure_timer:
            detected_loops = self._resolve_detected_loops(new_submap, max_loops, forced_loop_detection)

        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
            loop = detected_loops[0]

            query_img   = new_submap.get_img_names_at_index(loop.query_submap_frame)
            detected_img = (self.map.get_submap(loop.detected_submap_id)
                            .get_img_names_at_index(loop.detected_submap_frame))
            lc_img_names = [query_img, detected_img]

            with self.vggt_timer:
                prediction_lc = model.inference(lc_img_names)

            assert prediction_lc.extrinsics is not None and prediction_lc.intrinsics is not None

            self.graph.increment_loop_closure()

            lc_depth = prediction_lc.depth.astype(np.float32)[:, :, :, np.newaxis]  # (2, H, W, 1)
            lc_conf  = (prediction_lc.conf.astype(np.float32)
                        if prediction_lc.conf is not None
                        else np.ones(lc_depth.shape[:3], dtype=np.float32))
            lc_images_chw = (prediction_lc.processed_images.transpose(0, 3, 1, 2)
                             .astype(np.float32) / 255.0)

            pred_dict_lc = {
                "extrinsic_lc":    prediction_lc.extrinsics.astype(np.float32)[:, :3, :],
                "intrinsic_lc":    prediction_lc.intrinsics.astype(np.float32),
                "depth_lc":        lc_depth,
                "depth_conf_lc":   lc_conf,
                "frames_lc":       torch.from_numpy(lc_images_chw).to(device),
                "frames_lc_names": lc_img_names,
            }
        else:
            detected_loops = []

        # Floor segmentation for places layer — uses features already computed above.
        ground_seg = None
        if _feat_layers and prediction.aux:
            feat_key = f"feat_layer_{_seg_layer}"
            feat_np = prediction.aux.get(feat_key, None)
            if feat_np is not None:
                # feat_np: [S, ph, pw, 2*embed_dim] numpy (spatial format from DA3)
                S_f, ph, pw, dim_f = feat_np.shape
                feat_tensor = torch.from_numpy(feat_np).to(device, dtype=torch.bfloat16)
                # Reshape to token format: [1, S, N_patch, dim]
                tokens = feat_tensor.reshape(1, S_f, ph * pw, dim_f)
                x_img = images_tensor.unsqueeze(0).to(device, dtype=torch.bfloat16)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        floor_logits = model.seg_head(tokens, x_img)  # [1, S, H, W]
                ground_seg = (floor_logits.float().sigmoid() > self.floor_conf_threshold).squeeze(0).cpu()

        pred_dict = {
            "images":         images_chw,       # (S, 3, H, W) float32
            "depth":          depth,            # (S, H, W, 1) float32, metric
            "depth_conf":     conf,             # (S, H, W)   float32
            "extrinsic":      extrinsics_3x4,   # (S, 3, 4)   float32
            "intrinsic":      intrinsics,       # (S, 3, 3)   float32
            "detected_loops": detected_loops,
            "ground_seg":     ground_seg,       # (S, H, W) bool tensor or None
            **pred_dict_lc,
        }
        return pred_dict
