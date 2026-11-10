import threading
import open3d as o3d
import time
from collections import deque
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import viser
import viser.transforms as viser_tf

# 20 perceptually distinct colors for room overlay tiles (tab20 palette, RGB uint8)
_ROOM_PALETTE: list[tuple[int, int, int]] = [
    (31,  119, 180), (255, 127,  14), ( 44, 160,  44), (214,  39,  40),
    (148, 103, 189), (140,  86,  75), (227, 119, 194), (127, 127, 127),
    (188, 189,  34), ( 23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]

from foundit.places.clustering.base import ClusterResult, QueryClusterResult


from foundit.places.config import (
    FiltrationClusterConfig, ThresholdMode, ClusteringMode,
    DistanceMetric, DistanceAggregation,
)


class Viewer:
    def __init__(self, port: int = 8080):
        print(f"Starting viser server on port {port}")

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        # --- GUI Elements ---
        self.gui_show_frames = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        self.gui_show_frames.on_update(self._on_update_show_frames)

        # Add a button to trigger the walkthrough
        self.btn_walkthrough = self.server.gui.add_button("Play Walkthrough")
        self.btn_walkthrough.on_click(lambda _: self.run_walkthrough())

        self.gui_show_floor_seg = self.server.gui.add_checkbox("Show Floor Seg", initial_value=False)
        self.gui_show_floor_seg.on_update(self._on_update_show_floor_seg)

        self.gui_show_places = self.server.gui.add_checkbox("Show Places", initial_value=True)
        self.gui_show_places.on_update(self._on_update_show_places)

        # One-press toggle to hide/show all place-node edges (internal + cross),
        # independent of the tiles. Composes with "Show Places": an edge is shown
        # only when both are on.
        self._show_place_edges = True
        self._btn_toggle_place_edges = self.server.gui.add_button("Hide Place Edges")
        self._btn_toggle_place_edges.on_click(lambda _: self._toggle_place_edges())

        self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}
        # Submap ids that are loop-closure submaps — excluded from the walkthrough.
        self._lc_submap_ids: set[int] = set()
        # Optional callable → current camera world-position (3,) used as the
        # path-planning start.  Registered by the ROS wrapper to plan from the
        # odom-forward-propagated pose; None → fall back to the latest submap pose.
        self._planning_start_provider = None
        # World-space centre (3,) of the most recent room query's top region, used
        # by the "Go to Room" button as a navigation target. None until a query runs.
        self._last_room_center: Optional[np.ndarray] = None
        self._floor_seg_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}
        self._place_handles: list = []  # tile meshes + edge line segments
        self._place_edge_handles: list = []  # edge line segments only (for the edge toggle)

        # Optional robot URDF visualization (see add_robot_urdf / set_robot_pose).
        self._robot_urdf = None
        self._robot_base_frame = None
        self._robot_root_offset = np.eye(4)  # root-in-base, folded into set_robot_pose

        self._cluster_tile_names: list[str] = []
        self._cluster_legend_folder: Optional[viser.GuiFolderHandle] = None

        self.num_rand_colors = 250
        np.random.seed(100)
        self.random_colors = np.random.randint(0, 256, size=(self.num_rand_colors, 3), dtype=np.uint8)
        self.submap_id_to_color = dict()
        self.obj_id = 0
        self._obj_scene_names: list[str] = []  # tracks all object viz elements for clearing
        self._obj_pcd_store: dict[str, tuple] = {}   # name → (raw_points, colors, point_size)
        self._obj_obb_store: dict[str, tuple] = {}   # name → (center, extent, rotation, color, line_width)
        self.gui_obj_elevation: Optional[viser.GuiSliderHandle] = None  # set in add_object_query_gui

        # --- Places layer elevation ---
        # Maps submap_id → raw (unshifted) geometry dict for re-upload on slider change
        self._places_tile_store: dict[str, dict] = {}
        # Raw (unshifted) path segments [N, 2, 3] for re-upload on elevation change
        self._planned_path_segments: Optional[np.ndarray] = None
        # Waypoint sphere handles and positions for follow-path visualization
        self._waypoint_sphere_handles: list = []
        self._waypoint_sphere_positions: Optional[np.ndarray] = None
        # Blue sphere marking the path's actual stop point (after the stop-short trim)
        self._stop_point_sphere_handle = None
        # Frontier state — reapplied whenever elevation changes
        self._frontier_blocked_set: set[int] = set()
        self._frontier_unexplored_set: set[int] = set()
        self._frontier_near_boundary_set: set[int] = set()
        self._frontier_place_graph = None
        self._frontier_submap_rotations: dict = {}
        self._frontier_place_size: float = 0.0
        with self.server.gui.add_folder("Places Layer"):
            self.gui_places_elevation = self.server.gui.add_slider(
                "Elevation",
                min=-5.0, max=5.0, step=0.05, initial_value=0.0,
            )
            self.gui_places_elevation.on_update(lambda _: self._redraw_places_tiles())

        # --- Room overlay (elevated cluster tiles) ---
        # Each query is stored as (query_id, result, places) so all queries persist.
        self._cluster_render_history: list = []
        self._room_query_counter: int = 0
        # Maps (query_id, inst_idx) → RGB color so colors are stable across redraws
        self._room_color_assignments: dict[tuple[int, int], tuple[int, int, int]] = {}
        self._room_palette_idx: int = 0
        # Maps graph node index → (query_id, inst_idx) of the latest query that claimed it.
        # Older queries lose any nodes that a newer query also covers.
        self._node_owner: dict[int, tuple[int, int]] = {}
        with self.server.gui.add_folder("Room Overlay"):
            self.gui_room_elevation = self.server.gui.add_slider(
                "Elevation",
                min=-1.0, max=8.0, step=0.05, initial_value=3.0,
            )
            self.gui_room_elevation.on_update(lambda _: self._redraw_room_tiles())
            self._btn_clear_rooms = self.server.gui.add_button(
                "Clear Room Overlays", color="red"
            )
            self._btn_clear_rooms.on_click(lambda _: self._clear_room_overlays())

        # --- Point cloud store + height clipping ---
        # Maps point cloud name → (points [N,3], colors [N,3], point_size)
        self._pcd_store: Dict[str, tuple] = {}
        # Maps point cloud name → scene handle (for visibility toggling)
        self._pcd_handles: Dict[str, object] = {}

        self.gui_show_pointcloud = self.server.gui.add_checkbox("Show Point Cloud", initial_value=True)
        self.gui_show_pointcloud.on_update(self._on_update_show_pointcloud)

        with self.server.gui.add_folder("Height Clipping"):
            self.gui_clip_min = self.server.gui.add_slider(
                "Min Y (raise to cut ceiling)",
                min=-20.0, max=20.0, step=0.05, initial_value=-20.0,
            )
            self.gui_clip_max = self.server.gui.add_slider(
                "Max Y (lower to cut floor)",
                min=-20.0, max=20.0, step=0.05, initial_value=20.0,
            )
            self.gui_clip_min.on_update(lambda _: self._redraw_point_clouds())
            self.gui_clip_max.on_update(lambda _: self._redraw_point_clouds())

        # --- First Person View (FPV) ---
        # Movement buttons snap to local floor height; mouse orbit is left alone.
        _FPV_LOOK_DIST = 0.1
        self._FPV_LOOK_DIST = _FPV_LOOK_DIST
        with self.server.gui.add_folder("First Person View"):
            self.gui_fpv_enabled = self.server.gui.add_checkbox("FPV Mode", initial_value=False)
            self.gui_fpv_height = self.server.gui.add_slider(
                "Height above floor (m)", min=0.1, max=3.0, step=0.05, initial_value=1.0,
            )
            self.gui_fpv_step = self.server.gui.add_slider(
                "Step size (m)", min=0.01, max=2.0, step=0.01, initial_value=0.3,
            )
            self.server.gui.add_markdown("*Click to move · mouse to look*")
            self._btn_fpv_forward = self.server.gui.add_button("↑  Forward")
            self._btn_fpv_back    = self.server.gui.add_button("↓  Back")
            self._btn_fpv_left    = self.server.gui.add_button("←  Strafe Left")
            self._btn_fpv_right   = self.server.gui.add_button("→  Strafe Right")

        self._fpv_snapping: bool = False
        self.gui_fpv_enabled.on_update(lambda _: self._enter_fpv() if self.gui_fpv_enabled.value else None)
        self._btn_fpv_forward.on_click(lambda _: self._fpv_move( 0,  1))
        self._btn_fpv_back   .on_click(lambda _: self._fpv_move( 0, -1))
        self._btn_fpv_left   .on_click(lambda _: self._fpv_move(-1,  0))
        self._btn_fpv_right  .on_click(lambda _: self._fpv_move( 1,  0))

        @self.server.on_client_connect
        def _on_client_connect(client) -> None:
            @client.camera.on_update
            def _(_) -> None:
                if not self.gui_fpv_enabled.value or self._fpv_snapping:
                    return
                pos = np.array(client.camera.position, dtype=float)
                floor_y = self._get_floor_y_at(np.array([pos[0], pos[2]]))
                if floor_y is None:
                    return
                target_y = floor_y - self.gui_places_elevation.value - self.gui_fpv_height.value
                if abs(pos[1] - target_y) > 0.05:
                    self._fpv_snapping = True
                    pos[1] = target_y
                    client.camera.position = pos
                    self._fpv_snapping = False

        # Adjust default orbit control for better viewing control
        self.server.scene.set_up_direction("-y")

    def visualize_frames(self, extrinsics: np.ndarray, submap_id: int, images_: np.ndarray = None,
                         is_loop_closure: bool = False, color_override=None) -> None:
        """
        Add camera frames and frustums to the scene for a specific submap.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W) or None to disable visualizing the images
        is_loop_closure: True for loop-closure submaps, which the walkthrough skips.
        color_override: (r, g, b) to color every frustum of this submap; when None
            (default) each submap keeps its own distinct random color.  Used to
            colour poses by session in multi-session (--load_session) runs.
        """

        if images_ is not None and isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        if is_loop_closure:
            self._lc_submap_ids.add(submap_id)
        else:
            self._lc_submap_ids.discard(submap_id)

        if submap_id not in self.submap_frames:
            next_id = len(self.submap_id_to_color) + 1
            self.submap_id_to_color[submap_id] = next_id
        self.submap_frames[submap_id] = []
        self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            # Add the coordinate frame
            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            # Convert image and add frustum
            fov = np.radians(90) # default FOV if no images provided
            img = None
            aspect = 1.0
            if images_ is not None:
                img = images_[img_id]
                img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
                h, w = img.shape[:2]
                fy = 1.1 * h
                fov = 2 * np.arctan2(h / 2, fy)
                aspect = w / h

            if color_override is not None:
                frustum_color = color_override
            else:
                rand_color_idx = self.submap_id_to_color[submap_id]
                if rand_color_idx >= self.num_rand_colors:
                    rand_color_idx = rand_color_idx % self.num_rand_colors
                frustum_color = self.random_colors[rand_color_idx]
            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect= aspect,
                scale=0.05,
                image=img,
                line_width=3.0,
                color=frustum_color
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)

    def _on_update_show_frames(self, _) -> None:
        """Toggle visibility of all camera frames and frustums across all submaps."""
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames:
                f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible

    def visualize_floor_seg(
        self,
        extrinsics: np.ndarray,
        images_: np.ndarray,
        ground_mask,
        submap_id: int,
    ) -> None:
        """Display per-frame ground-segmentation overlays as camera frustums.

        Parameters
        ----------
        extrinsics   : (S, 3, 4) world-from-camera transforms
        images_      : (S, 3, H, W) RGB frames in [0, 1]
        ground_mask  : (S, H, W) binary mask (torch.Tensor or np.ndarray)
        submap_id    : integer submap identifier
        """
        if isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()
        if isinstance(ground_mask, torch.Tensor):
            ground_mask = ground_mask.cpu().numpy()

        self._floor_seg_frustums[submap_id] = []
        show = self.gui_show_floor_seg.value

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            img = (images_[img_id].transpose(1, 2, 0) * 255).astype(np.uint8)  # (H, W, 3)
            mask = ground_mask[img_id].astype(bool)                              # (H, W)

            overlay = img.copy()
            overlay[mask] = (
                img[mask] * 0.45 + np.array([0, 210, 60], dtype=np.float32) * 0.55
            ).astype(np.uint8)

            h, w = overlay.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            frustum = self.server.scene.add_camera_frustum(
                f"floor_seg_{submap_id}/frame_{img_id}/frustum",
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=overlay,
                line_width=2.0,
                color=(0, 210, 60),
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
            )
            frustum.visible = show
            self._floor_seg_frustums[submap_id].append(frustum)

    def _on_update_show_floor_seg(self, _) -> None:
        """Toggle visibility of all floor-seg frustums."""
        visible = self.gui_show_floor_seg.value
        for frustums in self._floor_seg_frustums.values():
            for fr in frustums:
                fr.visible = visible

    def _on_update_show_places(self, _) -> None:
        """Toggle visibility of all place-node tiles and edges."""
        visible = self.gui_show_places.value
        for handle in self._place_handles:
            handle.visible = visible
        # Edges also obey the separate edge toggle (edges are in _place_handles too,
        # so re-apply their combined visibility after the blanket set above).
        self._apply_place_edge_visibility()

    def _apply_place_edge_visibility(self) -> None:
        """Set every place-edge handle visible only when the places layer is shown
        and edges are not toggled off."""
        visible = self.gui_show_places.value and self._show_place_edges
        for h in self._place_edge_handles:
            h.visible = visible

    def _toggle_place_edges(self) -> None:
        """Button handler: flip place-edge visibility and update the button label."""
        self._show_place_edges = not self._show_place_edges
        self._apply_place_edge_visibility()
        self._btn_toggle_place_edges.label = (
            "Hide Place Edges" if self._show_place_edges else "Show Place Edges"
        )

    def register_place_handles(self, *handles) -> None:
        """Register scene handles (tiles / edges) so the Show Places checkbox controls them."""
        for h in handles:
            if h is not None:
                self._place_handles.append(h)

    # ------------------------------------------------------------------
    # Places layer with elevation control
    # ------------------------------------------------------------------

    def add_places_tiles(
        self,
        sid: str,
        tile_vertices: Optional[np.ndarray],
        tile_faces: Optional[np.ndarray],
        internal_edge_points: Optional[np.ndarray],
        cross_edge_points: Optional[np.ndarray],
    ) -> None:
        """Store raw (unshifted) places geometry and upload at the current elevation."""
        entry = self._places_tile_store.setdefault(sid, {})
        if tile_vertices is not None:
            entry["tile_vertices"] = tile_vertices
            entry["tile_faces"] = tile_faces
        if internal_edge_points is not None:
            entry["internal_edge_points"] = internal_edge_points
        if cross_edge_points is not None:
            entry["cross_edge_points"] = cross_edge_points
        for h in self._upload_places_tiles_for(sid):
            if h is not None:
                self._place_handles.append(h)

    def update_places_cross_edges(self, sid: str, edge_points: np.ndarray) -> None:
        """Update just the cross-edge geometry for a submap (e.g. after loop closure)."""
        self._places_tile_store.setdefault(sid, {})["cross_edge_points"] = edge_points
        for h in self._upload_places_tiles_for(sid):
            if h is not None:
                self._place_handles.append(h)

    def _upload_places_tiles_for(self, sid: str):
        entry = self._places_tile_store.get(sid)
        if not entry:
            return (None, None, None)
        dy = self.gui_places_elevation.value  # subtract from Y to go up (-y is up)
        visible = self.gui_show_places.value

        h_nodes = h_edges = h_cross = None

        if "tile_vertices" in entry and entry["tile_vertices"] is not None:
            verts = entry["tile_vertices"].copy()
            verts[:, 1] -= dy
            h_nodes = self.server.scene.add_mesh_simple(
                name="graph_nodes_" + sid,
                vertices=verts,
                faces=entry["tile_faces"],
                color=(0, 255, 0),
                opacity=0.7,
                side="double",
            )
        if "internal_edge_points" in entry and entry["internal_edge_points"] is not None:
            pts = entry["internal_edge_points"].copy()
            pts[:, :, 1] -= dy
            h_edges = self.server.scene.add_line_segments(
                name="graph_edges_" + sid,
                points=pts,
                colors=(0, 255, 0),
                line_width=2.0,
            )
        if "cross_edge_points" in entry and entry["cross_edge_points"] is not None:
            pts = entry["cross_edge_points"].copy()
            pts[:, :, 1] -= dy
            h_cross = self.server.scene.add_line_segments(
                name="global_edges_" + sid,
                points=pts,
                colors=(0, 255, 0),
                line_width=2.0,
            )

        if h_nodes is not None:
            h_nodes.visible = visible
        # Edges are shown only when the layer is visible AND edges aren't toggled off.
        for h in (h_edges, h_cross):
            if h is not None:
                h.visible = visible and self._show_place_edges
                self._place_edge_handles.append(h)
        return h_nodes, h_edges, h_cross

    def set_planned_path(self, path_points: np.ndarray) -> None:
        """Store raw path points and draw them at the current places elevation.

        Parameters
        ----------
        path_points : (N, 3) array of world-space waypoints along the path.
        """
        if len(path_points) < 2:
            return
        segments = np.stack([path_points[:-1], path_points[1:]], axis=1)  # (N-1, 2, 3)
        self._planned_path_segments = segments
        self._upload_planned_path()

    def _upload_planned_path(self) -> None:
        if self._planned_path_segments is None:
            return
        dy = self.gui_places_elevation.value
        segs = self._planned_path_segments.copy()
        segs[:, :, 1] -= dy
        self.server.scene.add_line_segments(
            name="shortest_path",
            points=segs,
            colors=(0, 0, 0),
            line_width=6.0,
        )

    def _redraw_places_tiles(self) -> None:
        """Re-upload all stored places tiles at the updated elevation."""
        self._place_handles.clear()
        self._place_edge_handles.clear()
        for sid in self._places_tile_store:
            for h in self._upload_places_tiles_for(sid):
                if h is not None:
                    self._place_handles.append(h)
        if self._frontier_place_graph is not None:
            self._apply_frontier_colors()
        self._upload_planned_path()

    def visualize_obb(
        self,
        center: np.ndarray,
        extent: np.ndarray,
        rotation: np.ndarray,
        color = (255, 0, 0),
        line_width: float = 4.0,
        detection=None,
    ):
        """Visualize an oriented bounding box (OBB) in Viser, respecting object elevation.

        When ``detection`` (a Detection) is given, the box is linked to it so
        refresh_object_geometry() can re-read the moved bbox after a loop closure.
        """
        name = f"obb_{self.obj_id}"
        self.obj_id += 1
        self._obj_scene_names.append(name)
        self._obj_obb_store[name] = (center.copy(), extent.copy(), rotation.copy(),
                                     color, line_width, detection)
        self._upload_obj_obb(name)

    def _upload_obj_obb(self, name: str) -> None:
        """Draw one stored OBB at the current object elevation."""
        center, extent, rotation, color, line_width, _det = self._obj_obb_store[name]
        elev = self._get_obj_elevation()
        shifted_center = center.copy()
        shifted_center[1] -= elev

        dx, dy, dz = extent / 2.0
        corners_local = np.array([
            [-dx, -dy, -dz],
            [ dx, -dy, -dz],
            [ dx,  dy, -dz],
            [-dx,  dy, -dz],
            [-dx, -dy,  dz],
            [ dx, -dy,  dz],
            [ dx,  dy,  dz],
            [-dx,  dy,  dz],
        ], dtype=np.float32)

        corners_world = (rotation @ corners_local.T).T + shifted_center

        edges_idx = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ]
        segments = []
        for (i, j) in edges_idx:
            segments.append(corners_world[i])
            segments.append(corners_world[j])
        segments = np.array(segments, dtype=np.float32).reshape(-1, 2, 3)

        self.server.scene.add_line_segments(
            name=name,
            points=segments,
            colors=color,
            line_width=line_width,
            visible=True,
        )

    def visualize_object_pointcloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        point_size: float = 0.005,
        detection=None,
    ) -> None:
        """Display a dense object point cloud in Viser, respecting object elevation.

        When ``detection`` is given, the cloud is linked to it so
        refresh_object_geometry() can re-read the moved points after a loop closure.
        """
        if points is None or len(points) == 0:
            return
        name = f"obj_pts_{self.obj_id}"
        self.obj_id += 1
        self._obj_scene_names.append(name)
        self._obj_pcd_store[name] = (points.astype(np.float32), colors, point_size, detection)
        self._upload_obj_pcd(name)


    def _get_obj_elevation(self) -> float:
        return self.gui_obj_elevation.value if self.gui_obj_elevation is not None else 0.0

    def _upload_obj_pcd(self, name: str) -> None:
        """Draw one stored object point cloud at the current object elevation."""
        raw_pts, colors, point_size, _det = self._obj_pcd_store[name]
        pts = raw_pts.copy()
        pts[:, 1] -= self._get_obj_elevation()
        self.server.scene.add_point_cloud(
            name=name,
            points=pts,
            colors=colors,
            point_size=point_size,
            point_shape="circle",
        )

    def _redraw_objects(self) -> None:
        """Re-upload all stored object point clouds and OBBs at the updated elevation."""
        for name in self._obj_pcd_store:
            self._upload_obj_pcd(name)
        for name in self._obj_obb_store:
            self._upload_obj_obb(name)

    def refresh_object_geometry(self) -> None:
        """Re-read displayed object geometry from its linked Detections and redraw.

        Detections are delta-H moved in place on a loop closure (see
        Objects.move_objects); this pulls their updated bbox / points into the
        viewer stores so the rendered boxes and clouds follow the correction.
        Only currently-displayed objects (those in the stores) are touched, so a
        cleared view stays cleared.
        """
        for name, entry in list(self._obj_obb_store.items()):
            center, extent, rotation, color, line_width, det = entry
            if det is not None and det.bbox is not None:
                b = det.bbox
                self._obj_obb_store[name] = (np.asarray(b.center, dtype=np.float32).copy(),
                                             np.asarray(b.extent, dtype=np.float32).copy(),
                                             np.asarray(b.rotation, dtype=np.float32).copy(),
                                             color, line_width, det)
        for name, entry in list(self._obj_pcd_store.items()):
            points, colors, point_size, det = entry
            if det is not None and det.points is not None:
                self._obj_pcd_store[name] = (np.asarray(det.points, dtype=np.float32),
                                             det.colors, point_size, det)
        self._redraw_objects()

    def visualize_vo_tracks(
        self,
        pts3d: np.ndarray,
        camera_pos: Optional[np.ndarray] = None,
        point_size: float = 0.015,
    ) -> None:
        """Overwrite the VO debug point cloud each call.

        pts3d      : (N, 3) world-space tracked points
        camera_pos : (3,) current camera position — drawn as a bright marker if given
        """
        if pts3d is None or len(pts3d) == 0:
            return
        colors = np.full((len(pts3d), 3), (255, 80, 0), dtype=np.uint8)  # orange
        self.server.scene.add_point_cloud(
            name="vo_tracks",
            points=pts3d.astype(np.float32),
            colors=colors,
            point_size=point_size,
            point_shape="circle",
        )
        if camera_pos is not None:
            marker = np.array([camera_pos], dtype=np.float32)
            self.server.scene.add_point_cloud(
                name="vo_camera_pos",
                points=marker,
                colors=np.array([[0, 200, 255]], dtype=np.uint8),  # cyan
                point_size=point_size * 3,
                point_shape="circle",
            )

    def add_point_cloud(
        self,
        name: str,
        points: np.ndarray,
        colors: np.ndarray,
        point_size: float = 0.005,
    ) -> None:
        """Store and display a point cloud, respecting the current height clip bounds.

        Replaces any previously stored cloud with the same name so that
        re-clipping always works from the full (voxel-downsampled) dataset.
        """
        self._pcd_store[name] = (points, colors, point_size)
        self._upload_point_cloud(name, points, colors, point_size)

    def _upload_point_cloud(
        self,
        name: str,
        points: np.ndarray,
        colors: np.ndarray,
        point_size: float,
    ) -> None:
        """Apply current Y-clip bounds and upload to the scene."""
        y_min = self.gui_clip_min.value
        y_max = self.gui_clip_max.value
        mask = (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        handle = self.server.scene.add_point_cloud(
            name=name,
            points=points[mask],
            colors=colors[mask],
            point_size=point_size,
            point_shape="circle",
        )
        handle.visible = self.gui_show_pointcloud.value
        self._pcd_handles[name] = handle

    def _redraw_point_clouds(self) -> None:
        """Re-upload all stored point clouds with the updated clip bounds."""
        for name, (points, colors, point_size) in self._pcd_store.items():
            self._upload_point_cloud(name, points, colors, point_size)

    def _on_update_show_pointcloud(self, _) -> None:
        visible = self.gui_show_pointcloud.value
        for handle in self._pcd_handles.values():
            handle.visible = visible

    # ------------------------------------------------------------------
    # First Person View helpers
    # ------------------------------------------------------------------

    def _get_floor_y_at(self, xz: np.ndarray) -> Optional[float]:
        """Return the raw Y of the place tile centroid closest to the given XZ position."""
        best_dist = float("inf")
        best_y = None
        for entry in self._places_tile_store.values():
            verts = entry.get("tile_vertices")
            if verts is None or len(verts) == 0:
                continue
            n_tiles = len(verts) // 4
            for i in range(n_tiles):
                quad = verts[i * 4 : i * 4 + 4]
                cx, cy, cz = quad[:, 0].mean(), quad[:, 1].mean(), quad[:, 2].mean()
                dist = (cx - xz[0]) ** 2 + (cz - xz[1]) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_y = cy
        return best_y

    def _enter_fpv(self) -> None:
        """Snap all connected clients to the FPV height when FPV Mode is toggled on."""
        for client in self.server.get_clients().values():
            pos  = np.array(client.camera.position, dtype=float)
            look = np.array(client.camera.look_at,  dtype=float)
            xz   = np.array([pos[0], pos[2]])
            floor_y = self._get_floor_y_at(xz)
            if floor_y is None:
                continue
            new_y = floor_y - self.gui_places_elevation.value - self.gui_fpv_height.value
            # Preserve horizontal look direction
            fwd = look - pos
            fwd[1] = 0.0
            norm = np.linalg.norm(fwd)
            if norm > 1e-6:
                fwd /= norm
            else:
                fwd = np.array([0.0, 0.0, 1.0])
            new_pos  = np.array([pos[0], new_y, pos[2]])
            new_look = new_pos + fwd * self._FPV_LOOK_DIST
            client.camera.position = new_pos
            client.camera.look_at  = new_look

    def _fpv_move_for_client(self, client, right: int, forward: int) -> None:
        """Move a single client in the XZ plane, snapping Y to the local floor height."""
        if not self.gui_fpv_enabled.value:
            return
        step = self.gui_fpv_step.value
        pos  = np.array(client.camera.position, dtype=float)
        look = np.array(client.camera.look_at,  dtype=float)

        fwd = look - pos
        fwd[1] = 0.0
        norm = np.linalg.norm(fwd)
        if norm < 1e-6:
            return
        fwd /= norm
        rgt = np.array([fwd[2], 0.0, -fwd[0]])

        delta  = fwd * (forward * step) + rgt * (right * step)
        new_xz = np.array([pos[0] + delta[0], pos[2] + delta[2]])

        floor_y = self._get_floor_y_at(new_xz)
        new_y   = (
            floor_y - self.gui_places_elevation.value - self.gui_fpv_height.value
            if floor_y is not None else pos[1]
        )

        new_pos  = np.array([new_xz[0], new_y, new_xz[1]])
        new_look = new_pos + fwd * self._FPV_LOOK_DIST
        client.camera.position = new_pos
        client.camera.look_at  = new_look

    def _fpv_move(self, right: int, forward: int) -> None:
        """Move all connected clients (delegates to _fpv_move_for_client)."""
        for client in self.server.get_clients().values():
            self._fpv_move_for_client(client, right, forward)

    def clear_objects(self) -> None:
        """Remove all object point clouds and OBBs from the scene."""
        for name in self._obj_scene_names:
            self.server.scene.remove_by_name(name)
        self._obj_scene_names.clear()
        self._obj_pcd_store.clear()
        self._obj_obb_store.clear()

    def set_planning_start_provider(self, provider) -> None:
        """Register a callable returning the current camera world-position (3,) to
        use as the path-planning start, or None to fall back to the latest submap
        pose.  Used by the ROS wrapper to plan from the odom-propagated pose.
        """
        self._planning_start_provider = provider

    def add_robot_urdf(self, urdf_path: str, base_frame_link: Optional[str] = None,
                       package_root: Optional[str] = None,
                       root_node_name: str = "/robot"):
        """Load a robot URDF and display it in the scene under ``root_node_name``.

        The robot is parented to a single frame so its whole pose can be updated
        with set_robot_pose().  ``package://<pkg>/...`` mesh references are
        resolved under ``package_root``; when None it defaults to the directory
        two levels above the URDF (i.e. the workspace root that contains the
        ``<pkg>`` folder — the standard ``<root>/<pkg>/urdf/<file>.urdf`` layout).

        ``base_frame_link`` is the URDF link that the pose passed to
        set_robot_pose() refers to (typically the odom base link).  When it
        differs from the URDF root (e.g. root ``footprint`` vs odom ``base_link``),
        the fixed root→base offset is folded in so the given pose lands on that
        link rather than the root.

        Returns the viser.extras.ViserUrdf handle, or None if loading failed
        (failure is logged and swallowed so it never breaks the viewer).
        """
        import os
        try:
            import yourdfpy
            from viser.extras import ViserUrdf
        except ImportError as e:
            print(f"[Viewer] Robot URDF disabled — missing dependency: {e}")
            return None

        if package_root is None:
            package_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(urdf_path))))

        def _filename_handler(fname: str) -> str:
            if fname.startswith("package://"):
                return os.path.join(package_root, fname[len("package://"):])
            return fname

        try:
            robot = yourdfpy.URDF.load(
                urdf_path, filename_handler=_filename_handler,
                load_meshes=True, build_collision_scene_graph=False,
                load_collision_meshes=False,
            )
            # Pose passed to set_robot_pose() refers to base_frame_link; the frame
            # we actually move is the URDF root, so store root-in-base and fold it
            # in later (identity when the root IS the base link).
            self._robot_root_offset = np.eye(4)
            if base_frame_link and base_frame_link in robot.link_map:
                self._robot_root_offset = np.asarray(
                    robot.get_transform(frame_to=robot.base_link,
                                        frame_from=base_frame_link), dtype=np.float64)
            elif base_frame_link:
                print(f"[Viewer] Robot URDF has no link '{base_frame_link}'; placing "
                      f"its root ('{robot.base_link}') at the given pose.")
            # Container frame so the whole robot can be repositioned at once.
            self._robot_base_frame = self.server.scene.add_frame(
                root_node_name, show_axes=False)
            self._robot_urdf = ViserUrdf(self.server, robot, root_node_name=root_node_name)
            print(f"[Viewer] Loaded robot URDF '{os.path.basename(urdf_path)}' "
                  f"({len(robot.link_map)} links).")
            return self._robot_urdf
        except Exception as e:  # noqa: BLE001 — never let a bad URDF kill the viewer
            import traceback
            traceback.print_exc()
            print(f"[Viewer] Failed to load robot URDF '{urdf_path}': {e}")
            return None

    def set_robot_pose(self, T_world_base: np.ndarray) -> None:
        """Move the loaded robot URDF so ``base_frame_link`` sits at the given 4×4
        base-to-world pose.

        No-op when no URDF has been loaded.
        """
        if self._robot_base_frame is None:
            return
        T_world_root = np.asarray(T_world_base, dtype=np.float64) @ self._robot_root_offset
        self._robot_base_frame.position = np.asarray(T_world_root[:3, 3], dtype=np.float32)
        self._robot_base_frame.wxyz = viser_tf.SO3.from_matrix(T_world_root[:3, :3]).wxyz

    def add_object_query_gui(self, objects_module, solver, data_lock) -> None:
        """Add an object query panel to the viser sidebar.

        Provides a text input + Search button so the user can query objects
        without touching the terminal.  Results are shown as markdown in the
        panel; matched point clouds / OBBs appear in the 3-D scene.
        """
        with self.server.gui.add_folder("Object Query"):
            gui_query  = self.server.gui.add_text("Query", initial_value="")
            gui_status = self.server.gui.add_markdown("*Enter a query and press Search.*")
            self.gui_obj_elevation = self.server.gui.add_slider(
                "Elevation", min=-2.0, max=5.0, step=0.05, initial_value=0.0,
            )
            self.gui_obj_elevation.on_update(lambda _: self._redraw_objects())
            btn_search = self.server.gui.add_button("Search", color="green")
            btn_clear  = self.server.gui.add_button("Clear Results", color="red")
            btn_plan   = self.server.gui.add_button("Plan Path to Object", color="blue")
            btn_missing = self.server.gui.add_button("Check for Missing Objects")
            gui_missing = self.server.gui.add_markdown("*Press to check for missing objects.*")
            btn_added = self.server.gui.add_button("Check for New Objects")
            gui_added = self.server.gui.add_markdown("*Press to check for new objects.*")

        @btn_search.on_click
        def _on_search(_) -> None:
            query = gui_query.value.strip()
            if not query:
                gui_status.content = "*Enter a query first.*"
                return
            gui_status.content = f"*Searching for '{query}'…*"
            with data_lock:
                detected = objects_module.get_3D_object(query, solver)
            if detected:
                gui_status.content = (
                    f"**Found {len(detected)} instance(s)** for *'{query}'*"
                )
            else:
                gui_status.content = f"*No results for '{query}'.*"

        @btn_clear.on_click
        def _on_clear(_) -> None:
            self.clear_objects()                 # scene geometry (pcds + OBBs)
            with data_lock:
                objects_module.clear_cache()     # cached queries, detections, keys, timings
            gui_status.content = "*Cleared.*"

        @btn_missing.on_click
        def _on_check_missing(_) -> None:
            gui_missing.content = "*Checking for missing objects…*"
            with data_lock:
                missing = objects_module.check_missing_object(solver)
            if missing:
                lines = "\n".join(f"- {name}" for name in missing)
                gui_missing.content = f"**Missing objects ({len(missing)}):**\n\n{lines}"
            else:
                gui_missing.content = "**None Detected.**"

        @btn_added.on_click
        def _on_check_added(_) -> None:
            gui_added.content = "*Checking for new objects…*"
            with data_lock:
                added = objects_module.check_added_object(solver)
            if added:
                lines = "\n".join(f"- {name}" for name in added)
                gui_added.content = f"**New objects ({len(added)}):**\n\n{lines}"
            else:
                gui_added.content = "**None Detected.**"

        @btn_plan.on_click
        def _on_plan_path(_) -> None:
            if objects_module.last_detected_object is None:
                gui_status.content = "*Search for an object first.*"
                return
            if solver.places is None:
                gui_status.content = "*Places layer not enabled. Run with --places.*"
                return
            # Prefer the registered start provider (e.g. the odom-propagated
            # current pose); otherwise fall back to the latest submap pose.
            camera_pos = None
            if self._planning_start_provider is not None:
                camera_pos = self._planning_start_provider()
            if camera_pos is None:
                with data_lock:
                    latest_submap = solver.map.get_latest_submap(ignore_loop_closure_submaps=True)
                    if latest_submap is None:
                        gui_status.content = "*No map data yet.*"
                        return
                    poses = latest_submap.get_all_poses_world(solver.graph)
                camera_pos = poses[-1, 0:3, 3]
            obj_pos = objects_module.last_detected_object.bbox.center
            gui_status.content = "*Planning path…*"
            objects_module.plan_path(camera_pos, obj_pos, solver)
            gui_status.content = "*Path planned to last queried object.*"

    def get_planned_path_points(self) -> "np.ndarray | None":
        """Return the raw (N, 3) world-space waypoints of the current planned path, or None."""
        if self._planned_path_segments is None:
            return None
        segs = self._planned_path_segments          # (N-1, 2, 3)
        return np.vstack([segs[:, 0], segs[-1:, 1]])  # (N, 3)

    def set_waypoint_spheres(self, waypoints: np.ndarray) -> None:
        """Draw each waypoint as a small orange icosphere. Call before starting to follow."""
        self.clear_waypoint_spheres()
        self._waypoint_sphere_positions = waypoints.copy()
        elevation = self.gui_places_elevation.value if self.gui_places_elevation is not None else 0.0
        for i, wp in enumerate(waypoints):
            pos = (float(wp[0]), float(wp[1]) + elevation, float(wp[2]))
            handle = self.server.scene.add_icosphere(
                f"waypoints/wp_{i}",
                radius=0.06,
                color=(255, 140, 0),  # orange
                position=pos,
            )
            self._waypoint_sphere_handles.append(handle)

    def mark_waypoint_reached(self, idx: int) -> None:
        """Turn waypoint idx green to indicate the robot has passed it."""
        if (self._waypoint_sphere_positions is None
                or idx >= len(self._waypoint_sphere_positions)):
            return
        wp = self._waypoint_sphere_positions[idx]
        elevation = self.gui_places_elevation.value if self.gui_places_elevation is not None else 0.0
        pos = (float(wp[0]), float(wp[1]) + elevation, float(wp[2]))
        # Re-call with the same name to overwrite color in place.
        handle = self.server.scene.add_icosphere(
            f"waypoints/wp_{idx}",
            radius=0.06,
            color=(0, 200, 80),  # green
            position=pos,
        )
        if idx < len(self._waypoint_sphere_handles):
            self._waypoint_sphere_handles[idx] = handle

    def set_stop_point_sphere(self, point) -> None:
        """Draw a blue sphere at the path's actual end point — where the robot will
        stop after the stop-short trim. Overwrites any previous one."""
        self.clear_stop_point_sphere()
        elevation = self.gui_places_elevation.value if self.gui_places_elevation is not None else 0.0
        pos = (float(point[0]), float(point[1]) + elevation, float(point[2]))
        self._stop_point_sphere_handle = self.server.scene.add_icosphere(
            "waypoints/stop_point",
            radius=0.09,          # a touch larger than the waypoints so it stands out
            color=(0, 90, 255),   # blue
            position=pos,
        )

    def clear_stop_point_sphere(self) -> None:
        """Remove the blue stop-point sphere from the scene."""
        if self._stop_point_sphere_handle is not None:
            try:
                self._stop_point_sphere_handle.remove()
            except Exception:
                pass
            self._stop_point_sphere_handle = None

    def clear_waypoint_spheres(self) -> None:
        """Remove all waypoint spheres (and the stop-point sphere) from the scene."""
        for h in self._waypoint_sphere_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._waypoint_sphere_handles = []
        self._waypoint_sphere_positions = None
        self.clear_stop_point_sphere()

    def add_follow_path_button(self, start_cb, stop_cb) -> None:
        """Add Follow Path / Stop Following buttons to the sidebar.

        Args:
            start_cb: called with (waypoints: np.ndarray shape (N,3)) when the
                      user clicks Follow Path.
            stop_cb:  called with no arguments when the user clicks Stop.
        """
        with self.server.gui.add_folder("Waypoint Following"):
            btn_follow = self.server.gui.add_button("Follow Path", color="green")
            btn_stop   = self.server.gui.add_button("Stop Following", color="red")
            gui_status = self.server.gui.add_markdown("*No path planned yet.*")

        @btn_follow.on_click
        def _on_follow(_) -> None:
            waypoints = self.get_planned_path_points()
            if waypoints is None or len(waypoints) < 2:
                gui_status.content = "⚠ Plan a path first."
                return
            gui_status.content = f"*Following — {len(waypoints)} waypoints.*"
            start_cb(waypoints)

        @btn_stop.on_click
        def _on_stop(_) -> None:
            gui_status.content = "*Stopped.*"
            stop_cb()

    def add_cancel_button(self, stop_event: threading.Event) -> None:
        """Add a Cancel Session button that signals the camera loop to stop."""
        btn = self.server.gui.add_button("Cancel Session", color="red")

        @btn.on_click
        def _on_cancel(_) -> None:
            stop_event.set()
            btn.disabled = True

    def add_get_object_pcd_button(self, request_cb) -> None:
        """Add a 'Get Object PCD in Cam' button.

        Clicking it calls ``request_cb()`` (no args).  The capture loop fulfils
        the request: it forces the current keyframe buffer into a submap so a
        current camera pose exists, then saves the last queried object's point
        cloud expressed in that camera's frame.  Fulfilment is asynchronous
        (handled on the loop's next iteration), so progress is printed to the
        console rather than reported back here.
        """
        with self.server.gui.add_folder("Object Capture"):
            btn = self.server.gui.add_button("Get Object PCD in Cam", color="blue")

        @btn.on_click
        def _on_get_object_pcd(_) -> None:
            request_cb()

    def add_place_object_button(self, callback) -> None:
        """Add a 'Place Object' button (used by the ROS wrapper).

        Clicking it calls ``callback(set_status)`` where ``set_status(msg: str)``
        writes this panel's markdown.  The callback publishes the last queried
        object's bounding-box centroid, expressed in the current robot/camera
        frame, as a ROS point.  Kept generic here so the non-ROS entry points that
        never wire it pay nothing.
        """
        with self.server.gui.add_folder("Place Object"):
            btn = self.server.gui.add_button("Place Object", color="blue")
            gui_status = self.server.gui.add_markdown(
                "*Press to publish the last queried object's location.*"
            )

        def _set_status(msg: str) -> None:
            gui_status.content = msg

        @btn.on_click
        def _on_place_object(_) -> None:
            callback(_set_status)

    def get_last_room_center(self) -> Optional[np.ndarray]:
        """World-space centre (3,) of the most recent room query's top region, or
        None if no room has been queried yet."""
        return self._last_room_center

    def add_go_to_room_button(self, callback) -> None:
        """Add a 'Go to Room' button (used by the ROS wrapper).

        Clicking it calls ``callback(set_status)`` where ``set_status(msg: str)``
        writes this panel's markdown.  The callback navigates the robot to the
        last queried region (see get_last_room_center).  Kept generic here so the
        non-ROS entry points that never wire it pay nothing.
        """
        with self.server.gui.add_folder("Go to Room"):
            btn = self.server.gui.add_button("Go to Room", color="blue")
            gui_status = self.server.gui.add_markdown(
                "*Press to navigate to the last queried room.*"
            )

        def _set_status(msg: str) -> None:
            gui_status.content = msg

        @btn.on_click
        def _on_go_to_room(_) -> None:
            callback(_set_status)

    def add_reset_scene_button(self, callback) -> None:
        """Add a 'Reset Scene' button that launches the autonomous patrol.

        Clicking it calls ``callback(set_status)`` where ``set_status(msg: str)``
        writes this panel's markdown, so the mission can report progress in the
        sidebar.  The callback is expected to run the mission on a background
        thread (the click handler returns immediately).  Kept generic here so the
        non-ROS entry points that never wire it pay nothing.
        """
        with self.server.gui.add_folder("Reset Scene"):
            btn = self.server.gui.add_button("Reset Scene", color="blue")
            gui_status = self.server.gui.add_markdown(
                "*Press to patrol cached objects and recapture moved ones.*"
            )

        def _set_status(msg: str) -> None:
            gui_status.content = msg

        @btn.on_click
        def _on_reset_scene(_) -> None:
            callback(_set_status)

    def add_room_query_gui(self, objects_module, solver, data_lock, top_region: bool = False) -> None:
        """Add a room query panel to the viser sidebar.

        Provides a text input + Search button so the user can query rooms/regions
        by CLIP text description while the SLAM run is live.  Results appear as
        coloured overlays above the places layer tiles (same as visualize_clusters).
        Requires --places and --run_os (for the CLIP model).
        """
        from foundit.places.clustering.query import create_query_clusterer, query_region
        from foundit.places.config import QueryClusterConfig

        _clusterer = [None]  # lazy-init, one instance shared across all clicks

        with self.server.gui.add_folder("Room Query"):
            gui_query  = self.server.gui.add_text("Query", initial_value="")
            gui_status = self.server.gui.add_markdown("*Enter a room description and press Search.*")
            btn_search  = self.server.gui.add_button("Search", color="green")
            btn_clear   = self.server.gui.add_button("Clear Overlays", color="red")

        @btn_search.on_click
        def _on_search(_) -> None:
            query = gui_query.value.strip()
            if not query:
                gui_status.content = "*Enter a query first.*"
                return
            if solver.places is None:
                gui_status.content = "*Places layer not enabled. Run with --places.*"
                return
            if objects_module.clip_model is None:
                gui_status.content = "*CLIP model not loaded. Run with --run_os.*"
                return
            gui_status.content = f"*Searching for '{query}'…*"

            if _clusterer[0] is None:
                _clusterer[0] = create_query_clusterer(
                    QueryClusterConfig(),
                    clip_model=objects_module.clip_model,
                    clip_tokenizer=objects_module.clip_tokenizer,
                )

            with data_lock:
                output = query_region(_clusterer[0], solver.places.graph, query,
                                      multi_region=not top_region)

            if output is None:
                gui_status.content = f"*No regions found for '{query}'.*"
                return

            n = len(output.regions)
            gui_status.content = f"**Found {n} region(s)** for *'{query}'*"
            # Remember the top region's centre (regions are sorted by confidence)
            # as the "Go to Room" navigation target.
            self._last_room_center = np.asarray(output.regions[0].centroid, dtype=np.float64)
            self.visualize_clusters(output.vis_result, solver.places)

        @btn_clear.on_click
        def _on_clear(_) -> None:
            self._clear_room_overlays()
            gui_status.content = "*Cleared.*"

    # ------------------------------------------------------------------
    # Frontier visualisation
    # ------------------------------------------------------------------

    def add_frontier_gui(self, solver, data_lock) -> None:
        """Add a Frontier panel to the viser sidebar.

        Classifies leaf nodes of the places graph as blocked (black, wall
        nearby) or unexplored (orange, open space), flags interior nodes hugging
        the boundary as near-boundary (red), and renders them as tile overlays.
        Requires --places.
        """
        with self.server.gui.add_folder("Frontier"):
            gui_status = self.server.gui.add_markdown("*Press Update to classify frontier nodes.*")
            btn_update = self.server.gui.add_button("Update Frontiers", color="blue")
            btn_clear  = self.server.gui.add_button("Clear Frontiers",  color="red")

        @btn_update.on_click
        def _on_update(_) -> None:
            try:
                if solver.places is None:
                    gui_status.content = "*Places layer not enabled. Run with --places.*"
                    return
                gui_status.content = "*Classifying frontier nodes…*"
                with data_lock:
                    classification = solver.compute_frontier()
                    graph    = solver.places.graph
                    rotations = dict(solver.places.submap_plane_rotations)
                    place_size = solver.places.place_size
                print(f"[Frontier] blocked={len(classification.blocked)}  "
                      f"unexplored={len(classification.unexplored)}  "
                      f"near_boundary={len(classification.near_boundary)}")
                self.show_frontier_tiles(classification, graph, rotations, place_size)
                gui_status.content = (
                    f"**Frontier:** {len(classification.unexplored)} unexplored (orange), "
                    f"{len(classification.blocked)} blocked (black), "
                    f"{len(classification.near_boundary)} near-boundary (red)"
                )
            except Exception as e:
                import traceback
                msg = traceback.format_exc()
                print(f"[Frontier ERROR]\n{msg}")
                gui_status.content = f"*Error: {e}*"

        @btn_clear.on_click
        def _on_clear_frontier(_) -> None:
            self.clear_frontier_tiles()
            gui_status.content = "*Frontier cleared.*"

    def show_frontier_tiles(
        self,
        classification,
        place_graph,
        submap_plane_rotations: dict,
        place_size: float,
    ) -> None:
        """Recolor existing tiles from a FrontierClassification.

        blocked → black, unexplored → orange, near-boundary → red, interior → green.
        """
        self._frontier_blocked_set = set(classification.blocked)
        self._frontier_unexplored_set = set(classification.unexplored)
        self._frontier_near_boundary_set = set(classification.near_boundary)
        self._frontier_place_graph = place_graph
        self._frontier_submap_rotations = submap_plane_rotations
        self._frontier_place_size = place_size
        self._apply_frontier_colors()

    def _apply_frontier_colors(self) -> None:
        """Split each submap's tile mesh into interior/blocked/unexplored/near-boundary and recolor."""
        from foundit.places.frontier import build_frontier_mesh

        place_graph = self._frontier_place_graph
        if place_graph is None:
            return

        R_map = self._frontier_submap_rotations
        place_size = self._frontier_place_size
        dy = self.gui_places_elevation.value
        visible = self.gui_show_places.value

        blocked = self._frontier_blocked_set
        unexplored = self._frontier_unexplored_set
        near_boundary = self._frontier_near_boundary_set

        _, all_active = place_graph.get_active_node_data()
        active_set = set(all_active)

        for sid in list(self._places_tile_store.keys()):
            node_indices = [
                i for i in place_graph.get_submap_node_indices(sid) if i in active_set
            ]
            if not node_indices:
                continue
            R = R_map.get(sid, np.eye(3))

            # The classification sets are mutually disjoint, so each node lands in
            # exactly one bucket below; anything left over is ordinary interior.
            for nodes, color, suffix in [
                ([i for i in node_indices
                  if i not in blocked and i not in unexplored and i not in near_boundary],
                 (0, 255, 0),   ""),
                ([i for i in node_indices if i in blocked],
                 (10, 10, 10),  "_blocked"),
                ([i for i in node_indices if i in unexplored],
                 (255, 140, 0), "_unexplored"),
                ([i for i in node_indices if i in near_boundary],
                 (255, 0, 0),   "_near_boundary"),
            ]:
                mesh_name = f"graph_nodes_{sid}{suffix}"
                if nodes:
                    centers = np.array([place_graph.get_position(i) for i in nodes], dtype=np.float32)
                    verts, faces = build_frontier_mesh(centers, R, place_size)
                    verts[:, 1] -= dy
                    h = self.server.scene.add_mesh_simple(
                        name=mesh_name, vertices=verts, faces=faces,
                        color=color, opacity=0.7, side="double",
                    )
                else:
                    h = self.server.scene.add_mesh_simple(
                        name=mesh_name,
                        vertices=np.zeros((3, 3), dtype=np.float32),
                        faces=np.zeros((1, 3), dtype=np.int32),
                        color=color, opacity=0.0,
                    )
                if h is not None:
                    h.visible = visible

    def clear_frontier_tiles(self) -> None:
        """Restore all tile meshes to solid green and remove frontier coloring."""
        if self._frontier_place_graph is None:
            return
        for sid in list(self._places_tile_store.keys()):
            for suffix in ("_blocked", "_unexplored", "_near_boundary"):
                self.server.scene.add_mesh_simple(
                    name=f"graph_nodes_{sid}{suffix}",
                    vertices=np.zeros((3, 3), dtype=np.float32),
                    faces=np.zeros((1, 3), dtype=np.int32),
                    color=(0, 0, 0), opacity=0.0,
                )
        self._frontier_blocked_set = set()
        self._frontier_unexplored_set = set()
        self._frontier_near_boundary_set = set()
        self._frontier_place_graph = None
        self._redraw_places_tiles()

    def visualize_clusters(self, result: ClusterResult, places) -> None:
        """Overlay elevated, coloured room tiles above the existing green places tiles.

        The green floor-level places tiles are left untouched.  A new set of
        coloured tiles is added above the scene at the current Room Overlay
        elevation, and a legend is shown in the sidebar.  Drag the
        "Elevation above scene" slider to raise or lower them.

        For ``QueryClusterResult`` with ``node_scores``, tiles are coloured
        per-node using a viridis colormap; non-query results get one colour
        per cluster instance.

        Parameters
        ----------
        result : ClusterResult from any AbstractClusterer
        places : Places facade (provides all_graph_nodes / submap_plane_rotations)
        """
        if not result.cluster_instances:
            print("[Clusters] Nothing to visualise -- run cluster() first.")
            return

        # Assign a unique query id and append to history (previous queries stay visible)
        query_id = self._room_query_counter
        self._room_query_counter += 1
        self._cluster_render_history.append((query_id, result, places))

        # Claim ownership of every node in this query, overriding any prior owner.
        # Affected old tiles will be rebuilt without those nodes in _redraw_room_tiles.
        for inst_idx, (_, node_indices) in enumerate(result.cluster_instances):
            for gidx in node_indices:
                self._node_owner[gidx] = (query_id, inst_idx)

        # Full redraw so old tiles are updated to shed any nodes this query took over
        self._redraw_room_tiles()
        print(f"[Clusters] Added query {query_id} — "
              f"{len(result.cluster_instances)} cluster(s). "
              f"Total queries: {len(self._cluster_render_history)}.")

    def _get_room_color(self, query_id: int, inst_idx: int) -> tuple[int, int, int]:
        """Return the stable color for this (query, instance) pair, assigning one if new."""
        key = (query_id, inst_idx)
        if key not in self._room_color_assignments:
            color = _ROOM_PALETTE[self._room_palette_idx % len(_ROOM_PALETTE)]
            self._room_color_assignments[key] = color
            self._room_palette_idx += 1
        return self._room_color_assignments[key]

    def _redraw_room_tiles(self) -> None:
        """Re-draw all accumulated room queries at the updated elevation."""
        if not self._cluster_render_history:
            return

        for name in self._cluster_tile_names:
            self.server.scene.remove_by_name(name)
        self._cluster_tile_names.clear()

        elevation = self.gui_room_elevation.value
        for query_id, result, places in self._cluster_render_history:
            hs = (places.place_size / 2.0) * 0.85
            base_node_verts = np.array([
                [-hs, -hs, 0.0],
                [ hs, -hs, 0.0],
                [ hs,  hs, 0.0],
                [-hs,  hs, 0.0],
            ], dtype=np.float32)
            is_query = isinstance(result, QueryClusterResult) and result.node_scores is not None
            if is_query:
                self._visualize_query_clusters(query_id, result, places, base_node_verts, elevation)
            else:
                self._visualize_standard_clusters(query_id, result, places, base_node_verts, elevation)

        self._rebuild_legend()

    def _rebuild_legend(self) -> None:
        """Rebuild the Room Legend folder to reflect all current queries."""
        if self._cluster_legend_folder is not None:
            self._cluster_legend_folder.remove()
            self._cluster_legend_folder = None

        if not self._cluster_render_history:
            return

        lines = []
        for query_id, result, _places in self._cluster_render_history:
            is_query = isinstance(result, QueryClusterResult) and result.node_scores is not None
            if is_query:
                label = result.label_names[0] if result.label_names else "query"
                lines.append(f"**Q{query_id}: {label}** — turbo (low→high conf)")
            else:
                used = sorted({li for li, _ in result.cluster_instances})
                for li in used:
                    label = result.label_names[li] if li < len(result.label_names) else f"region_{li}"
                    r, g, b = (int(c) for c in result.label_colors[li])
                    lines.append(f"**Q{query_id}: {label}** `#{r:02X}{g:02X}{b:02X}`")

        self._cluster_legend_folder = self.server.gui.add_folder("Room Legend")
        with self._cluster_legend_folder:
            self.server.gui.add_markdown("\n\n".join(f"- {l}" for l in lines))

    def _clear_room_overlays(self) -> None:
        """Remove all accumulated room overlay tiles and reset history."""
        for name in self._cluster_tile_names:
            self.server.scene.remove_by_name(name)
        self._cluster_tile_names.clear()
        if self._cluster_legend_folder is not None:
            self._cluster_legend_folder.remove()
            self._cluster_legend_folder = None
        self._cluster_render_history.clear()
        self._room_query_counter = 0
        self._room_color_assignments.clear()
        self._room_palette_idx = 0
        self._node_owner.clear()
        print("[Places] All room overlays cleared.")

    def _visualize_query_clusters(
        self,
        query_id: int,
        result: QueryClusterResult,
        places,
        base_node_verts: np.ndarray,
        elevation: float,
    ) -> None:
        """Draw query-cluster tiles, one unique color per room instance."""
        for inst_idx, (label_idx, node_indices) in enumerate(result.cluster_instances):
            # Only draw nodes still owned by this (query, instance)
            owned = [g for g in node_indices if self._node_owner.get(g) == (query_id, inst_idx)]
            if not owned:
                continue

            color = self._get_room_color(query_id, inst_idx)

            verts_list, faces_list = [], []
            for flat_i, gidx in enumerate(owned):
                world_center, submap_id = places.all_graph_nodes[gidx]
                elevated_center = world_center.copy()
                elevated_center[1] -= elevation
                R = places.submap_plane_rotations.get(submap_id, np.eye(3))
                verts = (base_node_verts @ R.T) + elevated_center
                b = flat_i * 4
                verts_list.append(verts.astype(np.float32))
                faces_list += [[b, b+1, b+2], [b, b+2, b+3]]

            tile_name = f"room_tiles_q{query_id}_{inst_idx}"
            self._cluster_tile_names.append(tile_name)
            self.server.scene.add_mesh_simple(
                name=tile_name,
                vertices=np.vstack(verts_list),
                faces=np.array(faces_list, dtype=np.uint32),
                color=color,
                opacity=0.85,
                side="double",
            )

    def _visualize_standard_clusters(
        self,
        query_id: int,
        result: ClusterResult,
        places,
        base_node_verts: np.ndarray,
        elevation: float,
    ) -> None:
        """Draw standard cluster tiles, one unique color per room instance."""
        for inst_idx, (label_idx, node_indices) in enumerate(result.cluster_instances):
            # Only draw nodes still owned by this (query, instance)
            owned = [g for g in node_indices if self._node_owner.get(g) == (query_id, inst_idx)]
            if not owned:
                continue

            color = self._get_room_color(query_id, inst_idx)

            verts_list, faces_list = [], []
            for flat_i, gidx in enumerate(owned):
                world_center, submap_id = places.all_graph_nodes[gidx]
                elevated_center = world_center.copy()
                elevated_center[1] -= elevation
                R = places.submap_plane_rotations.get(submap_id, np.eye(3))
                verts = (base_node_verts @ R.T) + elevated_center
                b = flat_i * 4
                verts_list.append(verts.astype(np.float32))
                faces_list += [[b, b+1, b+2], [b, b+2, b+3]]

            tile_name = f"room_tiles_q{query_id}_{inst_idx}"
            self._cluster_tile_names.append(tile_name)
            self.server.scene.add_mesh_simple(
                name=tile_name,
                vertices=np.vstack(verts_list),
                faces=np.array(faces_list, dtype=np.uint32),
                color=color,
                opacity=0.85,
                side="double",
            )

    def restore_full_places(self, places) -> None:
        """Remove room overlay tiles/legend.

        The green floor-level places tiles are always kept in the scene, so
        nothing needs to be rebuilt here.
        """
        for name in self._cluster_tile_names:
            self.server.scene.remove_by_name(name)
        self._cluster_tile_names.clear()
        if self._cluster_legend_folder is not None:
            self._cluster_legend_folder.remove()
            self._cluster_legend_folder = None
        self._cluster_render_history.clear()
        self._room_query_counter = 0
        print("[Places] Room overlay cleared.")

    def add_clustering_gui(
        self,
        initial_config: FiltrationClusterConfig,
        recluster_fn: Callable[[FiltrationClusterConfig], ClusterResult],
        places,
    ) -> None:
        """Add a viser GUI folder with controls for all FiltrationClusterConfig
        fields and a re-cluster button."""

        with self.server.gui.add_folder("Clustering"):
            gui_whiten = self.server.gui.add_checkbox(
                "Whiten Embeddings", initial_value=initial_config.whiten_embeddings,
            )
            gui_metric = self.server.gui.add_dropdown(
                "Distance Metric",
                options=[e.value for e in DistanceMetric],
                initial_value=initial_config.distance_metric.value,
            )
            gui_aggregation = self.server.gui.add_dropdown(
                "Distance Aggregation",
                options=[e.value for e in DistanceAggregation],
                initial_value=initial_config.distance_aggregation.value,
            )
            gui_threshold_mode = self.server.gui.add_dropdown(
                "Threshold Mode",
                options=[e.value for e in ThresholdMode],
                initial_value=initial_config.threshold_mode.value,
            )
            gui_plateau_ratio = self.server.gui.add_slider(
                "Plateau Ratio", min=0.0, max=1.0, step=0.01,
                initial_value=initial_config.plateau_ratio,
            )
            gui_min_dilation = self.server.gui.add_slider(
                "Min Dilation (m)", min=0.0, max=10.0, step=0.1,
                initial_value=initial_config.min_dilation_m,
            )
            gui_max_dilation = self.server.gui.add_slider(
                "Max Dilation (m)", min=0.0, max=10.0, step=0.1,
                initial_value=initial_config.max_dilation_m,
            )
            gui_min_component = self.server.gui.add_number(
                "Min Component Size", min=1, max=200, step=1,
                initial_value=initial_config.min_component_size,
            )
            gui_clustering_mode = self.server.gui.add_dropdown(
                "Clustering Mode",
                options=[e.value for e in ClusteringMode],
                initial_value=initial_config.clustering_mode.value,
            )
            gui_modularity_gamma = self.server.gui.add_slider(
                "Modularity Gamma", min=0.1, max=5.0, step=0.1,
                initial_value=initial_config.modularity_gamma,
            )

            btn = self.server.gui.add_button("Re-cluster", color="green")

            @btn.on_click
            def _on_recluster(_) -> None:
                config = FiltrationClusterConfig(
                    whiten_embeddings=gui_whiten.value,
                    distance_metric=DistanceMetric(gui_metric.value),
                    distance_aggregation=DistanceAggregation(gui_aggregation.value),
                    threshold_mode=ThresholdMode(gui_threshold_mode.value),
                    plateau_ratio=gui_plateau_ratio.value,
                    min_dilation_m=gui_min_dilation.value,
                    max_dilation_m=gui_max_dilation.value,
                    min_component_size=int(gui_min_component.value),
                    clustering_mode=ClusteringMode(gui_clustering_mode.value),
                    modularity_gamma=gui_modularity_gamma.value,
                )
                t0 = time.time()
                result = recluster_fn(config)
                dt = time.time() - t0
                print(f"[Clustering GUI] Re-clustered in {dt:.3f}s")
                self.visualize_clusters(result, places)

    def run_walkthrough(self, fps: float = 20.0):
            """
            Walks through the map using the current live positions of all frames.
            This accounts for loop closures because it pulls data from the scene handles.
            """
            # 1. Gather non-loop-closure submap IDs and sort them for a logical sequence.
            #    Loop-closure submaps are bridging mini-submaps, not part of the tour.
            sorted_submap_ids = sorted(
                sid for sid in self.submap_frames if sid not in self._lc_submap_ids
            )

            if not sorted_submap_ids:
                print("No frames found to walk through.")
                return

            clients = self.server.get_clients()
            if not clients:
                print("No clients connected to perform walkthrough.")
                return

            print("Starting walkthrough of updated poses...")

            for sub_id in sorted_submap_ids:
                frames = self.submap_frames[sub_id]
                # Assumes frames were added in chronological order to the list
                for frame_handle in frames:
                    # Get the current world-space pose from the visualizer
                    # If a loop closure moved the submap, these values will be updated
                    current_pos = frame_handle.position
                    current_wxyz = frame_handle.wxyz

                    # Update all connected clients
                    for client in clients.values():
                        client.camera.position = current_pos
                        client.camera.wxyz = current_wxyz
                    
                    # Control speed (1/fps)
                    time.sleep(1.0 / fps)