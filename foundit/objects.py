import time
import os
import subprocess
import tempfile
import numpy as np
import torch
import cv2

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from torchvision.transforms.functional import to_pil_image

import foundit.slam_utils as utils

# --- Object-approach path planning (see Objects.plan_path) ---
# Length of the final straight leg into an object, aimed along the line toward
# the nearest safe (green) tile so the robot approaches from open space and does
# not graze the wall/obstacle. The follower stops short (see --stop_short), so
# only part of this is actually driven.
APPROACH_NORMAL_LEN = 0.80   # metres


def _show_image(img):
    """Display a PIL image, using Windows viewer on WSL2."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(path)
    win_path = subprocess.check_output(["wslpath", "-w", path], text=True).strip()
    subprocess.Popen(["cmd.exe", "/c", "start", "", win_path],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

@dataclass
class Bbox:
    center: np.ndarray
    extent: np.ndarray
    rotation: np.ndarray
    label: Optional[str] = None

class CachedMemory:

    def __init__(self):
        self.objects = dict()
        self.query_times = dict()

    def __contains__(self, item):
        """Allows use of the 'in' operator to check if query is in cache AND there are detections for the query"""
        return item in self.objects and len(self.objects[item]) > 0

    def __iter__(self):
        """Allows use of 'for' loops"""
        return iter(self.objects)
    
    def items(self):
        return self.objects.items()
    
    def get_queries(self):
        return self.objects.keys()
    
    def get_detection(self, query):
        return self.objects.get(query, [])
    
    def get_time_queried(self, query):
        # None when the query has never been run → callers treat it as "no time
        # cutoff" (search all frames), which is correct for a first-time query.
        return self.query_times.get(query)
    
    def clear(self):
        """Forget every cached query — detections AND the query keys/timings."""
        self.objects.clear()
        self.query_times.clear()

    def add_object(self, query, detection):
        current_time = time.time()
        if query in self.objects:
            if detection is not None:
                self.objects[query].append(detection)
            self.query_times[query] = current_time
        else:
            self.objects[query] = [] if detection is None else [detection]
        self.query_times[query] = current_time

@dataclass
class Detection:
    """A detected object: 3D bbox plus optional per-detection data."""
    bbox: Bbox
    mask: Optional[np.ndarray] = None       # (H,W) numpy, values in [0,1]
    source_image: Optional[Any] = None      # PIL Image
    points: Optional[np.ndarray] = None     # (N,3) object points in world frame (at query time)
    colors: Optional[np.ndarray] = None     # (N,3) uint8 RGB, row-aligned with points
    submap_id: Optional[int] = None         # id of the submap SAM3 detected the object in
    frame_index: Optional[int] = None       # local frame index within that submap
    ref_homography: Optional[np.ndarray] = None  # graph H of (submap_id+frame_index) when
                                            # points were last computed; used to delta-H
                                            # move the object after graph re-optimization

class Objects:
    def __init__(self, use_smaller_pe_model, device="cuda"):
        self.device = device
        self.cached_memory = CachedMemory()
        self.top_k = 16
        self.clip_model = None
        self.clip_tokenizer = None
        self.clip_preprocess = None
        self.sam3_model = None
        self.processor = None
        self.last_detected_object = None  # most recent successful detection for path planning
        # check_added_object collapses same-query detections within this many metres
        # (multiple frames reporting the same physical object -> one detection).
        self.added_dedup_dist = 0.5
        self.filter_conf = True  # set to False via --no_filter_objects to skip confidence thresholding
        self.use_smaller_pe_model = use_smaller_pe_model
        # Grasping de-occlusion: rigidly align the cached (more complete) object cloud
        # onto the live current-frame detection with ICP so the published grasp cloud
        # is not truncated by occlusion (see detect_last_object_in_current_frame).
        self.grasp_icp_min_points = 30       # min points in either cloud to attempt ICP
        self.grasp_icp_max_corr_dist = 0.10  # m, fine-stage correspondence distance
        self.grasp_icp_min_fitness = 0.30    # min ICP overlap to trust the alignment

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def load_models(self, sam3_threshold=0.50):
        # Loading models here so the user does not have to load them if not running open-set object detection
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model.vitdet import compute_axial_cis
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        # Load image embedding models
        if self.use_smaller_pe_model:
            self.clip_model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True) # Smaller model (uses about 1/3 the time of L14)
        else:
            self.clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
        self.clip_model = self.clip_model.to(self.device)
        self.clip_tokenizer = transforms.get_text_tokenizer(self.clip_model.context_length)
        self.clip_preprocess = transforms.get_image_transform(self.clip_model.image_size)

        # Load SAM 3 model
        self.sam3_model = build_sam3_image_model(device=self.device)

        # Update SAM 3 positional encodings for new resolution
        new_resolution = 518
        patch_size = 14
        new_grid = new_resolution // patch_size
        model_device = next(self.sam3_model.parameters()).device
        for idx in [7, 15, 23, 31]:
            attn = self.sam3_model.backbone.vision_backbone.trunk.blocks[idx].attn
            scale_pos = attn.rope_pt_size[0] / new_grid
            new_freqs_cis = compute_axial_cis(attn.head_dim, new_grid, new_grid, attn.rope_theta, scale_pos)
            attn.freqs_cis = new_freqs_cis.to(model_device)

        self.processor = Sam3Processor(self.sam3_model, resolution = new_resolution, confidence_threshold=sam3_threshold)

    def _sam3_detect(self, best_img, query, img_h, img_w, lower=False):
        """Run single-query SAM3 on one frame image → (masks, scores).

        Shared by the single-frame detectors (get_3D_object's search loop,
        check_missing_object, detect_last_object_in_current_frame) so the
        set_image / set_text_prompt dance lives in exactly one place.
        """
        print("RUNNING SAM 3")
        with torch.no_grad():
            img_pil = to_pil_image(best_img)

            # We resize the image so that we get consistency with SAM 3 performance.
            # Letting this be the original image size could reduce runtime or alter accuracy. 
            orig_w, orig_h = img_pil.size
            target_w = 518
            target_h = int((target_w / orig_w) * orig_h)
            img_pil = img_pil.resize((target_w, target_h))

            if lower:
                self.processor.set_confidence_threshold(0.15)

            state = self.processor.set_image(img_pil, h=img_h, w=img_w)
            output = self.processor.set_text_prompt(state=state, prompt=query)
            if lower:
                self.processor.set_confidence_threshold(0.5) # TODO Dominic do not hardcode this
        return output["masks"], output["scores"]

    def _points_colors_from_mask(self, submap, frame_index, mask, img_np, solver):
        """World points + row-aligned RGB colors for one SAM3 mask on one frame.

        Honours self.filter_conf (confidence masking). Returns
        (points_3d (N,3) world, colors (N,3) uint8) or (None, None) when the mask
        yields no points.
        """
        if self.filter_conf:
            points_3d, point_conf_mask = submap.get_points_in_mask(
                frame_index, mask, solver.graph, use_conf_mask=True)
            colors = img_np[mask * point_conf_mask]
        else:
            points_3d = submap.get_points_in_mask(
                frame_index, mask, solver.graph, use_conf_mask=False)
            colors = img_np[mask]
        if points_3d is None or len(points_3d) == 0:
            return None, None
        return points_3d, colors

    def _search_single_query(self, query, ranking, solver, detected_bboxes,
                             viz_color, show_masks=False) -> List["Detection"]:
        """Search ``ranking`` (this query's OWN frames, descending CLIP score) in
        batches of top_k, running SAM3 per frame, skipping frames that already view
        a detected bbox, and early-stopping a batch once SAM3 has run in its second
        half with no find.  New detections are cached and returned.

        Because ``ranking`` is in this query's relevance order, the second-half
        early-stop is valid: once we are past the query's best frames without a
        find, later frames are even less likely.  (This is why the search is done
        per query rather than over a unified max-across-queries ranking, where a
        weak-scoring query's good frames sink below other queries' and the query
        would be dropped before they are ever examined.)

        ``detected_bboxes`` is a list of Bbox seeding the duplicate-view skip; it
        is mutated in place as new detections are found.  ``viz_color`` is the OBB
        colour, and ``show_masks`` pops up the per-frame mask overlay when True.
        """
        objects: List[Detection] = []
        batch_start = 0
        while batch_start < len(ranking):
            batch_end = min(batch_start + self.top_k, len(ranking))
            batch = ranking[batch_start:batch_end]
            midpoint = len(batch) // 2
            found_in_second_half = False
            sam3_ran_in_second_half = False

            print(f"\n--- Batch: frames {batch_start}-{batch_end - 1} ---")

            for idx, (score, submap_id, frame_index) in enumerate(batch):
                found_submap = solver.map.get_submap(submap_id)
                proj_mat = found_submap.get_project_mat(frame_index, solver.graph)
                best_img = found_submap.get_frame_at_index(frame_index)
                img_h, img_w = found_submap.get_img_size_at_index(frame_index)

                # Skip frames that already view a detected object for this query.
                is_duplicate_view = False
                for bbox in detected_bboxes:
                    obb_corners = utils.get_obb_corners(bbox.center, bbox.extent, bbox.rotation)
                    if utils.is_object_visible(obb_corners, proj_mat, (img_w, img_h)):
                        is_duplicate_view = True
                        break
                if is_duplicate_view:
                    continue

                if idx >= midpoint:
                    sam3_ran_in_second_half = True

                print(f"Checking Submap {submap_id}, Frame {frame_index} (Score: {score})")
                t1 = time.time()
                masks, scores = self._sam3_detect(best_img, query, img_h, img_w)
                print(f"SAM3 inference time: {(time.time() - t1) * 1000:.2f} ms")

                if masks.shape[0] == 0:
                    print(f"No masks found by SAM3 in Submap {submap_id}, Frame {frame_index}. Skipping.")
                    continue

                if idx >= midpoint:
                    found_in_second_half = True

                best_img_pil = to_pil_image(best_img).resize((img_w, img_h))
                if show_masks:
                    utils.overlay_masks(best_img_pil, masks).show()
                img_np = np.array(best_img_pil)   # (H, W, 3) uint8

                for i in range(masks.shape[0]):
                    mask = masks[i].cpu().numpy().squeeze()

                    points_3d, colors = self._points_colors_from_mask(
                        found_submap, frame_index, mask, img_np, solver)
                    if points_3d is None:
                        continue

                    obb = utils.compute_obb_from_points(points_3d)
                    if obb is None:
                        continue  # denoising left no points — skip this detection
                    obb_center, obb_extent, obb_rotation = obb

                    # Add to exclusion list for future iterations
                    detected_bboxes.append(Bbox(obb_center, obb_extent, obb_rotation))

                    bbox = Bbox(obb_center, obb_extent, obb_rotation, query)
                    obj = Detection(bbox=bbox, mask=mask, source_image=best_img_pil,
                                    points=points_3d, colors=colors,
                                    submap_id=submap_id, frame_index=frame_index,
                                    ref_homography=solver.graph.get_homography(submap_id + frame_index))

                    # Link the viewer geometry to the Detection so it can be
                    # refreshed when the object is delta-H moved on a loop closure.
                    solver.viewer.visualize_obb(
                        center=obb_center,
                        extent=obb_extent,
                        rotation=obb_rotation,
                        color=viz_color,
                        line_width=8.0,
                        detection=obj,
                    )
                    solver.viewer.visualize_object_pointcloud(points_3d, colors, detection=obj)

                    objects.append(obj)
                    self.cached_memory.add_object(query, obj)

            # Early stop: SAM3 ran in second half but found nothing there.
            if sam3_ran_in_second_half and not found_in_second_half:
                print("No objects found in second half of batch. Stopping search.")
                break

            batch_start = batch_end

        return objects

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def _detect_queries(self, solver, queries, detected_bboxes) -> Dict[str, List["Detection"]]:
        """Detect each query independently, ranking that query's OWN frames by its
        CLIP score so the per-query early-stop stays valid (see _search_single_query).

        ``detected_bboxes`` maps each query to a seed list of Bbox to treat as
        already-found (e.g. the cached detections, so only NEW locations count).
        Returns {query: [Detection]}.
        """
        query_objects: Dict[str, List[Detection]] = {}
        for query in queries:
            text_emb = utils.compute_text_embeddings(self.clip_model, self.clip_tokenizer, query)
            time_last_queried = self.cached_memory.get_time_queried(query)
            ranking = solver.map.retrieve_best_semantic_frames(text_emb, time_cutoff=time_last_queried)
            query_objects[query] = self._search_single_query(
                query, ranking, solver, detected_bboxes[query],
                viz_color=(0, 255, 0), show_masks=False,
            )
        return query_objects

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def get_3D_object(self, query, solver) -> List[Detection]:

        """
        inputs:
            query: string query for object retrieval, e.g. "chair", "table", "doorway".
            solver: VGGT-SLAM solver
        returns objects, a list of Detection instances or an empty list if no objects found.
        """

        cached = self.cached_memory.get_detection(query)
        if len(cached) > 0:
            print(f"Found {len(cached)} cached object(s) for query: '{query}'")
            self.last_detected_object = cached[0]
            return cached

        start_time = time.time()
        text_emb = utils.compute_text_embeddings(self.clip_model, self.clip_tokenizer, query)
        # This query's own frames ranked by CLIP score (all frames, no time cutoff).
        ranking = solver.map.retrieve_best_semantic_frames(text_emb)
        print(f"Searching {len(ranking)} ranked frames in batches of {self.top_k}...")

        objects = self._search_single_query(
            query, ranking, solver, detected_bboxes=[], viz_color=(255, 165, 0),
        )

        print("Total time taken:", time.time() - start_time)
        print(f"Total objects found: {len(objects)}")

        if objects:
            self.last_detected_object = objects[0]
        else:
            # add query to cached memory without any detected objects
            self.cached_memory.add_object(query, None)

        return objects

    def clear_cache(self):
        """Fully clear the object cache: all cached queries, their detections, the
        query keys, and their timings, plus the last-detected pointer.  After this a
        re-query starts from scratch (no cached hits, no cached empty-query keys)."""
        self.cached_memory.clear()
        self.last_detected_object = None

    def last_object_loop_target(self) -> Optional[Tuple[int, int]]:
        """Return (submap_id, frame_index) of the last detected object for a forced
        loop closure, or None if no object / no frame info is available.
        """
        det = self.last_detected_object
        if det is None or det.submap_id is None or det.frame_index is None:
            return None
        return (det.submap_id, det.frame_index)

    def move_objects(self, graph) -> None:
        """Move every cached object into the corrected world frame after a graph
        re-optimization (e.g. a loop closure).

        Each detection's points came from graph node ``submap_id + frame_index``.
        We apply that node's delta homography ``H_now @ inv(H_ref)`` to the stored
        points, re-fit the oriented bbox, and update the reference — the same
        delta-H mechanism PlaceLoopClosure.move_submap_nodes uses for place nodes.
        Detections without frame provenance or points are skipped.
        """
        for detections in self.cached_memory.objects.values():
            for det in detections:
                if (det.points is None or len(det.points) == 0
                        or det.submap_id is None or det.frame_index is None):
                    continue
                H_now = graph.get_homography(det.submap_id + det.frame_index)
                if H_now is None:
                    continue
                if det.ref_homography is None:
                    det.ref_homography = H_now      # initialise tracking (e.g. loaded session)
                    continue
                delta = H_now @ np.linalg.inv(det.ref_homography)
                if np.allclose(delta, np.eye(4), atol=1e-9):
                    continue                        # node did not move

                pts = np.asarray(det.points, dtype=np.float64)
                pts_h = np.hstack([pts, np.ones((len(pts), 1))])
                pts_w = (delta @ pts_h.T).T
                det.points = (pts_w[:, :3] / pts_w[:, 3:]).astype(np.float32)

                obb = utils.compute_obb_from_points(det.points)
                if obb is not None:            # keep the prior bbox if refit fails
                    c, e, r = obb
                    det.bbox = Bbox(c, e, r, det.bbox.label)
                det.ref_homography = H_now

    def last_object_in_camera_frame(
        self, solver, submap
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], str]]:
        """Return the last queried object's cloud in a camera's reference frame.

        The cached object points are stored in world frame; they are re-expressed
        in the frame of ``submap``'s newest keyframe (the "current" camera, assumed
        to view the object) via world_to_cam = inv(cam_to_world).

        Returns (points_cam (N,3) float64, colors (N,3) uint8 or None, label), or
        None if there is no cached object / no valid submap pose yet.
        """
        det = self.last_detected_object
        if det is None or det.points is None or len(det.points) == 0:
            print("[ObjectPCD] No cached object point cloud available — query an object first.")
            return None

        poses = submap.get_all_poses_world(solver.graph)  # cam-to-world per keyframe
        if len(poses) == 0:
            print("[ObjectPCD] Submap has no valid world poses yet.")
            return None
        world_to_cam = np.linalg.inv(poses[-1].astype(np.float64))  # newest keyframe

        pts_world = np.asarray(det.points, dtype=np.float64)
        pts_h = np.hstack([pts_world, np.ones((len(pts_world), 1))])
        pts_cam = (world_to_cam @ pts_h.T).T[:, :3]

        colors = None
        if det.colors is not None and len(det.colors) == len(pts_cam):
            colors = np.asarray(det.colors, dtype=np.uint8)
        label = det.bbox.label if (det.bbox is not None and det.bbox.label) else "object"
        return pts_cam, colors, label

    def save_last_object_in_camera_frame(self, solver, submap, output_dir: str,
                                         fmt: str = "pcd") -> Optional[str]:
        """Save the last queried object's point cloud (in the newest keyframe's
        camera frame) to ``output_dir`` as ``object_in_cam_<label>_<timestamp>.<fmt>``.

        Returns the saved file path, or None if there is no object / no valid pose.
        """
        import open3d as o3d

        result = self.last_object_in_camera_frame(solver, submap)
        if result is None:
            return None
        pts_cam, colors, label = result

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_cam)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        os.makedirs(output_dir, exist_ok=True)
        safe_label = "".join(c if c.isalnum() else "_" for c in str(label))
        path = os.path.join(output_dir, f"object_in_cam_{safe_label}_{int(time.time())}.{fmt}")
        o3d.io.write_point_cloud(path, pcd)
        print(f"[ObjectPCD] Saved {len(pts_cam)} object points in camera frame to {path}")
        return path

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def detect_last_object_in_current_frame(
        self, solver, submap
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], str, int]]:
        """Freshly detect the last-queried object's label in ``submap``'s newest
        keyframe with SAM3 and return its cloud in that camera's frame.

        Unlike last_object_in_camera_frame (which re-projects the CACHED cloud),
        this runs live detection on the current view, so the published cloud
        reflects the object as it is seen right now.  The detection is NOT cached
        or visualised — it is purely for publishing.  The strongest-scoring mask
        is used (the single object the camera is looking at).

        Returns (pts_cam (N,3) float64, colors (N,3) uint8 or None, label,
        frame_index) or None if there is no query label, no valid pose, or the
        object is not found in the current frame.
        """
        print("RUNNING OBJECT DETECT IN CURRENT FRAME")
        det = self.last_detected_object
        if det is None or det.bbox is None or not det.bbox.label:
            print("[ObjectPCD] No queried object label available — query an object first.")
            return None
        label = det.bbox.label

        # The newest real keyframe is the "current" view the robot is looking at.
        frame_index = submap.get_last_non_loop_frame_index()
        poses = submap.get_all_poses_world(solver.graph)  # cam-to-world per keyframe
        if len(poses) == 0 or frame_index >= len(poses):
            print("[ObjectPCD] Submap has no valid world poses yet.")
            return None

        best_img = submap.get_frame_at_index(frame_index)
        img_h, img_w = submap.get_img_size_at_index(frame_index)

        masks, scores = self._sam3_detect(best_img, label, img_h, img_w, lower=True)
        if masks.shape[0] == 0:
            print(f"[ObjectPCD] '{label}' not detected in the current frame.")
            return None

        # Strongest detection only — the single object in view.
        best_i = int(torch.as_tensor(scores).argmax().item())
        mask = masks[best_i].cpu().numpy().squeeze()

        img_np = np.array(to_pil_image(best_img).resize((img_w, img_h)))
        pts_world, colors = self._points_colors_from_mask(
            submap, frame_index, mask, img_np, solver)
        if pts_world is None:
            print(f"[ObjectPCD] '{label}' detected but produced no 3-D points.")
            return None

        # De-occlusion: the live detection is often a partial view.  Rigidly align the
        # cached (more complete) object cloud onto it and publish the MOVED cached cloud
        # so grasping sees the full object.  The live detection is held fixed; only the
        # cached cloud is moved.  Falls back to the live cloud if ICP is unavailable.
        pub_world, pub_colors = pts_world, colors
        moved, moved_colors = self._icp_align_cached_to_detection(
            det.points, det.colors, pts_world)
        if moved is not None:
            pub_world, pub_colors = moved, moved_colors
            print(f"[ObjectPCD] De-occluded '{label}' via ICP: publishing {len(moved)} "
                  f"cached points (live view had {len(pts_world)}).")

        # Express the published points in the current keyframe's camera frame.
        world_to_cam = np.linalg.inv(poses[frame_index].astype(np.float64))
        pub_world = np.asarray(pub_world, dtype=np.float64)
        pts_h = np.hstack([pub_world, np.ones((len(pub_world), 1))])
        pts_cam = (world_to_cam @ pts_h.T).T[:, :3]

        colors = (np.asarray(pub_colors, dtype=np.uint8)
                  if pub_colors is not None and len(pub_colors) == len(pts_cam) else None)
        return pts_cam, colors, label, frame_index

    def _icp_align_cached_to_detection(
        self, cached_pts, cached_colors, detection_pts
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Rigidly move the CACHED object cloud onto the live current-frame detection.

        The live detection is the fixed reference; only the cached cloud moves.  Both
        clouds are in the world frame and — because the camera poses are accurate — are
        already roughly co-located, so a centroid translation is a strong ICP init.

        ICP is run as detection(partial) -> cached(complete): registering the partial
        live cloud into the complete cached cloud gives every source point a real
        correspondence (more stable than matching the complete cloud's occluded back
        against a partial target).  The cached cloud is then moved by the inverse
        transform to sit at the detection's pose.

        Returns (moved_cached_pts (N,3) world, colors (N,3) uint8 or None), or
        (None, None) if there are too few points or the alignment does not lock on.
        """
        return None, None
        if cached_pts is None or detection_pts is None:
            print("[ObjectPCD] ICP de-occlusion skipped: "
                  f"{'no cached cloud' if cached_pts is None else 'no live detection cloud'} "
                  "— using the live detection.")
            return None, None
        cached_pts = np.asarray(cached_pts, dtype=np.float64)
        detection_pts = np.asarray(detection_pts, dtype=np.float64)
        if (len(cached_pts) < self.grasp_icp_min_points
                or len(detection_pts) < self.grasp_icp_min_points):
            print(f"[ObjectPCD] ICP de-occlusion skipped: too few points "
                  f"(cached={len(cached_pts)}, live={len(detection_pts)}, "
                  f"need >= {self.grasp_icp_min_points} each) — using the live detection.")
            return None, None

        import open3d as o3d

        src = o3d.geometry.PointCloud()   # live detection (partial), moved during ICP
        src.points = o3d.utility.Vector3dVector(detection_pts)
        tgt = o3d.geometry.PointCloud()   # cached object (complete), held fixed in ICP
        tgt.points = o3d.utility.Vector3dVector(cached_pts)

        # Good init: translate the detection centroid onto the cached centroid (both in
        # the same world frame thanks to the known camera pose).
        init = np.eye(4)
        init[:3, 3] = cached_pts.mean(axis=0) - detection_pts.mean(axis=0)

        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        # Coarse-to-fine on the correspondence distance for robustness.
        T_det_to_cached = init
        result = None
        for scale in (4.0, 2.0, 1.0):
            result = o3d.pipelines.registration.registration_icp(
                src, tgt, self.grasp_icp_max_corr_dist * scale, T_det_to_cached,
                estimator, criteria)
            T_det_to_cached = result.transformation

        if result is None or result.fitness < self.grasp_icp_min_fitness:
            print(f"[ObjectPCD] ICP de-occlusion rejected (fitness "
                  f"{0.0 if result is None else result.fitness:.2f} < "
                  f"{self.grasp_icp_min_fitness}); using the live detection.")
            return None, None

        # Move the cached cloud to the detection's pose (inverse of detection->cached).
        T_cached_to_det = np.linalg.inv(T_det_to_cached)
        pts_h = np.hstack([cached_pts, np.ones((len(cached_pts), 1))])
        moved = (T_cached_to_det @ pts_h.T).T[:, :3]

        colors = (np.asarray(cached_colors, dtype=np.uint8)
                  if cached_colors is not None and len(cached_colors) == len(cached_pts) else None)
        print(f"[ObjectPCD] ICP de-occlusion accepted: fitness {result.fitness:.2f} "
              f"(>= {self.grasp_icp_min_fitness}), inlier_rmse {result.inlier_rmse:.4f} m; "
              f"moved {len(moved)} cached points onto the live detection.")
        return moved, colors

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def get_3D_objects(self, queries: List[str], solver) -> Dict[str, List[Detection]]:
        """
        Find multiple objects, each ranked and searched over its own frames.

        inputs:
            queries: list of string queries, e.g. ["chair", "table", "cup"]
            solver: VGGT-SLAM solver
        returns dict mapping query -> list of Detection instances.
        """
        results: Dict[str, List[Detection]] = {}
        uncached_queries = []

        # Check cache for each query
        for query in queries:
            cached = self.cached_memory.get_detection(query)
            contains_detections = len(cached) > 0
            if contains_detections:
                print(f"Found {len(cached)} cached object(s) for query: '{query}'")
                self.last_detected_object = cached[0]
            else:
                uncached_queries.append(query)

        if not uncached_queries:
            return results

        start_time = time.time()

        # Detect each query independently over its own ranking (no prior detections
        # to exclude, so the seed lists are empty).
        detected_bboxes: Dict[str, List[Bbox]] = {q: [] for q in uncached_queries}
        query_objects = self._detect_queries(solver, uncached_queries, detected_bboxes)

        # Merge cached and newly detected results
        for q in uncached_queries:
            results[q] = query_objects[q]

        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed:.2f}s")
        for q in uncached_queries:
            print(f"  '{q}': {len(query_objects[q])} object(s)")

        return results

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def check_missing_object(self, solver):

        """
        inputs:
            current_frame:image
            solver: VGGT-SLAM solver
        returns: 
            a list of Detection instances of objects that are no longer visible in the current 
            frame or an empty list if no objects changed in the current view.
        """

        changed_objects = []

        for cached_query, cached_objects_list in self.cached_memory.items():
            # No objects exist for cached query
            if not cached_objects_list:
                continue

            for cached_object in cached_objects_list:
                bbox = cached_object.bbox
                most_recent_submap = solver.map.get_latest_submap(ignore_loop_closure_submaps=True)
                most_recent_frame_index = most_recent_submap.get_last_non_loop_frame_index()
                current_frame = most_recent_submap.get_frame_at_index(most_recent_frame_index)
                proj_mat = most_recent_submap.get_project_mat(most_recent_frame_index, solver.graph)
                img_h, img_w = most_recent_submap.get_img_size_at_index(most_recent_frame_index)

                # Check if this object should be in view of the current frame
                prev_center = bbox.center
                prev_extent = bbox.extent
                prev_rotation = bbox.rotation
                obb_corners = utils.get_obb_corners(prev_center, prev_extent, prev_rotation)

                if utils.is_object_visible(obb_corners, proj_mat, (img_w, img_h)):
                    # Check if object is in the current frame
                    masks, scores = self._sam3_detect(current_frame, cached_query, img_h, img_w)
                    print(cached_query, scores)
                    if masks.shape[0] == 0:
                        changed_objects.append(cached_query)

        return changed_objects

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def check_added_object(self, solver):

        """
        inputs:
            solver: VGGT-SLAM solver
        returns: 
            a list of Detection instances of objects that are visible at new 
            locations since the last time their respective text was queried
        """

        results: Dict[str, List[Detection]] = {}
        # Snapshot the queries: detection appends to the cache during the search.
        existing_queries = list(self.cached_memory.get_queries())

        if not existing_queries:
            return results

        # Seed each query's exclusion list with its cached bboxes, so only NEW
        # locations are reported.  Each query is then searched over its own
        # ranking (see _detect_queries).
        detected_bboxes: Dict[str, List[Bbox]] = {
            q: [d.bbox for d in self.cached_memory.get_detection(q)] for q in existing_queries
        }

        query_objects = self._detect_queries(solver, existing_queries, detected_bboxes)

        # Merge cached and newly detected results, collapsing near-duplicate
        # detections of the same query (multiple frames often report the SAME
        # physical object at slightly different centroids -> one "new" object each).
        for q in existing_queries:
            deduped = self._dedup_detections(query_objects[q], self.added_dedup_dist)
            if len(deduped) > 0:
                results[q] = deduped

        return results

    @staticmethod
    def _dedup_detections(dets: List["Detection"], dist: float) -> List["Detection"]:
        """Collapse detections whose bbox centres lie within ``dist`` metres, keeping
        the first of each cluster (detections arrive in CLIP-rank order, best first)."""
        kept: List["Detection"] = []
        for d in dets:
            c = np.asarray(d.bbox.center, dtype=float)
            if all(np.linalg.norm(c - np.asarray(k.bbox.center, dtype=float)) > dist
                   for k in kept):
                kept.append(d)
        return kept

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def get_3D_object_pcd(self, query, solver) -> List[np.ndarray]:
        """Return the dense world-frame point cloud (N, 3) of each detected instance
        of ``query``.

        Thin wrapper over get_3D_object for Replica evaluation, which needs only
        the raw points rather than the full Detection objects.
        """
        return [det.points for det in self.get_3D_object(query, solver)]

    def _object_approach_point(self, object_floor, classification, solver):
        """Point APPROACH_NORMAL_LEN m from the object along the line toward the
        nearest safe (green) place tile.

        The direction object→nearest-free-tile points into open space (away from
        whatever the object rests against), giving a clean, roughly-normal final
        approach without fitting a surface line — robust at table corners and
        similar geometry.  Returns the 3D approach point, or None when there are
        no free tiles / the nearest one is coincident with the object (caller then
        routes straight to the object).
        """
        graph = solver.places.graph
        positions, indices = graph.get_active_node_data()
        keep_out = classification.keep_out
        # Green (safe) tiles = active nodes that are not boundary / near-boundary.
        green = np.array(
            [p for p, i in zip(positions, indices) if i not in keep_out],
            dtype=np.float64,
        )
        if len(green) == 0:
            return None

        nearest = green[int(np.argmin(np.linalg.norm(green - object_floor, axis=1)))]
        direction = nearest - object_floor
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            return None                                            # object sits on a free tile
        return object_floor + APPROACH_NORMAL_LEN * (direction / dist)

    def plan_path_to_point(self, point1, point2, solver):
        """Plan and visualize a shortest path from ``point1`` to ``point2`` over the
        places layer, ending at the nearest safe (green) node to ``point2``.

        Unlike plan_path (which is object-specific and adds a straight surface
        approach), this routes directly to an open-space target — used by the
        "Go to Room" navigation, where point2 is a region centre.  find_shortest_path
        snaps the end to the nearest traversable (non keep-out) node, which is
        exactly the "nearest safe green tile to the centre" target.

        Returns True if a path was found and set, False otherwise (so callers can
        avoid acting on the viewer's previous, stale path).
        """
        from foundit.places import find_shortest_path
        import shapelysmooth as ss
        # Remove obstacle-blocked nodes and refresh the frontier so keep_out is
        # current (same preparation plan_path uses).
        classification = solver.refresh_places_and_frontier()
        path_result = find_shortest_path(
            solver.places.graph, point1, point2,
            excluded_nodes=classification.keep_out,
        )
        if path_result is None:
            return False
        path_points, _ = path_result
        smoothed_path_points = np.array(path_points)
        smoothed_path_points[:, [0, 2]] = np.array(ss.taubin_smooth(smoothed_path_points[:, [0, 2]]))
        # path_points are target→camera order, matching the viewer/follower convention.
        if len(smoothed_path_points) <= 1:
            return False
        solver.viewer.set_planned_path(smoothed_path_points)
        return True

    def plan_path(self, point1, point2, solver):
        """Compute and visualize a shortest path between two 3D points over the places layer.

        Boundary and near-boundary place nodes are excluded so the path stays in
        the interior of the explored region.  The final leg into the object is
        aimed along the line toward the nearest safe (green) tile: the path is
        routed to a node ~APPROACH_NORMAL_LEN out along that line, then the object
        is appended as the true endpoint so the last leg is a straight approach
        from open space.
        """
        from foundit.places import find_shortest_path
        from foundit.places.frontier import _floor_frame
        import shapelysmooth as ss
        # Project all submap points onto the places layer, remove any node now
        # covered by an obstacle (e.g. an object moved onto a previously-clear
        # interior tile), rebuild the frontier/edges, and update the viewer. The
        # removed nodes leave the graph, so keep_out alone suffices below.
        classification = solver.refresh_places_and_frontier()

        graph = solver.places.graph
        positions, _ = graph.get_active_node_data()
        if len(positions) == 0:
            return

        # Object projected onto the floor plane (the true final endpoint). The
        # camera is tilted so the floor is not axis-aligned; use the fitted floor
        # normal and the on-floor graph node nearest the object as the reference.
        p2 = np.asarray(point2, dtype=np.float64)
        floor_frame = _floor_frame(solver.places.submap_plane_rotations)
        if floor_frame is not None:
            floor_normal = floor_frame[:, 2]
            floor_ref = positions[int(np.argmin(np.linalg.norm(positions - p2, axis=1)))]
            object_floor = p2 - np.dot(p2 - floor_ref, floor_normal) * floor_normal
        else:
            object_floor = p2.copy()

        # Route to an approach point along the line toward the nearest safe tile
        # (or straight to the object when no free tile is available).
        approach = self._object_approach_point(object_floor, classification, solver)
        route_target = approach if approach is not None else object_floor

        path_result = find_shortest_path(
            graph, point1, route_target,
            excluded_nodes=classification.keep_out,
        )
        if path_result is None:
            return
        path_points, total_distance = path_result
        smoothed_path_points = np.array(path_points)
        smoothed_path_points[:, [0, 2]] = np.array(ss.taubin_smooth(smoothed_path_points[:, [0, 2]]))

        # path_points are object→camera order; index 0 is the routed end node.
        # Pin it to the exact approach point so the final object→approach leg is
        # exactly the straight approach line (undo any smoothing drift on that one
        # endpoint).
        if approach is not None:
            smoothed_path_points[0] = approach

        obj_waypoint = object_floor.reshape(1, 3)
        full_path = np.vstack([obj_waypoint, smoothed_path_points])

        if len(full_path) > 1:
            solver.viewer.set_planned_path(full_path)
