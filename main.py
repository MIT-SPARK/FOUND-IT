import os
import glob
import time
import argparse
import threading

import numpy as np
import torch
try:
    torch.backends.cuda.preferred_linalg_library('cusolver')
except RuntimeError:
    pass  # cusolver not available; fall back to PyTorch default
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt


import foundit.slam_utils as utils
from foundit.colmap_odometry import ColmapOdometry
from foundit.session import save_session, load_session
from foundit.app_setup import (
    add_common_slam_args,
    validate_common_args,
    build_objects_module,
    build_solver,
    build_model,
    run_interactive_loop,
)

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
add_common_slam_args(parser)
# --- Frame-source-specific arguments (offline image folder + COLMAP ) ---
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--downsample_factor", type=int, default=1, help="Factor to reduce image size by 1/N")
parser.add_argument("--colmap_path", type=str, default=None, help="Path to COLMAP sparse model directory (e.g. /dataset/sparse/0) to use COLMAP poses as odometry factors")
parser.add_argument("--depth_folder", type=str, default=None, help="Path to folder containing external depth images (filenames must match color images). Replaces VGGT depth estimates.")
parser.add_argument("--colmap_scale", type=float, default=0.5, help="Scale factor applied to raw depth image values to convert them to metric (colmap) scale")

def load_depth_images_for_submap(image_paths, depth_folder):
    """Load raw (unscaled) depth images from depth_folder matching color image names.

    Returns a list of (H, W) float32 arrays or None for frames with no match.
    Supports 8-bit and 16-bit depth images via cv2.IMREAD_ANYDEPTH.
    """
    DEPTH_EXTENSIONS = ['.png']
    depth_images = []
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        stem = stem.replace("rgb", "depth")
        found = None
        for ext in DEPTH_EXTENSIONS:
            depth_path = os.path.join(depth_folder, stem + ext)
            if os.path.exists(depth_path):
                raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if raw is not None:
                    found = raw.astype(np.float32)
                break
        if found is None:
            print(f"[DepthLoader] No depth image found for '{os.path.basename(img_path)}'")
        depth_images.append(found)
    return depth_images

def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

    if not validate_common_args(args):
        return

    # ------------------------------------------------------------------
    # --load_session: skip all processing, restore from disk, then fall
    # through to the normal interactive query / visualisation loop below.
    # ------------------------------------------------------------------
    if args.load_session:
        objects_module = build_objects_module(args)
        solver = build_solver(args)
        solver.register_object_cache(objects_module)

        load_session(args.load_session, solver)

        solver.update_all_submap_vis()

        if args.rooms:
            from foundit.places import FiltrationClusterer, FiltrationClusterConfig
            config = FiltrationClusterConfig()
            result = FiltrationClusterer(config).cluster(solver.places.graph)
            solver.viewer.visualize_clusters(result, solver.places)
            def recluster(cfg):
                return FiltrationClusterer(cfg).cluster(solver.places.graph)
            solver.viewer.add_clustering_gui(config, recluster, solver.places)

        # Jump straight to the interactive loop (defined below as a shared helper)
        _run_interactive_loop(args, objects_module, solver)

        if args.report_memory:
            from foundit.measure_memory import report_memory
            report_memory(solver, objects_module)

        return

    use_optical_flow_downsample = True
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    objects_module = build_objects_module(args)
    solver = build_solver(args)
    solver.register_object_cache(objects_module)

    colmap_odom = None
    if args.colmap_path is not None:
        colmap_odom = ColmapOdometry(args.colmap_path)

    model = build_model(args, device)

    from foundit.measure_memory import rss_bytes, report_memory
    baseline = rss_bytes() if args.report_memory else None

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*"))
                   if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                   and "depth" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    image_names = utils.downsample_images(image_names, args.downsample_factor)
    print(f"Found {len(image_names)} images")

    if colmap_odom is not None:
        before = len(image_names)
        image_names = [p for p in image_names if colmap_odom.has_pose(p)]
        print(f"Filtered to {len(image_names)} images with COLMAP poses (removed {before - len(image_names)})")

    image_names_subset = []
    count = 0
    image_count = 0
    total_time_load_imgs = 0
    total_time_start = time.time()
    keyframe_time = utils.Accumulator()
    backend_time = utils.Accumulator()
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            with keyframe_time:
                t1 = time.time()
                img = cv2.imread(image_name)
                total_time_load_imgs += time.time() - t1
                # rescale image to have width 518
                scale_factor = 518 / img.shape[1] # TODO Dominic we could use this image for VGGT since its already the right size
                img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
                enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
                if enough_disparity:
                    image_names_subset.append(image_name)
                    image_count += 1
        else:
            image_names_subset.append(image_name)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + 1 or image_name == image_names[-1]:
            count += 1
            print(image_names_subset)
            t1 = time.time()
            if args.use_da:
                predictions = solver.run_predictions_da(image_names_subset, model, max_loops=args.max_loops, clip_model=objects_module.clip_model, clip_preprocess=objects_module.clip_preprocess)
            else:
                predictions = solver.run_predictions(image_names_subset, model, args.max_loops, objects_module.clip_model, objects_module.clip_preprocess)
            print("Solver total time", time.time()-t1)
            print(count, "submaps processed")

            # Compute per-submap relative odometry from COLMAP poses when available.
            relative_odom = None
            if colmap_odom is not None:
                relative_odom = colmap_odom.get_relative_odom(image_names_subset, args.colmap_scale)

            # Load external depth images for this submap window when provided.
            depth_images = None
            colmap_intrinsics = None
            if args.depth_folder is not None:
                depth_images = load_depth_images_for_submap(image_names_subset, args.depth_folder)
                if colmap_odom is not None:
                    target_h = predictions["depth"].shape[1]
                    target_w = predictions["depth"].shape[2]
                    colmap_intrinsics = colmap_odom.get_intrinsics(image_names_subset, target_h, target_w)

            solver.add_points(predictions, relative_odom_orig=relative_odom,
                              depth_images=depth_images, depth_scale=0.001,
                              colmap_intrinsics=colmap_intrinsics)

            with backend_time:
                solver.graph.optimize()

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                solver.update_latest_submap_vis()
                if loop_closure_detected:
                    solver.update_all_submap_vis()

            if args.places and args.vis_map and solver.viewer is not None:
                classification = solver.compute_frontier()
                solver.viewer.show_frontier_tiles(
                    classification,
                    solver.places.graph,
                    dict(solver.places.submap_plane_rotations),
                    solver.places.place_size,
                )

            # Reset for next submap.
            image_names_subset = image_names_subset[-1:]

    total_time = time.time() - total_time_start
    total_time -= total_time_load_imgs
    print("total_time_load_imgs", total_time_load_imgs)
    average_fps = total_time / image_count
    solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))
    print(image_count, "frames processed")
    print("Total time:", total_time)
    print(f"Total time for VGGT calls: {solver.vggt_timer.total_time:.4f}s")
    print("Average VGGT time per frame:", solver.vggt_timer.total_time / image_count)
    print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / image_count)
    print("Average keyframe selection time per frame:", keyframe_time.total_time / image_count)
    print("Average backend time per frame:", backend_time.total_time / image_count)
    print("Average semantic time per frame:", solver.clip_timer.total_time / image_count)
    print("Average places+region time per frame:", solver.places_timer.total_time / image_count)
    print("Average total time per frame:", total_time / image_count)
    print("Average FPS:", 1 / average_fps)
        
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    if args.places:
        print(f"Places graph: {len(solver.places.graph)} nodes, "
              f"{solver.places.graph.num_connected_components} connected components")

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.rooms:
        from foundit.places import FiltrationClusterer, FiltrationClusterConfig

        config = FiltrationClusterConfig()
        result = FiltrationClusterer(config).cluster(solver.places.graph)
        solver.viewer.visualize_clusters(result, solver.places)

        def recluster(cfg: FiltrationClusterConfig):
            return FiltrationClusterer(cfg).cluster(solver.places.graph)

        solver.viewer.add_clustering_gui(config, recluster, solver.places)

    if args.save_session:
        save_session(solver, args.save_session)

    _run_interactive_loop(args, objects_module, solver)

    if args.report_memory:
        report_memory(solver, objects_module, baseline)

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)
        if not args.skip_dense_log:
            solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))
        # Log the average FPS to a companion file so eval scripts can read it per scene.
        fps_path = args.log_path.replace(".txt", "_fps.txt")
        with open(fps_path, "w") as f:
            f.write(f"{1 / average_fps:.6f}\n")
        print(f"Logged average FPS ({1 / average_fps:.4f}) to {fps_path}")


def _run_interactive_loop(args, objects_module, solver):
    """Register the viser GUI panels, run the shared query loop, then tear down.

    main.py registers its object/frontier panels only once processing is
    finished (the offline run has no live viewer interaction beforehand), so
    the panel setup and viewer shutdown wrap the shared interactive loop here.
    """
    if solver.viewer is not None:
        data_lock = threading.Lock()
        solver.viewer.add_object_query_gui(objects_module, solver, data_lock)
        solver.viewer.add_frontier_gui(solver, data_lock)

    run_interactive_loop(args, objects_module, solver)

    if solver.viewer is not None:
        solver.viewer.server.stop()


if __name__ == "__main__":
    main()
