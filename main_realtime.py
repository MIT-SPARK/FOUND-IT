import cv2
import numpy as np
import os
import threading
import argparse
import time
import torch

from foundit.cameras import BACKENDS
from foundit.session import save_session, load_session
from foundit.app_setup import (
    add_common_slam_args,
    validate_common_args,
    build_objects_module,
    build_solver,
    build_model,
    run_interactive_loop,
)

# --- Thread Safety Primitives ---
solver_lock = threading.Lock()  # Ensures only one solver thread runs at a time
data_lock = threading.Lock()    # Protects shared SLAM state (solver)

# Set by the viser "Get Object PCD in Cam" button; fulfilled by the capture loop.
pcd_request = threading.Event()

parser = argparse.ArgumentParser(description="Scene Graph RealSense live demo")
add_common_slam_args(parser)
# --- Frame-source-specific arguments (live camera capture) ---
parser.add_argument("--keyframe_folder", type=str, default="keyframes")
parser.add_argument("--camera", type=str, default="realsense", choices=["realsense", "zed", "logitech"], help="Camera backend (default: realsense)")
parser.add_argument("--camera_index", type=int, default=0, help="V4L2 device index for the logitech/UVC webcam backend (e.g. 0 for /dev/video0, 2 for /dev/video2)")
parser.add_argument("--fallback_camera", type=str, default=None, choices=["realsense", "zed", "logitech"], help="If the primary camera disconnects and cannot be restarted, switch to this backend instead (e.g. start on logitech, fail over to realsense)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_keyframe(img: np.ndarray, folder: str, idx: int) -> str:
    filename = os.path.join(folder, f"frame_{idx:06d}.png")
    cv2.imwrite(filename, img)
    return filename


def make_camera(name: str, camera_index: int = 0):
    """Construct a camera backend by name, passing the device index to webcams."""
    if name == "logitech":
        return BACKENDS[name](device_index=camera_index)
    return BACKENDS[name]()


def restart_camera(camera, stop_event=None, retry_delay: float = 2.0, fallback=None):
    """Stop and re-start the camera, retrying until a device is streaming again.

    Used to recover from a mid-session disconnect (e.g. a RealSense USB drop or a
    webcam unplug), where ``capture()`` raises.  Honors ``stop_event`` so the user
    can still cancel while a camera is unplugged.

    If ``fallback`` (a zero-arg factory returning a new Camera) is given, the first
    time the current camera fails to restart it switches to the fallback device and
    keeps retrying that instead — e.g. start on a webcam and fail over to a
    RealSense.

    Returns the working camera object, or None if aborted via stop_event.
    """
    try:
        camera.stop()
    except Exception:
        pass  # device may already be gone; ignore teardown errors

    attempt = 0
    while stop_event is None or not stop_event.is_set():
        attempt += 1
        try:
            camera.start()
            print(f"[Camera] Reconnected after {attempt} attempt(s).")
            return camera
        except Exception as e:
            print(f"[Camera] Reconnect attempt {attempt} failed: {e}.")
            if fallback is not None:
                print("[Camera] Switching to fallback camera...")
                try:
                    camera.stop()
                except Exception:
                    pass
                camera = fallback()
                fallback = None       # only fail over once
                continue              # try starting the fallback right away
            print(f"[Camera] Retrying in {retry_delay:.0f}s...")
            time.sleep(retry_delay)
    print("[Camera] Reconnection aborted (session cancelled).")
    return None


def threaded_process_submap(image_names_subset, solver, model, objects_module, args,
                            forced_loop=None):
    """Background thread: run VGGT inference + graph optimisation for one submap.

    forced_loop, when given as (detected_submap_id, detected_frame), forces a loop
    closure from this submap's newest frame back to that frame (see the object
    capture flow).
    """
    try:
        print(f"[SLAM] Processing submap ({len(image_names_subset)} frames)...")
        if args.use_da:
            predictions = solver.run_predictions_da(
                image_names_subset, model, max_loops=args.max_loops,
                clip_model=objects_module.clip_model, clip_preprocess=objects_module.clip_preprocess,
                forced_loop_detection=forced_loop,
            )
        else:
            predictions = solver.run_predictions(
                image_names_subset, model, args.max_loops,
                objects_module.clip_model, objects_module.clip_preprocess,
                forced_loop_detection=forced_loop,
            )
        with data_lock:
            solver.add_points(predictions)
            solver.graph.optimize()
            if args.vis_map:
                solver.update_latest_submap_vis()
                if len(predictions.get("detected_loops", [])) > 0:
                    solver.update_all_submap_vis()
        print("[SLAM] Submap done.")
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


def handle_object_pcd_request(image_names_subset, solver, model, objects_module, args):
    """Force the current keyframe buffer into a submap, then save the last queried
    object's point cloud in that submap's newest camera frame.

    Runs synchronously (a user-triggered action) so the new submap — and hence
    the current camera pose — exists before the object cloud is transformed and
    written.  Returns the trimmed keyframe buffer to continue capture from.
    """
    if len(image_names_subset) < 2:
        print("[ObjectPCD] Not enough keyframes yet to form a submap.")
        return image_names_subset

    # Close a loop back to the frame the object was seen in, to cut drift between
    # then and the current camera (None if the object carries no frame info).
    forced_loop = objects_module.last_object_loop_target()

    print(f"[ObjectPCD] Forcing a submap from {len(image_names_subset)} keyframes...")
    solver_lock.acquire()  # blocking: serialise with any in-flight submap thread
    # threaded_process_submap releases solver_lock in its own finally block.
    threaded_process_submap(list(image_names_subset), solver, model, objects_module, args,
                            forced_loop=forced_loop)

    # Save one folder up from the keyframes dir (the run's root folder).
    output_dir = os.path.dirname(os.path.abspath(args.keyframe_folder))
    with data_lock:
        latest_sm = solver.map.get_latest_submap(ignore_loop_closure_submaps=True)
        if latest_sm is not None:
            objects_module.save_last_object_in_camera_frame(
                solver, latest_sm, output_dir, fmt="pcd")
    return image_names_subset[-1:]


def main():
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

    if not validate_common_args(args):
        return

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    camera = make_camera(args.camera, args.camera_index)
    # Optional fail-over device: built fresh each time so a fresh pipeline is opened.
    camera_fallback = (
        (lambda: make_camera(args.fallback_camera, args.camera_index))
        if args.fallback_camera else None
    )
    print(f"Initializing {args.camera} camera...")
    os.makedirs(args.keyframe_folder, exist_ok=True)
    camera.start()

    # Warm up the camera — first few frames can be None
    print("Waiting for first camera frame...")
    first_frame = None
    while first_frame is None:
        try:
            first_frame = camera.capture()
        except Exception as e:
            print(f"[Camera] Error during warm-up ({e}). Reconnecting...")
            camera = restart_camera(camera, fallback=camera_fallback)

    print("Camera ready.")

    objects_module = build_objects_module(args)
    solver = build_solver(args)
    solver.register_object_cache(objects_module)
    model = build_model(args, device)
    print("All models loaded. Starting SLAM loop.")

    from foundit.measure_memory import rss_bytes, report_memory
    mem_baseline = rss_bytes() if args.report_memory else None

    # Load a prior session (if requested) so the live run can relocalize against it.
    if args.load_session:
        load_session(args.load_session, solver, objects_module)
        solver.mark_prior_session()
        # Show the loaded map so the user sees it while the new session builds up.
        solver.update_all_submap_vis()

    # Register viser GUI panels so they're available during capture
    stop_event = threading.Event()
    if solver.viewer is not None:
        solver.viewer.add_cancel_button(stop_event)
        solver.viewer.add_object_query_gui(objects_module, solver, data_lock)
        solver.viewer.add_frontier_gui(solver, data_lock)
        solver.viewer.add_get_object_pcd_button(pcd_request.set)
        if args.places:
            solver.viewer.add_room_query_gui(objects_module, solver, data_lock, top_region=args.top_region)

    frame_count = 0
    image_names_subset = []
    target_size = args.submap_size + 1
    submap_count = 0
    last_status_frame = 0  # for console status throttle when display is off

    try:
        while True:
            if stop_event.is_set():
                print("\n[Main] Session cancelled via viser.")
                break

            try:
                img = camera.capture()
            except Exception as e:
                print(f"[Camera] Lost connection ({e}). Attempting to reconnect...")
                camera = restart_camera(camera, stop_event, fallback=camera_fallback)
                if camera is None:  # session cancelled while reconnecting
                    break
                continue

            if img is None:
                continue

            frame_count += 1

            added_keyframe = False
            if solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow):
                frame_path = save_keyframe(img, args.keyframe_folder, frame_count)
                image_names_subset.append(frame_path)
                added_keyframe = True

            # User asked (via viser) for the last object's cloud in the current
            # camera frame: make the current frame the newest keyframe (if it was
            # not already added this iteration), then force a submap and save.
            if pcd_request.is_set():
                pcd_request.clear()
                if not added_keyframe:
                    frame_path = save_keyframe(img, args.keyframe_folder, frame_count)
                    image_names_subset.append(frame_path)
                image_names_subset = handle_object_pcd_request(
                    image_names_subset, solver, model, objects_module, args)

            if len(image_names_subset) >= target_size:
                if solver_lock.acquire(blocking=False):
                    submap_count += 1
                    print(f"[Main] Launching submap {submap_count} (frame {frame_count})...")
                    t = threading.Thread(
                        target=threaded_process_submap,
                        args=(list(image_names_subset), solver, model, objects_module, args),
                        daemon=True,
                    )
                    t.start()
                    image_names_subset = image_names_subset[-1:]
                else:
                    if len(image_names_subset) > target_size * 2:
                        image_names_subset = image_names_subset[-target_size:]

            kf = len(image_names_subset)
            slam_busy = solver_lock.locked()

            if frame_count - last_status_frame >= 30:
                busy_str = "  [SLAM running]" if slam_busy else ""
                print(f"[Camera] frame={frame_count}  KFs={kf}/{target_size}"
                        f"  submaps={submap_count}{busy_str}")
                last_status_frame = frame_count

    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
    finally:
        if camera is not None:
            camera.stop()

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
            solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))


if __name__ == "__main__":
    main()
