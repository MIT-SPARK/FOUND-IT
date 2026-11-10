"""Shared CLI / setup helpers for the VGGT-SLAM entry points.

``main.py`` (offline image folders), ``main_realtime.py`` (live cameras) and
``main_ros2.py`` (ROS 2 topics) share the bulk of their command-line surface,
argument validation, model / solver construction and post-processing query
loop.  Historically each file carried its own verbatim copy of all four, so a
change (a new model flag, a new query command) had to be made in three places
and drifted between them.

This module is the single source of truth for that common behaviour, so the
three entry points differ only in how they *source frames*:

    parser = argparse.ArgumentParser(...)
    add_common_slam_args(parser)
    # ... add frame-source-specific args ...
    args = parser.parse_args()

    if not validate_common_args(args):
        return
    # ... any entry-point-specific validation ...

    objects_module = build_objects_module(args)
    solver         = build_solver(args)
    model          = build_model(args, device)
    ...
    run_interactive_loop(args, objects_module, solver)
"""

import os
import time

import torch

from foundit.solver import Solver
from foundit.objects import Objects
from vggt.models.vggt import VGGT
from foundit.places.clustering.query import create_query_clusterer, query_region
from foundit.places.config import QueryClusterConfig
from foundit.agent import AgentConfig, TaskAgent

# Repository root (this file lives in <repo>/foundit/).  The optional
# VGGT-Omega and Depth-Anything-3 checkouts live at the repo root and are added
# to sys.path lazily when their respective backends are selected.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def add_common_slam_args(parser) -> None:
    """Register the SLAM arguments shared by every entry point.

    Frame-source-specific arguments (image folder, camera backend, ROS topic,
    odometry, COLMAP, Gaussian splatting, ...) are added by each entry point
    separately after calling this.
    """
    # --- Visualization ---
    parser.add_argument("--vis_map", action="store_true",
                        help="Visualize the map in viser as it is built (otherwise only the final map)")
    parser.add_argument("--vis_flow", action="store_true",
                        help="Visualize optical flow used for keyframe selection")
    parser.add_argument("--vis_imgs", action="store_true",
                        help="Visualize input images in the viewer")
    parser.add_argument("--vis_voxel_size", type=float, default=None,
                        help="Voxel size (m) for downsampling the point cloud shown in the viewer")

    # --- Submap / keyframe selection ---
    parser.add_argument("--submap_size", type=int, default=16,
                        help="Number of new frames per submap (excludes overlap/loop-closure frames)")
    parser.add_argument("--min_disparity", type=float, default=50.0,
                        help="Minimum disparity to trigger a new keyframe")
    parser.add_argument("--max_loops", type=int, default=1,
                        help="ONLY DEFAULT OF 1 SUPPORTED. Max loop closures per submap")
    parser.add_argument("--lc_thres", type=float, default=0.95,
                        help="Image-retrieval threshold in [0, 1]; higher = more loop closures")

    # --- Point cloud filtering ---
    parser.add_argument("--conf_threshold", type=float, default=25.0,
                        help="Initial percentage of low-confidence points to filter out")
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="Voxel size (m) for the sparse submap point cloud; -1 disables downsampling")
    parser.add_argument("--voxel_min_points", type=int, default=10,
                        help="A voxel must contain at least this many points to survive "
                             "downsampling; >1 drops sparse/noisy voxels (default: 3)")
    parser.add_argument("--metric", action="store_true",
                        help="Enable approximate metric scale via depth-anything-3 on the first image")

    # --- Places / rooms ---
    parser.add_argument("--places", action="store_true",
                        help="Enable Places-layer computation and visualization")
    parser.add_argument("--places_full", action="store_true",
                        help="Use all points regardless of VGGT confidence for the places calculation")
    parser.add_argument("--rooms", action="store_true",
                        help="Cluster place nodes into semantic rooms (requires --places and --run_os)")
    parser.add_argument("--top_region", action="store_true",
                        help="Only display the top-scoring region for region queries")

    # --- Open-set semantics ---
    parser.add_argument("--run_os", action="store_true",
                        help="Enable open-set semantic search with CLIP and SAM3")
    parser.add_argument("--log_os", action="store_true",
                        help="Enable logging of open-set detection results")
    parser.add_argument("--sam3_threshold", type=float, default=0.50,
                        help="Confidence threshold for SAM 3 object detection in [0, 1]")
    parser.add_argument("--top_k", type=int, default=16,
                        help="Number of top CLIP-ranked frames to process with SAM3")
    parser.add_argument("--no_filter_objects", action="store_true",
                        help="Disable confidence thresholding when extracting dense 3D points for queried objects")
    parser.add_argument("--use_smaller_pe_model", action="store_true",
                    help="Use Perception Encoder model PE-Core-B16-224 instead of PE-Core-L14-336")

    # --- LLM agent ---
    parser.add_argument("--agent", action="store_true",
                        help="Enable LLM agent for long-horizon task planning")
    parser.add_argument("--agent_model", type=str, default="gpt-5-mini",
                        help="OpenAI model for the agent")
    parser.add_argument("--agent_visual_verification", action="store_true",
                        help="Enable multimodal visual verification of detections")

    # --- Backends: Depth Anything 3 / VGGT-Omega ---
    parser.add_argument("--use_da", action="store_true",
                        help="Use Depth Anything 3 instead of VGGT. Loop closure verification is disabled.")
    parser.add_argument("--da_model", type=str, default="depth-anything/DA3-LARGE-1.1",
                        help="Depth Anything 3 model name (must support pose estimation)")
    parser.add_argument("--da_seg", type=str, default=None,
                        help="Path to a trained DA3 FloorSegHead checkpoint (enables --places with --use_da)")
    parser.add_argument("--use_omega", action="store_true",
                        help="Use VGGT-Omega instead of VGGT. Loop closure verification is disabled.")
    parser.add_argument("--omega_checkpoint", type=str, default=None,
                        help="Path to a VGGT-Omega checkpoint file. Required when --use_omega is set.")
    parser.add_argument("--omega_seg", type=str, default=None,
                        help="Path to a trained Omega FloorSegHead checkpoint (enables --places with --use_omega)")
    parser.add_argument("--omega_resolution", type=int, default=512,
                        help="Image resolution for VGGT-Omega inference (default: 512)")
    parser.add_argument("--omega_register_mean", type=str, default=None,
                        help="Path to register_mean_layerXX.npy for loop closure verification with --use_omega")


    # --- Sessions / logging / device ---
    parser.add_argument("--save_session", type=str, default=None, metavar="DIR",
                        help="Save the complete session (map, poses, embeddings, places, objects) to DIR")
    parser.add_argument("--load_session", type=str, default=None, metavar="DIR",
                        help="Load a previously saved session from DIR")
    parser.add_argument("--log_results", action="store_true",
                        help="Save a txt file with poses (and, unless --skip_dense_log, a dense point cloud)")
    parser.add_argument("--skip_dense_log", action="store_true",
                        help="Skip dense point-cloud logging when --log_results is set")
    parser.add_argument("--log_path", type=str, default="poses.txt",
                        help="Path to save the log file")
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device index (e.g. 0 or 1). Ignored when running on CPU.")
    parser.add_argument("--report_memory", action="store_true",
                        help="At end of run, print the per-component map memory footprint "
                             "(RGB, depth, points, embeddings, places graph, objects cache; "
                             "models excluded) and the process RSS growth since models loaded.")


def validate_common_args(args) -> bool:
    """Validate the argument combinations that apply to every entry point.

    Returns True if the arguments are consistent, or prints an explanatory
    error and returns False otherwise.  Entry points perform their own extra
    validation (e.g. ROS odometry flags) after this passes.
    """
    checks = [
        (args.agent and not args.run_os,
         "--agent requires --run_os."),
        (args.rooms and not args.places,
         "--rooms requires --places."),
        (args.use_da and args.places and not args.da_seg,
         "--places with --use_da requires --da_seg."),
        (args.metric and "DA3NESTED-GIANT" in args.da_model,
         "--metric with DA3NESTED-GIANT is not needed since that model already predicts metric "
         "scale. Disable --metric when using DA3NESTED-GIANT to prevent running out of GPU memory."),
        (args.use_omega and args.omega_checkpoint is None,
         "--use_omega requires --omega_checkpoint to be set."),
        (args.use_omega and args.places and not args.omega_seg,
         "--places with --use_omega requires --omega_seg."),
        (args.use_omega and args.use_da,
         "--use_omega and --use_da are mutually exclusive."),
    ]
    for failed, message in checks:
        if failed:
            print(f"Error: {message}")
            return False
    return True


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build_objects_module(args) -> Objects:
    """Build the open-set Objects module, loading SAM3/CLIP only when needed."""
    objects_module = Objects(args.use_smaller_pe_model)
    objects_module.top_k = args.top_k
    objects_module.filter_conf = not args.no_filter_objects
    if args.run_os:
        objects_module.load_models(args.sam3_threshold)
    return objects_module


def build_solver(args) -> Solver:
    """Construct the Solver from the shared SLAM arguments."""
    return Solver(
        init_conf_threshold=args.conf_threshold,
        lc_thres=args.lc_thres,
        use_places_layer=args.places,
        use_all_points_for_places=args.places_full,
        vis_voxel_size=args.vis_voxel_size,
        use_metric_scale=args.metric,
        voxel_size=args.voxel_size,
        voxel_min_points=args.voxel_min_points,
        vis_imgs=args.vis_imgs,
        use_omega=args.use_omega,
        omega_resolution=args.omega_resolution,
    )


def build_model(args, device):
    """Instantiate and load the pose/depth model selected by the CLI flags.

    Dispatches to VGGT-Omega, Depth Anything 3, or the default VGGT backbone,
    optionally attaching a floor-segmentation head for the places layer.
    """
    if args.use_omega:
        import sys as _sys
        _sys.path.insert(0, os.path.join(_REPO_ROOT, "vggt-omega"))
        from vggt_omega.models import VGGTOmega
        print(f"Initializing VGGT-Omega model from {args.omega_checkpoint}...")
        model = VGGTOmega(enable_alignment=True).to(device).eval()
        model.load_state_dict(torch.load(args.omega_checkpoint, map_location="cpu"))

        if args.omega_register_mean:
            model.aggregator.load_register_mean(args.omega_register_mean)

        if args.places and args.omega_seg:
            from vggt_omega.models.heads.floor_seg_head import FloorSegHead as OmegaFloorSegHead
            seg_ckpt = torch.load(args.omega_seg, map_location="cpu")
            layer_idx = seg_ckpt.get("layer_idx", 11)
            dim_in = seg_ckpt.get("dim_in", 2 * 1024)
            patch_size = seg_ckpt.get("patch_size", 16)
            seg_head = OmegaFloorSegHead(dim_in=dim_in, patch_size=patch_size, layer_idx=layer_idx)
            seg_head.load_state_dict(seg_ckpt["seg_head_state_dict"])
            seg_head = seg_head.to(device).eval()
            model.seg_head = seg_head
            model.seg_layer_idx = layer_idx
            model.aggregator._cache_all_layers = True
            print(f"Loaded Omega floor seg head (layer={layer_idx}, dim_in={dim_in})")
        return model

    if args.use_da:
        print(f"Initializing Depth Anything 3 model ({args.da_model})...")
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3.from_pretrained(args.da_model).to(device)
        model.eval()

        if args.places and args.da_seg:
            import sys as _sys
            _sys.path.insert(0, os.path.join(_REPO_ROOT, "Depth-Anything-3", "src"))
            from depth_anything_3.model.floor_seg_head import FloorSegHead
            from foundit.solver import _get_da3_backbone
            ckpt = torch.load(args.da_seg, map_location="cpu")
            layer_idx = ckpt.get("layer_idx", 19)
            backbone = _get_da3_backbone(model)
            dim_in = ckpt.get("dim_in", 2 * backbone.pretrained.embed_dim)
            seg_head = FloorSegHead(dim_in=dim_in, patch_size=14)
            seg_head.load_state_dict(ckpt["seg_head_state_dict"])
            seg_head = seg_head.to(device).to(torch.bfloat16)
            seg_head.eval()
            model.seg_head = seg_head
            model.seg_layer_idx = layer_idx
            print(f"Loaded DA3 floor seg head (layer={layer_idx}, dim_in={dim_in})")
        return model

    print("Initializing and loading VGGT model...")
    model = VGGT(enable_floor_seg=args.places)
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL), strict=False)

    if args.places:
        from vggt.heads.floor_seg_head import FloorSegHead
        seg_ckpt = torch.load("checkpoints/ground_detect.pt", map_location="cpu")
        _dim_in = seg_ckpt.get("dim_in", 2048)
        _default_source = "both" if _dim_in == 2048 else "frame"
        seg_head = FloorSegHead(
            dim_in=_dim_in,
            patch_size=14,
            layer_idx=seg_ckpt.get("layer_idx", 10),
            token_source=seg_ckpt.get("token_source", _default_source),
        )
        seg_head.load_state_dict(seg_ckpt["seg_head_state_dict"])
        model.seg_head = seg_head

    model.eval()
    model = model.to(torch.bfloat16).to(device)
    return model


# ---------------------------------------------------------------------------
# Interactive query loop
# ---------------------------------------------------------------------------

def run_interactive_loop(args, objects_module, solver) -> None:
    """Post-processing query loop shared by all entry points.

    Depending on the flags this runs the LLM agent loop, the open-set
    text/region query loop, or simply blocks so a --rooms visualization stays
    open.  Frame capture and viser GUI-panel registration are the caller's
    responsibility; this only handles the interactive terminal session.
    """
    if args.run_os and args.agent:
        agent_config = AgentConfig(model_name=args.agent_model,
                                   visual_verification=args.agent_visual_verification)
        agent = TaskAgent(agent_config, objects_module, solver)

        while True:
            task = input("\n[Agent] Enter a task (or 'q' to quit): ").strip()
            if task == "q":
                print("Exiting.")
                break
            if not task:
                print("Empty task. Try again.")
                continue

            task_response, bboxes, regions = agent.run(task)

            print(f"\n{'='*60}")
            print(f"Reasoning: {task_response.reasoning}")
            print(f"\nDetected {len(task_response.objects)} object(s):")
            for obj in task_response.objects:
                print(f"  - {obj.label}: center={obj.center}, extent={obj.extent}")
            if task_response.regions:
                print(f"\nFound {len(task_response.regions)} region(s):")
                for reg in task_response.regions:
                    print(f"  - {reg.label}: centroid={reg.centroid}, extent={reg.extent}, "
                          f"nodes={reg.num_nodes}, confidence={reg.confidence:.3f}")
            print(f"{'='*60}")

    elif args.run_os and args.log_os:
        import scripts.run_eval as run_eval
        run_eval.run_object_evals(
            os.path.abspath("clio_tasks/tasks_apartment.yaml"),
            os.path.abspath("clio_tasks/est_bboxes_apartment.yaml"),
            objects_module,
            solver,
        )
        run_eval.run_object_evals(
            os.path.abspath("clio_tasks/tasks_cubicle.yaml"),
            os.path.abspath("clio_tasks/est_bboxes_cubicle.yaml"),
            objects_module,
            solver,
        )
        run_eval.run_object_evals(
            os.path.abspath("clio_tasks/tasks_office.yaml"),
            os.path.abspath("clio_tasks/est_bboxes_office.yaml"),
            objects_module,
            solver,
        )

    elif args.run_os:
        record_of_detections = []
        region_clusterer = None
        while True:
            query = input("\nEnter text query (or r: <query> for region; plan path; clear; q): ").strip()

            if query == "q":
                print("Exiting.")
                break
            if not query:
                print("Empty query. Try again.")
                continue

            if query == "plan path":
                if solver.places is None:
                    print("Places layer not enabled. Enable with --places flag and re-run.")
                    continue
                if len(record_of_detections) < 2:
                    print("Need to detect at least two objects before planning a path.")
                    continue
                point1 = record_of_detections[-2][0].bbox.center
                point2 = record_of_detections[-1][0].bbox.center
                objects_module.plan_path(point1, point2, solver)
                continue
            if query == "clear":
                record_of_detections = []
                print("Cleared record of detections.")
                continue
            if query.startswith("r: "):
                region_query = query[3:].strip()
                assert len(region_query) > 0, "Region query cannot be empty after 'r: ' prefix."
                assert solver.places is not None, (
                    "Places module not initialized. Run with --places to enable region queries."
                )
                if region_clusterer is None:
                    region_clusterer = create_query_clusterer(
                        QueryClusterConfig(),
                        clip_model=objects_module.clip_model,
                        clip_tokenizer=objects_module.clip_tokenizer,
                    )
                t1 = time.time()
                output = query_region(region_clusterer, solver.places.graph, region_query,
                                      multi_region=not args.top_region)
                print(f"Region query time: {time.time() - t1:.2f}s")
                if output is None:
                    print(f"No regions found for query: '{region_query}'")
                    continue
                print(f"\nFound {len(output.regions)} region(s) for query: '{region_query}'")
                for reg in output.regions:
                    print(f"  - {reg.label}: centroid={reg.centroid}, extent={reg.extent}, "
                          f"nodes={reg.num_nodes}, confidence={reg.confidence:.3f}")
                solver.viewer.visualize_clusters(output.vis_result, solver.places)
                continue
            if query == "_full":
                assert solver.places is not None, (
                    "Places module not initialized. Run with --places to enable full restore."
                )
                solver.viewer.restore_full_places(solver.places)
                continue

            t1 = time.time()
            detected_bboxes = objects_module.get_3D_object(query, solver)
            print(f"Query time: {time.time() - t1:.2f}s")
            if len(detected_bboxes) > 0:
                record_of_detections.append(detected_bboxes)

    elif args.rooms:
        input("\n[Rooms] Visualisation ready - press Enter to exit.\n")
