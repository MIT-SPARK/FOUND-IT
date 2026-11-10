"""Measure the in-RAM data footprint of the VGGT-SLAM map, by component.

Sums the *actual bytes* of every array held in the live SLAM state (numpy
`nbytes`, torch `element_size * nelement`, PIL images as their pixel array),
excluding model weights and framework/allocator overhead.  This is the
reproducible, format-independent "map memory" number to report in a paper;
it decomposes into RGB / depth / 3D points / colors / masks / embeddings /
places graph / objects cache.

Usage — run any entry point with ``--report_memory`` to print the table (and,
if psutil is installed, the RSS growth since models loaded) at the end of a run.

Or call it directly in-process where ``solver``/``objects_module`` exist:

    from foundit.measure_memory import measure_slam_memory
    measure_slam_memory(solver, objects_module)      # prints a table, returns dict

For disk, save a session and measure it directly:

    du -ab <session_dir> | sort -n        # per-file bytes, on-disk session
    du -sb <keyframe_folder>              # RGB PNGs (the map's image store)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
except Exception:            # torch always present in this repo, but stay defensive
    torch = None
try:
    from PIL import Image
except Exception:
    Image = None


def _nbytes(x) -> int:
    """Raw data bytes held by x (numpy / torch / PIL / list / None). 0 otherwise."""
    if x is None:
        return 0
    if isinstance(x, np.ndarray):
        return int(x.nbytes)
    if torch is not None and isinstance(x, torch.Tensor):
        return int(x.element_size() * x.nelement())
    if Image is not None and isinstance(x, Image.Image):
        return int(np.asarray(x).nbytes)
    if isinstance(x, (list, tuple)):
        return sum(_nbytes(e) for e in x)
    return 0


def _str_list_bytes(names) -> int:
    """Byte size of a list of path strings (RGB pixels themselves live on disk)."""
    return sum(len(s) for s in names) if names else 0


def _submap_bytes(submap) -> dict[str, int]:
    """Per-component bytes for one submap (fields defined in foundit/submap.py)."""
    return {
        "image_paths":           _str_list_bytes(submap.img_names),                  # img_names strings; RGB pixels are on disk (measure with `du` on keyframes/)
        "depth_maps":            _nbytes(submap.depth_maps),                          # (S,H,W) float16
        "conf_masks":            _nbytes(submap.conf_masks),                          # (S,H,W) bool
        "ground_seg":            _nbytes(submap.ground_seg),                          # (S,H,W) bool, optional
        "sparse_points":         _nbytes(submap.sparse_points),                       # (N,3) float32
        "sparse_colors":         _nbytes(submap.sparse_colors),                       # (N,3) uint8
        "sparse_index_arrays":   (_nbytes(submap.sparse_frame_indices)                # (N,) int8
                                  + _nbytes(submap.sparse_pixel_row)                  # (N,) uint16
                                  + _nbytes(submap.sparse_pixel_col)),                # (N,) uint16
        "non_ground_mask":       _nbytes(submap.non_ground_mask),                     # (N,) bool, optional

        "camera_params":         (_nbytes(submap.poses)                              # (S,4,4)
                                  + _nbytes(submap.proj_mats)),
        "place_recog_embeddings": _nbytes(submap.retrieval_vectors),                  # SALAD/DINO place-recognition vectors
        "frame_semantic_embeddings": _nbytes(submap.semantic_vectors),               # per-frame CLIP vectors
    }


def _places_bytes(places) -> dict[str, int]:
    """Bytes held by the places layer: node positions, node semantic embeddings,
    graph adjacency, and cached tile-mesh geometry (visualisation)."""
    g = places._place_graph
    n_edges = sum(len(v) for v in g._global_adjacency.values())  # directed entries (each undirected edge counted twice)
    ext = places._extractor
    return {
        "places_node_positions":  _nbytes(g._positions),
        "places_node_embeddings": _nbytes(g._embeddings),     # vMF semantic mean per place node
        "places_adjacency":       n_edges * 4,                # logical int32 per adjacency entry
        "places_tile_mesh":       (sum(_nbytes(v) for v in ext.submap_tile_verts.values())
                                   + sum(_nbytes(v) for v in ext.submap_tile_faces.values())
                                   + sum(_nbytes(v) for v in ext.submap_internal_edge_pts.values())),
    }


def _objects_bytes(objects_module) -> dict[str, int]:
    """Bytes held by the open-set objects cache (per-detection point clouds,
    colors, masks, and cached source images)."""
    out = {"objects_points": 0, "objects_colors": 0, "objects_masks": 0, "objects_source_images": 0}
    mem = objects_module.cached_memory
    for detections in mem.objects.values():
        for det in detections:
            out["objects_points"]        += _nbytes(getattr(det, "points", None))
            out["objects_colors"]        += _nbytes(getattr(det, "colors", None))
            out["objects_masks"]         += _nbytes(det.mask)
            out["objects_source_images"] += _nbytes(det.source_image)
    return out


def measure_slam_memory(solver, objects_module=None, verbose: bool = True) -> dict:
    """Return {component: bytes} for the live SLAM map and print a normalized table.

    Excludes model weights (only walks solver.map / solver.places / objects cache
    / pose graph).  Counts are raw array bytes — the format-independent footprint.
    """
    totals: dict[str, int] = {}

    def add(d: dict[str, int]) -> None:
        for k, v in d.items():
            totals[k] = totals.get(k, 0) + v

    submaps = list(solver.map.get_submaps())
    n_keyframes = 0
    for sm in submaps:
        add(_submap_bytes(sm))
        n_keyframes += len(sm.img_names) if sm.img_names else (len(sm.poses) if sm.poses is not None else 0)

    n_place_nodes = 0
    if getattr(solver, "places", None) is not None:
        add(_places_bytes(solver.places))
        n_place_nodes = len(solver.places._place_graph._positions)

    n_objects = 0
    if objects_module is not None:
        add(_objects_bytes(objects_module))
        n_objects = sum(len(v) for v in objects_module.cached_memory.objects.values())

    # Pose graph: SL4 (4x4 float64) per initialised node — small, approximate.
    n_pose_nodes = len(getattr(solver.graph, "initialized_nodes", []))
    totals["pose_graph"] = n_pose_nodes * 16 * 8

    grand = sum(totals.values())

    if verbose:
        mb = 1024 * 1024
        print("\n=== VGGT-SLAM map memory (RAM, models excluded) ===")
        print(f"{'component':<28}{'MB':>10}{'% total':>10}")
        print("-" * 48)
        for k in sorted(totals, key=lambda k: -totals[k]):
            print(f"{k:<28}{totals[k]/mb:>10.3f}{100*totals[k]/max(grand,1):>9.1f}%")
        print("-" * 48)
        print(f"{'TOTAL':<28}{grand/mb:>10.3f}{100:>9.1f}%")
        print(f"\nsubmaps={len(submaps)}  keyframes={n_keyframes}  "
              f"place_nodes={n_place_nodes}  cached_objects={n_objects}")
        if n_keyframes:
            print(f"per keyframe : {grand/n_keyframes/1024:.1f} KB")
        if submaps:
            print(f"per submap   : {grand/len(submaps)/mb:.3f} MB")

    totals["_TOTAL"] = grand
    totals["_n_submaps"] = len(submaps)
    totals["_n_keyframes"] = n_keyframes
    totals["_n_place_nodes"] = n_place_nodes
    totals["_n_objects"] = n_objects
    return totals


def rss_bytes() -> Optional[int]:
    """Current process resident set size in bytes, or None if psutil is missing.

    Call once right after models are built (baseline) and again at end-of-run;
    the difference is the SLAM state's contribution to RAM with models excluded.
    """
    try:
        import os
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        return None


def report_memory(solver, objects_module=None, baseline_rss: Optional[int] = None) -> dict:
    """End-of-run report: print the per-component map table and, when a baseline
    RSS (captured right after models loaded) is given, the process RSS growth
    since — a models-excluded estimate of total RAM incl. allocator overhead.
    """
    totals = measure_slam_memory(solver, objects_module, verbose=True)
    if baseline_rss is not None:
        now = rss_bytes()
        if now is not None:
            print(f"process RSS growth since models loaded: "
                  f"{(now - baseline_rss)/1024/1024:.1f} MB "
                  f"(SLAM state + allocator overhead; models excluded)")
    else:
        print("(pass --report_memory before models load to also get RSS growth; "
              "requires psutil)")
    return totals
