"""Session persistence — save and load a complete VGGT-SLAM session.

Saved layout
------------
session_dir/
  config.json                      # solver configuration scalars
  graph.pkl                        # {node_id: 4×4 SL4 matrix}
  keyframes/                       # RGB keyframe PNGs (RGB is read from disk, not
    <basename>.png                 #   held in RAM); referenced by submap img_names
  map/
    metadata.json                  # submap IDs, non_lc_submap_ids
    submap_<id>/
      metadata.json                # scalar / string fields; img_names are basenames
                                   #   resolved against the top-level keyframes/ dir
      poses.npy                    # (S, 4, 4) local poses
      proj_mats.npy                # (S, 4, 4) inverse intrinsics
      retrieval_vectors.npy        # (S, D) DINOv2-SALAD embeddings
      semantic_vectors.npy         # (S, D) CLIP embeddings
      conf_masks.npy               # (S, H, W) bool
      depth_maps.npy               # (S, H, W) float16
      sparse_points.npy            # (N, 3) float32
      sparse_colors.npy            # (N, 3) uint8
      sparse_frame_indices.npy     # (N,)  int8
      sparse_pixel_row.npy         # (N,)  uint16
      sparse_pixel_col.npy         # (N,)  uint16
      non_ground_mask.npy          # (N,)  bool          [optional]
      ground_seg.npy               # (S, H, W) bool      [optional]
  places.pkl                       # places graph + extractor state [optional]
  objects.pkl                      # cached object detections [optional]
"""

from __future__ import annotations

import os
import json
import pickle
import shutil

import numpy as np
import torch

import gtsam
from gtsam.symbol_shorthand import X

from foundit.submap import Submap
from foundit.map import GraphMap
from foundit.graph import PoseGraph


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_session(solver, session_dir: str, objects_module=None) -> None:
    """Persist the complete solver state to *session_dir*.

    When *objects_module* is given, its cached object detections are saved too so
    they survive into a resumed session.
    """
    os.makedirs(session_dir, exist_ok=True)
    print(f"[Session] Saving to '{session_dir}' …")

    # 1. Config
    config = {
        "init_conf_threshold": solver.init_conf_threshold,
        "voxel_size":          solver.voxel_size,
        "lc_thres":            solver.lc_thres,
        "use_places_layer":    solver.use_places_layer,
        "use_metric_scale":    solver.use_metric_scale,
        "vis_imgs":            solver.vis_imgs,
    }
    _write_json(os.path.join(session_dir, "config.json"), config)

    # 2. Graph (SL4 values only — factors are not needed after optimisation)
    _save_graph(solver.graph, os.path.join(session_dir, "graph.pkl"))

    # 3. Map / submaps.  RGB keyframes live on disk (not in RAM), so copy the
    #    images each submap references into a shared, session-local keyframes/
    #    folder and store img_names as basenames — makes the session self-contained.
    map_dir = os.path.join(session_dir, "map")
    os.makedirs(map_dir, exist_ok=True)
    keyframes_dir = os.path.join(session_dir, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)
    submap_ids = list(solver.map.submaps.keys())
    _write_json(os.path.join(map_dir, "metadata.json"), {
        "submap_ids":       submap_ids,
        "non_lc_submap_ids": solver.map.non_lc_submap_ids,
    })
    for sid, submap in solver.map.submaps.items():
        sub_dir = os.path.join(map_dir, f"submap_{sid}")
        _save_submap(submap, sub_dir, keyframes_dir)
        print(f"[Session]   submap {sid} saved")

    # 4. Places
    if solver.places is not None:
        _save_places(solver.places, solver._confirmed_interior,
                     os.path.join(session_dir, "places.pkl"))
        print("[Session]   Places graph saved")

    # 5. Cached object detections
    if objects_module is not None:
        _save_objects(objects_module, os.path.join(session_dir, "objects.pkl"))
        print("[Session]   Object cache saved")

    print(f"[Session] Done — {len(submap_ids)} submaps.")


def _save_graph(graph: PoseGraph, path: str) -> None:
    data = {"num_loop_closures": graph.num_loop_closures, "nodes": {}}
    for sym in graph.initialized_nodes:
        node_id = int(gtsam.Symbol(sym).index())
        data["nodes"][node_id] = graph.values.atSL4(sym).matrix()
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _save_submap(submap: Submap, sub_dir: str, keyframes_dir: str) -> None:
    os.makedirs(sub_dir, exist_ok=True)

    # RGB frames are stored on disk; copy the ones this submap references into the
    # session keyframes folder and record just their basenames (resolved back to
    # full paths on load).
    _copy_keyframes(submap.img_names, keyframes_dir)

    meta = {
        "submap_id":                 submap.submap_id,
        "frame_ids":                 [float(x) for x in submap.frame_ids],
        "img_names":                 [os.path.basename(p) for p in submap.img_names],
        "last_non_loop_frame_index": submap.last_non_loop_frame_index,
        "is_lc_submap":              submap.is_lc_submap,
        "time_created":              submap.time_created,
    }
    _write_json(os.path.join(sub_dir, "metadata.json"), meta)

    _save_npy(sub_dir, "poses",            submap.poses)
    _save_npy(sub_dir, "proj_mats",        submap.proj_mats)
    _save_npy(sub_dir, "conf_masks",       submap.conf_masks)
    _save_npy(sub_dir, "depth_maps",       submap.depth_maps)
    _save_npy(sub_dir, "sparse_points",    submap.sparse_points)
    _save_npy(sub_dir, "sparse_colors",    submap.sparse_colors)
    _save_npy(sub_dir, "sparse_frame_indices", submap.sparse_frame_indices)
    _save_npy(sub_dir, "sparse_pixel_row", submap.sparse_pixel_row)
    _save_npy(sub_dir, "sparse_pixel_col", submap.sparse_pixel_col)
    _save_npy(sub_dir, "non_ground_mask",  submap.non_ground_mask)

    # retrieval_vectors may be a torch Tensor or list of tensors
    rv = submap.retrieval_vectors
    if rv is not None:
        if isinstance(rv, torch.Tensor):
            rv = rv.cpu().numpy()
        elif isinstance(rv, list) and rv and isinstance(rv[0], torch.Tensor):
            rv = torch.stack(rv).cpu().numpy()
        else:
            rv = np.array(rv)
        _save_npy(sub_dir, "retrieval_vectors", rv)

    sv = submap.semantic_vectors
    if sv is not None and (not isinstance(sv, list) or len(sv) > 0):
        _save_npy(sub_dir, "semantic_vectors", np.array(sv, dtype=np.float32))

    if submap.ground_seg is not None:
        gs = submap.ground_seg
        if isinstance(gs, torch.Tensor):
            gs = gs.cpu().numpy()
        _save_npy(sub_dir, "ground_seg", gs.astype(np.bool_))


def _save_places(places, confirmed_interior, path: str) -> None:
    g   = places._place_graph
    ext = places._extractor

    embeddings_serial = []
    for emb in g._embeddings:
        embeddings_serial.append(emb.tolist() if emb is not None else None)

    data = {
        # PlaceGraph
        "positions":           [pos.tolist() for pos in g._positions],
        "submap_ids":          g._submap_ids,
        "embeddings":          embeddings_serial,
        "frame_counts":        list(g._frame_counts),
        "concentrations":      list(g._concentrations),
        "resultant_lengths":   list(g._resultant_lengths),
        "global_adjacency":    {str(k): list(v) for k, v in g._global_adjacency.items()},
        "internal_adjacency":  {str(k): list(v) for k, v in g._internal_adjacency.items()},
        "submap_node_indices": {k: list(v) for k, v in g._submap_node_indices.items()},
        "deleted":             list(g._deleted),
        # GroundExtractor
        "submap_plane_rotations":   {k: v.tolist() for k, v in ext.submap_plane_rotations.items()},
        "submap_ref_homographies":  {k: v.tolist() for k, v in ext.submap_ref_homographies.items()},
        "submap_tile_verts":        {k: v.tolist() for k, v in ext.submap_tile_verts.items()},
        "submap_tile_faces":        {k: v.tolist() for k, v in ext.submap_tile_faces.items()},
        # Solver-owned frontier state: node indices latched as interior so the
        # one-way frontier -> interior transition survives a save/load round trip.
        "confirmed_interior":       [int(i) for i in confirmed_interior],
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _save_objects(objects_module, path: str) -> None:
    """Serialise the object detection cache (CachedMemory).

    Detections hold numpy bbox arrays, an optional mask, and an optional PIL
    source image; the latter is stored as a uint8 array for robust pickling.
    """
    mem = objects_module.cached_memory
    data = {"objects": {}, "query_times": dict(mem.query_times)}
    for query, detections in mem.objects.items():
        serial = []
        for det in detections:
            bbox = det.bbox
            serial.append({
                "center":   np.asarray(bbox.center, dtype=np.float32),
                "extent":   np.asarray(bbox.extent, dtype=np.float32),
                "rotation": np.asarray(bbox.rotation, dtype=np.float32),
                "label":    bbox.label,
                "mask":     det.mask.astype(np.bool_) if det.mask is not None else None,
                "source_image": (np.asarray(det.source_image, dtype=np.uint8)
                                 if det.source_image is not None else None),
                "points":   (np.asarray(det.points, dtype=np.float32)
                             if det.points is not None else None),
                "colors":   (np.asarray(det.colors, dtype=np.uint8)
                             if det.colors is not None else None),
                "submap_id":   det.submap_id,
                "frame_index": det.frame_index,
                "ref_homography": (np.asarray(det.ref_homography, dtype=np.float64)
                                   if det.ref_homography is not None else None),
            })
        data["objects"][query] = serial
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_session(session_dir: str, solver, objects_module=None) -> None:
    """Populate *solver* from a previously saved session directory.

    The solver should be freshly constructed with the right flags
    (use_places_layer, vis_imgs, …) before calling this.  When *objects_module*
    is given, the cached object detections are restored into it.
    """
    print(f"[Session] Loading from '{session_dir}' …")

    # 1. Graph
    _load_graph(solver.graph, os.path.join(session_dir, "graph.pkl"))

    # 2. Map
    map_dir = os.path.join(session_dir, "map")
    keyframes_dir = os.path.join(session_dir, "keyframes")
    meta    = _read_json(os.path.join(map_dir, "metadata.json"))
    solver.map.non_lc_submap_ids = meta["non_lc_submap_ids"]
    for sid in meta["submap_ids"]:
        sub_dir = os.path.join(map_dir, f"submap_{sid}")
        submap  = _load_submap(sub_dir, keyframes_dir)
        solver.map.submaps[submap.submap_id] = submap
        print(f"[Session]   submap {sid} loaded")

    # 3. Places
    places_path = os.path.join(session_dir, "places.pkl")
    if os.path.exists(places_path) and solver.places is not None:
        solver._confirmed_interior = _load_places(places_path, solver.places)
        print("[Session]   Places graph loaded")

    # 4. Cached object detections
    objects_path = os.path.join(session_dir, "objects.pkl")
    if os.path.exists(objects_path) and objects_module is not None:
        _load_objects(objects_path, objects_module)
        print("[Session]   Object cache loaded")

    print(f"[Session] Done — {len(meta['submap_ids'])} submaps.")


def _load_graph(graph: PoseGraph, path: str) -> None:
    with open(path, "rb") as f:
        data = pickle.load(f)
    graph.num_loop_closures = data.get("num_loop_closures", 0)
    for node_id, mat in data["nodes"].items():
        graph.add_homography(int(node_id), mat)
        # Pin every loaded node with a stiff prior so the prior session acts as a
        # rigid reference: the saved factor graph stored only node values (not the
        # between/prior factors), so without this the optimiser would face
        # unconstrained variables once the current session is added.
        graph.add_prior_factor(int(node_id), mat)


def _load_submap(sub_dir: str, keyframes_dir: str) -> Submap:
    meta   = _read_json(os.path.join(sub_dir, "metadata.json"))
    sid    = meta["submap_id"]
    submap = Submap(sid)

    submap.frame_ids                = meta["frame_ids"]
    # img_names were saved as basenames; resolve them against the session's
    # keyframes/ folder so RGB is read from the copied images.
    submap.img_names                = [os.path.join(keyframes_dir, bn)
                                       for bn in meta["img_names"]]
    submap.last_non_loop_frame_index = meta["last_non_loop_frame_index"]
    submap.is_lc_submap             = meta["is_lc_submap"]
    submap.time_created             = meta.get("time_created")

    submap.poses           = _load_npy(sub_dir, "poses")
    submap.proj_mats       = _load_npy(sub_dir, "proj_mats")
    submap.conf_masks      = _load_npy(sub_dir, "conf_masks")
    submap.depth_maps      = _load_npy(sub_dir, "depth_maps")

    submap.sparse_points        = _load_npy(sub_dir, "sparse_points")
    submap.sparse_colors        = _load_npy(sub_dir, "sparse_colors")
    submap.sparse_frame_indices = _load_npy(sub_dir, "sparse_frame_indices")
    submap.sparse_pixel_row     = _load_npy(sub_dir, "sparse_pixel_row")
    submap.sparse_pixel_col     = _load_npy(sub_dir, "sparse_pixel_col")
    submap.non_ground_mask      = _load_npy(sub_dir, "non_ground_mask")

    rv = _load_npy(sub_dir, "retrieval_vectors")
    if rv is not None:
        submap.retrieval_vectors = torch.from_numpy(rv)

    sv = _load_npy(sub_dir, "semantic_vectors")
    if sv is not None:
        submap.semantic_vectors = sv

    gs                = _load_npy(sub_dir, "ground_seg")
    submap.ground_seg = torch.from_numpy(gs) if gs is not None else None

    return submap


def _load_places(path: str, places) -> set:
    """Restore the places graph and return the latched-interior node-index set.

    The returned set is empty for sessions saved before frontier latching
    existed (the ``confirmed_interior`` key is absent).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    g = places._place_graph
    g._positions          = [np.array(p, dtype=np.float32) for p in data["positions"]]
    g._submap_ids         = data["submap_ids"]
    g._embeddings         = [
        np.array(e, dtype=np.float32) if e is not None else None
        for e in data["embeddings"]
    ]
    g._frame_counts       = data["frame_counts"]
    g._concentrations     = data["concentrations"]
    g._resultant_lengths  = data["resultant_lengths"]
    g._global_adjacency   = {int(k): list(v) for k, v in data["global_adjacency"].items()}
    g._internal_adjacency = {int(k): list(v) for k, v in data["internal_adjacency"].items()}
    g._submap_node_indices = {k: list(v) for k, v in data["submap_node_indices"].items()}
    g._deleted            = set(data["deleted"])

    ext = places._extractor
    ext.submap_plane_rotations  = {k: np.array(v, dtype=np.float64) for k, v in data["submap_plane_rotations"].items()}
    ext.submap_ref_homographies = {k: np.array(v, dtype=np.float64) for k, v in data["submap_ref_homographies"].items()}
    ext.submap_tile_verts       = {k: np.array(v, dtype=np.float32) for k, v in data.get("submap_tile_verts", {}).items()}
    ext.submap_tile_faces       = {k: np.array(v, dtype=np.int64)   for k, v in data.get("submap_tile_faces", {}).items()}

    return set(data.get("confirmed_interior", []))


def _load_objects(path: str, objects_module) -> None:
    """Restore the object detection cache (CachedMemory) from disk."""
    from PIL import Image
    from foundit.objects import Bbox, Detection

    with open(path, "rb") as f:
        data = pickle.load(f)

    mem = objects_module.cached_memory
    for query, serial in data.get("objects", {}).items():
        detections = []
        for e in serial:
            bbox = Bbox(e["center"], e["extent"], e["rotation"], e.get("label"))
            src = e.get("source_image")
            detections.append(Detection(
                bbox=bbox,
                mask=e.get("mask"),
                source_image=Image.fromarray(src) if src is not None else None,
                points=e.get("points"),
                colors=e.get("colors"),
                submap_id=e.get("submap_id"),
                frame_index=e.get("frame_index"),
                ref_homography=e.get("ref_homography"),
            ))
        mem.objects[query] = detections
    mem.query_times.update(data.get("query_times", {}))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_keyframes(img_names, keyframes_dir: str) -> None:
    """Copy each referenced RGB keyframe into keyframes_dir, deduped by basename.

    Missing source files are skipped with a warning (the session then simply has
    no image for that frame).
    """
    for path in img_names:
        dst = os.path.join(keyframes_dir, os.path.basename(path))
        if os.path.exists(dst):
            continue                       # already copied (frames shared across submaps)
        if os.path.exists(path):
            shutil.copy2(path, dst)
        else:
            print(f"[Session]   WARNING: keyframe not found, skipping copy: {path}")


def _save_npy(directory: str, name: str, arr) -> None:
    if arr is None:
        return
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    np.save(os.path.join(directory, f"{name}.npy"), arr)


def _load_npy(directory: str, name: str):
    path = os.path.join(directory, f"{name}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=False)


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _read_json(path: str):
    with open(path) as f:
        return json.load(f)
