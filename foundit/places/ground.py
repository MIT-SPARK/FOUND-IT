from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
from scipy.spatial import cKDTree

from foundit.places.config import GroundConfig, SubmapTiles
from foundit.places.graph import PlaceGraph, estimate_vmf_kappa

def _build_tile_mesh(
    centers: np.ndarray,
    R: np.ndarray,
    place_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a single-sided quad mesh from tile center positions.

    Each center produces one quad (4 vertices, 2 triangles).
    Use side="double" on the renderer call for back-face visibility.

    Args:
        centers:    (N, 3) world-space tile centre positions.
        R:          (3, 3) rotation from the local plane frame to world frame.
        place_size: Tile side length in metres.

    Returns:
        (vertices, faces) arrays suitable for viser add_mesh_simple.
    """
    hs = (place_size / 2.0) * 0.85
    base_quad = np.array(
        [[-hs, -hs, 0], [hs, -hs, 0], [hs, hs, 0], [-hs, hs, 0]],
        dtype=np.float32,
    )
    rotated_quad = base_quad @ R.T

    verts_list, faces_list = [], []
    for i, center in enumerate(centers):
        verts_list.append(rotated_quad + center)
        b = i * 4
        faces_list += [
            [b, b+1, b+2], [b, b+2, b+3],
        ]
    return np.vstack(verts_list), np.array(faces_list)

class GroundExtractor:
    """Converts a submap into a set of navigable place nodes on the ground plane.

    Pipeline (per submap):
      A. RANSAC plane fit on ground-segmented points
      B. Project inliers to a 2D local frame aligned with the plane
      C. Sub-grid validation — a coarse tile (place_size x place_size) is accepted
         only if all four of its half-size quadrants contain at least one point
      D. Build intra-submap edges (8-connected grid) and cross-submap edges
         (radius search against all prior submaps)
      E. Return mesh geometry as SubmapTiles for the caller to render

    Caches plane rotations, homographies, and mesh geometry per submap so that
    PlaceLoopClosure can apply delta-H transforms after loop closures without
    rerunning the full pipeline.
    """

    def __init__(self, config: GroundConfig) -> None:
        self.config = config

        # Per-submap caches — shared with PlaceLoopClosure via reference so that
        # loop-closure delta transforms can update the same objects in-place.
        self.submap_plane_rotations: dict[str, np.ndarray] = {}
        self.submap_ref_homographies: dict[str, np.ndarray] = {}
        self.submap_tile_verts: dict[str, np.ndarray] = {}
        self.submap_tile_faces: dict[str, np.ndarray] = {}
        self.submap_internal_edge_pts: dict[str, Optional[np.ndarray]] = {}

        # Multi-session: submap ids (as strings) of a loaded prior session that is
        # not yet aligned to the current session.  While populated, merges and
        # cross-edges are not formed across this boundary (the two sessions live in
        # different frames, so proximity between them is meaningless).  Cleared once
        # a loop closure aligns the sessions.  Empty → single-session behaviour.
        self.isolated_submap_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def may_connect(self, sid_a: str, sid_b: str) -> bool:
        """Whether two submaps may merge / cross-edge.

        True unless exactly one of them belongs to a not-yet-aligned prior session
        (``isolated_submap_ids``); i.e. connections are allowed within the prior
        session and within the current session, but not across the boundary until
        alignment clears the set.  Always True when no session is isolated, so
        single-session behaviour is unchanged.
        """
        iso = self.isolated_submap_ids
        if not iso:
            return True
        return (str(sid_a) in iso) == (str(sid_b) in iso)

    def extract_tiles(
        self,
        submap,
        graph_backend,
        place_graph: PlaceGraph,
        use_all_points: bool = False,
        can_remove_ground_seg: bool = True
    ) -> SubmapTiles:
        """Run the full ground-extraction pipeline for one submap.

        Args:
            submap:         The submap to process.
            graph_backend:  Pose-graph backend used to transform points to world frame.
            place_graph:    Graph to add new nodes / merge into existing nodes.
            use_all_points: If True, use all ground-segmented pixels for RANSAC
                            (denser but slower). If False, use confidence-filtered
                            sparse points only.

        Returns:
            SubmapTiles with mesh geometry and edge data, or an empty SubmapTiles
            if the ground mask is absent / too sparse.
        """
        cfg = self.config
        submap_id = str(submap.get_id())

        # Load semantic vectors (used later for per-tile embeddings)
        semantic_vectors, has_semantic_vectors = self._load_semantic_vectors(submap)

        # Early exit if ground mask is absent or too sparse
        ground_seg = submap.get_ground_segmentation_mask()
        if ground_seg is None:
            return SubmapTiles(submap_id=submap_id)
        ground_seg_np = ground_seg.cpu().numpy()   # (S, H, W) bool
        if ground_seg_np.sum() <= 10:
            return SubmapTiles(submap_id=submap_id)

        # Extract world-frame ground and non-ground points
        ground_pts, ground_fi, non_ground_pts, n_sparse = self._extract_ground_points(
            submap, graph_backend, ground_seg_np, use_all_points,
        )

        # If we do not need to store ground seg for visualization or debugging then
        # we are now done with it and can remove it.
        if can_remove_ground_seg:
            submap.remove_ground_seg()
            
        # print(f"[Places {submap_id}] point extraction:           {time.time()-t0:.3f}s  "
        #       f"({n_sparse} sparse, {len(ground_pts)} ground pts)")

        # A. RANSAC plane fitting
        t0 = time.time()
        plane_centroid, plane_normal, inliers = _fit_plane_ransac(
            ground_pts,
            n_iters=cfg.ransac_n_iters,
            threshold=cfg.ransac_threshold,
            batch_size=cfg.ransac_batch_size,
        )
        # print(f"[Places {submap_id}] RANSAC plane fit:            {time.time()-t0:.3f}s")
        if inliers is None:
            # print(f"[Places {submap_id}] TOTAL:                      {time.time()-t_total:.3f}s")
            return SubmapTiles(submap_id=submap_id)

        clean_ground = ground_pts[inliers]
        clean_fi = ground_fi[inliers]

        # B. Build local coordinate frame aligned with the ground plane
        R_local_to_world = self._build_local_frame(plane_normal)
        self.submap_plane_rotations[submap_id] = R_local_to_world
        local_pts = (clean_ground - plane_centroid) @ R_local_to_world

        # C. Find fully covered tiles and remove obstacle-blocked ones
        t0 = time.time()
        unique_sub, fully_covered, coarse_x, coarse_y = self._find_fully_covered_tiles(
            local_pts, cfg.place_size, min_subcell=cfg.min_subcell_coverage,
        )
        blocked_tiles, z_axis_sign = self._find_blocked_tiles(
            non_ground_pts, plane_centroid, R_local_to_world,
            cfg.place_size, cfg.obstacle_height_limit,
        )
        if blocked_tiles and len(fully_covered) > 0:
            unblocked = np.array([tuple(t) not in blocked_tiles for t in fully_covered])
            fully_covered = fully_covered[unblocked]
        # print(f"[Places {submap_id}] tile discovery + obstacles:  {time.time()-t0:.3f}s  "
        #       f"({len(unique_sub)} sub-cells -> {len(fully_covered)} candidate tiles)")

        # Compute world-space center positions for all candidates
        world_centers = self._compute_tile_world_centers(
            fully_covered, R_local_to_world, plane_centroid, cfg.place_size, z_axis_sign,
        )

        # Compute per-tile semantic embeddings
        tile_emb = self._compute_tile_embeddings(
            fully_covered, clean_fi, coarse_x, coarse_y, semantic_vectors, has_semantic_vectors,
        )

        # Snapshot existing graph nodes for creation-time merge detection.  Nodes
        # this submap may not connect to (an unaligned prior session) are excluded,
        # so its tiles never merge across the session boundary.
        if len(place_graph) > 0:
            all_positions, all_indices = place_graph.get_active_node_data()
            keep = [self.may_connect(submap_id, place_graph.get_submap_id(i)) for i in all_indices]
            if any(keep):
                keep_arr = np.array(keep, dtype=bool)
                existing_positions = all_positions[keep_arr]
                existing_indices = [i for i, k in zip(all_indices, keep) if k]
                existing_tree = cKDTree(existing_positions[:, [0, 2]])
            else:
                existing_positions = None
                existing_tree, existing_indices = None, []
        else:
            existing_positions = None
            existing_tree, existing_indices = None, []
        merge_radius = cfg.place_size * cfg.merge_radius_factor

        # D. Merge candidate tiles into existing nodes or create new graph nodes
        t0 = time.time()
        valid_tiles, tile_centroids, local_to_global_idx, tile_to_nodes, n_merged = self._merge_and_assign_nodes(
            fully_covered, world_centers, tile_emb,
            place_graph, submap_id,
            existing_tree, existing_indices, existing_positions, merge_radius,
        )

        # Record current homography as the reference for future delta-H moves
        self.submap_ref_homographies[submap_id] = graph_backend.get_homography(submap.get_id())

        # E. Build intra-submap and cross-submap connectivity
        t0 = time.time()
        connect_dist = cfg.place_size * cfg.cross_edge_radius_factor
        internal_edges, cross_edges = self._build_connectivity(
            tile_to_nodes, local_to_global_idx, valid_tiles, tile_centroids,
            place_graph, submap_id, connect_dist,
        )
        self.submap_internal_edge_pts[submap_id] = np.array(internal_edges) if internal_edges else None

        # F. Build mesh geometry for rendering and cache it
        tile_vertices, tile_faces = None, None
        if valid_tiles:
            all_centroids = np.array(list(tile_centroids.values()))
            tile_vertices, tile_faces = _build_tile_mesh(all_centroids, R_local_to_world, cfg.place_size)
            self.submap_tile_verts[submap_id] = tile_vertices
            self.submap_tile_faces[submap_id] = tile_faces

        result = SubmapTiles(
            submap_id=submap_id,
            tile_vertices=tile_vertices,
            tile_faces=tile_faces,
            internal_edge_points=np.array(internal_edges) if internal_edges else None,
            cross_edge_points=np.array(cross_edges) if cross_edges else None,
            node_count=len(valid_tiles),
        )
        # print(f"[Places {submap_id}] TOTAL:                      {time.time()-t_total:.3f}s")
        return result

    # ------------------------------------------------------------------
    # Private helpers — each covers one stage of the pipeline
    # ------------------------------------------------------------------

    def _load_semantic_vectors(self, submap) -> tuple[Optional[np.ndarray], bool]:
        """Load per-frame semantic vectors from the submap.

        Returns (semantic_vectors, has_semantic_vectors). semantic_vectors is
        None when the submap has no valid vectors.
        """
        vecs = submap.get_all_semantic_vectors()
        if isinstance(vecs, list) and len(vecs) == 0:
            return None, False
        vecs = np.array(vecs)
        has = vecs.ndim == 2 and vecs.shape[0] > 0
        return (vecs if has else None), has

    def _extract_ground_points(
        self, submap, graph_backend, ground_seg_np: np.ndarray, use_all_points: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Split the submap's world-frame sparse points into ground and non-ground sets.

        Returns (ground_pts, ground_frame_indices, non_ground_pts, n_sparse_total).
        """
        sparse_world_pts, sparse_fi_arr = submap.get_world_points_and_frame_indices(graph_backend)
        fi = submap.sparse_frame_indices.astype(np.int32)
        pr = submap.sparse_pixel_row.astype(np.int32)
        pc = submap.sparse_pixel_col.astype(np.int32)
        ground_labels = ground_seg_np[fi, pr, pc]
        non_ground_mask = ~ground_labels
        non_ground_pts = sparse_world_pts[non_ground_mask]
        # Persist the mask on the submap so the whole-graph obstacle check
        # (solver.compute_blocked_nodes) can reuse only the non-ground points,
        # transformed to world with the current graph.
        submap.set_non_ground_mask(non_ground_mask)

        if use_all_points:
            ground_pts, ground_fi = submap.get_all_ground_points_in_world_frame(graph_backend)
        else:
            ground_pts = sparse_world_pts[ground_labels]
            ground_fi  = sparse_fi_arr[ground_labels]

        return ground_pts, ground_fi, non_ground_pts, len(sparse_world_pts)

    def _build_local_frame(self, plane_normal: np.ndarray) -> np.ndarray:
        """Build the rotation matrix from the local plane frame to world frame.

        The z-axis of the local frame equals the plane normal. The x/y axes are
        chosen to be orthogonal to it in a right-handed coordinate system.

        Returns:
            R_local_to_world: (3, 3) rotation matrix whose columns are [x, y, z].
        """
        z_axis = plane_normal
        ref    = np.array([0, 0, 1]) if np.abs(z_axis[2]) < 0.9 else np.array([0, 1, 0])
        x_axis = np.cross(z_axis, ref)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    def _find_fully_covered_tiles(
        self, local_pts: np.ndarray, place_size: float,
        min_subcell: int = 4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find coarse tiles where enough sub-cells contain ground points.

        The ground plane is divided into coarse tiles of size place_size x place_size,
        each sub-divided into a 2x2 grid of half-size cells. A tile is accepted if
        at least ``min_subcell`` of 4 sub-cells are occupied.

        Uses an int32->int64 packing trick for fast unique-row operations:
        viewing a contiguous (N, 2) int32 array as (N,) int64 lets np.unique sort
        a flat integer array instead of sorting 2D rows (~5-10x faster).

        Returns:
            unique_sub:    (M,) int64 packed unique sub-cell keys.
            fully_covered: (K, 2) int32 array of coarse tile indices [ix, iy].
            coarse_x:      (N,) per-point coarse x index (used for embedding grouping).
            coarse_y:      (N,) per-point coarse y index.
        """
        sub_h = place_size / 2.0
        sub_grid_indices = np.floor(local_pts[:, :2] / sub_h).astype(np.int32)
        coarse_x = sub_grid_indices[:, 0] >> 1   # floor-divide by 2 via arithmetic right-shift
        coarse_y = sub_grid_indices[:, 1] >> 1

        # Fast 1D unique via int32->int64 packing
        packed_sub = np.ascontiguousarray(sub_grid_indices).view(np.int64).ravel()
        unique_sub = np.unique(packed_sub)

        # Derive coarse tile for each unique sub-cell, then count sub-cells per tile
        coarse_i32 = np.ascontiguousarray(unique_sub.view(np.int32).reshape(-1, 2) >> 1).astype(np.int32)
        packed_coarse = np.ascontiguousarray(coarse_i32).view(np.int64).ravel()
        unique_coarse_packed, sc_counts = np.unique(packed_coarse, return_counts=True)
        fully_covered = (
            unique_coarse_packed[sc_counts >= min_subcell]
            .view(np.int32).reshape(-1, 2)
        )

        return unique_sub, fully_covered, coarse_x, coarse_y

    def _find_blocked_tiles(
        self,
        non_ground_pts: np.ndarray,
        plane_centroid: np.ndarray,
        R_local_to_world: np.ndarray,
        place_size: float,
        obstacle_height_limit: float,
    ) -> tuple[set, int]:
        """Project non-ground points into the local frame and find obstacle tiles.

        Non-ground points between 0.1 m and obstacle_height_limit above the plane
        are treated as obstacles. Their coarse tile indices are returned as a set.

        The plane normal may point downward depending on RANSAC orientation; z_axis_sign
        corrects this so height is always measured upward from the floor.

        Returns:
            blocked_tiles: set of (ix, iy) coarse tile tuples that contain obstacles.
            z_axis_sign:   +1 or -1 — multiply local z by this to get floor-upward height.
        """
        if len(non_ground_pts) == 0:
            return set(), 1   # no obstacles; +1 is an arbitrary up-sign default
        all_local = (non_ground_pts - plane_centroid) @ R_local_to_world
        z_axis_sign = -1 if np.mean(all_local[:, 2]) < 0 else 1
        z_local = z_axis_sign * all_local[:, 2]
        is_obstacle = (z_local > 0.10) & (z_local < obstacle_height_limit)

        if not np.any(is_obstacle):
            return set(), z_axis_sign

        obs_indices = np.floor(all_local[is_obstacle][:, :2] / place_size).astype(np.int32)
        void_dt = np.dtype((np.void, obs_indices.dtype.itemsize * obs_indices.shape[1]))
        b = np.ascontiguousarray(obs_indices).view(void_dt)
        _, idx = np.unique(b, return_index=True)
        return set(map(tuple, obs_indices[idx])), z_axis_sign

    def _compute_tile_world_centers(
        self,
        fully_covered: np.ndarray,
        R_local_to_world: np.ndarray,
        plane_centroid: np.ndarray,
        place_size: float,
        z_axis_sign: int,
    ) -> np.ndarray:
        """Compute world-space center positions for a set of coarse tile indices.

        Returns:
            world_centers: (K, 3) array, or empty (0, 3) if fully_covered is empty.
        """
        if len(fully_covered) == 0:
            return np.empty((0, 3))
        h = place_size
        ix_arr, iy_arr = fully_covered[:, 0], fully_covered[:, 1]
        local_centers = np.column_stack([
            ix_arr * h + h / 2,
            iy_arr * h + h / 2,
            np.full(len(ix_arr), z_axis_sign * 0.00),  
            # We used to set this to 0.05 so that the places layer was 5 cm above the ground for 
            # visualization, but that assumes the first camera is gravity aligned.
        ])
        return (local_centers @ R_local_to_world.T) + plane_centroid

    def _compute_tile_embeddings(
        self,
        fully_covered: np.ndarray,
        clean_frame_indices: np.ndarray,
        coarse_x: np.ndarray,
        coarse_y: np.ndarray,
        semantic_vectors: Optional[np.ndarray],
        has_semantic_vectors: bool,
    ) -> dict[tuple[int, int], tuple]:
        """Compute vMF embedding statistics for each candidate tile.

        Groups ground points by tile using a linear encoding + argsort to avoid
        O(K x N) boolean mask operations.

        Returns:
            dict mapping (ix, iy) -> (avg_emb, r_bar, frame_count, concentration).
            avg_emb is None when no semantic vectors are available.
        """
        tile_emb: dict[tuple[int, int], tuple] = {}

        if not (has_semantic_vectors and len(fully_covered) > 0 and len(clean_frame_indices) > 0):
            for tx, ty in fully_covered:
                tile_emb[(int(tx), int(ty))] = (None, 0.0, 0, 0.0)
            return tile_emb

        # Encode each point as a 1D tile key for grouping via argsort + split
        y_range  = int(coarse_y.max() - coarse_y.min()) + 2
        y_off    = int(coarse_y.min())
        x_off    = int(coarse_x.min())
        pt_enc   = (coarse_x - x_off) * y_range + (coarse_y - y_off)
        sort_ord = np.argsort(pt_enc, kind='stable')
        s_enc    = pt_enc[sort_ord]
        s_fi     = clean_frame_indices[sort_ord]
        bounds   = np.flatnonzero(np.diff(s_enc)) + 1
        groups   = np.split(s_fi, bounds)
        g_encs   = s_enc[np.r_[0, bounds]]
        enc_map  = {int(k): v for k, v in zip(g_encs, groups)}

        for tx, ty in fully_covered:
            enc    = (int(tx) - x_off) * y_range + (int(ty) - y_off)
            frames = np.unique(enc_map.get(enc, np.array([], dtype=np.int32)))
            if frames.size > 0:
                embs    = semantic_vectors[frames]
                avg_emb = embs.mean(axis=0)
                r_bar   = float(np.linalg.norm(avg_emb))
                avg_emb = avg_emb / r_bar if r_bar > 1e-8 else avg_emb
                fc      = int(frames.size)
                conc    = estimate_vmf_kappa(r_bar, embs.shape[1], fc)
                tile_emb[(int(tx), int(ty))] = (avg_emb, r_bar, fc, conc)
            else:
                tile_emb[(int(tx), int(ty))] = (None, 0.0, 0, 0.0)

        return tile_emb

    def _merge_and_assign_nodes(
        self,
        fully_covered: np.ndarray,
        world_centers: np.ndarray,
        tile_emb: dict,
        place_graph: PlaceGraph,
        submap_id: str,
        existing_tree,
        existing_indices: list,
        existing_positions,
        merge_radius: float,
    ) -> tuple[list, dict, dict, dict, int]:
        """Batch-query the existing-node KDTree and merge or create nodes.

        For each candidate tile:
          - If within merge_radius of an existing node: accumulate the embedding
            into that node (no new node created).
          - Otherwise: add a new node to the graph.

        Returns:
            valid_tiles:        list of (ix, iy) tuples for newly created tiles.
            tile_centroids:     dict mapping (ix, iy) -> world center (3,), new tiles only.
            local_to_global_idx: dict mapping (ix, iy) -> graph node index, new tiles only.
            tile_to_nodes:      dict mapping EVERY candidate (ix, iy) -> the list of
                                graph node indices it resolved to — [new node] for a
                                created tile, or every same-floor node it merged into
                                for a merged tile.  Used by connectivity to bridge merges.
            n_merged:           number of tiles merged into existing nodes.
        """
        if existing_tree is not None and len(world_centers) > 0:
            batch_hits = existing_tree.query_ball_point(world_centers[:, [0, 2]], r=merge_radius)
        else:
            batch_hits = [[] for _ in range(len(world_centers))]

        max_dy = self.config.cross_edge_max_dy

        valid_tiles: list[tuple[int, int]] = []
        tile_centroids: dict[tuple[int, int], np.ndarray] = {}
        local_to_global_idx: dict[tuple[int, int], int] = {}
        tile_to_nodes: dict[tuple[int, int], list[int]] = {}
        n_merged = 0

        for k in range(len(fully_covered)):
            ix, iy = int(fully_covered[k, 0]), int(fully_covered[k, 1])
            world_center = world_centers[k]
            avg_emb, r_bar, frame_count, concentration = tile_emb[(ix, iy)]

            if batch_hits[k]:
                new_y = world_center[1]
                same_floor_hits = [
                    hj for hj in batch_hits[k]
                    if abs(existing_positions[hj, 1] - new_y) <= max_dy
                ]
                if same_floor_hits:
                    # Every same-floor overlapping node is a valid merge target, so
                    # record them all — the tile's grid adjacency bridges to each of
                    # them (see _build_connectivity).  Fold the embedding into a
                    # single (oldest) node only, to avoid double-counting the
                    # observation across the co-located nodes.
                    merge_nodes = [existing_indices[hj] for hj in same_floor_hits]
                    place_graph.merge_embedding(
                        min(merge_nodes), avg_emb, frame_count,
                        new_resultant_length=r_bar,
                    )
                    tile_to_nodes[(ix, iy)] = merge_nodes
                    n_merged += 1
                    continue

            valid_tiles.append((ix, iy))
            tile_centroids[(ix, iy)] = world_center
            new_idx = place_graph.add_node(
                world_center, submap_id, avg_emb,
                frame_count=frame_count, concentration=concentration,
                resultant_length=r_bar,
            )
            local_to_global_idx[(ix, iy)] = new_idx
            tile_to_nodes[(ix, iy)] = [new_idx]

        return valid_tiles, tile_centroids, local_to_global_idx, tile_to_nodes, n_merged

    def _build_connectivity(
        self,
        tile_to_nodes: dict,
        local_to_global_idx: dict,
        valid_tiles: list,
        tile_centroids: dict,
        place_graph: PlaceGraph,
        submap_id: str,
        connect_dist: float,
    ) -> tuple[list, list]:
        """Build grid connectivity for this submap, bridging across merged tiles.

        Every candidate tile has already been resolved to one or more graph nodes
        in ``tile_to_nodes`` — ``[new node]`` for a created tile, or every
        same-floor node it merged into for a merged tile.  We walk the 8-connected
        tile grid over *all* candidate tiles and connect every node of one tile to
        every node of each adjacent tile:

          * Two new tiles produce an ordinary intra-submap internal edge.
          * A tile that merged into existing node(s) carries its adjacency onto
            each of them.  So if new tiles C, D sit between existing nodes A, B and
            merge (C->A, D->B), the C-D adjacency becomes the A-B edge that a plain
            merge would otherwise lose — and if C also overlapped A', C-D bridges
            D's node(s) to A' as well.

        Edges are added as *internal* edges because they encode topological
        adjacency (not mere proximity) and must survive cross-edge rebuilds,
        which recompute proximity edges by radius and would drop a genuine A-B
        link that is too far apart to re-find.  Self-loops (both tiles merged
        into the same node), duplicate edges, and cross-floor links are skipped.

        A radius search then adds the usual cross-submap proximity edges from new
        tiles to nearby prior-submap nodes, skipping any pair the grid step
        already connected.

        Returns:
            internal_edges: list of [p1, p2] world-point pairs for visualisation.
            cross_edges:    list of [p1, p2] world-point pairs for visualisation.
        """
        internal_edges: list = []
        cross_edges: list    = []
        max_dy = self.config.cross_edge_max_dy

        # Grid connectivity over ALL candidate tiles, routed through merges.
        offsets = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        visited_pairs: set = set()
        for (ix, iy), nodes1 in tile_to_nodes.items():
            for dx, dy in offsets:
                neighbor_tile = (ix + dx, iy + dy)
                nodes2 = tile_to_nodes.get(neighbor_tile)
                if nodes2 is None:
                    continue
                pair_key = ((ix, iy), neighbor_tile) if (ix, iy) < neighbor_tile \
                    else (neighbor_tile, (ix, iy))
                if pair_key in visited_pairs:
                    continue
                visited_pairs.add(pair_key)
                # Each tile may resolve to several overlapping nodes (a merged tile
                # keeps every same-floor node within merge_radius), so bridge every
                # node of one tile to every node of the adjacent tile.
                for node1 in nodes1:
                    for node2 in nodes2:
                        if node1 == node2:
                            continue  # same node — no self-loop
                        if node2 in place_graph.get_neighbors(node1):
                            continue  # edge already exists — no duplicates
                        p1 = place_graph.get_position(node1)
                        p2 = place_graph.get_position(node2)
                        if abs(p1[1] - p2[1]) > max_dy:
                            continue  # reject cross-floor links
                        place_graph.add_internal_edge(node1, node2)
                        internal_edges.append([p1, p2])

        # Cross-submap: radius search against all prior-submap nodes. The
        # kd-tree is built on XZ only; we then reject pairs whose vertical
        # separation exceeds config.cross_edge_max_dy to prevent
        # cross-floor edges (e.g. kitchen above living-room).  Nodes across an
        # unaligned session boundary are excluded (see may_connect).
        other_indices = [i for i in place_graph.get_nodes_not_in_submap(submap_id)
                         if self.may_connect(submap_id, place_graph.get_submap_id(i))]
        if other_indices and valid_tiles:
            other_positions = place_graph.get_positions(other_indices)
            new_positions   = np.array([tile_centroids[(ix, iy)] for ix, iy in valid_tiles])
            hits = cKDTree(other_positions[:, [0, 2]]).query_ball_point(new_positions[:, [0, 2]], r=connect_dist, workers=-1)
            other_y = other_positions[:, 1]
            for tile_i, (ix, iy) in enumerate(valid_tiles):
                hs = hits[tile_i]
                if not hs:
                    continue
                new_y = new_positions[tile_i, 1]
                idx1 = local_to_global_idx[(ix, iy)]
                for tree_j in hs:
                    if abs(other_y[tree_j] - new_y) > max_dy:
                        continue
                    other_idx = other_indices[tree_j]
                    if other_idx in place_graph.get_neighbors(idx1):
                        continue  # already linked (e.g. by the grid step) — no duplicates
                    cross_edges.append([tile_centroids[(ix, iy)], other_positions[tree_j]])
                    place_graph.add_cross_edge(idx1, other_idx)

        return internal_edges, cross_edges


def _fit_plane_ransac(
    points: np.ndarray,
    n_iters: int = 2000,
    threshold: float = 0.05,
    batch_size: int = 100,
) -> tuple:
    """Fit a plane to a point cloud using batched GPU RANSAC.

    Processes `batch_size` random 3-point samples per GPU kernel launch to
    amortise CUDA overhead, then refines the best candidate with SVD.

    Args:
        points:     (N, 3) float32 point cloud in world frame.
        n_iters:    Total number of random samples to evaluate.
        threshold:  Inlier distance threshold in metres.
        batch_size: Number of samples per GPU batch.

    Returns:
        (centroid, normal, inlier_mask) where centroid and normal are (3,)
        numpy arrays and inlier_mask is a boolean (N,) array. Returns
        (None, None, None) if fitting fails.
    """
    n_points = points.shape[0]
    if n_points < 3:
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pts_gpu = torch.from_numpy(points).float().to(device)

    best_inlier_count = -1
    best_plane_params = None

    for i in range(0, n_iters, batch_size):
        current_batch_size = min(batch_size, n_iters - i)

        # Sample random triples of points and compute their plane normals
        idx = torch.randint(0, n_points, (current_batch_size, 3), device=device)
        pts_samples = pts_gpu[idx]
        v1 = pts_samples[:, 1] - pts_samples[:, 0]
        v2 = pts_samples[:, 2] - pts_samples[:, 0]
        normals = torch.cross(v1, v2, dim=1)
        norm_mags = torch.norm(normals, dim=1, keepdim=True)

        # Skip degenerate samples (collinear points produce a zero-length normal)
        valid_mask = (norm_mags.squeeze() > 1e-6)
        if not valid_mask.any():
            continue

        normals = normals / norm_mags
        d_vals = -(normals * pts_samples[:, 0]).sum(dim=1, keepdim=True)

        # Count inliers for every plane hypothesis in the batch simultaneously
        batch_normals = normals[valid_mask]
        batch_d = d_vals[valid_mask].T
        dists = torch.abs(pts_gpu @ batch_normals.T + batch_d)
        inlier_counts = (dists < threshold).sum(dim=0)

        batch_best_idx = torch.argmax(inlier_counts)
        batch_best_count = inlier_counts[batch_best_idx].item()
        if batch_best_count > best_inlier_count:
            best_inlier_count = batch_best_count
            best_plane_params = (
                batch_normals[batch_best_idx],
                batch_d[0, batch_best_idx],
            )

    if best_plane_params is None or best_inlier_count < 3:
        return None, None, None

    # Refine normal with SVD on the full inlier set
    best_n, best_d = best_plane_params
    final_inlier_mask = torch.abs(pts_gpu @ best_n + best_d) < threshold
    inlier_pts = pts_gpu[final_inlier_mask]

    centroid = inlier_pts.mean(dim=0)
    H = (inlier_pts - centroid).T @ (inlier_pts - centroid)
    # SVD on the 3×3 covariance matrix — run on CPU to avoid the deprecated MAGMA
    # GPU path; the cost is negligible for a 3×3 matrix.
    final_normal = torch.linalg.svd(H.cpu()).Vh[-1].to(inlier_pts.device)

    return (
        centroid.cpu().numpy(),
        final_normal.cpu().numpy(),
        final_inlier_mask.cpu().numpy(),
    )