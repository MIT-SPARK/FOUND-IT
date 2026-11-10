from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from foundit.places.config import GroundConfig, SubmapTiles
from foundit.places.graph import PlaceGraph
from foundit.places.ground import GroundExtractor, _build_tile_mesh


def _apply_homography(delta: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an (N, 3) point array."""
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    pts_new = (delta @ pts_h.T).T
    return pts_new[:, :3] / pts_new[:, 3:]


class PlaceLoopClosure:
    """Delta-H node movement and cross-edge rebuild after loop closures.

    Shares per-submap caches with GroundExtractor via reference.
    """

    def __init__(self, config: GroundConfig, extractor: GroundExtractor) -> None:
        self.config = config
        self._ext = extractor

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def merge_submap_nodes(
        self,
        submap,
        place_graph: PlaceGraph,
    ):
        sid = str(submap.get_id())
        if sid not in self._ext.submap_ref_homographies:
            return None

        merge_radius = self.config.place_size * self.config.merge_radius_factor
        n_merged = self._merge_overlapping_nodes(sid, place_graph, merge_radius)
        if n_merged > 0:
            print(f"[Places {sid}] loop-closure merge: merged {n_merged} node(s)")

        tile_vertices, tile_faces, internal_edge_points = \
                self._rebuild_submap_mesh_from_graph(sid, place_graph)

        return SubmapTiles(
            submap_id=sid,
            tile_vertices=tile_vertices,
            tile_faces=tile_faces,
            internal_edge_points=internal_edge_points,
            node_count=len(place_graph.get_submap_node_indices(sid)),
        )


    def move_submap_nodes(
        self,
        submap,
        graph_backend,
        place_graph: PlaceGraph,
    ) -> None:
        """Apply the loop-closure delta homography to this submap's node positions.

        Only the graph node positions are moved; the tile mesh is rebuilt from
        those positions later by merge_submap_nodes / rebuild_all_cross_edges.
        No-op if places have not been computed for this submap yet.
        """
        sid = str(submap.get_id())
        if sid not in self._ext.submap_ref_homographies:
            return

        delta = self._compute_delta(sid, graph_backend, submap)
        self._move_nodes(sid, place_graph, delta)

    def rebuild_all_cross_edges(
        self,
        place_graph: PlaceGraph,
    ) -> list[tuple[str, np.ndarray]]:
        """Rebuild every cross-submap edge from scratch using a global KD-tree.

        Runs a global dedup pass first to remove cross-submap duplicate tiles
        that survived the per-submap merge (due to independent grid quantization).

        Returns list of (submap_id, edge_points_array) tuples for the caller to render.
        """
        if len(place_graph) == 0:
            return []

        t0 = time.time()
        place_graph.reset_cross_edges()

        n_deduped, affected_sids = self._global_dedup(place_graph)
        for sid in affected_sids:
            self._rebuild_submap_mesh_from_graph(sid, place_graph)

        all_positions, active_indices = place_graph.get_active_node_data()
        connect_dist = self.config.place_size * self.config.cross_edge_radius_factor
        max_dy = self.config.cross_edge_max_dy
        pairs = cKDTree(all_positions[:, [0, 2]]).query_pairs(connect_dist)

        cross_edges_by_sid: dict[str, list[list[np.ndarray]]] = {}
        n_cross = 0
        n_dy_rejected = 0
        for ai, aj in pairs:
            i, j = active_indices[ai], active_indices[aj]
            sid_i = place_graph.get_submap_id(i)
            sid_j = place_graph.get_submap_id(j)
            if sid_i == sid_j:
                continue
            if not self._ext.may_connect(sid_i, sid_j):
                continue  # across an unaligned session boundary — no cross edge
            if j in place_graph.get_neighbors(i):
                continue  # already linked (e.g. a bridging internal edge) — no duplicate
            if abs(all_positions[ai, 1] - all_positions[aj, 1]) > max_dy:
                n_dy_rejected += 1
                continue
            place_graph.add_cross_edge(i, j)
            cross_edges_by_sid.setdefault(sid_j, []).append(
                [all_positions[ai], all_positions[aj]]
            )
            n_cross += 1

        print(f"[Places] rebuild_all_cross_edges:         {time.time()-t0:.3f}s  "
              f"({n_deduped} deduped, {n_cross} cross edges, "
              f"{n_dy_rejected} dy-rejected across {len(place_graph)} nodes)")

        return [
            (sid, np.array(edges))
            for sid, edges in cross_edges_by_sid.items()
        ]

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _compute_delta(self, sid: str, graph_backend, submap) -> np.ndarray:
        """Compute the delta homography from the cached reference to the current pose."""
        H_old = self._ext.submap_ref_homographies[sid]
        H_new = graph_backend.get_homography(submap.get_id())
        self._ext.submap_ref_homographies[sid] = H_new
        return H_new @ np.linalg.inv(H_old)

    def _move_nodes(
        self, sid: str, place_graph: PlaceGraph, delta: np.ndarray
    ) -> None:
        """Apply delta to every node position in the submap."""
        indices = place_graph.get_submap_node_indices(sid)
        if not indices:
            return
        positions = place_graph.get_positions(indices)
        for idx, new_pos in zip(indices, _apply_homography(delta, positions)):
            place_graph.update_node_position(idx, new_pos)

    def _global_dedup(
        self, place_graph: PlaceGraph
    ) -> tuple[int, set[str]]:
        """Remove cross-submap duplicate tiles via global KDTree + Union-Find.

        Two submaps observing the same area produce grids offset by up to
        place_size * sqrt(2)/2.  Tiles within merge_radius of each other across
        submaps are grouped transitively; the node with highest frame_count
        survives and absorbs the others' vMF statistics.

        Returns (n_removed, affected_submap_ids).
        """
        all_positions, active_indices = place_graph.get_active_node_data()
        if len(all_positions) < 2:
            return 0, set()

        merge_radius = self.config.place_size * self.config.merge_radius_factor
        max_dy = self.config.cross_edge_max_dy
        pairs = cKDTree(all_positions[:, [0, 2]]).query_pairs(merge_radius)

        # Filter to cross-submap pairs on the same floor
        cross_pairs: list[tuple[int, int]] = []
        n_dy_rejected = 0
        for ai, aj in pairs:
            i, j = active_indices[ai], active_indices[aj]
            sid_i, sid_j = place_graph.get_submap_id(i), place_graph.get_submap_id(j)
            if sid_i == sid_j:
                continue
            if not self._ext.may_connect(sid_i, sid_j):
                continue  # across an unaligned session boundary — do not dedup-merge
            if abs(all_positions[ai, 1] - all_positions[aj, 1]) > max_dy:
                n_dy_rejected += 1
                continue
            cross_pairs.append((i, j))
        if n_dy_rejected > 0:
            print(f"[Places] global_dedup: {n_dy_rejected} cross-floor pairs rejected")

        if not cross_pairs:
            return 0, set()

        # Union-Find with path compression
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i, j in cross_pairs:
            union(i, j)

        # Group by root
        involved = set()
        for i, j in cross_pairs:
            involved.add(i)
            involved.add(j)
        groups: dict[int, list[int]] = {}
        for idx in involved:
            groups.setdefault(find(idx), []).append(idx)

        # Merge: survivor = highest frame_count
        n_removed = 0
        affected_sids: set[str] = set()
        for members in groups.values():
            if len(members) <= 1:
                continue
            survivor = max(members, key=lambda i: place_graph.get_frame_count(i))
            for idx in members:
                if idx != survivor:
                    affected_sids.add(place_graph.get_submap_id(idx))
                    # Merge idx into survivor, carrying idx's edges over.
                    place_graph.merge_node(idx, survivor)
                    n_removed += 1

        if n_removed > 0:
            print(f"[Places] global dedup: removed {n_removed} duplicate nodes "
                  f"across {len(affected_sids)} submaps")
        return n_removed, affected_sids

    def _merge_overlapping_nodes(
        self, sid: str, place_graph: PlaceGraph, merge_radius: float
    ) -> int:
        """Merge each node in submap sid into the nearest cross-submap node within
        merge_radius (XZ plane distance).  Returns the number of nodes removed."""
        other_indices = [i for i in place_graph.get_nodes_not_in_submap(sid)
                         if self._ext.may_connect(sid, place_graph.get_submap_id(i))]
        if not other_indices:
            return 0

        own_indices = list(place_graph.get_submap_node_indices(sid))  # snapshot
        if not own_indices:
            return 0

        other_pos = place_graph.get_positions(other_indices)
        tree = cKDTree(other_pos[:, [0, 2]])   # XZ plane

        # Batch query: one call instead of N individual queries
        own_pos = place_graph.get_positions(own_indices)
        all_hits = tree.query_ball_point(own_pos[:, [0, 2]], r=merge_radius)

        max_dy = self.config.cross_edge_max_dy
        n_merged = 0
        n_dy_rejected = 0
        deleted = place_graph._deleted
        for idx, hits, own_y in zip(own_indices, all_hits, own_pos[:, 1]):
            if idx in deleted:   # may have been removed in an earlier iteration
                continue
            if not hits:
                continue
            same_floor_hits = [h for h in hits if abs(other_pos[h, 1] - own_y) <= max_dy]
            if not same_floor_hits:
                n_dy_rejected += 1
                continue
            survivor = other_indices[min(same_floor_hits)]
            # Merge idx into survivor, carrying idx's edges over (see merge_node).
            place_graph.merge_node(idx, survivor)
            n_merged += 1
        if n_dy_rejected > 0:
            print(f"[Places {sid}] _merge_overlapping_nodes: {n_dy_rejected} cross-floor rejected")
        return n_merged

    def _rebuild_submap_mesh_from_graph(
        self, sid: str, place_graph: PlaceGraph
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Rebuild tile mesh and internal edge segments from live graph state.

        Called after node removals so the cached geometry stays consistent.
        Updates the extractor's caches in-place.
        Returns (tile_vertices, tile_faces, internal_edge_points).
        """
        surviving = place_graph.get_submap_node_indices(sid)
        if not surviving or sid not in self._ext.submap_plane_rotations:
            self._ext.submap_tile_verts.pop(sid, None)
            self._ext.submap_tile_faces.pop(sid, None)
            self._ext.submap_internal_edge_pts[sid] = None
            return None, None, None

        centers = place_graph.get_positions(surviving)
        R = self._ext.submap_plane_rotations[sid]
        tile_vertices, tile_faces = _build_tile_mesh(centers, R, self.config.place_size)
        self._ext.submap_tile_verts[sid] = tile_vertices
        self._ext.submap_tile_faces[sid] = tile_faces

        internal_edge_points = self._collect_internal_edge_segments(surviving, place_graph)
        self._ext.submap_internal_edge_pts[sid] = internal_edge_points
        return tile_vertices, tile_faces, internal_edge_points

    @staticmethod
    def _collect_internal_edge_segments(
        node_indices: list[int], place_graph: PlaceGraph
    ) -> Optional[np.ndarray]:
        """Return an (E, 2, 3) array of line segment endpoints for internal edges
        touching the given nodes, or None if there are none.

        Most internal edges are intra-submap, but a merge can bridge two prior
        submaps (see GroundExtractor._build_connectivity), so a neighbour may live
        in another submap and not be in ``node_indices``; look its position up
        directly in that case.
        """
        # Cache positions upfront to avoid repeated list indexing per edge
        pos = {n: place_graph.get_position(n) for n in node_indices}
        edge_segs = []
        visited: set[tuple[int, int]] = set()
        for nidx in node_indices:
            p1 = pos[nidx]
            for neighbor in place_graph.get_internal_neighbors(nidx):
                key = (min(nidx, neighbor), max(nidx, neighbor))
                if key not in visited:
                    visited.add(key)
                    p2 = pos[neighbor] if neighbor in pos else place_graph.get_position(neighbor)
                    edge_segs.append([p1, p2])
        return np.array(edge_segs) if edge_segs else None