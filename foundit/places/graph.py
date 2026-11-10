from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np


def estimate_vmf_kappa(r_bar: float, d: int, k: int, kappa_max: float = 500.0) -> float:
    """MLE of vMF concentration parameter (Banerjee et al. 2005).

    Args:
        r_bar: Mean resultant length (norm of the averaged unit vectors).
        d: Embedding dimensionality.
        k: Number of contributing frames.
        kappa_max: Upper clamp for numerical stability.
    """
    if k <= 1:
        return kappa_max
    denom = 1.0 - r_bar * r_bar
    if denom < 1e-12:
        return kappa_max
    kappa = r_bar * (d - r_bar * r_bar) / denom
    return min(float(kappa), kappa_max)


class PlaceGraph:
    """Pure adjacency data structure for place-graph nodes.

    No geometry computation, no visualization. Each node stores a 3D position,
    the submap it belongs to, and an optional embedding vector.
    """

    def __init__(self) -> None:
        self._positions: list[np.ndarray] = []
        self._submap_ids: list[str] = []
        self._embeddings: list[Optional[np.ndarray]] = []
        self._frame_counts: list[int] = []
        self._concentrations: list[float] = []
        self._resultant_lengths: list[float] = []
        self._global_adjacency: dict[int, list[int]] = {}
        self._internal_adjacency: dict[int, list[int]] = {}
        self._submap_node_indices: dict[str, list[int]] = {}
        self._deleted: set[int] = set()

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def add_node(self, position: np.ndarray, submap_id: str,
                 embedding: Optional[np.ndarray] = None,
                 frame_count: int = 0, concentration: float = 0.0,
                 resultant_length: float = 0.0) -> int:
        idx = len(self._positions)
        self._positions.append(position)
        self._submap_ids.append(submap_id)
        self._embeddings.append(embedding)
        self._frame_counts.append(frame_count)
        self._concentrations.append(concentration)
        self._resultant_lengths.append(resultant_length)
        self._global_adjacency[idx] = []
        self._internal_adjacency[idx] = []
        self._submap_node_indices.setdefault(submap_id, []).append(idx)
        return idx

    def add_internal_edge(self, i: int, j: int) -> None:
        self._global_adjacency[i].append(j)
        self._global_adjacency[j].append(i)
        self._internal_adjacency[i].append(j)
        self._internal_adjacency[j].append(i)

    def add_cross_edge(self, i: int, j: int) -> None:
        self._global_adjacency[i].append(j)
        self._global_adjacency[j].append(i)

    # ------------------------------------------------------------------ #
    # Read accessors
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._positions) - len(self._deleted)

    def get_position(self, idx: int) -> np.ndarray:
        return self._positions[idx]

    def get_positions_array(self) -> np.ndarray:
        """Return positions of all active (non-deleted) nodes.

        NOTE: array index does NOT equal graph node index when deletions exist.
        Use get_active_node_data() when the graph-index mapping is needed.
        """
        positions, _ = self.get_active_node_data()
        return positions

    def get_positions(self, indices: list[int]) -> np.ndarray:
        """Return an (N, 3) array of positions for the given node indices."""
        if len(indices) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.array([self._positions[i] for i in indices])

    def get_active_node_data(self) -> tuple[np.ndarray, list[int]]:
        """Return (positions_array, graph_indices) for all non-deleted nodes.

        array_index  →  graph_indices[array_index]  is the guaranteed mapping.
        """
        active = [i for i in range(len(self._positions)) if i not in self._deleted]
        if not active:
            return np.empty((0, 3), dtype=np.float64), []
        return np.array([self._positions[i] for i in active]), active

    def get_embedding(self, idx: int) -> Optional[np.ndarray]:
        return self._embeddings[idx]

    def get_frame_count(self, idx: int) -> int:
        return self._frame_counts[idx]

    def get_concentration(self, idx: int) -> float:
        return self._concentrations[idx]

    def get_resultant_length(self, idx: int) -> float:
        return self._resultant_lengths[idx]

    def get_submap_id(self, idx: int) -> str:
        return self._submap_ids[idx]

    def get_neighbors(self, idx: int) -> list[int]:
        return self._global_adjacency.get(idx, [])

    def get_internal_neighbors(self, idx: int) -> list[int]:
        return self._internal_adjacency.get(idx, [])

    def get_submap_node_indices(self, submap_id: str) -> list[int]:
        return self._submap_node_indices.get(submap_id, [])

    def get_nodes_not_in_submap(self, submap_id: str) -> list[int]:
        # Iterate per-submap index lists (which exclude deleted nodes)
        result = []
        for sid, indices in self._submap_node_indices.items():
            if sid != submap_id:
                result.extend(indices)
        return result

    def get_all_submap_ids(self) -> list[str]:
        return list(self._submap_node_indices.keys())

    @property
    def active_indices(self) -> list[int]:
        """Graph indices of all non-deleted nodes."""
        return [i for i in range(len(self._positions)) if i not in self._deleted]

    @property
    def num_connected_components(self) -> int:
        g = nx.Graph()
        g.add_nodes_from(self._global_adjacency.keys())
        for node, neighbors in self._global_adjacency.items():
            for neighbor in neighbors:
                g.add_edge(node, neighbor)
        return nx.number_connected_components(g)

    # ------------------------------------------------------------------ #
    # Mutation helpers (loop closure)
    # ------------------------------------------------------------------ #

    def update_node_position(self, idx: int, new_pos: np.ndarray) -> None:
        self._positions[idx] = new_pos

    def merge_embedding(self, idx: int,
                        new_emb: Optional[np.ndarray],
                        new_frame_count: int,
                        new_resultant_length: float = 0.0) -> None:
        """Merge a new observation into an existing node using vMF sufficient statistics.

        Recovers the original vector sums via mu * r_bar * fc, combines them,
        and re-estimates (mu, r_bar, kappa) from the combined sum.
        """
        if new_emb is not None:
            old_emb = self._embeddings[idx]
            old_fc = self._frame_counts[idx]
            old_rbar = self._resultant_lengths[idx]

            if old_emb is not None and old_fc > 0:
                sum_old = old_emb * old_rbar * old_fc
                sum_new = new_emb * new_resultant_length * new_frame_count
                combined = sum_old + sum_new
                norm = float(np.linalg.norm(combined))
                total_fc = old_fc + new_frame_count

                r_bar = norm / total_fc if total_fc > 0 else 0.0
                self._embeddings[idx] = combined / norm if norm > 1e-8 else combined
                self._resultant_lengths[idx] = r_bar
                self._concentrations[idx] = estimate_vmf_kappa(
                    r_bar, len(combined), total_fc
                )
            else:
                self._embeddings[idx] = new_emb
                self._resultant_lengths[idx] = new_resultant_length
        self._frame_counts[idx] += new_frame_count

    def remove_node(self, idx: int) -> None:
        """Remove a node from the graph (soft-delete).

        Cleans up all adjacency references to the node and removes it from its
        submap's index list.  The underlying position/embedding slots are kept
        so that existing integer indices into _positions remain stable.
        """
        # Remove idx from every neighbour's adjacency lists
        for neighbor in list(self._global_adjacency.get(idx, [])):
            nbr = self._global_adjacency.get(neighbor)
            if nbr is not None:
                try:
                    nbr.remove(idx)
                except ValueError:
                    pass
        for neighbor in list(self._internal_adjacency.get(idx, [])):
            nbr = self._internal_adjacency.get(neighbor)
            if nbr is not None:
                try:
                    nbr.remove(idx)
                except ValueError:
                    pass
        self._global_adjacency.pop(idx, None)
        self._internal_adjacency.pop(idx, None)

        # Remove from submap index
        sid = self._submap_ids[idx]
        idx_list = self._submap_node_indices.get(sid, [])
        try:
            idx_list.remove(idx)
        except ValueError:
            pass

        self._deleted.add(idx)

    def _connect(self, i: int, j: int, internal: bool) -> None:
        """Add an i–j edge with no duplicate global entry.

        internal=True also records it in the internal adjacency (so it survives
        reset_cross_edges); an existing cross-only edge is upgraded to internal.
        """
        if j not in self._global_adjacency[i]:
            self._global_adjacency[i].append(j)
            self._global_adjacency[j].append(i)
        if internal and j not in self._internal_adjacency[i]:
            self._internal_adjacency[i].append(j)
            self._internal_adjacency[j].append(i)

    def merge_node(self, source_idx: int, target_idx: int) -> None:
        """Absorb ``source_idx`` into ``target_idx``, then soft-delete the source.

        Folds the source's vMF sufficient statistics into the target, then moves
        the source's edges onto the target — preserving each edge's internal vs
        cross type and skipping self-loops, duplicates, and already-deleted
        neighbours — before removing the source.  Transferring the edges is what
        keeps the merged node's connectivity intact; a bare merge_embedding +
        remove_node would silently drop it.

        No-op when source == target or the source is already deleted.
        """
        if source_idx == target_idx or source_idx in self._deleted:
            return

        self.merge_embedding(
            target_idx,
            self._embeddings[source_idx],
            self._frame_counts[source_idx],
            new_resultant_length=self._resultant_lengths[source_idx],
        )

        internal_nbrs = set(self._internal_adjacency.get(source_idx, []))
        for nbr in set(self._global_adjacency.get(source_idx, [])):
            if nbr in (source_idx, target_idx) or nbr in self._deleted:
                continue  # skip self-loops, the merge partner, and dead neighbours
            self._connect(target_idx, nbr, internal=nbr in internal_nbrs)

        self.remove_node(source_idx)

    def reset_cross_edges(self) -> None:
        """Restore global adjacency to internal-only edges."""
        for idx in self._global_adjacency:
            self._global_adjacency[idx] = list(self._internal_adjacency.get(idx, []))

    # ------------------------------------------------------------------ #
    # Backward-compat properties
    # ------------------------------------------------------------------ #

    @property
    def all_graph_nodes(self) -> list[tuple[np.ndarray, str]]:
        return list(zip(self._positions, self._submap_ids))

    @property
    def global_adjacency(self) -> dict[int, list[int]]:
        return self._global_adjacency

    @property
    def all_node_embeddings(self) -> list[Optional[np.ndarray]]:
        return self._embeddings
