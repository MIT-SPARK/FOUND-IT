"""Graph-filtration clustering
(note Nico: this is pretty close to Hydra's approach)

"""
from __future__ import annotations

import heapq
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import networkx as nx
from networkx.algorithms.community import louvain_communities

from foundit.places.clustering.base import AbstractClusterer, ClusterResult
from foundit.places.config import (
    ClusteringMode,
    DistanceAggregation,
    DistanceMetric,
    FiltrationClusterConfig,
    ThresholdMode,
)
from foundit.places.graph import PlaceGraph


_REGION_PALETTE = np.array([
    [228,  26,  28], [  55, 126, 184], [  77, 175,  74], [152,  78, 163],
    [255, 127,   0], [255, 255,  51], [166,  86,  40], [247, 129, 191],
    [153, 153, 153], [  0, 210, 213], [179, 222, 105], [253, 180,  98],
    [190, 186, 218], [128, 177, 211], [251, 128, 114], [179, 205, 227],
    [204, 235, 197], [222, 203, 228], [254, 217, 166], [255, 255, 204],
], dtype=np.uint8)


class FiltrationClusterer(AbstractClusterer):
    """Persistent-homology graph filtration, ported from Hydra (https://github.com/MIT-SPARK/Hydra).
    
    Algorithm pipeline:
        1. Compute edge distances (embedding similarity; edges without embeddings are skipped)
        2. Compute node distances (aggregate neighbor edge scores)
        3. Graph filtration (union-find persistent homology)
        4. Threshold selection (REPEATED / LONGEST_LIFETIME / PLATEAU)
        5. Component extraction (connected components above threshold)
        6. Refinement (NEIGHBORS / MODULARITY / NONE)
    """

    def __init__(self, config: FiltrationClusterConfig) -> None:
        self.config = config

    def cluster(self, graph: PlaceGraph) -> ClusterResult:
        n = len(graph)
        if n == 0:
            return _empty_result()

        # optioal: PCA whitening of node embeddings
        whitened: Optional[dict[int, np.ndarray]] = None
        if self.config.whiten_embeddings:
            valid_count = sum(1 for i in range(n) if graph.get_embedding(i) is not None)
            if valid_count >= 2:
                whitened = _whiten_embeddings(graph, n)

        # 1. Edge distances (use whitened embeddings if available)
        edge_dists = _compute_edge_distances(graph, self.config.distance_metric, whitened)

        # 2. Node distances
        node_dists = _compute_node_distances(
            graph, edge_dists, n, self.config.distance_aggregation
        )

        # 3. Graph filtration
        barcodes = _graph_filtration(graph, edge_dists, node_dists, n)

        # 4. Threshold selection
        threshold = _select_threshold(
            barcodes,
            self.config.threshold_mode,
            self.config.plateau_ratio,
            self.config.min_dilation_m,
            self.config.max_dilation_m,
        )

        # 5. Component extraction
        raw_labels = _extract_components(
            graph, node_dists, threshold, n,
            self.config.min_component_size,
        )

        # 6. Refinement
        labels = _refine_labels(
            graph, raw_labels, n, self.config.clustering_mode,
            self.config.modularity_gamma, edge_dists,
        )

        return _build_result(labels, n)

    def update_incremental(self, graph: PlaceGraph,
                           new_node_indices: list[int]) -> ClusterResult:
        # v1: full recompute. Graph is small enough (~5k nodes) that union-find + Dijkstra completes in <100ms. Optimization opportunity for future lazy recompute on subgraph deltas.
        return self.cluster(graph)

def _whiten_embeddings(
    graph: PlaceGraph,
    n: int,
) -> dict[int, np.ndarray]:
    """PCA-whiten node embeddings to increase cosine distance dynamic range.

    Returns a mapping from node index to whitened unit-norm embedding.
    Does NOT mutate the graph's stored embeddings.
    """
    valid_indices: list[int] = []
    valid_embs: list[np.ndarray] = []
    for i in range(n):
        emb = graph.get_embedding(i)
        if emb is not None:
            valid_indices.append(i)
            valid_embs.append(emb)

    X = np.stack(valid_embs, axis=0)  # (K, D)
    K, D = X.shape

    # Center
    mean = X.mean(axis=0)
    X_centered = X - mean

    # PCA via covariance eigen-decomposition
    #   when K < D, use gram matrix trick: O(K^2 D) instead of O(D^3)
    if K >= D:
        cov = (X_centered.T @ X_centered) / K
        eigvals, eigvecs = np.linalg.eigh(cov)
    else:
        gram = (X_centered @ X_centered.T) / K
        eigvals, eigvecs_gram = np.linalg.eigh(gram)
        eigvecs = X_centered.T @ eigvecs_gram
        col_norms = np.linalg.norm(eigvecs, axis=0)
        eigvecs = eigvecs / np.maximum(col_norms, 1e-12)

    order = np.argsort(eigvals)[::-1] # descending (eigh returns ascending)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # keep components with non-trivial variance (auto-determines effective rank)
    eps = 1e-5
    max_eigval = eigvals[0]
    assert max_eigval > 0, "All embeddings are identical (zero variance)"
    effective = int(np.sum(eigvals > eps * max_eigval))

    # project onto effective components, scale by 1/sqrt(eigenvalue)
    scale = 1.0 / np.sqrt(eigvals[:effective])
    whitened = (X_centered @ eigvecs[:, :effective]) * scale  # (K, effective)

    # re-normalize to unit sphere
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    whitened = whitened / np.maximum(norms, 1e-12)

    return {node_idx: whitened[pos] for pos, node_idx in enumerate(valid_indices)}


def _compute_edge_distances(
    graph: PlaceGraph,
    metric: DistanceMetric,
    whitened: Optional[dict[int, np.ndarray]] = None,
) -> dict[tuple[int, int], float]:
    """Compute pairwise distances for every edge in the graph."""
    n = len(graph)
    edge_dists: dict[tuple[int, int], float] = {}

    for i in range(n):
        for j in graph.get_neighbors(i):
            key = (min(i, j), max(i, j))
            if key in edge_dists:
                continue

            if whitened is not None:
                emb_i = whitened.get(i)
                emb_j = whitened.get(j)
            else:
                emb_i = graph.get_embedding(i)
                emb_j = graph.get_embedding(j)

            if emb_i is None or emb_j is None:
                continue
            if metric == DistanceMetric.VMF:
                kappa_i = graph.get_concentration(i)
                kappa_j = graph.get_concentration(j)
                edge_dists[key] = _vmf_distance(emb_i, kappa_i, emb_j, kappa_j)
            else:
                edge_dists[key] = _embedding_distance(emb_i, emb_j, metric)

    return edge_dists


def _embedding_distance(a: np.ndarray, b: np.ndarray, metric: DistanceMetric) -> float:
    if metric == DistanceMetric.COSINE:
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))

        cos_dist = 1.0 - dot # cosine distance
        assert cos_dist >= 0, "Dijkstra could have infinite loop with negative weights!!!"
        return cos_dist  
    elif metric == DistanceMetric.L2:
        return float(np.linalg.norm(a - b))
    elif metric == DistanceMetric.L1:
        return float(np.sum(np.abs(a - b)))
    assert False, f"Unknown metric: {metric}"


def _vmf_distance(
    mu_i: np.ndarray, kappa_i: float,
    mu_j: np.ndarray, kappa_j: float,
) -> float:
    """Concentration-weighted cosine distance on the unit sphere.

    d_vmf(i,j) = 1 - ||kappa_i * mu_i + kappa_j * mu_j|| / (kappa_i + kappa_j)

    Properties:
      - Non-negative, in [0, 1] (Dijkstra-safe)
      - = 0 when mu_i = mu_j
      - Reduces to 1 - cos(theta/2) when kappas equal
      - Both kappas near zero => maximally uncertain, merge freely (return 0)
    """
    kappa_sum = kappa_i + kappa_j
    if kappa_sum < 1e-12:
        return 0.0
    combined = kappa_i * mu_i + kappa_j * mu_j
    dist = 1.0 - float(np.linalg.norm(combined)) / kappa_sum
    assert dist > -1e-6, f"vMF distance negative: {dist}"
    return max(dist, 0.0)


def _compute_node_distances(
    graph: PlaceGraph,
    edge_dists: dict[tuple[int, int], float],
    n: int,
    aggregation: DistanceAggregation,
) -> np.ndarray:
    """Aggregate neighbor edge scores per node -> node's filtration attribute."""
    node_dists = np.zeros(n, dtype=np.float64)

    for i in range(n):
        neighbors = graph.get_neighbors(i)
        if not neighbors:
            node_dists[i] = 0.0
            continue

        scores = []
        for j in neighbors:
            key = (min(i, j), max(i, j))
            if key in edge_dists:
                scores.append(edge_dists[key])

        if not scores:
            node_dists[i] = 0.0
            continue

        arr = np.array(scores)
        if aggregation == DistanceAggregation.AVERAGE:
            node_dists[i] = float(np.mean(arr))
        elif aggregation == DistanceAggregation.MAX:
            node_dists[i] = float(np.max(arr))
        elif aggregation == DistanceAggregation.MIN:
            node_dists[i] = float(np.min(arr))
        elif aggregation == DistanceAggregation.MEDIAN:
            node_dists[i] = float(np.median(arr))

    return node_dists

class _UnionFind:
    __slots__ = ('parent', 'rank')

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def _graph_filtration(
    graph: PlaceGraph,
    edge_dists: dict[tuple[int, int], float],
    node_dists: np.ndarray,
    n: int,
) -> list[tuple[float, float]]:
    """Union-find based persistent homology.

    Returns barcodes: list of (birth, death) for each merged component.
    """
    # sorted event list: nodes and edges by their distance attribute
    events: list[tuple[float, int, int, int]] = []  # (dist, type, i, j)
    # type 0 = node, type 1 = edge

    for i in range(n):
        events.append((node_dists[i], 0, i, -1))

    for (i, j), d in edge_dists.items():
        events.append((d, 1, i, j))

    events.sort(key=lambda x: (x[0], x[1]))

    uf = _UnionFind(n)
    born: dict[int, float] = {}  # component root -> birth distance
    barcodes: list[tuple[float, float]] = []
    active = set()  # active node indices

    for dist, etype, i, j in events:
        if etype == 0:
            # Node birth
            active.add(i)
            born[i] = dist
        else:
            # edge event: merge compnents if both endpoints are active
            if i not in active or j not in active:
                continue
            ri, rj = uf.find(i), uf.find(j)
            if ri == rj:
                continue

            # younger component dies (higher birth time)
            birth_ri = born.get(ri, 0.0)
            birth_rj = born.get(rj, 0.0)
            if birth_ri < birth_rj:
                dying, surviving = rj, ri
            else:
                dying, surviving = ri, rj

            barcodes.append((born.get(dying, 0.0), dist))
            uf.union(surviving, dying)
            # The surviving root keeps its birth
            new_root = uf.find(surviving)
            born[new_root] = born.get(surviving, 0.0)

    return barcodes


def _select_threshold(
    barcodes: list[tuple[float, float]],
    mode: ThresholdMode,
    plateau_ratio: float,
    min_val: float,
    max_val: float,
) -> float:
    """Pick threshold from filtration curve."""
    if not barcodes:
        return 0.0

    lifetimes = [(death - birth, birth, death) for birth, death in barcodes]

    if mode == ThresholdMode.LONGEST_LIFETIME:
        lifetimes.sort(key=lambda x: -x[0])
        best = lifetimes[0]
        threshold = (best[1] + best[2]) / 2.0

    elif mode == ThresholdMode.REPEATED:
        # Longest constant-component-count run across all death values
        death_values = sorted(set(d for _, d in barcodes))
        if len(death_values) < 2:
            return death_values[0] if death_values else 0.0

        longest_run_start = death_values[0]
        longest_run_len = 0
        run_start = death_values[0]
        run_len = 1

        for k in range(1, len(death_values)):
            # Count components at this threshold
            # If same count as previous, extend run
            count_prev = sum(1 for b, d in barcodes if d > death_values[k-1])
            count_curr = sum(1 for b, d in barcodes if d > death_values[k])
            if count_curr == count_prev:
                run_len += 1
            else:
                if run_len > longest_run_len:
                    longest_run_len = run_len
                    longest_run_start = run_start
                run_start = death_values[k]
                run_len = 1

        if run_len > longest_run_len:
            longest_run_start = run_start

        threshold = longest_run_start

    elif mode == ThresholdMode.PLATEAU:
        lifetimes.sort(key=lambda x: -x[0])
        max_lifetime = lifetimes[0][0] if lifetimes else 0.0
        cutoff = plateau_ratio * max_lifetime
        # Find broadest run above cutoff
        above = [(lt, b, d) for lt, b, d in lifetimes if lt >= cutoff]
        if above:
            threshold = (above[0][1] + above[0][2]) / 2.0
        else:
            threshold = 0.0
    else:
        assert False, f"Unknown threshold mode: {mode}"

    return np.clip(threshold, min_val, max_val)

def _extract_components(
    graph: PlaceGraph,
    node_dists: np.ndarray,
    threshold: float,
    n: int,
    min_component_size: int,
) -> np.ndarray:
    """Connected components among nodes with distance > threshold."""
    labels = np.full(n, -1, dtype=np.int32)
    active_nodes = np.where(node_dists >= threshold)[0]

    if len(active_nodes) == 0:
        return labels

    active_set = set(active_nodes.tolist())
    idx_map = {node: i for i, node in enumerate(active_nodes)}

    rows, cols = [], []
    for node in active_nodes:
        for nb in graph.get_neighbors(node):
            if nb in active_set:
                rows.append(idx_map[node])
                cols.append(idx_map[nb])

    m = len(active_nodes)
    if rows:
        data = np.ones(len(rows), dtype=np.int8)
        adj = csr_matrix((data, (rows, cols)), shape=(m, m))
        n_comp, comp_labels = connected_components(adj, directed=False)
    else:
        n_comp = m
        comp_labels = np.arange(m)

    # Filter by min_component_size
    valid_labels: dict[int, int] = {}
    next_label = 0
    for c in range(n_comp):
        members = np.where(comp_labels == c)[0]
        if len(members) >= min_component_size:
            valid_labels[c] = next_label
            next_label += 1

    for local_i, node in enumerate(active_nodes):
        c = comp_labels[local_i]
        if c in valid_labels:
            labels[node] = valid_labels[c]

    return labels


def _refine_labels(
    graph: PlaceGraph,
    labels: np.ndarray,
    n: int,
    mode: ClusteringMode,
    modularity_gamma: float,
    edge_dists: dict[tuple[int, int], float],
) -> np.ndarray:
    """Absorb unassigned boundary nodes into nearby clusters."""
    if mode == ClusteringMode.NONE:
        return labels

    unassigned = np.where(labels == -1)[0]
    if len(unassigned) == 0:
        return labels

    if mode == ClusteringMode.NEIGHBORS:
        return _refine_dijkstra(graph, labels, n, edge_dists)
    elif mode == ClusteringMode.MODULARITY:
        return _refine_modularity(graph, labels, n, modularity_gamma, edge_dists)

    return labels


def _refine_dijkstra(
    graph: PlaceGraph,
    labels: np.ndarray,
    n: int,
    edge_dists: dict[tuple[int, int], float],
) -> np.ndarray:
    """Dijkstra expansion from labeled components into unassigned nodes."""
    refined = labels.copy()

    # Min-heap: (distance, node_idx, cluster_label)
    queue: list[tuple[float, int, int]] = []
    distances = np.full(n, np.inf)

    # Seed from all labeled nodes
    for i in range(n):
        if labels[i] >= 0:
            distances[i] = 0.0
            heapq.heappush(queue, (0.0, i, int(labels[i])))

    while queue:
        d, u, lbl = heapq.heappop(queue)
        if d > distances[u]:
            continue

        for v in graph.get_neighbors(u):
            key = (min(u, v), max(u, v))
            if key not in edge_dists:
                continue
            w = edge_dists[key]
            alt = d + w
            if alt < distances[v]:
                distances[v] = alt
                refined[v] = lbl
                heapq.heappush(queue, (alt, v, lbl))

    return refined


def _refine_modularity(
    graph: PlaceGraph,
    labels: np.ndarray,
    n: int,
    gamma: float,
    edge_dists: dict[tuple[int, int], float],
) -> np.ndarray:
    """Louvain community detection via networkx, seeded with existing labels."""

    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for (i, j), d in edge_dists.items():
        G.add_edge(i, j, weight=1.0 / (d + 1e-8))

    # Seed initial partition from existing labels
    communities = louvain_communities(G, resolution=gamma, seed=42)

    refined = labels.copy()
    for label_idx, community in enumerate(communities):
        for node in community:
            if refined[node] == -1:
                # Find most common label among already-labeled community members
                labeled = [int(refined[m]) for m in community if refined[m] >= 0]
                if labeled:
                    counts = np.bincount(labeled)
                    refined[node] = int(np.argmax(counts))
                else:
                    refined[node] = label_idx

    return refined


def _build_result(labels: np.ndarray, n: int) -> ClusterResult:
    unique_labels = sorted(set(labels.tolist()) - {-1})
    n_clusters = len(unique_labels)

    label_names = [f"region_{i}" for i in range(n_clusters)]

    palette = _REGION_PALETTE
    if n_clusters > len(palette):
        extra = np.random.RandomState(42).randint(0, 256, (n_clusters - len(palette), 3)).astype(np.uint8)
        palette = np.vstack([palette, extra])

    label_colors = palette[:n_clusters]

    # Remap labels to contiguous 0..K-1
    remap = {old: new for new, old in enumerate(unique_labels)}
    remapped = np.array([remap.get(int(l), -1) for l in labels], dtype=np.int32)

    # Build cluster instances via connected components per label
    cluster_instances: list[tuple[int, list[int]]] = []
    for new_label in range(n_clusters):
        members = np.where(remapped == new_label)[0].tolist()
        if members:
            cluster_instances.append((new_label, members))

    return ClusterResult(
        node_labels=remapped,
        cluster_instances=cluster_instances,
        label_names=label_names,
        label_colors=label_colors,
    )


def _empty_result() -> ClusterResult:
    return ClusterResult(
        node_labels=np.array([], dtype=np.int32),
        cluster_instances=[],
        label_names=[],
        label_colors=np.empty((0, 3), dtype=np.uint8),
    )
