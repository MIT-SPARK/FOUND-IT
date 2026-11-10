from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import foundit.slam_utils as utils
from foundit.places.clustering.base import AbstractClusterer, ClusterResult
from foundit.places.config import SemanticClusterConfig
from foundit.places.graph import PlaceGraph


_DEFAULT_COLORS = np.array([
    [ 70, 130, 180],   # bedroom          - steel blue
    [144, 238, 144],   # office cubicle   - light green
    [255, 165,   0],   # kitchen          - orange
    [135, 206, 235],   # bathroom         - sky blue
    [188, 143, 143],   # workshop         - rosy brown
    [255,  99,  71],   # living room      - tomato
    [255, 235,  59],   # hallway          - yellow
    [147, 112, 219],   # dining room      - medium purple
    [169, 169, 169],   # garage           - dark grey
    [210, 180, 140],   # library          - tan
    [127, 255, 212],   # conference room  - aquamarine
    [255,  20, 147],   # gym              - deep pink
    [176, 196, 222],   # laundry room     - light steel blue
    [240, 230, 140],   # closet           - khaki
    [152, 251, 152],   # laboratory       - pale green
    [255, 218, 185],   # stairwell        - peach puff
    [192, 192, 192],   # storage room     - silver
    [255, 160, 122],   # break room       - light salmon
], dtype=np.uint8)


class SemanticClusterer(AbstractClusterer):
    """Zero-shot room clustering using CLIP text/image cosine similarity. CLOSED-SET SO NOT A GOOD GENERAL SOLUTION.

    1. Encode room-type text prompts with the same PE CLIP that produced node embeddings.
    2. Vectorized zero-shot classification: batch matmul of node embeddings x text embeddings.
    3. Majority-vote smoothing over the adjacency graph.
    4. Connected-component grouping per label.
    """

    def __init__(
        self,
        config: SemanticClusterConfig,
        clip_model=None,
        clip_tokenizer=None,
    ) -> None:
        self.config = config
        self._clip_model = clip_model
        self._clip_tokenizer = clip_tokenizer

    def cluster(self, graph: PlaceGraph) -> ClusterResult:
        n_nodes = len(graph)
        labels = self.config.room_labels
        n_labels = len(labels)

        if n_nodes == 0:
            print("[SemanticClusterer] No place nodes to cluster.")
            return _empty_result(labels)

        embeddings = graph.all_node_embeddings
        valid_count = sum(1 for e in embeddings if e is not None)
        if valid_count == 0:
            print("[SemanticClusterer] No semantic embeddings found. "
                  "Run with --run_os to populate CLIP embeddings per node.")
            return _empty_result(labels)

        clip_model, clip_tokenizer = _ensure_models(
            self._clip_model, self._clip_tokenizer, self.config.device
        )
        self._clip_model = clip_model
        self._clip_tokenizer = clip_tokenizer

        # 1. Encode text labels (vectorized)
        print(f"[SemanticClusterer] Encoding {n_labels} room-type labels...")
        text_embs = np.stack([
            utils.compute_text_embeddings(
                clip_model, clip_tokenizer,
                self.config.prompt_template.format(label=label),
                device=self.config.device,
            ).squeeze(0)
            for label in labels
        ], axis=0)  # (L, D)

        # 2. Vectorized nearest-label assignment
        # Stack all valid embeddings into a matrix, batch matmul
        valid_mask = np.array([e is not None for e in embeddings])
        valid_indices = np.where(valid_mask)[0]
        node_emb_matrix = np.stack([embeddings[i] for i in valid_indices], axis=0)  # (K, D)

        sims = node_emb_matrix @ text_embs.T  # (K, L)
        best_labels_valid = np.argmax(sims, axis=1)  # (K,)

        label_indices = np.full(n_nodes, -1, dtype=np.int32)
        label_indices[valid_indices] = best_labels_valid

        # 3. Majority-vote smoothing
        for _ in range(self.config.smooth_iterations):
            new_labels = label_indices.copy()
            for node_idx in range(n_nodes):
                if label_indices[node_idx] < 0:
                    continue
                neighbor_indices = graph.get_neighbors(node_idx)
                votes = [label_indices[node_idx]]
                for nb in neighbor_indices:
                    if label_indices[nb] >= 0:
                        votes.append(int(label_indices[nb]))
                counts = np.bincount(votes, minlength=n_labels)
                new_labels[node_idx] = int(np.argmax(counts))
            label_indices = new_labels

        # 4. Connected-component grouping per label
        cluster_instances: list[tuple[int, list[int]]] = []
        for label_idx in range(n_labels):
            label_nodes = [i for i in range(n_nodes) if label_indices[i] == label_idx]
            if not label_nodes:
                continue

            node_set = set(label_nodes)
            sub = {n: s for s, n in enumerate(label_nodes)}
            rows, cols = [], []
            for n in label_nodes:
                for nb in graph.get_neighbors(n):
                    if nb in node_set:
                        rows.append(sub[n])
                        cols.append(sub[nb])

            if rows:
                data = np.ones(len(rows), dtype=np.int8)
                adj = csr_matrix(
                    (data, (rows, cols)),
                    shape=(len(label_nodes), len(label_nodes)),
                )
                n_comp, comp_labels = connected_components(adj, directed=False)
                for c in range(n_comp):
                    members = [label_nodes[s] for s in np.where(comp_labels == c)[0]]
                    cluster_instances.append((label_idx, members))
            else:
                for n in label_nodes:
                    cluster_instances.append((label_idx, [n]))

        # Summary
        label_counts: dict[str, int] = {}
        for li, nodes in cluster_instances:
            lname = labels[li]
            label_counts[lname] = label_counts.get(lname, 0) + 1

        print(f"[SemanticClusterer] Found {len(cluster_instances)} room instance(s) "
              f"across {len(label_counts)} label type(s):")
        for name, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  {name}: {cnt} region(s)")

        # Build color array: extend _DEFAULT_COLORS if more labels than colors
        color_arr = _DEFAULT_COLORS
        if n_labels > len(color_arr):
            extra = np.random.RandomState(42).randint(0, 256, (n_labels - len(color_arr), 3)).astype(np.uint8)
            color_arr = np.vstack([color_arr, extra])

        return ClusterResult(
            node_labels=label_indices,
            cluster_instances=cluster_instances,
            label_names=list(labels),
            label_colors=color_arr[:n_labels],
        )

    def update_incremental(self, graph: PlaceGraph,
                           new_node_indices: list[int]) -> ClusterResult:
        # Full recluster -- fast enough for typical graph sizes (<10k nodes)
        return self.cluster(graph)


def _empty_result(labels: list[str]) -> ClusterResult:
    return ClusterResult(
        node_labels=np.array([], dtype=np.int32),
        cluster_instances=[],
        label_names=list(labels),
        label_colors=_DEFAULT_COLORS[:len(labels)],
    )


def _ensure_models(clip_model, clip_tokenizer, device: str):
    """Return (clip_model, clip_tokenizer), auto-loading PE CLIP if needed."""
    if clip_model is not None:
        return clip_model, clip_tokenizer

    print("[SemanticClusterer] Loading PE CLIP model for text encoding...")
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms

    model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True).to(device)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    return model, tokenizer
