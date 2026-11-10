from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import foundit.slam_utils as utils
from foundit.places.clustering.base import (
    AbstractQueryClusterer,
    QueryClusterResult,
    _QUERY_COLOR,
    empty_query_result,
)
from foundit.places.config import QueryClusterConfig
from foundit.places.graph import PlaceGraph


def _ensure_models(clip_model, clip_tokenizer, device: str):
    """Return (clip_model, clip_tokenizer), auto-loading PE CLIP if needed."""
    if clip_model is not None:
        return clip_model, clip_tokenizer

    print("[VmfPropagation] Loading PE CLIP model for text encoding...")
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms

    model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True).to(device)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    return model, tokenizer


class VmfPropagationClusterer(AbstractQueryClusterer):
    """Open-vocabulary region extraction via vMF-weighted graph propagation.

    All internal computation uses compact arrays indexed [0, n) where n is the
    number of active (non-deleted) nodes. A graph-to-compact mapping translates
    between the PlaceGraph's sparse indices and the dense working arrays.

    Algorithm:
    1. Compute per-node vMF log-likelihood s_i = kappa_i * dot(mu_i, q).
    2. Early-reject: if max score < median_kappa * early_reject_ratio.
    3. Propagate scores with adaptive alpha mixing: high-kappa nodes anchor
       to their original score, low-kappa nodes diffuse from neighbors.
    4. Fit a 1D two-component GMM (log-space EM) to binarize fg/bg.
    5. Extract connected components from the foreground mask, filter by
       minimum component size.
    """

    def __init__(
        self,
        config: QueryClusterConfig,
        clip_model=None,
        clip_tokenizer=None,
    ) -> None:
        self.config = config
        self._clip_model = clip_model
        self._clip_tokenizer = clip_tokenizer

    def cluster_region(self, graph: PlaceGraph, query: str) -> QueryClusterResult:
        active = graph.active_indices
        n = len(active)

        # Step 0: guards
        if n == 0:
            print(f"[VmfPropagation] Empty graph, nothing to cluster for '{query}'.")
            return empty_query_result(query, 0)

        # Build compact arrays indexed [0, n)
        embeddings = [graph.get_embedding(i) for i in active]
        valid_mask = np.array([e is not None for e in embeddings])
        if not valid_mask.any():
            print(f"[VmfPropagation] No embeddings in graph for '{query}'.")
            return empty_query_result(query, 0)

        graph_to_compact = {gi: ci for ci, gi in enumerate(active)}

        clip_model, clip_tokenizer = _ensure_models(
            self._clip_model, self._clip_tokenizer, self.config.device
        )
        self._clip_model = clip_model
        self._clip_tokenizer = clip_tokenizer

        # Step 1: encode query
        prompt = self.config.prompt_template.format(label=query)
        q_emb = utils.compute_text_embeddings(
            clip_model, clip_tokenizer, prompt, device=self.config.device
        ).squeeze(0)  # (D,)

        # Step 2: vMF affinity scores (compact arrays)
        kappas = np.array([graph.get_concentration(i) for i in active])
        scores = self._compute_vmf_scores(embeddings, valid_mask, q_emb, kappas, n)

        # Step 3: early rejection
        valid_kappas = kappas[valid_mask]
        median_kappa = float(np.median(valid_kappas)) if len(valid_kappas) > 0 else 0.0
        if np.max(scores) < median_kappa * self.config.early_reject_ratio:
            print(f"[VmfPropagation] No significant match for '{query}' (early reject).")
            return empty_query_result(query, 0)

        # Step 4-5: propagate (compact arrays)
        propagated = self._propagate_scores(
            graph, active, graph_to_compact, scores, kappas, n,
        )

        # Step 6: GMM binarization
        foreground_mask = self._fit_gmm_1d(propagated)
        if foreground_mask is None:
            print(f"[VmfPropagation] No separable foreground for '{query}'.")
            return empty_query_result(query, 0)

        # Step 7: connected components + size filter (compact indices)
        cluster_instances_compact = self._extract_foreground_components(
            graph, active, graph_to_compact, foreground_mask,
        )

        if not cluster_instances_compact:
            print(f"[VmfPropagation] All foreground components too small for '{query}'.")
            return empty_query_result(query, 0)

        # Map compact members back to graph indices
        cluster_instances = [
            (label, [active[ci] for ci in compact_members])
            for label, compact_members in cluster_instances_compact
        ]

        # Build sparse output arrays sized to max graph index + 1
        max_idx = max(active) + 1
        node_labels = np.full(max_idx, -1, dtype=np.int32)
        for _, members in cluster_instances:
            for idx in members:
                node_labels[idx] = 0

        node_scores = np.zeros(max_idx, dtype=np.float64)
        for ci, gi in enumerate(active):
            node_scores[gi] = propagated[ci]

        total_nodes = sum(len(m) for _, m in cluster_instances)
        print(f"[VmfPropagation] '{query}': {len(cluster_instances)} region(s), "
              f"{total_nodes} node(s)")

        return QueryClusterResult(
            node_labels=node_labels,
            cluster_instances=cluster_instances,
            label_names=[query],
            label_colors=_QUERY_COLOR,
            node_scores=node_scores,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_vmf_scores(
        embeddings: list,
        valid_mask: np.ndarray,
        q_emb: np.ndarray,
        kappas: np.ndarray,
        n: int,
    ) -> np.ndarray:
        scores = np.zeros(n, dtype=np.float64)
        valid_indices = np.where(valid_mask)[0]
        emb_matrix = np.stack([embeddings[i] for i in valid_indices], axis=0)  # (K, D)
        cosines = emb_matrix @ q_emb  # (K,)
        scores[valid_indices] = kappas[valid_indices] * cosines
        return scores

    def _propagate_scores(
        self,
        graph: PlaceGraph,
        active: list[int],
        graph_to_compact: dict[int, int],
        scores: np.ndarray,
        kappas: np.ndarray,
        n: int,
    ) -> np.ndarray:
        # per-node degree (only counting active neighbors)
        degrees = np.zeros(n, dtype=np.float64)
        for ci, gi in enumerate(active):
            neighbors = graph.get_neighbors(gi)
            degrees[ci] = sum(1 for nb in neighbors if nb in graph_to_compact)

        # Auto-scale lambda from kappa/degree distributions
        pos_kappas = kappas[kappas > 0]
        pos_degrees = degrees[degrees > 0]
        if len(pos_kappas) == 0 or len(pos_degrees) == 0:
            return scores.copy()
        lam = float(np.median(pos_kappas) / np.median(pos_degrees))

        # per-node alpha: confidence-based mixing weight
        neighbor_weight = lam * degrees
        alpha = np.where(
            kappas + neighbor_weight > 0,
            kappas / (kappas + neighbor_weight),
            0.0,
        )

        f = scores.copy()
        cfg = self.config
        for iteration in range(cfg.max_propagation_iters):
            f_new = np.empty_like(f)
            for ci, gi in enumerate(active):
                compact_neighbors = [
                    graph_to_compact[nb]
                    for nb in graph.get_neighbors(gi)
                    if nb in graph_to_compact
                ]
                if compact_neighbors:
                    neighbor_mean = np.mean([f[j] for j in compact_neighbors])
                else:
                    neighbor_mean = 0.0
                f_new[ci] = alpha[ci] * scores[ci] + (1.0 - alpha[ci]) * neighbor_mean
            delta = np.max(np.abs(f_new - f))
            f = f_new
            if delta < cfg.convergence_eps:
                break
        return f

    def _fit_gmm_1d(self, values: np.ndarray) -> np.ndarray | None:
        """Fit a 2-component 1D Gaussian mixture via EM. Returns boolean
        foreground mask, or None if no separable foreground found."""
        x = values.astype(np.float64)
        n = len(x)
        assert n >= 2, "Need at least 2 nodes for GMM"

        # Initialize: split at median
        median = float(np.median(x))
        mu = np.array([np.mean(x[x <= median]) if np.any(x <= median) else median - 1.0,
                        np.mean(x[x > median]) if np.any(x > median) else median + 1.0])
        var = np.array([np.var(x[x <= median]) if np.sum(x <= median) > 1 else 1.0,
                         np.var(x[x > median]) if np.sum(x > median) > 1 else 1.0])
        var = np.maximum(var, 1e-10)
        pi = np.array([0.5, 0.5])

        cfg = self.config
        for _ in range(cfg.gmm_max_iters):
            # E-step: compute responsibilities
            log_resp = np.zeros((n, 2), dtype=np.float64)
            for k in range(2):
                log_resp[:, k] = (np.log(pi[k] + 1e-300)
                                  - 0.5 * np.log(var[k] + 1e-300)
                                  - 0.5 * (x - mu[k]) ** 2 / (var[k] + 1e-300))

            # Numerical stability: subtract max per row
            log_resp -= log_resp.max(axis=1, keepdims=True)
            resp = np.exp(log_resp)
            resp /= resp.sum(axis=1, keepdims=True) + 1e-300

            # M-step
            nk = resp.sum(axis=0) + 1e-300
            pi_new = nk / n
            mu_new = (resp * x[:, None]).sum(axis=0) / nk
            var_new = (resp * (x[:, None] - mu_new) ** 2).sum(axis=0) / nk
            var_new = np.maximum(var_new, 1e-10)

            if np.max(np.abs(mu_new - mu)) < 1e-6:
                mu, var, pi = mu_new, var_new, pi_new
                break
            mu, var, pi = mu_new, var_new, pi_new

        # Foreground = component with higher mean
        fg_idx = int(np.argmax(mu))
        bg_idx = 1 - fg_idx

        if mu[fg_idx] <= mu[bg_idx]:
            return None

        foreground = resp[:, fg_idx] > 0.5
        if not foreground.any():
            return None
        return foreground

    def _extract_foreground_components(
        self,
        graph: PlaceGraph,
        active: list[int],
        graph_to_compact: dict[int, int],
        foreground_mask: np.ndarray,
    ) -> list[tuple[int, list[int]]]:
        """Extract connected components from foreground. Returns compact indices."""
        fg_compact = np.where(foreground_mask)[0]
        if len(fg_compact) == 0:
            return []

        fg_set = set(fg_compact.tolist())
        idx_to_sub = {ci: s for s, ci in enumerate(fg_compact)}
        rows, cols = [], []
        for ci in fg_compact:
            gi = active[ci]
            for nb in graph.get_neighbors(gi):
                nb_compact = graph_to_compact.get(nb)
                if nb_compact is not None and nb_compact in fg_set:
                    rows.append(idx_to_sub[ci])
                    cols.append(idx_to_sub[nb_compact])

        m = len(fg_compact)
        if rows:
            data = np.ones(len(rows), dtype=np.int8)
            adj = csr_matrix((data, (rows, cols)), shape=(m, m))
            n_comp, comp_labels = connected_components(adj, directed=False)
        else:
            n_comp = m
            comp_labels = np.arange(m)

        result: list[tuple[int, list[int]]] = []
        for c in range(n_comp):
            members = [int(fg_compact[s]) for s in np.where(comp_labels == c)[0]]
            if len(members) >= self.config.min_component_size:
                result.append((0, members))  # label_idx=0, the query label
        return result
