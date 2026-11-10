"""Factory and registry for query-conditioned clustering implementations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foundit.places.clustering.base import AbstractQueryClusterer, QueryClusterResult
from foundit.places.clustering.propagation import VmfPropagationClusterer
from foundit.places.config import QueryClusterConfig
from foundit.places.graph import PlaceGraph

_REGISTRY: dict[str, type[AbstractQueryClusterer]] = {
    "vmf_propagation": VmfPropagationClusterer,
}


def create_query_clusterer(config: QueryClusterConfig, **kwargs) -> AbstractQueryClusterer:
    """Construct a query clusterer from config.method.

    Keyword arguments are forwarded to the concrete class constructor
    (e.g. clip_model, clip_tokenizer).
    """
    assert config.method in _REGISTRY, (
        f"Unknown query clustering method '{config.method}'. "
        f"Available: {list(_REGISTRY.keys())}"
    )
    cls = _REGISTRY[config.method]
    return cls(config=config, **kwargs)


# Backward-compat alias
QueryClusterer = VmfPropagationClusterer


# ------------------------------------------------------------------ #
# Shared region query logic (used by both agent and interactive loop)
# ------------------------------------------------------------------ #


@dataclass
class RegionInfo:
    label: str
    centroid: list[float]
    extent: list[float]
    num_nodes: int
    confidence: float


@dataclass
class RegionQueryOutput:
    regions: list[RegionInfo]
    vis_result: QueryClusterResult


def query_region(
    clusterer: AbstractQueryClusterer,
    graph: PlaceGraph,
    query: str,
    multi_region: bool = True,
) -> RegionQueryOutput | None:
    """Run a text-conditioned region query and return structured results.

    Returns None if no clusters are found.
    """
    result = clusterer.cluster_region(graph, query)

    if not result.cluster_instances:
        return None

    regions: list[RegionInfo] = []
    for _, members in result.cluster_instances:
        member_pos = np.array([graph.get_position(idx) for idx in members])
        centroid = member_pos.mean(axis=0)
        extent = member_pos.max(axis=0) - member_pos.min(axis=0)
        mean_score = float(np.mean([result.node_scores[idx] for idx in members]))
        regions.append(RegionInfo(
            label=query,
            centroid=[float(c) for c in centroid],
            extent=[float(e) for e in extent],
            num_nodes=len(members),
            confidence=mean_score,
        ))

    regions.sort(key=lambda r: r.confidence, reverse=True)

    if not multi_region:
        regions = regions[:1]

    # Build a filtered QueryClusterResult containing only surviving clusters
    instance_scores = [
        float(np.mean([result.node_scores[idx] for idx in members]))
        for _, members in result.cluster_instances
    ]
    ranked = sorted(range(len(instance_scores)), key=lambda i: instance_scores[i], reverse=True)
    keep = ranked[:len(regions)]
    vis_result = QueryClusterResult(
        node_labels=result.node_labels,
        cluster_instances=[result.cluster_instances[i] for i in keep],
        label_names=result.label_names,
        label_colors=result.label_colors,
        node_scores=result.node_scores,
    )

    return RegionQueryOutput(regions=regions, vis_result=vis_result)
