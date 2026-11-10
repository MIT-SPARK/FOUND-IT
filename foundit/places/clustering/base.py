from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from foundit.places.graph import PlaceGraph


@dataclass
class ClusterResult:
    node_labels: np.ndarray            # (N,) int32, -1 = unlabeled
    cluster_instances: list[tuple[int, list[int]]]  # [(label_idx, [node_indices]), ...]
    label_names: list[str]
    label_colors: np.ndarray           # (L, 3) uint8


@dataclass
class QueryClusterResult(ClusterResult):
    node_scores: np.ndarray = None  # (N,) float64, propagated vMF scores


_QUERY_COLOR = np.array([[220, 50, 50]], dtype=np.uint8)  # red fallback


class AbstractClusterer(ABC):

    @abstractmethod
    def cluster(self, graph: PlaceGraph) -> ClusterResult:
        ...

    @abstractmethod
    def update_incremental(self, graph: PlaceGraph,
                           new_node_indices: list[int]) -> ClusterResult:
        ...


class AbstractQueryClusterer(ABC):

    @abstractmethod
    def cluster_region(self, graph: PlaceGraph, query: str) -> QueryClusterResult:
        ...


def empty_query_result(query: str, n: int) -> QueryClusterResult:
    return QueryClusterResult(
        node_labels=np.full(n, -1, dtype=np.int32) if n > 0 else np.array([], dtype=np.int32),
        cluster_instances=[],
        label_names=[query],
        label_colors=_QUERY_COLOR,
        node_scores=np.zeros(n, dtype=np.float64) if n > 0 else np.array([], dtype=np.float64),
    )
