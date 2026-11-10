from __future__ import annotations

from typing import Optional

from foundit.places.clustering import (
    AbstractClusterer,
    AbstractQueryClusterer,
    ClusterResult,
    FiltrationClusterer,
    QueryClusterer,
    QueryClusterResult,
    RegionInfo,
    RegionQueryOutput,
    SemanticClusterer,
    VmfPropagationClusterer,
    create_query_clusterer,
    query_region,
)
from foundit.places.config import (
    ClusteringMode,
    DistanceAggregation,
    DistanceMetric,
    FiltrationClusterConfig,
    GroundConfig,
    PlacesConfig,
    QueryClusterConfig,
    SemanticClusterConfig,
    SubmapTiles,
    ThresholdMode,
)
from foundit.places.frontier import (
    FrontierClassification,
    classify_frontier_nodes,
    find_near_boundary_nodes,
)
from foundit.places.graph import PlaceGraph
from foundit.places.ground import GroundExtractor
from foundit.places.loop_closure import PlaceLoopClosure
from foundit.places.pathfinding import find_shortest_path

__all__ = [
    # Facade
    "Places",
    # Core types
    "PlaceGraph",
    "GroundExtractor",
    "PlaceLoopClosure",
    "find_shortest_path",
    "classify_frontier_nodes",
    "find_near_boundary_nodes",
    "FrontierClassification",
    # Clustering
    "AbstractClusterer",
    "AbstractQueryClusterer",
    "ClusterResult",
    "QueryClusterResult",
    "FiltrationClusterer",
    "VmfPropagationClusterer",
    "QueryClusterer",
    "RegionInfo",
    "RegionQueryOutput",
    "create_query_clusterer",
    "query_region",
    "SemanticClusterer",
    # Config
    "PlacesConfig",
    "GroundConfig",
    "SemanticClusterConfig",
    "FiltrationClusterConfig",
    "QueryClusterConfig",
    "SubmapTiles",
    "ThresholdMode",
    "ClusteringMode",
    "DistanceMetric",
    "DistanceAggregation",
]


class Places:
    """Facade maintaining the old API shape for solver.py integration.

    Key change from the old monolithic Places class: **no viewer parameter** in
    any method. Returns SubmapTiles / edge data for the caller to render.
    """

    def __init__(self, config: Optional[PlacesConfig] = None) -> None:
        config = config or PlacesConfig()
        self._place_graph = PlaceGraph()
        self._extractor = GroundExtractor(config.ground)
        self._loop_closure = PlaceLoopClosure(config.ground, self._extractor)
        self._config = config

    # ------------------------------------------------------------------ #
    # Public methods (return data, no rendering)
    # ------------------------------------------------------------------ #

    def compute_places(
        self,
        submap,
        graph,
        use_all_points: bool = False,
        can_remove_ground_seg: bool = True,
    ) -> SubmapTiles:
        return self._extractor.extract_tiles(
            submap, graph, self._place_graph, use_all_points, can_remove_ground_seg
        )
    
    def merge_submap_places(self, submap) -> Optional[SubmapTiles]:
        return self._loop_closure.merge_submap_nodes(
            submap, self._place_graph
        )

    def move_submap_places(self, submap, graph):
        return self._loop_closure.move_submap_nodes(
            submap, graph, self._place_graph
        )

    def rebuild_all_cross_edges(self) -> list[tuple[str, ...] | tuple]:
        return self._loop_closure.rebuild_all_cross_edges(self._place_graph)

    def set_session_isolation(self, submap_ids) -> None:
        """Hold ``submap_ids`` (a prior, not-yet-aligned session) apart from the
        current session: no merges or cross-edges are formed across the boundary
        until this is cleared (pass an empty set) once a loop closure aligns them.
        """
        self._extractor.isolated_submap_ids = {str(s) for s in submap_ids}

    def remove_nodes(self, node_indices) -> tuple[list[SubmapTiles], list]:
        """Soft-delete the given graph nodes, then rebuild the geometry they touch.

        For each submap that lost a node the tile mesh + internal edges are rebuilt
        from the surviving nodes, and all cross-submap edges are rebuilt globally
        (the removal can open gaps that change proximity connectivity).

        Returns ``(submap_tiles, cross_edges)`` for the caller to render:
        ``submap_tiles`` is a list of SubmapTiles (one per affected submap) and
        ``cross_edges`` is the list of ``(submap_id, edge_points)`` from
        rebuild_all_cross_edges.
        """
        graph = self._place_graph
        active = set(graph.active_indices)
        to_remove = [i for i in node_indices if i in active]
        if not to_remove:
            return [], []

        affected_sids = {graph.get_submap_id(i) for i in to_remove}
        for i in to_remove:
            graph.remove_node(i)

        tiles: list[SubmapTiles] = []
        for sid in affected_sids:
            tv, tf, iep = self._loop_closure._rebuild_submap_mesh_from_graph(sid, graph)
            tiles.append(SubmapTiles(
                submap_id=sid, tile_vertices=tv, tile_faces=tf,
                internal_edge_points=iep,
                node_count=len(graph.get_submap_node_indices(sid)),
            ))

        cross_edges = self.rebuild_all_cross_edges()
        return tiles, cross_edges

    # ------------------------------------------------------------------ #
    # Properties delegating to internals
    # ------------------------------------------------------------------ #

    @property
    def graph(self) -> PlaceGraph:
        """Direct access for clusterers and pathfinding."""
        return self._place_graph

    @property
    def place_size(self) -> float:
        return self._config.ground.place_size

    @property
    def near_boundary_threshold(self) -> float:
        return self._config.ground.near_boundary_threshold

    @property
    def obstacle_height_limit(self) -> float:
        return self._config.ground.obstacle_height_limit

    @property
    def submap_plane_rotations(self) -> dict:
        return self._extractor.submap_plane_rotations

    @property
    def submap_ref_homographies(self) -> dict:
        return self._extractor.submap_ref_homographies

    # Backward-compat: these delegate to PlaceGraph
    @property
    def all_graph_nodes(self) -> list:
        return self._place_graph.all_graph_nodes

    @property
    def global_adjacency(self) -> dict:
        return self._place_graph.global_adjacency

    @property
    def all_node_embeddings(self) -> list:
        return self._place_graph.all_node_embeddings
