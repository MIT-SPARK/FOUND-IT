from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# enums

class ThresholdMode(Enum):
    REPEATED = "repeated"
    LONGEST_LIFETIME = "longest_lifetime"
    PLATEAU = "plateau"


class ClusteringMode(Enum):
    MODULARITY = "modularity"
    NEIGHBORS = "neighbors"
    NONE = "none"


class DistanceMetric(Enum):
    COSINE = "cosine"
    L2 = "l2"
    L1 = "l1"
    VMF = "vmf"


class DistanceAggregation(Enum):
    MAX = "max"
    MIN = "min"
    AVERAGE = "average"
    MEDIAN = "median"


# config

@dataclass
class GroundConfig:
    place_size: float = 0.15
    obstacle_height_limit: float = 1.5
    ransac_n_iters: int = 2000
    ransac_threshold: float = 0.05
    ransac_batch_size: int = 100
    cross_edge_radius_factor: float = 1.5
    cross_edge_max_dy: float = 0.25
    merge_radius_factor: float = 0.7  # merge_radius = place_size * 0.7 = 10.5cm
    min_subcell_coverage: int = 4  # out of 4 sub-cells; 3 relaxes for sparse areas
    near_boundary_threshold: float = 0.40  # metres; interior nodes this close to a boundary node are flagged near-boundary


@dataclass
class SemanticClusterConfig:
    room_labels: list[str] = field(default_factory=lambda: [
        "bedroom", "office cubicle", "kitchen", "bathroom", "workshop",
        "living room", "hallway", "dining room", "garage", "library",
        "conference room", "gym", "laundry room", "closet", "laboratory",
        "stairwell", "storage room", "break room",
    ])
    smooth_iterations: int = 2
    prompt_template: str = "a photo of a {label}"
    device: str = "cuda"


@dataclass
class FiltrationClusterConfig:
    min_dilation_m: float = 0.0
    max_dilation_m: float = 5.0
    min_component_size: int = 20
    threshold_mode: ThresholdMode = ThresholdMode.PLATEAU
    plateau_ratio: float = 0.15
    clustering_mode: ClusteringMode = ClusteringMode.NEIGHBORS
    modularity_gamma: float = 1.0
    distance_metric: DistanceMetric = DistanceMetric.VMF
    distance_aggregation: DistanceAggregation = DistanceAggregation.AVERAGE
    whiten_embeddings: bool = True
    vmf_kappa_max: float = 500.0


@dataclass
class QueryClusterConfig:
    method: str = "vmf_propagation"
    prompt_template: str = "a photo of a {label}"
    min_component_size: int = 20
    max_propagation_iters: int = 20
    convergence_eps: float = 1e-4
    early_reject_ratio: float = 0.05
    gmm_max_iters: int = 50
    device: str = "cuda"


@dataclass
class PlacesConfig:
    ground: GroundConfig = field(default_factory=GroundConfig)


# return type 

@dataclass
class SubmapTiles:
    submap_id: str
    tile_vertices: Optional[np.ndarray] = None   # (N*4, 3) quad vertices
    tile_faces: Optional[np.ndarray] = None       # (M, 3) triangle faces
    internal_edge_points: Optional[np.ndarray] = None  # (E, 2, 3) line segment endpoints
    cross_edge_points: Optional[np.ndarray] = None     # (C, 2, 3) line segment endpoints
    node_count: int = 0
