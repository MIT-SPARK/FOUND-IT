from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from foundit.places.graph import PlaceGraph

@dataclass(frozen=True)
class FrontierClassification:
    """Immutable classification of place-graph nodes relative to the map boundary.

    Each field holds graph node indices, and the three sets are mutually
    disjoint.  Any active node absent from all three is an ordinary interior
    node.

        blocked        boundary (frontier) nodes with a wall/obstacle just beyond.
        unexplored     boundary (frontier) nodes opening onto unseen space.
        near_boundary  interior nodes within GroundConfig.near_boundary_threshold
                       of any boundary node.  Boundary nodes are excluded here —
                       they already carry a blocked/unexplored label.
    """

    blocked: frozenset[int]
    unexplored: frozenset[int]
    near_boundary: frozenset[int]

    @property
    def boundary(self) -> frozenset[int]:
        """Graph indices of all boundary (frontier) nodes: blocked ∪ unexplored."""
        return self.blocked | self.unexplored

    @property
    def keep_out(self) -> frozenset[int]:
        """Graph indices a path should avoid: boundary ∪ near-boundary nodes."""
        return self.blocked | self.unexplored | self.near_boundary


# TODO Dominic, we already make a plane when building the places. We probably can avoid fitting a plane again here.

def _debug_visualize_non_floor(
    world_points: np.ndarray,
    non_floor_pts: np.ndarray,
    frontier_pos: np.ndarray,
) -> None:
    import open3d as o3d

    # Force independent float64 copies — ascontiguousarray may return a view
    # when the source is already float64/contiguous, causing Open3D to read
    # stale/zeroed memory if the source is freed before rendering completes.
    wp  = world_points.astype(np.float64,  copy=True)
    nfp = non_floor_pts.astype(np.float64, copy=True)
    fp  = frontier_pos.astype(np.float64,  copy=True)

    print(f"[Frontier debug] {len(nfp)} non-floor pts / {len(wp)} total pts / {len(fp)} frontier nodes")

    geoms = []

    # All world points — grey
    if len(wp) > 0:
        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(wp)
        pcd_all.paint_uniform_color([0.6, 0.6, 0.6])
        geoms.append(pcd_all)

    # Non-floor points — red # TODO Dominic, sometimes the nfp do not show up because they are colored over by the grey wp points
    if len(nfp) > 0:
        print(f"[Frontier debug] Non-floor points sample: {nfp[:5]} ...")
        pcd_nf = o3d.geometry.PointCloud()
        pcd_nf.points = o3d.utility.Vector3dVector(nfp)
        pcd_nf.paint_uniform_color([1.0, 0.0, 0.0])
        geoms.append(pcd_nf)

    # Frontier tile centres — blue spheres
    for pos in fp:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(pos)
        sphere.paint_uniform_color([0.0, 0.4, 1.0])
        geoms.append(sphere)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geoms.append(axes)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Frontier debug: non-floor pts (red) | all pts (grey) | frontier nodes (blue)",
    )


def _has_uncovered_side(
    node_position: np.ndarray,
    neighbor_positions: np.ndarray,
    plane_xy: np.ndarray,
    max_gap_deg: float,
) -> bool:
    """Return True if the node's edges leave a side of the tile uncovered.

    Edges are projected onto the floor plane (spanned by the two columns of
    ``plane_xy``) and sorted by bearing.  If the largest angular gap between
    consecutive edges exceeds ``max_gap_deg``, one side of the tile has no
    outgoing edge — the node sits on an opening and is therefore a frontier.
    With the default 90° a node is "surrounded" only when it has at least one
    edge into every quadrant (i.e. one per side).

    Projecting onto the actual floor plane (rather than world XZ) keeps the
    bearings undistorted when the camera — and hence the world frame — is tilted.

    A node with fewer than two usable edges always has an uncovered side.

    Args:
        node_position:      (3,) world position of the node.
        neighbor_positions: (K, 3) world positions of its graph neighbours.
        plane_xy:           (3, 2) orthonormal basis spanning the floor plane;
                            edge vectors are projected onto it.
        max_gap_deg:        Largest angular gap (degrees) still counted as covered.
    """
    if len(neighbor_positions) == 0:
        return True

    deltas = (neighbor_positions - node_position) @ plane_xy   # (K, 2) in floor plane
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    deltas = deltas[lengths > 1e-6]           # drop coincident neighbours
    if len(deltas) < 2:
        return True

    angles = np.sort(np.arctan2(deltas[:, 1], deltas[:, 0]))
    gaps = np.diff(angles)
    wrap_gap = 2.0 * np.pi - (angles[-1] - angles[0])   # across the -pi/pi seam
    max_gap = max(gaps.max(), wrap_gap)
    return bool(max_gap > np.deg2rad(max_gap_deg) + 1e-6)   # tolerate exact 90° as covered


# Projects a world point (x, y, z) onto (x, z) — the world XZ plane.  Used as the
# floor basis when no submap plane rotations are available, reproducing the
# original (tilt-unaware) behaviour.
_WORLD_XZ_BASIS = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])


def _floor_frame(submap_plane_rotations: "dict | None") -> "np.ndarray | None":
    """Build a single world→floor rotation from the per-submap plane fits.

    Every submap's plane frame (``R_local_to_world`` in ``submap_plane_rotations``)
    stores the fitted floor normal as its third column.  All submaps fit the same
    real floor in the shared world frame, so these normals agree up to sign; we
    flip them to a common hemisphere and average for robustness against a single
    bad fit, then build an orthonormal in-plane basis.  Planar distances and
    angular gaps are invariant to the arbitrary in-plane orientation, so any
    consistent basis is equivalent — one averaged frame suffices for every node.

    Returns a (3, 3) matrix ``[x_axis | y_axis | normal]`` with the normal
    oriented "up" (toward -Y in VGGT convention), or None when no rotations are
    available (the caller then falls back to world XZ).
    """
    if not submap_plane_rotations:
        return None

    up = np.array([0.0, -1.0, 0.0])   # -Y is "up" in VGGT camera convention
    normals = []
    for R in submap_plane_rotations.values():
        n = np.asarray(R, dtype=np.float64)[:, 2]
        normals.append(-n if n @ up < 0.0 else n)   # collapse to a common hemisphere
    normal = np.mean(normals, axis=0)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return None
    normal = normal / norm

    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(normal, ref)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)   # orthonormal since normal, x_axis are unit & orthogonal
    return np.column_stack([x_axis, y_axis, normal])


def classify_frontier_nodes(
    place_graph: PlaceGraph,
    world_points: np.ndarray,
    place_size: float,
    *,
    min_neighbors: int = 5,
    max_edge_gap_deg: float = 100.0,
    known_interior: frozenset[int] = frozenset(),
    submap_plane_rotations: "dict | None" = None,
    blocked_radius: float | None = None,
    wall_height_threshold: float = 0.15, # TODO Dominic we need to consider that we raise the tiles above the floor
    min_num_points_for_blocked: int = 5, # TODO Dominic this needs to consider whether or not we sparsify the 3D points
) -> tuple[list[int], list[int]]:
    """Classify place-graph leaf (frontier) nodes as blocked or unexplored.

    A node is a *frontier* if it is not fully surrounded, i.e. either:

    * it has fewer than ``min_neighbors`` edges in the place graph, or
    * its edges leave a side of the tile uncovered — the largest angular gap
      between consecutive edges (in the floor plane) exceeds ``max_edge_gap_deg``.
      This catches nodes that have many edges bunched on one side yet open onto
      unexplored space on another.

    All horizontal reasoning (edge bearings and the blocked-radius search) is
    done in the true floor plane derived from ``submap_plane_rotations``, so a
    tilted camera — which tilts the world XZ plane — does not distort it.  When
    no rotations are supplied the code falls back to world XZ and a PCA floor
    normal (the original, tilt-unaware behaviour).

    A node listed in ``known_interior`` is never a frontier, regardless of the
    geometric tests.  This latches the interior state: a caller that feeds back
    the nodes it has already seen as non-frontier guarantees that once a node is
    classified interior it stays interior (frontier -> interior is one-way).

    Among frontiers:

    * **blocked** – SLAM sparse points that are above the floor (walls /
      obstacles) exist within ``blocked_radius`` in the floor plane.
    * **unexplored** – no such non-floor points nearby; open space lies
      beyond this tile.

    Args:
        place_graph:           The active place graph.
        world_points:          (N, 3) all SLAM sparse points in world space.
        place_size:            Tile side length in metres (from GroundConfig).
        min_neighbors:         Nodes with fewer graph edges than this are
                               considered frontier.
        max_edge_gap_deg:      A node is surrounded only if no angular gap between
                               its edges exceeds this (default 90° = one edge per
                               side); a larger gap marks it a frontier.
        known_interior:        Graph node indices already latched as interior;
                               they are excluded from the frontier set.
        submap_plane_rotations: {submap_id: R_local_to_world} plane fits used to
                               build the floor frame for horizontal projections.
                               None → project onto world XZ instead.
        blocked_radius:        Floor-plane search radius for wall points.
                               Defaults to 2 × place_size.
        wall_height_threshold: Minimum signed distance above the floor plane for
                               a point to be counted as a non-floor (wall) point.

    Returns:
        (blocked_indices, unexplored_indices) – lists of graph node indices.
    """
    if blocked_radius is None:
        blocked_radius = place_size * 2.0   # walls can sit a tile or two beyond the last floor tile

    positions, indices = place_graph.get_active_node_data()
    if len(positions) == 0:
        return [], []

    # Floor frame for all horizontal (in-plane) reasoning.  Falls back to world
    # XZ when no submap plane fits are available.
    floor_frame = _floor_frame(submap_plane_rotations)
    plane_xy = floor_frame[:, :2] if floor_frame is not None else _WORLD_XZ_BASIS

    # ------------------------------------------------------------------ #
    # 1. Detect frontier nodes: too few edges, or a side left uncovered.
    #    Nodes already latched as interior can never become frontiers again.
    # ------------------------------------------------------------------ #
    frontier_arr_idx = []
    for i, node_idx in enumerate(indices):
        if node_idx in known_interior:
            continue   # latched interior — never a frontier again
        neighbors = place_graph.get_neighbors(node_idx)
        if len(neighbors) < min_neighbors:
            frontier_arr_idx.append(i)   # too sparse to be surrounded
            continue
        neighbor_positions = np.array([place_graph.get_position(n) for n in neighbors])
        if _has_uncovered_side(positions[i], neighbor_positions, plane_xy, max_edge_gap_deg):
            frontier_arr_idx.append(i)
    frontier_node_idx = [indices[i] for i in frontier_arr_idx]

    if not frontier_node_idx:
        return [], []

    frontier_pos = positions[frontier_arr_idx]  # (F, 3)

    # ------------------------------------------------------------------ #
    # 2. Classify: blocked (wall nearby) vs unexplored (open space).
    # ------------------------------------------------------------------ #
    if world_points is not None and len(world_points) > 0:
        # Reference floor point (frontier nodes all lie on the floor).
        centroid = frontier_pos.mean(axis=0)
        if floor_frame is not None:
            normal = floor_frame[:, 2]   # calibrated floor normal (oriented up)
        else:
            # Fallback: estimate the floor normal via PCA on the frontier nodes.
            _, _, Vt = np.linalg.svd(frontier_pos - centroid, full_matrices=False)
            normal = Vt[-1]  # smallest-variance direction = plane normal
            # Orient normal to point upward (negative Y in viser's Y-down convention)
            if normal[1] > 0:
                normal = -normal
        # Signed distance of each world point above the floor plane.
        # Positive = above the floor (wall / obstacle side).
        dists = (world_points - centroid) @ normal
        non_floor_mask = dists > wall_height_threshold
        non_floor_pts = world_points[non_floor_mask]

        # _debug_visualize_non_floor(world_points, non_floor_pts, frontier_pos)

        if len(non_floor_pts) > 0:
            # Project into the floor plane so the radius search measures true
            # in-plane distance regardless of world tilt.
            nf_tree = cKDTree(non_floor_pts @ plane_xy)
            hit_counts = nf_tree.query_ball_point(
                frontier_pos @ plane_xy, r=blocked_radius, return_length=True
            )
            blocked_mask = hit_counts > min_num_points_for_blocked
        else:
            blocked_mask = np.zeros(len(frontier_node_idx), dtype=bool)
    else:
        blocked_mask = np.zeros(len(frontier_node_idx), dtype=bool)

    blocked = [frontier_node_idx[i] for i in range(len(frontier_node_idx)) if blocked_mask[i]]
    unexplored = [frontier_node_idx[i] for i in range(len(frontier_node_idx)) if not blocked_mask[i]]
    return blocked, unexplored


def find_obstacle_blocked_nodes(
    place_graph: PlaceGraph,
    world_points: np.ndarray,
    place_size: float,
    *,
    submap_plane_rotations: "dict | None" = None,
    obstacle_min_height: float = 0.20,
    obstacle_max_height: float = 1.5,
    radius: float | None = None,
    min_points: int = 5,
) -> list[int]:
    """Return active place-node indices that now have an obstacle sitting on them.

    Plan-time, whole-graph analogue of ``GroundExtractor._find_blocked_tiles``:
    every world point whose height above the floor plane falls in the obstacle
    band ``[obstacle_min_height, obstacle_max_height]`` is treated as an obstacle,
    and any active node with at least ``min_points`` such points within ``radius``
    in the floor plane is flagged blocked.

    Unlike ``classify_frontier_nodes`` — which only inspects frontier (leaf)
    nodes — this checks **every** active node, so an obstacle that appears on a
    previously-clear INTERIOR tile (e.g. an object that was moved into the path)
    is caught.  Horizontal reasoning is done in the true floor plane derived from
    ``submap_plane_rotations`` (falling back to world XZ), matching the frontier
    pass.

    Args:
        place_graph:            The active place graph.
        world_points:           (N, 3) all SLAM points in world space.
        place_size:             Tile side length in metres (from GroundConfig).
        submap_plane_rotations: {submap_id: R_local_to_world} plane fits for the
                                floor frame; None → world XZ.
        obstacle_min_height:    Min signed height above the floor to count as an
                                obstacle (matches ground's 0.10 m; ignores floor).
        obstacle_max_height:    Max height (ignore the ceiling / tall structure
                                above head height); GroundConfig.obstacle_height_limit.
        radius:                 Floor-plane search radius; defaults to a tile
                                footprint (place_size / 2), mirroring ground's
                                point-in-tile binning without bleeding into
                                neighbouring tiles.
        min_points:             Points-in-radius needed to flag a node blocked.

    Returns:
        List of blocked active graph node indices.
    """
    positions, indices = place_graph.get_active_node_data()
    if len(positions) == 0 or world_points is None or len(world_points) == 0:
        return []
    if radius is None:
        radius = place_size / 2.0

    floor_frame = _floor_frame(submap_plane_rotations)
    plane_xy = floor_frame[:, :2] if floor_frame is not None else _WORLD_XZ_BASIS
    centroid = positions.mean(axis=0)   # reference floor point (nodes lie on the floor)
    if floor_frame is not None:
        normal = floor_frame[:, 2]      # calibrated floor normal (oriented up)
    else:
        # Fallback: PCA floor normal from the node positions (as in the frontier pass).
        _, _, Vt = np.linalg.svd(positions - centroid, full_matrices=False)
        normal = Vt[-1]
        if normal[1] > 0:               # orient "up" (−Y in viser's Y-down convention)
            normal = -normal

    # Keep only points within the obstacle height band above the floor.
    heights = (world_points - centroid) @ normal
    obstacle_pts = world_points[(heights > obstacle_min_height) & (heights < obstacle_max_height)]
    if len(obstacle_pts) == 0:
        return []

    # Count obstacle points under each node's footprint in the floor plane.
    tree = cKDTree(obstacle_pts @ plane_xy)
    counts = tree.query_ball_point(positions @ plane_xy, r=radius, return_length=True)
    return [indices[i] for i in range(len(indices)) if counts[i] >= min_points]


def find_near_boundary_nodes(
    place_graph: PlaceGraph,
    boundary_node_indices: Sequence[int],
    threshold: float,
) -> list[int]:
    """Return active interior nodes lying within ``threshold`` metres of a boundary node.

    A *boundary node* is a frontier node (see classify_frontier_nodes).  This
    flags the interior nodes hugging the explored region's edge — e.g. a caution
    buffer a vehicle should keep clear of.  Distance is the full 3D Euclidean
    distance between place-node centres.

    Boundary nodes are never returned: a boundary node is trivially within
    ``threshold`` of itself, but it already carries a blocked/unexplored label,
    so including it here would be redundant.

    Args:
        place_graph:           The active place graph.
        boundary_node_indices: Graph indices of the boundary (frontier) nodes.
        threshold:             Maximum distance in metres
                               (GroundConfig.near_boundary_threshold).

    Returns:
        Graph node indices classified as near-boundary, in ascending order.
        Empty when there are no boundary nodes or no other active nodes.
    """
    boundary_set = set(boundary_node_indices)
    if not boundary_set:
        return []

    positions, indices = place_graph.get_active_node_data()
    if len(positions) == 0:
        return []

    boundary_positions = np.array(
        [place_graph.get_position(i) for i in boundary_set], dtype=np.float64
    )
    # For each active node, count how many boundary nodes fall within threshold.
    # A positive count means the node hugs the boundary.
    within_counts = cKDTree(boundary_positions).query_ball_point(
        positions, r=threshold, return_length=True
    )
    return [
        node_idx
        for array_idx, node_idx in enumerate(indices)
        if node_idx not in boundary_set and within_counts[array_idx] > 0
    ]


def build_frontier_mesh(
    centers: np.ndarray,
    R: np.ndarray,
    place_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a quad mesh for a set of frontier tile centres.

    Args:
        centers:    (N, 3) world-space tile centre positions.
        R:          (3, 3) rotation from local plane frame to world frame.
        place_size: Tile side length in metres.

    Returns:
        (vertices (N*4, 3), faces (N*2, 3))
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
        faces_list += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    return np.vstack(verts_list).astype(np.float32), np.array(faces_list, dtype=np.int32)
