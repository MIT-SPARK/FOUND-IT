from __future__ import annotations

import heapq
from typing import Optional

import numpy as np

from foundit.places.graph import PlaceGraph


def find_shortest_path(
    place_graph: PlaceGraph,
    start_point: np.ndarray,
    end_point: np.ndarray,
    excluded_nodes: frozenset[int] = frozenset(),
) -> Optional[tuple[list[np.ndarray], float]]:
    """Dijkstra shortest path between two 3D points on the place graph.

    Planning is restricted to *traversable* nodes: active graph nodes whose
    index is not in ``excluded_nodes``.  Start and end are snapped to the
    nearest traversable node, and Dijkstra never steps onto an excluded node.
    Pass the boundary and near-boundary node indices here to keep the path in
    the interior of the explored region.

    Returns (path_points, total_distance) or None if no path exists — including
    when every graph node is excluded.

    TODO Nico: this is not a bottleneck but for places in general we can see if we can improve performance by using packages like networkx.
    """
    if len(place_graph) == 0:
        print("No graph nodes available.")
        return None

    all_positions, active_indices = place_graph.get_active_node_data()

    # Restrict planning to traversable nodes (drop boundary / near-boundary).
    keep = [idx not in excluded_nodes for idx in active_indices]
    if not any(keep):
        print("No traversable graph nodes available (all excluded).")
        return None
    positions = all_positions[np.array(keep, dtype=bool)]
    indices = [idx for idx, keep_it in zip(active_indices, keep) if keep_it]

    dists1 = np.linalg.norm(positions - start_point, axis=1)
    start_node_idx = indices[int(np.argmin(dists1))]

    dists2 = np.linalg.norm(positions - end_point, axis=1)
    end_node_idx = indices[int(np.argmin(dists2))]

    queue: list[tuple[float, int]] = [(0.0, start_node_idx)]
    distances: dict[int, float] = {start_node_idx: 0.0}
    previous: dict[int, Optional[int]] = {start_node_idx: None}
    found = False

    while queue:
        current_dist, u = heapq.heappop(queue)

        if u == end_node_idx:
            found = True
            break

        if current_dist > distances.get(u, float('inf')):
            continue

        for v in place_graph.get_neighbors(u):
            if v in excluded_nodes:
                continue  # never route through boundary / near-boundary nodes
            pos_u = place_graph.get_position(u)
            pos_v = place_graph.get_position(v)
            weight = float(np.linalg.norm(pos_u - pos_v))

            alt = current_dist + weight
            if alt < distances.get(v, float('inf')):
                distances[v] = alt
                previous[v] = u
                heapq.heappush(queue, (alt, v))

    if not found:
        print("No path found between these locations.")
        return None

    path_points: list[np.ndarray] = []
    curr: Optional[int] = end_node_idx
    while curr is not None:
        path_points.append(place_graph.get_position(curr))
        curr = previous[curr]

    if len(path_points) <= 1:
        print("Start and End are the same node.")
        return None

    total_distance = distances[end_node_idx]
    print(f"Path found! Length: {len(path_points)} nodes, distance: {total_distance:.3f}")
    return path_points, total_distance