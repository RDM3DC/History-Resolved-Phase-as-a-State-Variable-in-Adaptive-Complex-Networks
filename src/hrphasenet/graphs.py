"""
Graph factory functions for the three primary graph families (Section 10.1).

ring_graph    - ring graph with optional defect edge
diamond_graph - diamond (two-path) graph
path_graph    - simple source-to-sink path
"""

from __future__ import annotations


def ring_graph(n: int, defect: bool = False) -> tuple[list[tuple[int, int]], int, int]:
    """Create a ring graph with n nodes.

    The ring has edges 0→1, 1→2, ..., (n-2)→(n-1), (n-1)→0.
    Forcing: source = 0, sink = n // 2.

    When defect=True an extra edge is added from node 0 to node n//2
    to introduce a shortcut / defect path.

    Parameters
    ----------
    n:
        Number of nodes (>= 3).
    defect:
        Whether to add a defect shortcut edge.

    Returns
    -------
    (edges, source_node, sink_node)
    """
    if n < 3:
        raise ValueError("Ring graph requires >= 3 nodes")
    edges = [(i, (i + 1) % n) for i in range(n)]
    if defect:
        mid = n // 2
        if (0, mid) not in edges:
            edges.append((0, mid))
    source = 0
    sink = n // 2
    return edges, source, sink


def diamond_graph() -> tuple[list[tuple[int, int]], int, int]:
    """Create the diamond (two-path) graph.

    Topology:
        0 -> 1 -> 3
        0 -> 2 -> 3

    Source = 0, sink = 3.

    Returns
    -------
    (edges, source_node, sink_node)
    """
    edges = [(0, 1), (1, 3), (0, 2), (2, 3)]
    return edges, 0, 3


def path_graph(n: int) -> tuple[list[tuple[int, int]], int, int]:
    """Create a simple directed path graph: 0 -> 1 -> ... -> n-1.

    Parameters
    ----------
    n:
        Number of nodes (>= 2).

    Returns
    -------
    (edges, source_node, sink_node)
    """
    if n < 2:
        raise ValueError("Path graph requires >= 2 nodes")
    edges = [(i, i + 1) for i in range(n - 1)]
    return edges, 0, n - 1


def complete_graph(n: int) -> tuple[list[tuple[int, int]], int, int]:
    """Create a complete directed graph (all ordered pairs).

    Source = 0, sink = n - 1.

    Parameters
    ----------
    n:
        Number of nodes (>= 2).

    Returns
    -------
    (edges, source_node, sink_node)
    """
    if n < 2:
        raise ValueError("Complete graph requires >= 2 nodes")
    edges = [(u, v) for u in range(n) for v in range(n) if u != v]
    return edges, 0, n - 1
