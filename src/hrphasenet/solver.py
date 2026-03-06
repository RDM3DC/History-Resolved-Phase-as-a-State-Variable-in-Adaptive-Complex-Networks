"""
Nodal potential solver for the adaptive complex network (Section 3.2).

The graph admittance system is:

    Y @ phi = b

where Y is the nodal admittance matrix built from edge conductances,
phi is the vector of node potentials, and b encodes source/sink forcing.

Implements:
  build_admittance_matrix  - assemble the complex sparse admittance matrix
  nodal_solve              - solve for node potentials given sources
  edge_current             - compute per-edge complex current
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_admittance_matrix(
    edges: list[tuple[int, int]],
    conductances: np.ndarray,
    n_nodes: int,
) -> sp.csr_matrix:
    """Assemble the nodal admittance matrix Y.

    For edge e = (u, v) with conductance G_e:
      Y[u, u] += G_e
      Y[v, v] += G_e
      Y[u, v] -= G_e
      Y[v, u] -= G_e

    Parameters
    ----------
    edges:
        List of (u, v) directed edge pairs (0-indexed).
    conductances:
        Complex conductance per edge, shape (m,).
    n_nodes:
        Number of nodes in the graph.

    Returns
    -------
    Sparse CSR admittance matrix of shape (n_nodes, n_nodes), complex128.
    """
    m = len(edges)
    rows = []
    cols = []
    data = []

    for idx, ((u, v), G) in enumerate(zip(edges, conductances)):
        rows += [u, v, u, v]
        cols += [u, v, v, u]
        data += [G, G, -G, -G]

    Y = sp.coo_matrix(
        (data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=complex
    ).tocsr()
    return Y


def nodal_solve(
    edges: list[tuple[int, int]],
    conductances: np.ndarray,
    n_nodes: int,
    source_node: int,
    sink_node: int,
    source_value: complex = 1.0 + 0j,
    ground_node: int | None = None,
) -> tuple[np.ndarray, bool]:
    """Solve nodal potentials under a source-sink forcing protocol.

    The system is solved by pinning one node as ground (reference potential
    phi = 0) and injecting source_value current at source_node, draining
    at sink_node.  A small regularisation term is added for robustness
    near ill-conditioned configurations.

    Parameters
    ----------
    edges:
        List of (u, v) directed edge pairs (0-indexed).
    conductances:
        Complex edge conductances, shape (m,).
    n_nodes:
        Number of nodes.
    source_node:
        Index of the source node.
    sink_node:
        Index of the sink node.
    source_value:
        Complex amplitude injected at source_node (drained at sink_node).
    ground_node:
        Node pinned to phi = 0.  Defaults to sink_node when None.

    Returns
    -------
    (phi, converged):
        phi: complex node potentials, shape (n_nodes,).
        converged: True if the linear solve succeeded without warnings.
    """
    if ground_node is None:
        ground_node = sink_node

    Y = build_admittance_matrix(edges, conductances, n_nodes)

    # Build RHS
    b = np.zeros(n_nodes, dtype=complex)
    b[source_node] += source_value
    b[sink_node] -= source_value

    # Ground the reference node: replace row/col with identity
    Y = Y.tolil()
    Y[ground_node, :] = 0.0
    Y[:, ground_node] = 0.0
    Y[ground_node, ground_node] = 1.0
    b[ground_node] = 0.0

    # Small regularisation for near-singular matrices
    n = Y.shape[0]
    reg = 1e-12 * sp.eye(n, dtype=complex, format="csr")
    Y_reg = Y.tocsr() + reg

    converged = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            phi = spla.spsolve(Y_reg, b)
    except (sp.linalg.MatrixRankWarning, Exception):
        converged = False
        # Fallback: dense solve
        try:
            phi = np.linalg.solve(Y_reg.toarray(), b)
        except np.linalg.LinAlgError:
            phi = np.zeros(n_nodes, dtype=complex)

    return phi, converged


def edge_current(
    edges: list[tuple[int, int]],
    conductances: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Compute per-edge complex current I_e = G_e * (phi_u - phi_v).

    Parameters
    ----------
    edges:
        List of (u, v) edge pairs (0-indexed).
    conductances:
        Complex conductances, shape (m,).
    phi:
        Node potentials, shape (n_nodes,).

    Returns
    -------
    Complex edge currents, shape (m,).
    """
    edges_arr = np.asarray(edges, dtype=int)
    u = edges_arr[:, 0]
    v = edges_arr[:, 1]
    delta_phi = phi[u] - phi[v]
    return conductances * delta_phi
