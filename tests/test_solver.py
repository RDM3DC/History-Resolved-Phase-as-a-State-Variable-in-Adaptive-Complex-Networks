"""Tests for the nodal solver (solver.py)."""

from __future__ import annotations

import numpy as np
import pytest

from hrphasenet.solver import build_admittance_matrix, edge_current, nodal_solve


class TestBuildAdmittanceMatrix:
    def test_single_edge(self):
        edges = [(0, 1)]
        G = np.array([2.0 + 0j])
        Y = build_admittance_matrix(edges, G, n_nodes=2)
        Y_dense = Y.toarray()
        # Y[0,0] = Y[1,1] = 2, Y[0,1] = Y[1,0] = -2
        np.testing.assert_allclose(Y_dense[0, 0], 2.0)
        np.testing.assert_allclose(Y_dense[1, 1], 2.0)
        np.testing.assert_allclose(Y_dense[0, 1], -2.0)
        np.testing.assert_allclose(Y_dense[1, 0], -2.0)

    def test_symmetry(self):
        edges = [(0, 1), (1, 2), (0, 2)]
        G = np.array([1.0 + 0j, 2.0 + 0j, 0.5 + 0j])
        Y = build_admittance_matrix(edges, G, n_nodes=3).toarray()
        np.testing.assert_allclose(Y, Y.T)

    def test_complex_conductance(self):
        edges = [(0, 1)]
        G = np.array([1.0 + 2.0j])
        Y = build_admittance_matrix(edges, G, n_nodes=2).toarray()
        np.testing.assert_allclose(Y[0, 0], 1.0 + 2.0j)
        np.testing.assert_allclose(Y[0, 1], -(1.0 + 2.0j))


class TestNodalSolve:
    def test_single_resistor(self):
        # Two nodes, one resistor of conductance G=1.
        # source at 0, sink at 1, source_value=1.
        # Expected: phi[0] = 1, phi[1] = 0
        edges = [(0, 1)]
        G = np.array([1.0 + 0j])
        phi, converged = nodal_solve(edges, G, n_nodes=2,
                                     source_node=0, sink_node=1,
                                     source_value=1.0 + 0j)
        assert converged
        np.testing.assert_allclose(phi[1], 0.0, atol=1e-10)
        np.testing.assert_allclose(phi[0], 1.0, atol=1e-6)

    def test_voltage_divider(self):
        # Three nodes in series: 0--G1--1--G2--2
        # Inject source_value=1 A at node 0, drain at node 2.
        # With G1=G2=1 (R=1 each), R_total=2 ohms,
        # phi[0]=2, phi[1]=1, phi[2]=0 (ground).
        edges = [(0, 1), (1, 2)]
        G = np.array([1.0 + 0j, 1.0 + 0j])
        phi, converged = nodal_solve(edges, G, n_nodes=3,
                                     source_node=0, sink_node=2,
                                     source_value=1.0 + 0j)
        assert converged
        np.testing.assert_allclose(phi[2], 0.0, atol=1e-10)
        np.testing.assert_allclose(np.real(phi[1]), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.real(phi[0]), 2.0, atol=1e-6)

    def test_diamond_graph(self):
        # Diamond: 0->1->3, 0->2->3; G all 1; source=0, sink=3
        edges = [(0, 1), (1, 3), (0, 2), (2, 3)]
        G = np.ones(4, dtype=complex)
        phi, converged = nodal_solve(edges, G, n_nodes=4,
                                     source_node=0, sink_node=3,
                                     source_value=1.0 + 0j)
        assert converged
        np.testing.assert_allclose(phi[3], 0.0, atol=1e-10)
        # By symmetry phi[1] == phi[2]
        np.testing.assert_allclose(phi[1], phi[2], atol=1e-10)

    def test_returns_correct_shape(self):
        edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        G = np.ones(4, dtype=complex)
        phi, _ = nodal_solve(edges, G, n_nodes=4,
                              source_node=0, sink_node=3)
        assert phi.shape == (4,)


class TestEdgeCurrent:
    def test_ohms_law(self):
        # G=2, phi[0]=1, phi[1]=0 -> I = 2 * 1 = 2
        edges = [(0, 1)]
        G = np.array([2.0 + 0j])
        phi = np.array([1.0 + 0j, 0.0 + 0j])
        I = edge_current(edges, G, phi)
        np.testing.assert_allclose(I[0], 2.0 + 0j)

    def test_sign_convention(self):
        # Current flows from 0 to 1 when phi[0] > phi[1]
        edges = [(0, 1)]
        G = np.array([1.0 + 0j])
        phi = np.array([2.0 + 0j, 1.0 + 0j])
        I = edge_current(edges, G, phi)
        assert np.real(I[0]) > 0

    def test_array_edges(self):
        edges = [(0, 1), (1, 2)]
        G = np.array([1.0 + 0j, 1.0 + 0j])
        phi = np.array([1.0 + 0j, 0.5 + 0j, 0.0 + 0j])
        I = edge_current(edges, G, phi)
        np.testing.assert_allclose(I[0], 0.5 + 0j, atol=1e-12)
        np.testing.assert_allclose(I[1], 0.5 + 0j, atol=1e-12)
