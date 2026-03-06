"""Tests for graph factory functions (graphs.py) and drive generators (drives.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hrphasenet.graphs import complete_graph, diamond_graph, path_graph, ring_graph
from hrphasenet.drives import (
    chirp_drive,
    constant_drive,
    periodic_drive,
    reversal_drive,
)


class TestRingGraph:
    def test_basic_ring(self):
        edges, src, snk = ring_graph(4)
        assert len(edges) == 4
        assert src == 0
        assert snk == 2

    def test_ring_with_defect(self):
        edges_plain, _, _ = ring_graph(6, defect=False)
        edges_defect, _, _ = ring_graph(6, defect=True)
        assert len(edges_defect) == len(edges_plain) + 1

    def test_ring_connectivity(self):
        edges, _, _ = ring_graph(5)
        nodes = set()
        for u, v in edges:
            nodes.add(u)
            nodes.add(v)
        assert nodes == {0, 1, 2, 3, 4}

    def test_ring_too_small(self):
        with pytest.raises(ValueError):
            ring_graph(2)


class TestDiamondGraph:
    def test_structure(self):
        edges, src, snk = diamond_graph()
        assert src == 0
        assert snk == 3
        assert len(edges) == 4
        # Paths: 0->1->3 and 0->2->3
        assert (0, 1) in edges
        assert (1, 3) in edges
        assert (0, 2) in edges
        assert (2, 3) in edges


class TestPathGraph:
    def test_structure(self):
        edges, src, snk = path_graph(5)
        assert len(edges) == 4
        assert src == 0
        assert snk == 4

    def test_too_small(self):
        with pytest.raises(ValueError):
            path_graph(1)


class TestCompleteGraph:
    def test_edge_count(self):
        edges, _, _ = complete_graph(4)
        assert len(edges) == 4 * 3  # n*(n-1) directed edges

    def test_too_small(self):
        with pytest.raises(ValueError):
            complete_graph(1)


class TestConstantDrive:
    def test_all_equal(self):
        values = list(constant_drive(2.0 + 1j, 5))
        assert all(v == 2.0 + 1j for v in values)

    def test_length(self):
        assert len(list(constant_drive(1.0, 7))) == 7


class TestPeriodicDrive:
    def test_length(self):
        assert len(list(periodic_drive(1.0, omega=1.0, dt=0.1, n_steps=10))) == 10

    def test_unit_amplitude(self):
        values = list(periodic_drive(1.0, omega=1.0, dt=0.1, n_steps=10))
        for v in values:
            assert abs(abs(v) - 1.0) < 1e-10

    def test_phase_advances(self):
        values = list(periodic_drive(1.0, omega=2 * math.pi, dt=1.0, n_steps=2))
        # At t=0: phase=0; at t=1: phase=2pi → same value
        assert abs(values[0] - values[1]) < 1e-10


class TestChirpDrive:
    def test_length(self):
        assert len(list(chirp_drive(1.0, 0.5, 2.0, 0.01, 20))) == 20

    def test_unit_amplitude(self):
        values = list(chirp_drive(1.0, 0.5, 5.0, 0.01, 30))
        for v in values:
            assert abs(abs(v) - 1.0) < 1e-10


class TestReversalDrive:
    def test_correct_phases(self):
        values = list(reversal_drive(1.0 + 0j, -1.0 + 0j, n_A=3, n_BA=2, n_restore=3))
        assert values[:3] == [1.0 + 0j] * 3
        assert values[3:5] == [-1.0 + 0j] * 2
        assert values[5:] == [1.0 + 0j] * 3

    def test_total_length(self):
        values = list(reversal_drive(1.0, 2.0, n_A=5, n_BA=3, n_restore=7))
        assert len(values) == 15
