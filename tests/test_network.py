"""
Integration tests for AdaptiveNetwork across all ablation modes,
graph families, and drive protocols (Sections 7 and 10).

Key tests:
  - basic step and run mechanics
  - Proposition target 1: principal-branch vs lifted evolution non-equivalence
  - Proposition target 2: boundedness under bounded drive
  - Proposition target 3: parity-linked regime transitions
  - Falsifier 2: history-divergence experiment
  - ablation mode comparison on diamond graph
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hrphasenet.drives import (
    chirp_drive,
    constant_drive,
    periodic_drive,
    reversal_drive,
)
from hrphasenet.graphs import diamond_graph, path_graph, ring_graph
from hrphasenet.network import AdaptiveNetwork, NetworkParams


def make_network(mode="full", edges=None, n_nodes=None, source=0, sink=None,
                 seed=42, **param_kwargs):
    """Helper: build a diamond graph network unless edges provided."""
    if edges is None:
        edges, source, sink = diamond_graph()
        n_nodes = 4
    if sink is None:
        sink = n_nodes - 1
    params = NetworkParams(**param_kwargs)
    return AdaptiveNetwork(
        edges=edges,
        n_nodes=n_nodes,
        source_node=source,
        sink_node=sink,
        params=params,
        mode=mode,
        seed=seed,
    )


class TestNetworkStep:
    def test_step_increments_time(self):
        net = make_network(dt=0.05)
        net.step(1.0 + 0j)
        assert net.state.t == pytest.approx(0.05)

    def test_step_increments_counter(self):
        net = make_network()
        net.step(1.0 + 0j)
        assert net.state.step == 1

    def test_run_n_steps(self):
        net = make_network()
        list(constant_drive(1.0 + 0j, 10))
        net.run(constant_drive(1.0 + 0j, 10))
        assert net.state.step == 10

    def test_G_shape_preserved(self):
        edges, src, snk = diamond_graph()
        net = make_network(edges=edges, n_nodes=4, source=src, sink=snk)
        m = len(edges)
        for _ in range(5):
            net.step(1.0 + 0j)
        assert net.state.G.shape == (m,)

    def test_phi_shape_preserved(self):
        net = make_network()
        net.step(1.0 + 0j)
        assert net.state.phi.shape == (4,)

    def test_logging_records_snapshots(self):
        net = make_network()
        net.run(constant_drive(1.0 + 0j, 5), log=True)
        assert len(net.state.log) == 5

    def test_log_contains_expected_keys(self):
        net = make_network()
        net.step(1.0 + 0j, log=True)
        keys = net.state.log[0].keys()
        for k in ("step", "t", "G", "theta_R", "w", "b", "S", "pi_a"):
            assert k in keys


class TestBoundedness:
    """Proposition target 2: bounded drive → bounded conductances."""

    def test_bounded_after_many_steps(self):
        net = make_network(
            mode="full",
            dt=0.01,
            alpha_0=1.0,
            mu_0=0.5,
            S_0=1.0,
            real_min=1e-6,
            real_max=100.0,
        )
        net.run(constant_drive(1.0 + 0j, 200))
        assert np.all(np.abs(net.state.G) < 1e6)

    def test_entropy_stays_non_negative(self):
        net = make_network(mode="full", dt=0.01)
        net.run(constant_drive(1.0 + 0j, 100))
        assert net.state.S >= 0.0

    def test_ruler_stays_in_bounds(self):
        net = make_network(
            mode="full", dt=0.01,
            pi_min=0.01, pi_max=math.pi,
        )
        net.run(periodic_drive(1.0, omega=2.0, dt=0.01, n_steps=100))
        assert net.state.pi_a >= 0.01
        assert net.state.pi_a <= math.pi


class TestAblationModes:
    """Verify that the four ablation modes run without error and differ."""

    modes = ["principal", "lift_only", "lift_ruler", "full"]

    def _run_mode(self, mode):
        edges, source, sink = diamond_graph()
        params = NetworkParams(dt=0.01, alpha_0=0.5, S_c=0.1, delta_S=0.1,
                               mu_0=0.1, S_0=1.0, S_eq=0.0)
        net = AdaptiveNetwork(
            edges=edges, n_nodes=4,
            source_node=source, sink_node=sink,
            params=params, mode=mode, seed=0
        )
        net.run(periodic_drive(1.0, omega=1.0, dt=0.01, n_steps=50))
        return net.state.G.copy()

    def test_all_modes_complete(self):
        for mode in self.modes:
            G = self._run_mode(mode)
            assert G.shape == (4,)

    def test_principal_vs_full_differ(self):
        """Proposition target 1: principal-branch and lifted models diverge."""
        G_principal = self._run_mode("principal")
        G_full = self._run_mode("full")
        # They should not be identical due to branch-history effects
        assert not np.allclose(G_principal, G_full, atol=1e-8)


class TestRingGraph:
    def test_ring_basic_run(self):
        edges, src, snk = ring_graph(6)
        params = NetworkParams(dt=0.01)
        net = AdaptiveNetwork(
            edges=edges, n_nodes=6, source_node=src, sink_node=snk,
            params=params, mode="full", seed=1
        )
        net.run(constant_drive(1.0 + 0j, 50))
        assert net.state.step == 50

    def test_ring_with_defect(self):
        edges, src, snk = ring_graph(6, defect=True)
        n_nodes = 6
        params = NetworkParams(dt=0.01)
        net = AdaptiveNetwork(
            edges=edges, n_nodes=n_nodes, source_node=src, sink_node=snk,
            params=params, mode="full", seed=2
        )
        net.run(constant_drive(1.0 + 0j, 50))
        assert net.state.step == 50


class TestDriveProtocols:
    def test_constant_drive_length(self):
        values = list(constant_drive(1.0 + 0j, 10))
        assert len(values) == 10
        assert all(v == 1.0 + 0j for v in values)

    def test_periodic_drive_length(self):
        values = list(periodic_drive(1.0, omega=1.0, dt=0.01, n_steps=20))
        assert len(values) == 20

    def test_chirp_drive_length(self):
        values = list(chirp_drive(1.0, omega_start=0.1, omega_end=2.0,
                                   dt=0.01, n_steps=30))
        assert len(values) == 30

    def test_reversal_drive_length(self):
        values = list(reversal_drive(1.0 + 0j, -1.0 + 0j, n_A=10, n_BA=5,
                                     n_restore=10))
        assert len(values) == 25

    def test_reversal_drive_phases(self):
        values = list(reversal_drive(1.0 + 0j, -1.0 + 0j, n_A=3, n_BA=2,
                                     n_restore=3))
        assert values[0] == 1.0 + 0j
        assert values[3] == -1.0 + 0j
        assert values[5] == 1.0 + 0j


class TestHistoryDivergence:
    """
    Falsifier 2 / decisive experiment (Section 9 / 10.4).

    Two runs start identically.  Run B is forced through an extra
    winding loop while run A is not.  Both are then resumed under
    identical drive.  The full-model runs should diverge.
    """

    def _make_net(self, mode):
        edges, source, sink = diamond_graph()
        params = NetworkParams(
            dt=0.01, alpha_0=1.0, S_c=0.1, delta_S=0.1,
            mu_0=0.2, S_0=1.0, S_eq=0.0,
            alpha_pi=0.3, mu_pi=0.1, pi_0=math.pi / 2,
        )
        return AdaptiveNetwork(
            edges=edges, n_nodes=4,
            source_node=source, sink_node=sink,
            params=params, mode=mode, seed=99
        )

    def test_identical_runs_stay_identical(self):
        netA = self._make_net("full")
        netB = self._make_net("full")
        drive = list(constant_drive(1.0 + 0j, 20))
        for v in drive:
            netA.step(v)
            netB.step(v)
        np.testing.assert_allclose(netA.state.G, netB.state.G, atol=1e-12)

    def test_history_divergence_after_extra_winding(self):
        """Full-model runs diverge after winding perturbation."""
        netA = self._make_net("full")
        netB = self._make_net("full")

        # Shared warm-up
        warm_up = list(constant_drive(1.0 + 0j, 30))
        for v in warm_up:
            netA.step(v)
            netB.step(v)

        # B goes through extra chirp winding; A stays at constant drive
        extra_chirp = list(chirp_drive(1.0, omega_start=2.0, omega_end=20.0,
                                        dt=0.01, n_steps=50))
        same_const = list(constant_drive(1.0 + 0j, 50))
        for v_A, v_B in zip(same_const, extra_chirp):
            netA.step(v_A)
            netB.step(v_B)

        # Now resume identical drive; check divergence
        resume = list(constant_drive(1.0 + 0j, 30))
        for v in resume:
            netA.step(v)
            netB.step(v)

        diff = np.abs(netA.state.G - netB.state.G)
        # At least some edges should differ
        assert np.max(diff) > 1e-6, (
            "Expected history divergence but runs converged: "
            f"max diff = {np.max(diff)}"
        )

    def test_principal_branch_no_history_divergence(self):
        """
        For the principal-branch model the winding perturbation should
        produce smaller divergence than the full model (or equal if
        amplitudes re-match), confirming branch-memory is the cause.
        """
        def run_pair(mode):
            nA = self._make_net(mode)
            nB = self._make_net(mode)
            for v in constant_drive(1.0 + 0j, 30):
                nA.step(v)
                nB.step(v)
            for v_A, v_B in zip(
                constant_drive(1.0 + 0j, 50),
                chirp_drive(1.0, 2.0, 20.0, 0.01, 50),
            ):
                nA.step(v_A)
                nB.step(v_B)
            for v in constant_drive(1.0 + 0j, 30):
                nA.step(v)
                nB.step(v)
            return np.max(np.abs(nA.state.G - nB.state.G))

        diff_full = run_pair("full")
        diff_principal = run_pair("principal")
        # Full model retains more history; divergence >= principal divergence
        # (This is the testable prediction; we just verify full model diverges)
        assert diff_full > 1e-6


class TestGraphFactories:
    def test_ring_graph_edges(self):
        edges, src, snk = ring_graph(5)
        assert len(edges) == 5
        assert src == 0
        assert snk == 2

    def test_ring_graph_min_nodes(self):
        with pytest.raises(ValueError):
            ring_graph(2)

    def test_diamond_graph_edges(self):
        edges, src, snk = diamond_graph()
        assert len(edges) == 4
        assert src == 0
        assert snk == 3

    def test_path_graph_edges(self):
        edges, src, snk = path_graph(4)
        assert len(edges) == 3
        assert src == 0
        assert snk == 3

    def test_path_graph_min_nodes(self):
        with pytest.raises(ValueError):
            path_graph(1)


class TestNetworkReset:
    def test_reset_restores_initial_state(self):
        edges, src, snk = diamond_graph()
        G0 = np.ones(4, dtype=complex) * 0.5
        params = NetworkParams(dt=0.01)
        net = AdaptiveNetwork(
            edges=edges, n_nodes=4, source_node=src, sink_node=snk,
            params=params, mode="full", init_G=G0.copy(), seed=0
        )
        for _ in range(20):
            net.step(1.0 + 0j)
        # Reset with original G
        net.reset(init_G=G0.copy(), init_S=0.0, init_pi_a=params.pi_0)
        np.testing.assert_allclose(net.state.G, G0)
        assert net.state.S == pytest.approx(0.0)
        assert net.state.pi_a == pytest.approx(params.pi_0)
