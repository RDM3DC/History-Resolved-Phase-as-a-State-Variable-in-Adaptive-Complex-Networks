"""Tests for entropy proxy and adaptive ruler (entropy.py)."""

from __future__ import annotations

import pytest

from hrphasenet.entropy import (
    EntropyRulerState,
    branch_slip_term,
    entropy_step,
    ruler_step,
    transport_dissipation,
)
import numpy as np


class TestEntropyRulerState:
    def test_valid_state(self):
        s = EntropyRulerState(S=1.0, pi_a=1.0)
        assert s.S == 1.0
        assert s.pi_a == 1.0

    def test_negative_S_raises(self):
        with pytest.raises(ValueError, match="S must be >= 0"):
            EntropyRulerState(S=-0.1, pi_a=1.0)

    def test_zero_pi_a_raises(self):
        with pytest.raises(ValueError, match="pi_a must be > 0"):
            EntropyRulerState(S=0.0, pi_a=0.0)


class TestEntropyStep:
    def test_steady_state(self):
        # At equilibrium with no dissipation: S stays near S_eq
        S_eq = 0.5
        S = entropy_step(S_eq, T1=0.0, T2=0.0, gamma=1.0, S_eq=S_eq, dt=0.01)
        assert S == pytest.approx(S_eq)

    def test_above_equilibrium_decays(self):
        S0 = 2.0
        S1 = entropy_step(S0, T1=0.0, T2=0.0, gamma=1.0, S_eq=0.0, dt=0.01)
        assert S1 < S0

    def test_below_equilibrium_grows(self):
        S0 = 0.0
        S1 = entropy_step(S0, T1=0.0, T2=0.0, gamma=1.0, S_eq=1.0, dt=0.1)
        assert S1 > S0

    def test_clamped_at_zero(self):
        S = entropy_step(0.0, T1=0.0, T2=0.0, gamma=10.0, S_eq=0.0, dt=1.0)
        assert S >= 0.0

    def test_dissipation_increases_S(self):
        S0 = 0.0
        S1 = entropy_step(S0, T1=5.0, T2=0.0, gamma=0.0, S_eq=0.0, dt=0.1)
        assert S1 > S0

    def test_branch_slip_increases_S(self):
        S0 = 0.0
        S1 = entropy_step(S0, T1=0.0, T2=3.0, gamma=0.0, S_eq=0.0, dt=0.1)
        assert S1 > S0


class TestRulerStep:
    def test_steady_state_at_pi0(self):
        # When S=0 and ruler is at pi_0, ruler stays at pi_0
        pi_a = ruler_step(
            pi_a=1.0, S=0.0, alpha_pi=0.1, mu_pi=0.2, pi_0=1.0,
            pi_min=0.01, pi_max=4.0, dt=0.01
        )
        assert pi_a == pytest.approx(1.0)

    def test_high_entropy_grows_ruler(self):
        pi_a = ruler_step(
            pi_a=0.5, S=10.0, alpha_pi=0.5, mu_pi=0.01, pi_0=0.5,
            pi_min=0.01, pi_max=4.0, dt=0.1
        )
        assert pi_a > 0.5

    def test_low_entropy_ruler_relaxes_to_pi0(self):
        pi_a = ruler_step(
            pi_a=2.0, S=0.0, alpha_pi=0.1, mu_pi=1.0, pi_0=1.0,
            pi_min=0.01, pi_max=4.0, dt=0.1
        )
        assert pi_a < 2.0

    def test_clip_at_pi_max(self):
        pi_a = ruler_step(
            pi_a=3.9, S=100.0, alpha_pi=10.0, mu_pi=0.0, pi_0=1.0,
            pi_min=0.01, pi_max=4.0, dt=1.0
        )
        assert pi_a <= 4.0

    def test_clip_at_pi_min(self):
        pi_a = ruler_step(
            pi_a=0.02, S=0.0, alpha_pi=0.0, mu_pi=10.0, pi_0=1.0,
            pi_min=0.01, pi_max=4.0, dt=1.0
        )
        assert pi_a >= 0.01


class TestTransportDissipation:
    def test_positive_result(self):
        I = np.array([1.0 + 0j, 0.5 + 0.5j])
        G = np.array([2.0 + 0j, 1.0 + 0j])
        T1 = transport_dissipation(I, G)
        assert T1 >= 0.0

    def test_zero_current(self):
        I = np.zeros(3, dtype=complex)
        G = np.ones(3, dtype=complex)
        assert transport_dissipation(I, G) == 0.0

    def test_real_resistor(self):
        # |I|^2 / Re(G) = 4 / 2 = 2 for one edge
        I = np.array([2.0 + 0j])
        G = np.array([2.0 + 0j])
        T1 = transport_dissipation(I, G)
        assert T1 == pytest.approx(2.0)


class TestBranchSlipTerm:
    def test_no_slip(self):
        delta_w = np.zeros(4, dtype=int)
        assert branch_slip_term(delta_w) == 0.0

    def test_one_slip(self):
        delta_w = np.array([0, 1, 0, -1])
        assert branch_slip_term(delta_w, kappa=1.0) == pytest.approx(2.0)

    def test_kappa_scaling(self):
        delta_w = np.array([1])
        assert branch_slip_term(delta_w, kappa=3.0) == pytest.approx(3.0)
