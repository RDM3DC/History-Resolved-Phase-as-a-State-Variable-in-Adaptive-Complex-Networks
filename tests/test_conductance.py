"""Tests for adaptive conductance dynamics (conductance.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hrphasenet.conductance import (
    alpha_G,
    apply_conductance_clamps,
    conductance_step,
    mu_G,
    suppression_term,
)


class TestAlphaG:
    def test_at_S_c_gives_half_alpha0(self):
        a = alpha_G(S=1.0, alpha_0=2.0, S_c=1.0, delta_S=0.1)
        assert a == pytest.approx(1.0, rel=1e-3)

    def test_far_above_threshold_gives_alpha0(self):
        a = alpha_G(S=100.0, alpha_0=2.0, S_c=1.0, delta_S=0.1)
        assert a == pytest.approx(2.0, rel=1e-3)

    def test_far_below_threshold_gives_near_zero(self):
        a = alpha_G(S=-100.0, alpha_0=2.0, S_c=1.0, delta_S=0.1)
        assert a < 0.01

    def test_monotone_in_S(self):
        alphas = [alpha_G(S=float(s), alpha_0=1.0, S_c=1.0, delta_S=0.1) for s in range(5)]
        assert all(alphas[i] <= alphas[i + 1] for i in range(len(alphas) - 1))


class TestMuG:
    def test_zero_S(self):
        assert mu_G(S=0.0, mu_0=1.0, S_0=1.0) == 0.0

    def test_proportional_to_S(self):
        m1 = mu_G(S=1.0, mu_0=1.0, S_0=1.0)
        m2 = mu_G(S=2.0, mu_0=1.0, S_0=1.0)
        assert m2 == pytest.approx(2.0 * m1)

    def test_non_negative(self):
        assert mu_G(S=-5.0, mu_0=1.0, S_0=1.0) >= 0.0


class TestSuppressionTerm:
    def test_zero_lambda(self):
        G = np.array([1.0 + 0j, 2.0 + 0j])
        theta_R = np.array([0.1, 0.5])
        result = suppression_term(G, theta_R, pi_a=math.pi, lambda_s=0.0)
        np.testing.assert_array_equal(result, np.zeros_like(G))

    def test_shape_preserved(self):
        G = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
        theta_R = np.zeros(3)
        result = suppression_term(G, theta_R, pi_a=math.pi, lambda_s=1.0)
        assert result.shape == (3,)

    def test_zero_at_zero_theta(self):
        # sin^2(0 / (2*pi_a)) = 0
        G = np.array([1.0 + 0j])
        theta_R = np.array([0.0])
        result = suppression_term(G, theta_R, pi_a=math.pi, lambda_s=1.0)
        np.testing.assert_allclose(np.abs(result), 0.0, atol=1e-12)

    def test_maximal_at_pi_times_pi_a(self):
        # sin^2(theta_R / (2*pi_a)) = 1 when theta_R / (2*pi_a) = pi/2
        # i.e. theta_R = pi * pi_a
        G = np.array([2.0 + 0j])
        pi_a = 1.0
        theta_R = np.array([math.pi * pi_a])
        result = suppression_term(G, theta_R, pi_a=pi_a, lambda_s=1.0)
        np.testing.assert_allclose(np.abs(result), 2.0, atol=1e-10)


class TestConductanceStep:
    def test_output_shape(self):
        m = 4
        G = np.ones(m, dtype=complex)
        I_mag = np.ones(m)
        theta_R = np.zeros(m)
        G_new = conductance_step(
            G, I_mag, theta_R, S=1.0, pi_a=math.pi,
            dt=0.01, alpha_0=1.0, S_c=0.5, delta_S=0.1,
            mu_0=0.1, S_0=1.0
        )
        assert G_new.shape == (m,)

    def test_real_part_growth_with_zero_phase(self):
        # With theta_R=0: reinforce = alpha * I_mag * 1 (real), G grows if alpha > mu*G
        G = np.array([0.1 + 0j])
        I_mag = np.array([1.0])
        theta_R = np.array([0.0])
        G_new = conductance_step(
            G, I_mag, theta_R, S=10.0, pi_a=math.pi,
            dt=0.1, alpha_0=1.0, S_c=0.5, delta_S=0.1,
            mu_0=0.0, S_0=1.0
        )
        # With mu=0, G should only grow (reinforce only)
        assert np.real(G_new[0]) > np.real(G[0])

    def test_decay_dominates_at_high_S_mu(self):
        # With alpha=0 and small dt, large mu should shrink G
        G = np.array([10.0 + 0j])
        I_mag = np.array([0.001])  # tiny current
        theta_R = np.array([0.0])
        G_new = conductance_step(
            G, I_mag, theta_R, S=1.0, pi_a=math.pi,
            dt=0.01, alpha_0=0.0, S_c=0.5, delta_S=0.1,
            mu_0=2.0, S_0=1.0
        )
        # mu_G(S=1, mu_0=2, S_0=1) = 2; dG = 0 - 2*10 = -20; G_new = 10 - 0.2 = 9.8
        assert np.abs(G_new[0]) < np.abs(G[0])


class TestApplyConductanceClamps:
    def test_real_min_enforced(self):
        G = np.array([-0.5 + 0j, 0.0 + 0j, 0.5 + 0j])
        G_clamped = apply_conductance_clamps(G, real_min=0.01)
        assert np.all(np.real(G_clamped) >= 0.01)

    def test_real_max_enforced(self):
        G = np.array([5.0 + 0j, 10.0 + 0j])
        G_clamped = apply_conductance_clamps(G, real_min=0.0, real_max=3.0)
        assert np.all(np.real(G_clamped) <= 3.0)

    def test_imag_max_enforced(self):
        G = np.array([1.0 + 5j, 1.0 - 5j])
        G_clamped = apply_conductance_clamps(G, imag_max=2.0)
        assert np.all(np.abs(np.imag(G_clamped)) <= 2.0)

    def test_budget_normalisation(self):
        G = np.array([3.0 + 0j, 3.0 + 0j])
        G_clamped = apply_conductance_clamps(G, budget=4.0)
        assert np.sum(np.real(G_clamped)) == pytest.approx(4.0, rel=1e-6)

    def test_noise_changes_values(self):
        rng = np.random.default_rng(42)
        G = np.ones(10, dtype=complex)
        G_noisy = apply_conductance_clamps(G, noise_std=0.01, rng=rng)
        assert not np.allclose(G_noisy, G)

    def test_no_change_when_all_unconstrained(self):
        G = np.array([2.0 + 1j, 1.0 - 0.5j])
        G_clamped = apply_conductance_clamps(
            G, real_min=0.0, real_max=None, imag_max=None,
            budget=None, noise_std=0.0
        )
        np.testing.assert_array_equal(G_clamped, G)
