"""Tests for phase lifting, winding, and parity (phase.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hrphasenet.phase import lifted_phase_update, winding_and_parity, wrap


class TestWrap:
    def test_zero(self):
        assert wrap(0.0) == 0.0

    def test_pi_maps_to_pi(self):
        assert wrap(math.pi) == pytest.approx(math.pi)

    def test_minus_pi_maps_to_pi(self):
        """(-pi, pi] convention: -pi wraps to +pi."""
        assert wrap(-math.pi) == pytest.approx(math.pi)

    def test_two_pi_maps_to_zero(self):
        assert wrap(2.0 * math.pi) == pytest.approx(0.0)

    def test_three_pi_maps_to_pi(self):
        assert wrap(3.0 * math.pi) == pytest.approx(math.pi)

    def test_minus_three_halves_pi(self):
        # -3pi/2 -> equivalent to +pi/2
        assert wrap(-1.5 * math.pi) == pytest.approx(0.5 * math.pi)

    def test_array(self):
        angles = np.array([0.0, math.pi, -math.pi, 2.0 * math.pi])
        result = wrap(angles)
        expected = np.array([0.0, math.pi, math.pi, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_large_positive(self):
        assert wrap(100.0 * math.pi) == pytest.approx(0.0)


class TestLiftedPhaseUpdate:
    def test_small_increment(self):
        # Small forward step: no clipping
        theta_ref = 0.0
        theta_raw = 0.3
        result = lifted_phase_update(theta_ref, theta_raw, pi_a=math.pi)
        assert result == pytest.approx(0.3)

    def test_clip_at_ruler(self):
        # Increment larger than ruler should be clipped
        theta_ref = 0.0
        theta_raw = 1.5  # > pi/4
        pi_a = math.pi / 4.0
        result = lifted_phase_update(theta_ref, theta_raw, pi_a=pi_a)
        assert result == pytest.approx(pi_a)

    def test_branch_continuity_across_pi(self):
        # Start just below -pi, step across the branch cut
        # theta_ref near +pi, raw phase just crosses to negative side
        theta_ref = math.pi - 0.1
        theta_raw = -(math.pi - 0.05)  # wraps to pi + 0.05 -> increment ~ 0.15
        result = lifted_phase_update(theta_ref, theta_raw, pi_a=math.pi)
        # Expected: wrap(-pi+0.05 - (pi-0.1)) = wrap(-2pi+0.15) = wrap(0.15) ≈ 0.15
        # theta_ref + 0.15 ≈ pi - 0.1 + 0.15 = pi + 0.05
        assert result == pytest.approx(math.pi + 0.05, abs=1e-10)

    def test_freeze_near_zero(self):
        # When |z| < z_min, phase should not update
        theta_ref = 1.0
        theta_raw = 2.0
        result = lifted_phase_update(
            theta_ref, theta_raw, pi_a=math.pi, z_mag=0.0, z_min=1e-9
        )
        assert result == pytest.approx(theta_ref)

    def test_no_freeze_above_z_min(self):
        theta_ref = 1.0
        theta_raw = 1.5
        result = lifted_phase_update(
            theta_ref, theta_raw, pi_a=math.pi, z_mag=1.0, z_min=1e-9
        )
        assert result == pytest.approx(1.5)

    def test_array_input(self):
        theta_ref = np.array([0.0, math.pi])
        theta_raw = np.array([0.5, math.pi + 0.2])
        result = lifted_phase_update(theta_ref, theta_raw, pi_a=math.pi)
        assert result[0] == pytest.approx(0.5)
        # wrap(0.2) = 0.2; theta_ref[1] + 0.2
        assert result[1] == pytest.approx(math.pi + 0.2)

    def test_monotone_accumulation_over_winding(self):
        """Full winding should accumulate 2*pi in resolved phase."""
        theta_ref = 0.0
        n_steps = 100
        d_theta = 2.0 * math.pi / n_steps
        for _ in range(n_steps):
            theta_raw = wrap(theta_ref + d_theta)
            theta_ref = lifted_phase_update(theta_ref, theta_raw, pi_a=math.pi)
        assert theta_ref == pytest.approx(2.0 * math.pi, rel=1e-3)


class TestWindingAndParity:
    def test_zero_phase(self):
        w, b = winding_and_parity(0.0)
        assert w == 0
        assert b == 1

    def test_one_winding(self):
        w, b = winding_and_parity(2.0 * math.pi)
        assert w == 1
        assert b == -1

    def test_two_windings(self):
        w, b = winding_and_parity(4.0 * math.pi)
        assert w == 2
        assert b == 1

    def test_negative_winding(self):
        w, b = winding_and_parity(-2.0 * math.pi)
        assert w == -1
        assert b == -1

    def test_array(self):
        theta_R = np.array([0.0, 2.0 * math.pi, 4.0 * math.pi, -2.0 * math.pi])
        w, b = winding_and_parity(theta_R)
        np.testing.assert_array_equal(w, [0, 1, 2, -1])
        np.testing.assert_array_equal(b, [1, -1, 1, -1])

    def test_parity_alternates(self):
        """Parity must alternate with each full winding."""
        parities = []
        for k in range(6):
            _, b = winding_and_parity(k * 2.0 * math.pi)
            parities.append(b)
        for i in range(len(parities) - 1):
            assert parities[i] != parities[i + 1]

    def test_falsifier_1_full_winding_returns_to_known_state(self):
        """Falsifier 1 (Section 9): scalar one-loop phase test.

        Starting from theta_R = 0, a full positive winding should yield
        winding number 1 and parity -1.  The state should NOT return to
        the same branch as the start.
        """
        theta_ref = 0.0
        n_steps = 100
        d_theta = 2.0 * math.pi / n_steps
        for _ in range(n_steps):
            theta_raw = wrap(theta_ref + d_theta)
            theta_ref = lifted_phase_update(theta_ref, theta_raw, pi_a=math.pi)

        w, b = winding_and_parity(theta_ref)
        # After one full winding: w=1, b=-1 (different from initial w=0, b=+1)
        assert w == 1
        assert b == -1
