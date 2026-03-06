"""
Entropy proxy and adaptive phase ruler dynamics (Sections 6.1 and 6.2).

EntropyRulerState holds:
  S     - entropy-like scalar (>= 0)
  pi_a  - adaptive phase ruler half-width (> 0)

entropy_step:  discrete Euler update for S
ruler_step:    discrete Euler update for pi_a
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EntropyRulerState:
    """Joint state for the entropy proxy and adaptive phase ruler.

    Attributes
    ----------
    S:
        Entropy proxy scalar (non-negative).
    pi_a:
        Adaptive phase ruler half-width in radians (positive).
    """

    S: float
    pi_a: float

    def __post_init__(self) -> None:
        if self.S < 0:
            raise ValueError(f"Entropy proxy S must be >= 0, got {self.S}")
        if self.pi_a <= 0:
            raise ValueError(f"Adaptive ruler pi_a must be > 0, got {self.pi_a}")


def entropy_step(
    S: float,
    T1: float,
    T2: float,
    gamma: float,
    S_eq: float,
    dt: float,
) -> float:
    """Discrete Euler update for the entropy proxy (Section 6.1).

    S^+ = max(0, S + dt * [T1 + T2 - gamma * (S - S_eq)])

    Parameters
    ----------
    S:
        Current entropy value.
    T1:
        Transport dissipation term (>= 0).
    T2:
        Branch-slip / winding-event term (>= 0).
    gamma:
        Relaxation rate toward equilibrium (>= 0).
    S_eq:
        Equilibrium entropy value.
    dt:
        Time step.

    Returns
    -------
    Updated entropy S^+ (clamped to >= 0).
    """
    dS = T1 + T2 - gamma * (S - S_eq)
    return max(0.0, S + dt * dS)


def ruler_step(
    pi_a: float,
    S: float,
    alpha_pi: float,
    mu_pi: float,
    pi_0: float,
    pi_min: float,
    pi_max: float,
    dt: float,
) -> float:
    """Discrete Euler update for the adaptive phase ruler (Section 6.2).

    d(pi_a)/dt = alpha_pi * S - mu_pi * (pi_a - pi_0)
    Result is clipped to [pi_min, pi_max].

    Parameters
    ----------
    pi_a:
        Current ruler value.
    S:
        Current entropy value.
    alpha_pi:
        Entropy-drive strength (>= 0).
    mu_pi:
        Relaxation rate back to pi_0 (>= 0).
    pi_0:
        Baseline ruler value.
    pi_min:
        Minimum allowed ruler value (> 0).
    pi_max:
        Maximum allowed ruler value.
    dt:
        Time step.

    Returns
    -------
    Updated ruler pi_a^+ clipped to [pi_min, pi_max].
    """
    d_pi = alpha_pi * S - mu_pi * (pi_a - pi_0)
    pi_new = pi_a + dt * d_pi
    return float(np.clip(pi_new, pi_min, pi_max))


def transport_dissipation(
    currents: np.ndarray,
    conductances: np.ndarray,
) -> float:
    """Compute transport dissipation term T1 = sum |I_e|^2 / |G_e|.

    Uses the real part of conductance as the resistive component.
    Falls back to |G_e| when Re(G_e) <= 0 to avoid division by zero.

    Parameters
    ----------
    currents:
        Complex edge currents, shape (m,).
    conductances:
        Complex edge conductances, shape (m,).

    Returns
    -------
    Scalar transport dissipation T1 >= 0.
    """
    denom = np.where(
        np.real(conductances) > 0,
        np.real(conductances),
        np.abs(conductances),
    )
    denom = np.where(denom > 0, denom, 1.0)
    return float(np.sum(np.abs(currents) ** 2 / denom))


def branch_slip_term(
    delta_w: np.ndarray,
    kappa: float = 1.0,
) -> float:
    """Compute branch-slip term T2 = kappa * sum |delta_w_e|.

    Parameters
    ----------
    delta_w:
        Change in winding number per edge this step (integer array).
    kappa:
        Scaling coefficient.

    Returns
    -------
    Scalar branch-slip contribution T2 >= 0.
    """
    return float(kappa * np.sum(np.abs(delta_w)))
