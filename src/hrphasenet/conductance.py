"""
Adaptive complex conductance dynamics (Section 5).

Implements:
  alpha_G  - entropy-gated gain function
  mu_G     - entropy-dependent decay function
  conductance_step - discrete Euler update for G_e
  apply_conductance_clamps - real/imag bounds and budget
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid sigma(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def alpha_G(
    S: float,
    alpha_0: float,
    S_c: float,
    delta_S: float,
) -> float:
    """Entropy-gated gain function (Section 5.1).

    alpha_G(S) = alpha_0 * sigma((S - S_c) / delta_S)

    Parameters
    ----------
    S:
        Current entropy value.
    alpha_0:
        Maximum gain amplitude.
    S_c:
        Entropy threshold for half-activation.
    delta_S:
        Width of the activation transition (> 0).

    Returns
    -------
    Gain value in (0, alpha_0).
    """
    return float(alpha_0 * _sigmoid((S - S_c) / delta_S))


def mu_G(
    S: float,
    mu_0: float,
    S_0: float,
) -> float:
    """Entropy-dependent decay function (Section 5.1).

    mu_G(S) = mu_0 * S / S_0

    Clamped to >= 0 to avoid negative decay.

    Parameters
    ----------
    S:
        Current entropy value.
    mu_0:
        Base decay rate.
    S_0:
        Normalisation entropy scale (> 0).

    Returns
    -------
    Decay rate >= 0.
    """
    return float(max(0.0, mu_0 * S / S_0))


def suppression_term(
    G: np.ndarray,
    theta_R: np.ndarray,
    pi_a: float,
    lambda_s: float,
) -> np.ndarray:
    """Optional suppression term (Section 5.2).

    Penalty = lambda_s * G_e * sin^2(theta_R_e / (2 * pi_a))

    Subtracted from the conductance derivative.

    Parameters
    ----------
    G:
        Complex conductance array, shape (m,).
    theta_R:
        Resolved phase array, shape (m,).
    pi_a:
        Adaptive ruler value (> 0).
    lambda_s:
        Suppression strength (>= 0).

    Returns
    -------
    Suppression contribution, shape (m,) complex.
    """
    if lambda_s == 0.0:
        return np.zeros_like(G)
    factor = np.sin(theta_R / (2.0 * pi_a)) ** 2
    return lambda_s * G * factor


def conductance_step(
    G: np.ndarray,
    I_mag: np.ndarray,
    theta_R: np.ndarray,
    S: float,
    pi_a: float,
    dt: float,
    alpha_0: float,
    S_c: float,
    delta_S: float,
    mu_0: float,
    S_0: float,
    lambda_s: float = 0.0,
    use_suppression: bool = False,
) -> np.ndarray:
    """Discrete Euler update for complex edge conductances (Section 5).

    G_e^+ = G_e + dt * (alpha_G(S) * |I_e| * exp(i*theta_R_e)
                         - mu_G(S) * G_e
                         - suppression)

    Parameters
    ----------
    G:
        Current complex conductances, shape (m,).
    I_mag:
        Current magnitudes |I_e|, shape (m,).
    theta_R:
        Resolved phase per edge, shape (m,).
    S:
        Current entropy value.
    pi_a:
        Current adaptive ruler value.
    dt:
        Time step.
    alpha_0, S_c, delta_S:
        Parameters for alpha_G.
    mu_0, S_0:
        Parameters for mu_G.
    lambda_s:
        Suppression strength (Section 5.2).
    use_suppression:
        Whether to apply the suppression term.

    Returns
    -------
    Updated complex conductances, shape (m,).
    """
    a = alpha_G(S, alpha_0, S_c, delta_S)
    m = mu_G(S, mu_0, S_0)

    reinforce = a * I_mag * np.exp(1j * theta_R)
    decay = m * G

    dG = reinforce - decay
    if use_suppression:
        dG = dG - suppression_term(G, theta_R, pi_a, lambda_s)

    return G + dt * dG


def apply_conductance_clamps(
    G: np.ndarray,
    real_min: float = 1e-6,
    real_max: float | None = None,
    imag_max: float | None = None,
    budget: float | None = None,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply clamps, budgets, and optional noise to conductances (Section 5.3).

    Parameters
    ----------
    G:
        Complex conductances, shape (m,).
    real_min:
        Lower bound for real part.
    real_max:
        Upper bound for real part (None = no bound).
    imag_max:
        Symmetric bound for imaginary part (None = no bound).
    budget:
        Total real conductance budget; if sum exceeds budget, normalise.
    noise_std:
        Standard deviation of complex Gaussian noise to add.
    rng:
        Random number generator (used only when noise_std > 0).

    Returns
    -------
    Clamped / budgeted / noised conductances, shape (m,).
    """
    G = G.copy()

    # Apply real-part lower bound
    re = np.real(G)
    re = np.clip(re, real_min, real_max)
    G = re + 1j * np.imag(G)

    # Apply imaginary-part symmetric bound
    if imag_max is not None:
        im = np.clip(np.imag(G), -imag_max, imag_max)
        G = np.real(G) + 1j * im

    # Budget normalisation (real part)
    if budget is not None:
        total_re = np.sum(np.real(G))
        if total_re > budget:
            scale = budget / total_re
            G = np.real(G) * scale + 1j * np.imag(G)

    # Optional symmetry-breaking noise
    if noise_std > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.standard_normal(G.shape) + 1j * rng.standard_normal(G.shape)
        G = G + noise_std * noise

    return G
