"""
History-resolved phase lifting module.

Implements:
  - wrap: wraps an angle increment to (-pi, pi]
  - lifted_phase_update: branch-consistent lifted phase evolution
  - winding_and_parity: winding number and parity from resolved phase
"""

from __future__ import annotations

import numpy as np


def wrap(delta: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to the half-open interval (-pi, pi].

    Uses the identity: wrap(x) = ((x + pi) mod 2*pi) - pi, then
    maps exactly -pi to +pi so the interval is (-pi, pi].

    Parameters
    ----------
    delta:
        Angle increment in radians (scalar or array).

    Returns
    -------
    Wrapped angle in (-pi, pi].
    """
    result = (np.asarray(delta) + np.pi) % (2.0 * np.pi) - np.pi
    # Move the -pi boundary to +pi
    result = np.where(result == -np.pi, np.pi, result)
    if np.ndim(delta) == 0:
        return float(result)
    return result


def lifted_phase_update(
    theta_ref: float | np.ndarray,
    theta_raw: float | np.ndarray,
    pi_a: float,
    z_mag: float | np.ndarray | None = None,
    z_min: float = 1e-9,
) -> float | np.ndarray:
    """Compute the history-resolved (lifted) phase update.

    Algorithm (Section 4.1 of the paper):
      1. Compute wrapped increment  delta = wrap(theta_raw - theta_ref)
      2. Clip increment to [-pi_a, +pi_a] (adaptive ruler)
      3. New resolved phase = theta_ref + clipped delta
      4. If |z| < z_min, freeze the update (return theta_ref unchanged)

    Parameters
    ----------
    theta_ref:
        Prior resolved phase (scalar or array), in radians.
    theta_raw:
        Raw (principal-branch) phase from the current signal, radians.
    pi_a:
        Adaptive phase ruler half-width (> 0), radians.
    z_mag:
        Magnitude of the signal producing theta_raw.  When provided,
        updates are frozen wherever |z| < z_min.  Pass None to skip.
    z_min:
        Magnitude threshold below which the phase update is frozen.

    Returns
    -------
    New resolved phase (same shape as theta_ref / theta_raw).
    """
    theta_ref = np.asarray(theta_ref, dtype=float)
    theta_raw = np.asarray(theta_raw, dtype=float)

    delta = wrap(theta_raw - theta_ref)
    delta_clip = np.clip(delta, -pi_a, pi_a)
    theta_new = theta_ref + delta_clip

    if z_mag is not None:
        z_mag = np.asarray(z_mag, dtype=float)
        frozen = z_mag < z_min
        theta_new = np.where(frozen, theta_ref, theta_new)

    if theta_new.ndim == 0:
        return float(theta_new)
    return theta_new


def winding_and_parity(
    theta_R: float | np.ndarray,
) -> tuple[int | np.ndarray, int | np.ndarray]:
    """Compute winding number and parity from resolved phase.

    Definitions (Section 4.2):
      w = round(theta_R / (2*pi))
      b = (-1)^w  (parity: +1 for even winding, -1 for odd)

    Parameters
    ----------
    theta_R:
        Resolved phase in radians (scalar or array).

    Returns
    -------
    (w, b):
        w: winding number (int or integer array),
        b: parity in {-1, +1} (int or integer array).
    """
    theta_R = np.asarray(theta_R, dtype=float)
    w = np.round(theta_R / (2.0 * np.pi)).astype(int)
    b = np.where(w % 2 == 0, 1, -1).astype(int)
    if theta_R.ndim == 0:
        return int(w), int(b)
    return w, b
