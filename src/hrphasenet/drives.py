"""
Drive protocol generators (Section 10.2).

Each generator yields a sequence of complex source values for a given
number of steps.

constant_drive  - steady source-sink amplitude
periodic_drive  - sinusoidally modulated amplitude or phase
chirp_drive     - frequency-swept phase signal (repeated branch crossings)
reversal_drive  - A→B, B→A, restore A→B protocol
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np


def constant_drive(
    amplitude: complex,
    n_steps: int,
) -> Generator[complex, None, None]:
    """Constant source-sink drive.

    Yields *amplitude* for *n_steps* steps.

    Parameters
    ----------
    amplitude:
        Complex source value (e.g. 1.0+0j for real unit injection).
    n_steps:
        Number of time steps.
    """
    for _ in range(n_steps):
        yield amplitude


def periodic_drive(
    amplitude: float,
    omega: float,
    dt: float,
    n_steps: int,
    phase_offset: float = 0.0,
) -> Generator[complex, None, None]:
    """Periodically modulated complex drive.

    source(t) = amplitude * exp(i * (omega * t + phase_offset))

    Parameters
    ----------
    amplitude:
        Real amplitude (>= 0).
    omega:
        Angular frequency (rad / time_unit).
    dt:
        Time step.
    n_steps:
        Number of steps.
    phase_offset:
        Initial phase offset in radians.
    """
    for k in range(n_steps):
        t = k * dt
        yield amplitude * np.exp(1j * (omega * t + phase_offset))


def chirp_drive(
    amplitude: float,
    omega_start: float,
    omega_end: float,
    dt: float,
    n_steps: int,
) -> Generator[complex, None, None]:
    """Linear frequency-sweep (chirp) drive for repeated branch crossings.

    Instantaneous frequency: omega(t) = omega_start + (omega_end - omega_start) * t / T
    Source value: amplitude * exp(i * integrated_phase(t))

    Parameters
    ----------
    amplitude:
        Real amplitude (>= 0).
    omega_start:
        Starting angular frequency.
    omega_end:
        Ending angular frequency.
    dt:
        Time step.
    n_steps:
        Number of steps.
    """
    T = (n_steps - 1) * dt
    for k in range(n_steps):
        t = k * dt
        if T > 0:
            integrated_phase = omega_start * t + 0.5 * (omega_end - omega_start) * t**2 / T
        else:
            integrated_phase = omega_start * t
        yield amplitude * np.exp(1j * integrated_phase)


def reversal_drive(
    amplitude_A: complex,
    amplitude_B: complex,
    n_A: int,
    n_BA: int,
    n_restore: int,
) -> Generator[complex, None, None]:
    """Reversal protocol: A for n_A steps, B for n_BA steps, restore A.

    This is the key protocol for testing history dependence (Section 10.4):
    1. Drive A for n_A steps.
    2. Drive B for n_BA steps (reversed or altered).
    3. Restore drive A for n_restore steps.

    Parameters
    ----------
    amplitude_A:
        Complex source for the A phase.
    amplitude_B:
        Complex source for the B (reversed/altered) phase.
    n_A:
        Steps for initial A drive.
    n_BA:
        Steps for B drive.
    n_restore:
        Steps for final A restore.
    """
    for _ in range(n_A):
        yield amplitude_A
    for _ in range(n_BA):
        yield amplitude_B
    for _ in range(n_restore):
        yield amplitude_A
