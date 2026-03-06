"""
AdaptiveNetwork: main simulation class for the history-resolved phase model
(Appendix A recipe and Section 7 ablation families).

NetworkParams: dataclass holding all model hyper-parameters.

Ablation modes
--------------
  "principal"   - principal-branch baseline (no lifting, fixed ruler)
  "lift_only"   - history-resolved phase, fixed ruler
  "lift_ruler"  - history-resolved phase + adaptive ruler
  "full"        - all features: lifted phase, entropy gate, ruler, suppression
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .conductance import apply_conductance_clamps, conductance_step
from .entropy import (
    EntropyRulerState,
    branch_slip_term,
    entropy_step,
    ruler_step,
    transport_dissipation,
)
from .phase import lifted_phase_update, winding_and_parity, wrap
from .solver import edge_current, nodal_solve

AblationMode = Literal["principal", "lift_only", "lift_ruler", "full"]


@dataclass
class NetworkParams:
    """All hyper-parameters for the adaptive network simulation.

    Conductance dynamics
    --------------------
    alpha_0:   maximum gain amplitude
    S_c:       entropy threshold for half-gain activation
    delta_S:   width of the activation sigmoid
    mu_0:      base decay rate scale
    S_0:       normalisation entropy for decay

    Entropy / ruler
    ---------------
    gamma:     entropy relaxation rate
    S_eq:      entropy equilibrium value
    kappa:     branch-slip scaling coefficient
    alpha_pi:  entropy drive on ruler
    mu_pi:     ruler relaxation rate
    pi_0:      baseline ruler value
    pi_min:    minimum ruler value
    pi_max:    maximum ruler value

    Phase / suppression
    -------------------
    z_min:      near-zero magnitude threshold for phase freeze
    lambda_s:   suppression strength

    Simulation
    ----------
    dt:         time step
    real_min:   lower bound for Re(G)
    real_max:   upper bound for Re(G) (None = unbounded)
    imag_max:   symmetric bound for Im(G) (None = unbounded)
    budget:     total Re(G) budget (None = no budget)
    noise_std:  conductance noise standard deviation
    """

    # Conductance
    alpha_0: float = 1.0
    S_c: float = 0.5
    delta_S: float = 0.1
    mu_0: float = 0.5
    S_0: float = 1.0

    # Entropy / ruler
    gamma: float = 0.1
    S_eq: float = 0.0
    kappa: float = 1.0
    alpha_pi: float = 0.1
    mu_pi: float = 0.2
    pi_0: float = np.pi / 2.0
    pi_min: float = 1e-3
    pi_max: float = np.pi

    # Phase / suppression
    z_min: float = 1e-9
    lambda_s: float = 0.1

    # Simulation
    dt: float = 0.01
    real_min: float = 1e-6
    real_max: float | None = None
    imag_max: float | None = None
    budget: float | None = None
    noise_std: float = 0.0


@dataclass
class NetworkState:
    """Mutable simulation state for all edges and scalars.

    Attributes
    ----------
    G:         complex conductances, shape (m,)
    theta_R:   resolved phase, shape (m,)
    w:         winding numbers, shape (m,) int
    b:         parity, shape (m,) int  {-1, +1}
    S:         entropy proxy scalar
    pi_a:      adaptive phase ruler scalar
    phi:       node potentials from last solve, shape (n_nodes,)
    converged: whether last linear solve converged
    t:         current simulation time
    step:      step counter
    log:       list of per-step snapshot dicts (if logging enabled)
    """

    G: np.ndarray
    theta_R: np.ndarray
    w: np.ndarray
    b: np.ndarray
    S: float
    pi_a: float
    phi: np.ndarray
    converged: bool = True
    t: float = 0.0
    step: int = 0
    log: list[dict] = field(default_factory=list)


class AdaptiveNetwork:
    """Adaptive complex network with history-resolved phase.

    Parameters
    ----------
    edges:
        List of (u, v) directed edge pairs (0-indexed).
    n_nodes:
        Number of nodes.
    source_node:
        Index of the injection node.
    sink_node:
        Index of the drain node.
    params:
        Hyper-parameters.  Defaults to NetworkParams() if None.
    mode:
        Ablation mode:
          "principal"  - no phase lifting, no adaptive ruler
          "lift_only"  - phase lifting, fixed ruler
          "lift_ruler" - phase lifting + dynamic ruler
          "full"       - all features
    init_G:
        Initial conductances.  Defaults to 1+0j per edge.
    init_S:
        Initial entropy.  Defaults to params.S_eq.
    init_pi_a:
        Initial ruler.  Defaults to params.pi_0.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        edges: list[tuple[int, int]],
        n_nodes: int,
        source_node: int,
        sink_node: int,
        params: NetworkParams | None = None,
        mode: AblationMode = "full",
        init_G: np.ndarray | None = None,
        init_S: float | None = None,
        init_pi_a: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.edges = list(edges)
        self.n_nodes = n_nodes
        self.source_node = source_node
        self.sink_node = sink_node
        self.params = params or NetworkParams()
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        m = len(self.edges)

        G0 = (
            np.ones(m, dtype=complex)
            if init_G is None
            else np.asarray(init_G, dtype=complex).copy()
        )
        S0 = self.params.S_eq if init_S is None else float(init_S)
        pi0 = self.params.pi_0 if init_pi_a is None else float(init_pi_a)

        # Initialise resolved phase from initial conductance argument
        theta_R0 = np.angle(G0)
        w0, b0 = winding_and_parity(theta_R0)

        self.state = NetworkState(
            G=G0,
            theta_R=theta_R0,
            w=np.asarray(w0, dtype=int),
            b=np.asarray(b0, dtype=int),
            S=S0,
            pi_a=pi0,
            phi=np.zeros(n_nodes, dtype=complex),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(
        self,
        source_value: complex = 1.0 + 0j,
        log: bool = False,
    ) -> NetworkState:
        """Advance the network by one time step.

        Follows Appendix A recipe:
          3. Solve node potentials.
          4. Compute edge currents and raw phase.
          5. Update resolved phase.
          6. Update winding and parity.
          7. Update entropy.
          8. Update ruler.
          9. Update conductances.
         10. Apply clamps, budgets, log.

        Parameters
        ----------
        source_value:
            Complex current injected at source_node this step.
        log:
            If True, append a snapshot dict to state.log.

        Returns
        -------
        Updated NetworkState (modified in-place, also returned).
        """
        p = self.params
        s = self.state

        # --- 3. Nodal solve ---
        phi, converged = nodal_solve(
            self.edges,
            s.G,
            self.n_nodes,
            self.source_node,
            self.sink_node,
            source_value=source_value,
        )
        s.phi = phi
        s.converged = converged

        # --- 4. Edge currents and raw phase ---
        I = edge_current(self.edges, s.G, phi)
        I_mag = np.abs(I)
        theta_raw = np.angle(I)

        # --- 5. Update resolved phase ---
        w_prev = s.w.copy()

        if self.mode == "principal":
            # Principal-branch baseline: use raw angle directly
            theta_R_new = theta_raw.copy()
        else:
            # History-resolved lifting (modes: lift_only, lift_ruler, full)
            pi_a_for_clip = s.pi_a  # adaptive for lift_ruler/full, fixed for lift_only
            theta_R_new = lifted_phase_update(
                s.theta_R,
                theta_raw,
                pi_a=pi_a_for_clip,
                z_mag=I_mag,
                z_min=p.z_min,
            )

        s.theta_R = np.asarray(theta_R_new, dtype=float)

        # --- 6. Winding and parity ---
        w_new, b_new = winding_and_parity(s.theta_R)
        s.w = np.asarray(w_new, dtype=int)
        s.b = np.asarray(b_new, dtype=int)
        delta_w = s.w - w_prev

        # --- 7. Entropy update ---
        if self.mode in ("lift_ruler", "full", "lift_only"):
            T1 = transport_dissipation(I, s.G)
            T2 = branch_slip_term(delta_w, kappa=p.kappa)
        else:
            T1 = transport_dissipation(I, s.G)
            T2 = 0.0

        s.S = entropy_step(s.S, T1, T2, p.gamma, p.S_eq, p.dt)

        # --- 8. Ruler update ---
        if self.mode in ("lift_ruler", "full"):
            s.pi_a = ruler_step(
                s.pi_a,
                s.S,
                p.alpha_pi,
                p.mu_pi,
                p.pi_0,
                p.pi_min,
                p.pi_max,
                p.dt,
            )
        # For principal / lift_only: ruler stays fixed at pi_0

        # --- 9. Conductance update ---
        use_suppression = self.mode == "full"
        s.G = conductance_step(
            s.G,
            I_mag,
            s.theta_R,
            s.S,
            s.pi_a,
            p.dt,
            p.alpha_0,
            p.S_c,
            p.delta_S,
            p.mu_0,
            p.S_0,
            lambda_s=p.lambda_s,
            use_suppression=use_suppression,
        )

        # --- 10. Clamps, budgets, noise ---
        s.G = apply_conductance_clamps(
            s.G,
            real_min=p.real_min,
            real_max=p.real_max,
            imag_max=p.imag_max,
            budget=p.budget,
            noise_std=p.noise_std,
            rng=self.rng,
        )

        # Advance time
        s.t += p.dt
        s.step += 1

        # --- Logging ---
        if log:
            s.log.append(self._snapshot(source_value, I, theta_raw, delta_w))

        return s

    def run(
        self,
        drive,
        log: bool = False,
    ) -> NetworkState:
        """Run the network over a sequence of drive values.

        Parameters
        ----------
        drive:
            Iterable of complex source values (one per step).
        log:
            Whether to record per-step snapshots in state.log.

        Returns
        -------
        Final NetworkState.
        """
        for source_value in drive:
            self.step(source_value, log=log)
        return self.state

    def reset(
        self,
        init_G: np.ndarray | None = None,
        init_S: float | None = None,
        init_pi_a: float | None = None,
    ) -> None:
        """Reset the network state (useful for ablation comparisons).

        Parameters
        ----------
        init_G:
            New initial conductances.  Keeps current if None.
        init_S:
            New initial entropy.  Resets to params.S_eq if None.
        init_pi_a:
            New initial ruler.  Resets to params.pi_0 if None.
        """
        p = self.params
        m = len(self.edges)

        G0 = (
            self.state.G.copy()
            if init_G is None
            else np.asarray(init_G, dtype=complex).copy()
        )
        S0 = p.S_eq if init_S is None else float(init_S)
        pi0 = p.pi_0 if init_pi_a is None else float(init_pi_a)

        theta_R0 = np.angle(G0)
        w0, b0 = winding_and_parity(theta_R0)

        self.state = NetworkState(
            G=G0,
            theta_R=theta_R0,
            w=np.asarray(w0, dtype=int),
            b=np.asarray(b0, dtype=int),
            S=S0,
            pi_a=pi0,
            phi=np.zeros(self.n_nodes, dtype=complex),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _snapshot(
        self,
        source_value: complex,
        I: np.ndarray,
        theta_raw: np.ndarray,
        delta_w: np.ndarray,
    ) -> dict:
        s = self.state
        return {
            "step": s.step,
            "t": s.t,
            "source": source_value,
            "G": s.G.copy(),
            "theta_R": s.theta_R.copy(),
            "theta_raw": theta_raw.copy(),
            "w": s.w.copy(),
            "b": s.b.copy(),
            "delta_w": delta_w.copy(),
            "S": s.S,
            "pi_a": s.pi_a,
            "I": I.copy(),
            "converged": s.converged,
        }
