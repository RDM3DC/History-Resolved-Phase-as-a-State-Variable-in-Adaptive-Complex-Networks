"""
hrphasenet: History-Resolved Phase as a State Variable in Adaptive Complex Networks.

Implements the adaptive complex-network model from the paper:
  - History-resolved phase lifting (branch-consistent updates)
  - Entropy-gated adaptive conductance dynamics
  - Adaptive phase ruler controlling per-step phase change
  - Winding number and parity observables
  - Ablation families: principal-branch, lift-only, lift+ruler, full model
"""

from .phase import wrap, lifted_phase_update, winding_and_parity
from .entropy import EntropyRulerState, entropy_step, ruler_step
from .conductance import alpha_G, mu_G, conductance_step
from .solver import nodal_solve, edge_current
from .network import AdaptiveNetwork, NetworkParams
from .graphs import ring_graph, diamond_graph, path_graph
from .drives import (
    constant_drive,
    periodic_drive,
    chirp_drive,
    reversal_drive,
)

__all__ = [
    "wrap",
    "lifted_phase_update",
    "winding_and_parity",
    "EntropyRulerState",
    "entropy_step",
    "ruler_step",
    "alpha_G",
    "mu_G",
    "conductance_step",
    "nodal_solve",
    "edge_current",
    "AdaptiveNetwork",
    "NetworkParams",
    "ring_graph",
    "diamond_graph",
    "path_graph",
    "constant_drive",
    "periodic_drive",
    "chirp_drive",
    "reversal_drive",
]
