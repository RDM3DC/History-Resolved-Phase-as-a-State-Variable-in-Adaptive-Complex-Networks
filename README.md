# History-Resolved Phase as a State Variable in Adaptive Complex Networks

`hrphasenet` is a Python simulation package for adaptive complex networks whose
edge conductances are governed by a **history-resolved phase** — a lifted,
branch-consistent representation of the complex current phase that tracks
winding numbers and parity across time.  The framework implements the full
theoretical model described in the companion paper, including four ablation
modes that let you isolate the contribution of each component.

---

## Table of Contents

1. [Background](#background)
2. [Installation](#installation)
3. [Package layout](#package-layout)
4. [Key concepts](#key-concepts)
5. [Ablation modes](#ablation-modes)
6. [Quick start](#quick-start)
7. [API overview](#api-overview)
   - [NetworkParams](#networkparams)
   - [AdaptiveNetwork](#adaptivenetwork)
   - [Graph factories](#graph-factories)
   - [Drive protocols](#drive-protocols)
   - [Low-level modules](#low-level-modules)
8. [Documentation](#documentation)
9. [Benchmarks](#benchmarks)
10. [Running the tests](#running-the-tests)
11. [License](#license)

---

## Background

In classical adaptive-network models (e.g. Hebbian learning, Hebb–Oja, Physarum
polycephalum) each edge carries a real-valued conductance that grows with flow
and decays otherwise.  When the current is *complex* the naive choice is to use
the principal-branch angle `arg(I) ∈ (-π, π]`, which introduces artificial
discontinuities every time the phase wraps.

This package replaces the principal-branch angle with a **history-resolved
phase** `θ_R` that:

* accumulates phase increments continuously (no wrap-around jumps),
* tracks the **winding number** `w = round(θ_R / 2π)` and **parity**
  `b = (-1)^w` of each edge,
* clips each step's increment to an **adaptive ruler** `π_a ∈ (0, π]` so that
  spurious jumps during near-zero crossings are suppressed,
* couples to an **entropy proxy** `S` that measures transport dissipation and
  branch-slip events and feeds back into both the ruler and the conductance
  gain/decay rates.

---

## Installation

```bash
# Standard install (runtime dependencies: numpy, scipy)
pip install hrphasenet

# Editable install with test dependencies
pip install -e ".[dev]"
```

Python ≥ 3.9 and NumPy ≥ 1.24 / SciPy ≥ 1.10 are required.

---

## Package layout

```
src/hrphasenet/
├── __init__.py        # public re-exports
├── network.py         # AdaptiveNetwork, NetworkParams, NetworkState
├── phase.py           # wrap, lifted_phase_update, winding_and_parity
├── entropy.py         # entropy_step, ruler_step, transport_dissipation, …
├── conductance.py     # conductance_step, apply_conductance_clamps, …
├── solver.py          # nodal_solve, edge_current, build_admittance_matrix
├── graphs.py          # ring_graph, diamond_graph, path_graph, complete_graph
└── drives.py          # constant_drive, periodic_drive, chirp_drive, reversal_drive
```

---

## Key concepts

| Symbol | Description |
|--------|-------------|
| `G_e ∈ ℂ` | Complex conductance of edge *e* |
| `θ_R` | History-resolved (lifted) phase, unbounded |
| `w` | Winding number: `round(θ_R / 2π)` |
| `b` | Parity: `(-1)^w` |
| `S` | Entropy proxy (transport dissipation + branch-slip events) |
| `π_a` | Adaptive phase ruler half-width (clips incoming increments) |

### History-resolved phase lifting

At each time step the raw current phase `θ_raw = arg(I_e)` is compared to the
previous resolved phase `θ_R`.  The increment `Δ = wrap(θ_raw − θ_R)` is
clipped to `[−π_a, +π_a]` and added to `θ_R`:

```
θ_R ← θ_R + clip(wrap(θ_raw − θ_R), −π_a, +π_a)
```

If `|I_e| < z_min` the update is frozen (near-zero suppression).

### Entropy proxy

```
S⁺ = max(0,  S + dt · [T₁ + T₂ − γ(S − S_eq)])
```

where `T₁ = Σ |I_e|² / Re(G_e)` is transport dissipation and
`T₂ = κ · Σ |Δw_e|` counts branch-slip events.

### Adaptive ruler

```
π_a⁺ = clip(π_a + dt · [α_π · S − μ_π(π_a − π₀)], π_min, π_max)
```

High entropy widens the ruler; low entropy relaxes it back toward `π₀`.

### Conductance update

```
G_e⁺ = G_e + dt · [α_G(S) · |I_e| · exp(i θ_R) − μ_G(S) · G_e − suppression]
```

where `α_G(S) = α₀ · σ((S − S_c) / δ_S)` is an entropy-gated gain and
`μ_G(S) = μ₀ · S / S₀` is an entropy-proportional decay.

---

## Ablation modes

Four modes let you benchmark the contribution of each component:

| Mode | Phase lifting | Adaptive ruler | Entropy gate | Suppression |
|------|:---:|:---:|:---:|:---:|
| `"principal"` | ✗ | ✗ | ✓ | ✗ |
| `"lift_only"` | ✓ | ✗ | ✓ | ✗ |
| `"lift_ruler"` | ✓ | ✓ | ✓ | ✗ |
| `"full"` | ✓ | ✓ | ✓ | ✓ |

---

## Quick start

```python
import numpy as np
from hrphasenet.network import AdaptiveNetwork, NetworkParams
from hrphasenet.graphs import diamond_graph
from hrphasenet.drives import constant_drive, reversal_drive

# --- Build a diamond graph (4 nodes, 4 edges) ---
edges, src, snk = diamond_graph()

params = NetworkParams(
    alpha_0=1.0,
    S_c=0.5,
    delta_S=0.1,
    mu_0=0.5,
    gamma=0.1,
    dt=0.01,
)

net = AdaptiveNetwork(
    edges=edges,
    n_nodes=4,
    source_node=src,
    sink_node=snk,
    params=params,
    mode="full",
    seed=42,
)

# --- Run 500 steps with a constant drive, recording snapshots ---
state = net.run(constant_drive(amplitude=1.0 + 0j, n_steps=500), log=True)

print(f"Final conductances : {state.G}")
print(f"Entropy proxy      : {state.S:.4f}")
print(f"Adaptive ruler     : {state.pi_a:.4f}")
print(f"Winding numbers    : {state.w}")

# --- Reversal protocol (tests history dependence) ---
net.reset()
drive = reversal_drive(
    amplitude_A=1.0 + 0j,
    amplitude_B=-1.0 + 0j,
    n_A=300,
    n_BA=100,
    n_restore=200,
)
state = net.run(drive, log=True)
```

---

## API overview

### NetworkParams

`NetworkParams` is a dataclass holding every hyper-parameter of the simulation.
All fields have sensible defaults so you only need to override what you want.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_0` | `1.0` | Maximum conductance gain |
| `S_c` | `0.5` | Entropy threshold for half-gain |
| `delta_S` | `0.1` | Width of gain sigmoid |
| `mu_0` | `0.5` | Base conductance decay rate |
| `S_0` | `1.0` | Normalisation entropy for decay |
| `gamma` | `0.1` | Entropy relaxation rate |
| `S_eq` | `0.0` | Entropy equilibrium |
| `kappa` | `1.0` | Branch-slip scaling |
| `alpha_pi` | `0.1` | Entropy drive on ruler |
| `mu_pi` | `0.2` | Ruler relaxation rate |
| `pi_0` | `π/2` | Baseline ruler value |
| `pi_min` | `1e-3` | Minimum ruler value |
| `pi_max` | `π` | Maximum ruler value |
| `z_min` | `1e-9` | Phase-freeze threshold |
| `lambda_s` | `0.1` | Suppression strength |
| `dt` | `0.01` | Time step |
| `real_min` | `1e-6` | Lower bound for Re(G) |
| `real_max` | `None` | Upper bound for Re(G) |
| `imag_max` | `None` | Symmetric bound for Im(G) |
| `budget` | `None` | Total Re(G) budget |
| `noise_std` | `0.0` | Conductance noise std dev |

### AdaptiveNetwork

```python
AdaptiveNetwork(
    edges,          # list of (u, v) directed edge pairs (0-indexed)
    n_nodes,        # number of nodes
    source_node,    # injection node index
    sink_node,      # drain node index
    params=None,    # NetworkParams (defaults to NetworkParams())
    mode="full",    # ablation mode
    init_G=None,    # initial conductances (default: 1+0j per edge)
    init_S=None,    # initial entropy (default: params.S_eq)
    init_pi_a=None, # initial ruler (default: params.pi_0)
    seed=None,      # random seed
)
```

Key methods:

| Method | Description |
|--------|-------------|
| `net.step(source_value, log=False)` | Advance one time step; returns `NetworkState` |
| `net.run(drive, log=False)` | Run over an iterable of source values; returns final `NetworkState` |
| `net.reset(init_G, init_S, init_pi_a)` | Reset state (useful for ablation comparisons) |

`NetworkState` fields: `G`, `theta_R`, `w`, `b`, `S`, `pi_a`, `phi`,
`converged`, `t`, `step`, `log`.

### Graph factories

```python
from hrphasenet.graphs import ring_graph, diamond_graph, path_graph, complete_graph

edges, src, snk = ring_graph(n=8)               # 8-node ring
edges, src, snk = ring_graph(n=8, defect=True)  # ring with one additional diagonal shortcut edge (0 → n//2)
edges, src, snk = diamond_graph()               # 4-node two-path graph
edges, src, snk = path_graph(n=5)              # 5-node linear path
edges, src, snk = complete_graph(n=4)          # fully-connected digraph
```

All factories return `(edges, source_node, sink_node)`.

### Drive protocols

```python
from hrphasenet.drives import (
    constant_drive,
    periodic_drive,
    chirp_drive,
    reversal_drive,
)

# Steady injection
drive = constant_drive(amplitude=1+0j, n_steps=1000)

# Sinusoidal phase modulation
drive = periodic_drive(amplitude=1.0, omega=2.0, dt=0.01, n_steps=1000)

# Frequency sweep (triggers repeated branch crossings)
drive = chirp_drive(amplitude=1.0, omega_start=0.5, omega_end=5.0,
                    dt=0.01, n_steps=2000)

# A → B → A reversal (history-dependence test)
drive = reversal_drive(amplitude_A=1+0j, amplitude_B=-1+0j,
                       n_A=300, n_BA=100, n_restore=200)
```

All generators are lazy (Python `Generator`) and can be passed directly to
`net.run(drive)`.

### Low-level modules

The following modules are available for direct use or extension:

| Module | Key exports |
|--------|------------|
| `hrphasenet.phase` | `wrap`, `lifted_phase_update`, `winding_and_parity` |
| `hrphasenet.entropy` | `entropy_step`, `ruler_step`, `transport_dissipation`, `branch_slip_term`, `EntropyRulerState` |
| `hrphasenet.conductance` | `conductance_step`, `apply_conductance_clamps`, `alpha_G`, `mu_G`, `suppression_term` |
| `hrphasenet.solver` | `nodal_solve`, `edge_current`, `build_admittance_matrix` |

---

## Documentation

The `docs/` directory contains the full research documentation:

| Document | Description |
|----------|-------------|
| [White Paper](docs/white_paper.md) | Comprehensive 13-section white paper covering the full theory, empirical results, safe claims, and next steps |
| [Derivation](docs/derivation.md) | Step-by-step derivation of the history-resolved phase conductance update equation |
| [Benchmark Report](docs/benchmark_report.md) | Summary of the 9-benchmark validation suite results |
| [Parity Lock Table](docs/parity_lock_deformation_table.md) | Parity-lock deformation data under varying drive amplitudes |
| [TopEquations Submission](docs/topequations_submission.md) | Full submission record for the TopEquations leaderboard (score: 102) |

---

## Benchmarks

The `benchmarks/` directory contains the validation infrastructure:

```bash
# Run the full 9-benchmark suite
python benchmarks/run_benchmarks.py
```

Benchmarks included:

| Benchmark | What it tests |
|-----------|---------------|
| **monodromy** | Parity-lock deformation under varying amplitudes |
| **freeze_near_zero** | Phase freeze for sub-threshold currents |
| **boundedness** | Conductance stays bounded (Re(G) > 0) |
| **ablation_modes** | Score ordering: full > lift_ruler > lift_only > principal |
| **history_divergence** | Reversal protocol divergence from principal branch |
| **matched_present_state_separation** | Identical present states diverge when history differs |
| **operational_memory_gap** | Memory gap between full and principal modes |
| **memory_threshold_sweep** | Memory retention across ruler-width sweep |
| **memory_onset_phase_diagram** | Phase diagram of memory onset in (α₀, κ) space |

Pre-computed results are in `benchmarks/benchmark_report.json`.

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The test suite (≈ 118 tests) runs in roughly 1–2 seconds and covers all
modules: phase lifting, entropy dynamics, conductance updates, nodal solver,
graph factories, drive generators, and the full `AdaptiveNetwork` integration.

---

## License

See [LICENSE](LICENSE) for details.