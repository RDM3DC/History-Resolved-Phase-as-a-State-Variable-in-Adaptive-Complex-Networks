"""
GPU-accelerated History-Resolved Phase benchmark suite.

Runs the full hrphasenet model on GPU(s) via PyTorch tensors,
scaling experiments that are infeasible on CPU:

  1. Large onset-map sweep (configurable grid, default 100x100)
  2. Large-network matched-present divergence (configurable N, default 500)
  3. Parallel ablation comparison across all four modes

Requires:  torch (with CUDA), numpy, hrphasenet (editable install)
Hardware:  tested on dual RTX 3090 Ti (24 GB each)

Usage:
    python benchmarks/hr_phase_gpu_benchmark.py
    python benchmarks/hr_phase_gpu_benchmark.py --onset-grid 500 --network-size 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# GPU-native phase / entropy / conductance kernels
# ---------------------------------------------------------------------------

def _wrap_t(delta: torch.Tensor) -> torch.Tensor:
    """wrap angle to (-pi, pi] on GPU."""
    result = (delta + math.pi) % (2.0 * math.pi) - math.pi
    result[result == -math.pi] = math.pi
    return result


def _lifted_phase_update_t(
    theta_ref: torch.Tensor,
    theta_raw: torch.Tensor,
    pi_a: float,
    z_mag: torch.Tensor | None = None,
    z_min: float = 1e-9,
) -> torch.Tensor:
    delta = _wrap_t(theta_raw - theta_ref)
    delta_clip = delta.clamp(-pi_a, pi_a)
    theta_new = theta_ref + delta_clip
    if z_mag is not None:
        frozen = z_mag < z_min
        theta_new = torch.where(frozen, theta_ref, theta_new)
    return theta_new


def _winding_parity_t(theta_R: torch.Tensor):
    w = torch.round(theta_R / (2.0 * math.pi)).to(torch.int64)
    b = torch.where(w % 2 == 0, 1, -1).to(torch.int64)
    return w, b


def _suppression_t(G: torch.Tensor, theta_R: torch.Tensor, pi_a: float, lambda_s: float) -> torch.Tensor:
    if lambda_s == 0.0:
        return torch.zeros_like(G)
    return lambda_s * G * torch.sin(theta_R / (2.0 * pi_a)) ** 2


# ---------------------------------------------------------------------------
# GPU graph helpers
# ---------------------------------------------------------------------------

@dataclass
class GPUNetParams:
    alpha_0: float = 1.0
    S_c: float = 0.1
    delta_S: float = 0.1
    mu_0: float = 0.2
    S_0: float = 1.0
    S_eq: float = 0.0
    gamma: float = 0.1
    kappa: float = 1.0
    alpha_pi: float = 0.3
    mu_pi: float = 0.1
    pi_0: float = math.pi / 2.0
    pi_min: float = 1e-3
    pi_max: float = math.pi
    z_min: float = 1e-9
    lambda_s: float = 0.1
    dt: float = 0.01
    real_min: float = 1e-6


def _diamond_edges(device: torch.device):
    """Diamond (two-path) graph: 0→1→3, 0→2→3. 4 nodes, 4 edges."""
    src = torch.tensor([0, 1, 0, 2], device=device, dtype=torch.long)
    dst = torch.tensor([1, 3, 2, 3], device=device, dtype=torch.long)
    return src, dst, 0, 3, 4  # src, dst, source, sink, n_nodes


def _wide_diamond_edges(n_paths: int, device: torch.device):
    """Wide diamond: source=0, sink=n_paths+1, n_paths intermediate nodes.

    Topology:  0 → i → n_paths+1  for i in 1..n_paths
    Total: n_paths+2 nodes, 2*n_paths edges.
    """
    n_nodes = n_paths + 2
    ints = torch.arange(1, n_paths + 1, device=device, dtype=torch.long)
    src = torch.cat([torch.zeros_like(ints), ints])
    dst = torch.cat([ints, torch.full_like(ints, n_nodes - 1)])
    return src, dst, 0, n_nodes - 1, n_nodes


def _sigmoid_t(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x.clamp(-500, 500))


class GPUNetwork:
    """Vectorised adaptive network running entirely on a single CUDA device."""

    def __init__(self, n_nodes: int, device: torch.device, params: GPUNetParams | None = None,
                 mode: str = "full", seed: int = 99,
                 edges: tuple[torch.Tensor, torch.Tensor] | None = None,
                 source: int | None = None, sink: int | None = None):
        self.p = params or GPUNetParams()
        self.mode = mode
        self.device = device
        self.n_nodes = n_nodes

        if edges is not None:
            self.src_idx, self.dst_idx = edges
            self.source = source if source is not None else 0
            self.sink = sink if sink is not None else n_nodes - 1
        else:
            # Default: diamond graph (matches CPU benchmark)
            src, dst, self.source, self.sink, n = _diamond_edges(device)
            self.src_idx = src
            self.dst_idx = dst
            self.n_nodes = n
        self.m = len(self.src_idx)

        torch.manual_seed(seed)
        self.G = torch.ones(self.m, dtype=torch.complex128, device=device)
        self.theta_R = torch.zeros(self.m, dtype=torch.float64, device=device)
        self.w = torch.zeros(self.m, dtype=torch.int64, device=device)
        self.b = torch.ones(self.m, dtype=torch.int64, device=device)
        self.S = self.p.S_eq
        self.pi_a = self.p.pi_0

        # Precompute solver helpers
        self._diag_range = torch.arange(self.n_nodes, device=device)
        self._reg = 1e-12 * torch.eye(self.n_nodes, dtype=torch.complex128, device=device)

    def _solve_and_current(self, source_value: complex):
        """Direct solve for node potentials using current injection (matches CPU solver)."""
        n = self.n_nodes

        # Build admittance matrix Y
        Y = torch.zeros(n, n, dtype=torch.complex128, device=self.device)

        # Diagonal: Y[i,i] = sum of G for edges touching node i
        diag = torch.zeros(n, dtype=torch.complex128, device=self.device)
        diag.scatter_add_(0, self.src_idx, self.G)
        diag.scatter_add_(0, self.dst_idx, self.G)
        Y[self._diag_range, self._diag_range] = diag

        # Off-diagonal: Y[u,v] -= G_e and Y[v,u] -= G_e
        Y[self.src_idx, self.dst_idx] -= self.G
        Y[self.dst_idx, self.src_idx] -= self.G

        # Current injection RHS: inject at source, drain at sink
        sv = torch.tensor(source_value, dtype=torch.complex128, device=self.device)
        b = torch.zeros(n, dtype=torch.complex128, device=self.device)
        b[self.source] = sv
        b[self.sink] = -sv

        # Ground the sink node: zero row+column, diagonal=1, b=0
        Y[self.sink, :] = 0.0
        Y[:, self.sink] = 0.0
        Y[self.sink, self.sink] = 1.0
        b[self.sink] = 0.0

        # Regularise and direct-solve
        phi = torch.linalg.solve(Y + self._reg, b)

        # Edge currents: I_e = G_e * (phi_u - phi_v)
        I = self.G * (phi[self.src_idx] - phi[self.dst_idx])
        return phi, I

    def step(self, source_value: complex = 1.0 + 0j):
        p = self.p
        phi, I = self._solve_and_current(source_value)
        I_mag = I.abs()
        theta_raw = I.angle()

        w_prev = self.w.clone()

        # Phase update
        if self.mode == "principal":
            self.theta_R = theta_raw.to(torch.float64)
        else:
            self.theta_R = _lifted_phase_update_t(
                self.theta_R, theta_raw.to(torch.float64),
                pi_a=self.pi_a, z_mag=I_mag.to(torch.float64), z_min=p.z_min,
            )

        self.w, self.b = _winding_parity_t(self.theta_R)
        delta_w = self.w - w_prev

        # Entropy
        re_G = self.G.real
        denom = torch.where(re_G > 0, re_G, self.G.abs())
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        T1 = float((I_mag ** 2 / denom).sum())
        T2 = float(p.kappa * delta_w.abs().sum())
        if self.mode == "principal":
            T2 = 0.0
        dS = T1 + T2 - p.gamma * (self.S - p.S_eq)
        self.S = max(0.0, self.S + p.dt * dS)

        # Ruler
        if self.mode in ("lift_ruler", "full"):
            d_pi = p.alpha_pi * self.S - p.mu_pi * (self.pi_a - p.pi_0)
            self.pi_a = float(np.clip(self.pi_a + p.dt * d_pi, p.pi_min, p.pi_max))

        # Conductance
        a = float(p.alpha_0 * torch.sigmoid(torch.tensor((self.S - p.S_c) / p.delta_S)))
        mu = max(0.0, p.mu_0 * self.S / p.S_0)
        reinforce = a * I_mag.to(torch.complex128) * torch.exp(1j * self.theta_R.to(torch.complex128))
        decay = mu * self.G
        dG = reinforce - decay
        if self.mode == "full":
            dG = dG - _suppression_t(self.G, self.theta_R, self.pi_a, p.lambda_s).to(torch.complex128)
        self.G = self.G + p.dt * dG
        # Clamp real part
        re = self.G.real.clamp(min=p.real_min)
        self.G = re + 1j * self.G.imag

    def run_drive(self, drive_values):
        for v in drive_values:
            self.step(v)


def _constant_drive(amplitude, n_steps):
    return [amplitude] * n_steps


def _chirp_drive(amplitude, omega_start, omega_end, dt, n_steps):
    T = (n_steps - 1) * dt
    out = []
    for k in range(n_steps):
        t = k * dt
        if T > 0:
            phase = omega_start * t + 0.5 * (omega_end - omega_start) * t ** 2 / T
        else:
            phase = omega_start * t
        out.append(amplitude * np.exp(1j * phase))
    return out


# ---------------------------------------------------------------------------
# Batched diamond network for onset-map sweep
# ---------------------------------------------------------------------------

class BatchedDiamondNetwork:
    """B independent 4-node diamond networks on GPU, fully batched.

    All B networks share the diamond topology but may have different pi_0.
    Solves all B systems simultaneously via torch.linalg.solve on (B,4,4).
    """

    # Diamond: edges (0,1), (1,3), (0,2), (2,3)
    _SRC = [0, 1, 0, 2]
    _DST = [1, 3, 2, 3]

    def __init__(self, B: int, device: torch.device, mode: str,
                 pi_0_values: torch.Tensor, seed: int = 99):
        p = GPUNetParams()
        self.B = B
        self.device = device
        self.mode = mode
        self.n = 4
        self.m = 4

        self.src_idx = torch.tensor(self._SRC, device=device, dtype=torch.long)
        self.dst_idx = torch.tensor(self._DST, device=device, dtype=torch.long)

        torch.manual_seed(seed)
        self.G = torch.ones(B, self.m, dtype=torch.complex128, device=device)
        self.theta_R = torch.zeros(B, self.m, dtype=torch.float64, device=device)
        self.w = torch.zeros(B, self.m, dtype=torch.int64, device=device)
        self.b_parity = torch.ones(B, self.m, dtype=torch.int64, device=device)
        self.S = torch.full((B,), p.S_eq, dtype=torch.float64, device=device)
        self.pi_a = pi_0_values.clone().to(dtype=torch.float64, device=device)
        self.pi_0 = pi_0_values.clone().to(dtype=torch.float64, device=device)

        # Scalar params
        self.alpha_0 = p.alpha_0
        self.S_c = p.S_c
        self.delta_S = p.delta_S
        self.mu_0 = p.mu_0
        self.S_0 = p.S_0
        self.gamma = p.gamma
        self.S_eq = p.S_eq
        self.kappa = p.kappa
        self.alpha_pi = p.alpha_pi
        self.mu_pi = p.mu_pi
        self.pi_min = p.pi_min
        self.pi_max = p.pi_max
        self.z_min = p.z_min
        self.lambda_s = p.lambda_s
        self.dt = p.dt
        self.real_min = p.real_min

        self._reg = 1e-12 * torch.eye(4, dtype=torch.complex128, device=device)

    def step(self, drive_values: torch.Tensor):
        """Advance all B networks by one time step.

        drive_values: (B,) complex tensor.
        """
        B, n, m = self.B, self.n, self.m

        # --- Batched admittance matrix build ---
        Y = torch.zeros(B, n, n, dtype=torch.complex128, device=self.device)
        for k in range(m):
            i, j = self._SRC[k], self._DST[k]
            Gk = self.G[:, k]
            Y[:, i, i] += Gk
            Y[:, j, j] += Gk
            Y[:, i, j] -= Gk
            Y[:, j, i] -= Gk

        # Current injection b vector
        b = torch.zeros(B, n, dtype=torch.complex128, device=self.device)
        b[:, 0] = drive_values        # inject at source (node 0)
        b[:, 3] = -drive_values       # drain at sink (node 3)

        # Ground sink: zero row/col 3, diagonal=1
        Y[:, 3, :] = 0.0
        Y[:, :, 3] = 0.0
        Y[:, 3, 3] = 1.0
        b[:, 3] = 0.0

        # Batched solve
        phi = torch.linalg.solve(Y + self._reg.unsqueeze(0), b)

        # Edge currents (B, m)
        I = self.G * (phi[:, self.src_idx] - phi[:, self.dst_idx])
        I_mag = I.abs()
        theta_raw = I.angle().to(torch.float64)

        w_prev = self.w.clone()

        # --- Phase update ---
        if self.mode == "principal":
            self.theta_R = theta_raw
        else:
            delta = _wrap_t(theta_raw - self.theta_R)
            pi_b = self.pi_a.unsqueeze(1)  # (B, 1)
            delta_clip = delta.clamp(-pi_b, pi_b)
            theta_new = self.theta_R + delta_clip
            frozen = I_mag < self.z_min
            self.theta_R = torch.where(frozen, self.theta_R, theta_new)

        # --- Winding / parity ---
        self.w = torch.round(self.theta_R / (2 * math.pi)).to(torch.int64)
        self.b_parity = torch.where(self.w % 2 == 0, 1, -1).to(torch.int64)
        delta_w = self.w - w_prev

        # --- Entropy ---
        re_G = self.G.real
        denom = torch.where(re_G > 0, re_G, self.G.abs())
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        T1 = (I_mag ** 2 / denom).sum(dim=1)
        T2 = (self.kappa * delta_w.abs().to(torch.float64)).sum(dim=1)
        if self.mode == "principal":
            T2 = torch.zeros_like(T2)
        dS = T1 + T2 - self.gamma * (self.S - self.S_eq)
        self.S = (self.S + self.dt * dS).clamp(min=0.0)

        # --- Ruler ---
        if self.mode in ("lift_ruler", "full"):
            d_pi = self.alpha_pi * self.S - self.mu_pi * (self.pi_a - self.pi_0)
            self.pi_a = (self.pi_a + self.dt * d_pi).clamp(self.pi_min, self.pi_max)

        # --- Conductance ---
        sig_arg = ((self.S - self.S_c) / self.delta_S).clamp(-500, 500)
        a = (self.alpha_0 * torch.sigmoid(sig_arg)).unsqueeze(1)  # (B, 1)
        mu = (self.mu_0 * self.S / self.S_0).clamp(min=0.0).unsqueeze(1)  # (B, 1)

        reinforce = a * I_mag.to(torch.complex128) * torch.exp(1j * self.theta_R.to(torch.complex128))
        decay = mu * self.G
        dG = reinforce - decay
        if self.mode == "full":
            pi_b = self.pi_a.unsqueeze(1)
            supp = self.lambda_s * self.G * torch.sin(self.theta_R / (2 * pi_b)) ** 2
            dG = dG - supp.to(torch.complex128)
        self.G = self.G + self.dt * dG
        re = self.G.real.clamp(min=self.real_min)
        self.G = re + 1j * self.G.imag


# ---------------------------------------------------------------------------
# Protocol pair (matched-present divergence)
# ---------------------------------------------------------------------------

def _history_protocol_pair(mode: str, device: torch.device, n_nodes: int | None = None,
                           omega_end: float = 20.0, pi_0: float = math.pi / 2.0,
                           graph: str = "diamond"):
    """Create a matched-present protocol pair.

    graph: "diamond" (4 nodes, matches CPU benchmark) or "wide_diamond" (scaled).
    n_nodes is used only for wide_diamond to set the number of intermediate paths.
    """
    params = GPUNetParams(pi_0=pi_0)

    if graph == "diamond":
        src, dst, source, sink, nn = _diamond_edges(device)
    elif graph == "wide_diamond":
        n_paths = (n_nodes or 10) - 2
        src, dst, source, sink, nn = _wide_diamond_edges(max(2, n_paths), device)
    else:
        raise ValueError(f"Unknown graph type: {graph}")

    def make_net():
        return GPUNetwork(nn, device, params=params, mode=mode, seed=99,
                          edges=(src.clone(), dst.clone()), source=source, sink=sink)

    net_a = make_net()
    net_b = make_net()

    warmup = _constant_drive(1.0 + 0j, 30)
    net_a.run_drive(warmup)
    net_b.run_drive(warmup)

    chirp = _chirp_drive(1.0, 2.0, omega_end, 0.01, 50)
    const = _constant_drive(1.0 + 0j, 50)
    for va, vb in zip(const, chirp):
        net_a.step(va)
        net_b.step(vb)

    resume = _constant_drive(1.0 + 0j, 30)
    net_a.run_drive(resume)
    net_b.run_drive(resume)

    return net_a, net_b


# ---------------------------------------------------------------------------
# Benchmark 1: Large onset-map sweep
# ---------------------------------------------------------------------------

def onset_map_benchmark(grid_size: int, device: torch.device) -> dict:
    """Sweep (pi_0, omega_end) to map the memory-onset boundary.

    Uses BatchedDiamondNetwork to run all grid_size^2 protocol pairs
    simultaneously on the GPU — one batched torch.linalg.solve per step.
    """
    pi_0_vals = np.linspace(math.pi / 6, 5 * math.pi / 6, grid_size)
    omega_vals = np.linspace(4.0, 24.0, grid_size)

    N_pairs = grid_size * grid_size
    B = 2 * N_pairs  # net_a and net_b per pair

    # Build per-pair parameter arrays
    pi_0_per_pair = []
    omega_end_per_pair = []
    for pi_0 in pi_0_vals:
        for omega in omega_vals:
            pi_0_per_pair.append(pi_0)
            omega_end_per_pair.append(omega)

    # pi_0 for all B networks: [a_0..a_{N-1}, b_0..b_{N-1}]
    pi_0_all = torch.tensor(pi_0_per_pair + pi_0_per_pair,
                            dtype=torch.float64, device=device)
    omega_end_t = torch.tensor(omega_end_per_pair, dtype=torch.float64, device=device)

    batch_net = BatchedDiamondNetwork(B, device, mode="full",
                                      pi_0_values=pi_0_all, seed=99)

    dt = 0.01
    const_drive = torch.full((B,), 1.0 + 0j, dtype=torch.complex128, device=device)

    t0 = time.time()

    # 1) Warmup: 30 steps, all networks get constant drive
    for _ in range(30):
        batch_net.step(const_drive)

    # 2) Diverge: 50 steps — net_a gets constant, net_b gets chirp
    T_chirp = 49 * dt
    omega_start = 2.0
    for k in range(50):
        t = k * dt
        phase = omega_start * t + 0.5 * (omega_end_t - omega_start) * t ** 2 / T_chirp
        drive = const_drive.clone()
        drive[N_pairs:] = torch.exp(1j * phase.to(torch.complex128))
        batch_net.step(drive)

    # 3) Resume: 30 steps, all networks get constant drive
    for _ in range(30):
        batch_net.step(const_drive)

    elapsed = time.time() - t0
    print(f"  batched sweep ({B} networks × 110 steps) done in {elapsed:.1f}s")

    # --- Extract results ---
    theta_R_a = batch_net.theta_R[:N_pairs]  # (N, 4)
    theta_R_b = batch_net.theta_R[N_pairs:]
    theta_gap = (theta_R_a - theta_R_b).abs().max(dim=1).values  # (N,)

    G_a = batch_net.G[:N_pairs]
    G_b = batch_net.G[N_pairs:]
    pi_a_a = batch_net.pi_a[:N_pairs].unsqueeze(1)
    pi_a_b = batch_net.pi_a[N_pairs:].unsqueeze(1)
    supp_a = batch_net.lambda_s * G_a * torch.sin(theta_R_a / (2 * pi_a_a)) ** 2
    supp_b = batch_net.lambda_s * G_b * torch.sin(theta_R_b / (2 * pi_a_b)) ** 2
    supp_gap = (supp_a - supp_b).abs().max(dim=1).values  # (N,)

    memory_on = (theta_gap > math.pi) & (supp_gap > 1e-3)

    # Move to CPU for JSON/CSV
    results = []
    theta_gap_cpu = theta_gap.cpu().numpy()
    supp_gap_cpu = supp_gap.cpu().numpy()
    memory_on_cpu = memory_on.cpu().numpy()
    idx = 0
    for pi_0 in pi_0_vals:
        for omega in omega_vals:
            results.append({
                "pi_0": float(pi_0),
                "omega_end": float(omega),
                "theta_R_gap": float(theta_gap_cpu[idx]),
                "suppression_gap": float(supp_gap_cpu[idx]),
                "memory_on": bool(memory_on_cpu[idx]),
            })
            idx += 1

    on_count = int(memory_on.sum())
    return {
        "name": "onset_map_gpu",
        "grid_size": grid_size,
        "total_points": N_pairs,
        "memory_on_count": on_count,
        "memory_off_count": N_pairs - on_count,
        "elapsed_seconds": round(elapsed, 2),
        "device": str(device),
        "pass": on_count > 0 and (N_pairs - on_count) > 0,
        "rows": results,
    }


# ---------------------------------------------------------------------------
# Benchmark 2: Large-network matched-present divergence
# ---------------------------------------------------------------------------

def large_network_benchmark(n_nodes: int, device: torch.device) -> dict:
    """Run matched-present divergence on a wide-diamond graph."""
    t0 = time.time()

    print(f"  building {n_nodes}-node wide-diamond networks (full + principal)...")
    net_full_a, net_full_b = _history_protocol_pair("full", device, n_nodes=n_nodes,
                                                     graph="wide_diamond")
    net_princ_a, net_princ_b = _history_protocol_pair("principal", device, n_nodes=n_nodes,
                                                       graph="wide_diamond")

    full_theta_gap = float((net_full_a.theta_R - net_full_b.theta_R).abs().max())
    princ_theta_gap = float((net_princ_a.theta_R - net_princ_b.theta_R).abs().max())

    full_G_gap = float((net_full_a.G - net_full_b.G).abs().max())
    princ_G_gap = float((net_princ_a.G - net_princ_b.G).abs().max())

    full_supp_a = _suppression_t(net_full_a.G, net_full_a.theta_R, net_full_a.pi_a, net_full_a.p.lambda_s)
    full_supp_b = _suppression_t(net_full_b.G, net_full_b.theta_R, net_full_b.pi_a, net_full_b.p.lambda_s)
    full_supp_gap = float((full_supp_a - full_supp_b).abs().max())

    max_G = float(net_full_a.G.abs().max())

    elapsed = time.time() - t0
    n_paths = max(2, n_nodes - 2)
    print(f"  {n_nodes}-node ({n_paths} paths) benchmark done in {elapsed:.1f}s")

    return {
        "name": "large_network_matched_present",
        "n_nodes": n_nodes,
        "n_paths": n_paths,
        "n_edges": 2 * n_paths,
        "full_theta_R_gap": full_theta_gap,
        "principal_theta_R_gap": princ_theta_gap,
        "full_G_gap": full_G_gap,
        "principal_G_gap": princ_G_gap,
        "full_suppression_gap": full_supp_gap,
        "max_abs_G": max_G,
        "elapsed_seconds": round(elapsed, 2),
        "device": str(device),
        "pass": (
            full_theta_gap > math.pi
            and princ_theta_gap < 1e-6
            and full_supp_gap > 1e-3
            and max_G < 1e6
        ),
    }


# ---------------------------------------------------------------------------
# Benchmark 3: Ablation sweep
# ---------------------------------------------------------------------------

def ablation_benchmark(device: torch.device) -> dict:
    """Run all four ablation modes and compare."""
    modes = ["principal", "lift_only", "lift_ruler", "full"]
    results = {}
    t0 = time.time()

    for mode in modes:
        net_a, net_b = _history_protocol_pair(mode, device, graph="diamond")
        theta_gap = float((net_a.theta_R - net_b.theta_R).abs().max())
        G_gap = float((net_a.G - net_b.G).abs().max())
        supp_a = _suppression_t(net_a.G, net_a.theta_R, net_a.pi_a, net_a.p.lambda_s)
        supp_b = _suppression_t(net_b.G, net_b.theta_R, net_b.pi_a, net_b.p.lambda_s)
        supp_gap = float((supp_a - supp_b).abs().max())
        results[mode] = {
            "theta_R_gap": theta_gap,
            "G_gap": G_gap,
            "suppression_gap": supp_gap,
            "S_a": net_a.S,
            "S_b": net_b.S,
            "pi_a_a": net_a.pi_a,
            "pi_a_b": net_b.pi_a,
        }

    elapsed = time.time() - t0

    # Full should have the largest gaps
    full_theta = results["full"]["theta_R_gap"]
    princ_theta = results["principal"]["theta_R_gap"]
    return {
        "name": "ablation_sweep",
        "modes": results,
        "elapsed_seconds": round(elapsed, 2),
        "device": str(device),
        "pass": full_theta > math.pi and princ_theta < 1e-6,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPU History-Resolved Phase benchmarks")
    parser.add_argument("--onset-grid", type=int, default=100,
                        help="Grid size for onset map (NxN sweep)")
    parser.add_argument("--network-size", type=int, default=500,
                        help="Number of nodes for large-network test")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (default 0)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: benchmarks/)")
    args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        gpu_name = torch.cuda.get_device_name(args.gpu)
        gpu_mem = torch.cuda.get_device_properties(args.gpu).total_memory / 1e9
        n_gpus = torch.cuda.device_count()
        print(f"Using GPU {args.gpu}: {gpu_name} ({gpu_mem:.1f} GB)")
        if n_gpus > 1:
            print(f"  ({n_gpus} GPUs total — this script uses device {args.gpu})")
    else:
        device = torch.device("cpu")
        gpu_name = "CPU"
        gpu_mem = 0
        n_gpus = 0
        print("CUDA not available, running on CPU (will be slow)")

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_memory_gb": round(gpu_mem, 1),
        "n_gpus_available": n_gpus,
        "benchmarks": [],
    }

    # --- Benchmark 1: Onset map ---
    print(f"\n{'='*60}")
    print(f"Benchmark 1: Onset map ({args.onset_grid}x{args.onset_grid} sweep)")
    print(f"{'='*60}")
    onset = onset_map_benchmark(args.onset_grid, device)
    report["benchmarks"].append({k: v for k, v in onset.items() if k != "rows"})
    print(f"  Result: {onset['memory_on_count']} ON / {onset['memory_off_count']} OFF "
          f"in {onset['elapsed_seconds']}s — {'PASS' if onset['pass'] else 'FAIL'}")

    # Save onset CSV
    csv_path = out_dir / f"onset_map_gpu_{args.onset_grid}x{args.onset_grid}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pi_0", "omega_end", "theta_R_gap", "suppression_gap", "memory_on"])
        writer.writeheader()
        for row in onset["rows"]:
            writer.writerow(row)
    print(f"  Saved: {csv_path}")

    # --- Benchmark 2: Large network ---
    print(f"\n{'='*60}")
    print(f"Benchmark 2: {args.network_size}-node ring matched-present divergence")
    print(f"{'='*60}")
    large = large_network_benchmark(args.network_size, device)
    report["benchmarks"].append(large)
    print(f"  full θ_R gap:  {large['full_theta_R_gap']:.6f}")
    print(f"  princ θ_R gap: {large['principal_theta_R_gap']:.6e}")
    print(f"  full supp gap: {large['full_suppression_gap']:.6e}")
    print(f"  max |G|:       {large['max_abs_G']:.6f}")
    print(f"  Result: {'PASS' if large['pass'] else 'FAIL'} in {large['elapsed_seconds']}s")

    # --- Benchmark 3: Ablation ---
    print(f"\n{'='*60}")
    print(f"Benchmark 3: Ablation sweep (4 modes)")
    print(f"{'='*60}")
    ablation = ablation_benchmark(device)
    report["benchmarks"].append(ablation)
    for mode, data in ablation["modes"].items():
        print(f"  {mode:12s}: θ_R gap={data['theta_R_gap']:.6e}  "
              f"supp gap={data['suppression_gap']:.6e}")
    print(f"  Result: {'PASS' if ablation['pass'] else 'FAIL'} in {ablation['elapsed_seconds']}s")

    # --- Summary ---
    all_pass = all(b["pass"] for b in report["benchmarks"])
    report["all_passed"] = all_pass

    json_path = out_dir / "gpu_benchmark_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"All passed: {all_pass}")
    print(f"Report: {json_path}")
    print(f"Onset CSV: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
