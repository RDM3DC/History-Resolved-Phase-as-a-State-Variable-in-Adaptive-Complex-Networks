"""
Generate publication-quality onset-map heatmaps from GPU benchmark CSV data.

Usage:
    python benchmarks/plot_onset_map.py                          # uses 100x100
    python benchmarks/plot_onset_map.py --csv benchmarks/onset_map_gpu_500x500.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_grid(csv_path: str | Path):
    df = pd.read_csv(csv_path)
    pi_0 = np.sort(df["pi_0"].unique())
    omega = np.sort(df["omega_end"].unique())
    n_pi, n_om = len(pi_0), len(omega)

    theta_map = np.full((n_pi, n_om), np.nan)
    supp_map = np.full((n_pi, n_om), np.nan)
    onset_map = np.zeros((n_pi, n_om), dtype=int)

    pi_idx = {v: i for i, v in enumerate(pi_0)}
    om_idx = {v: i for i, v in enumerate(omega)}

    for _, row in df.iterrows():
        i = pi_idx[row["pi_0"]]
        j = om_idx[row["omega_end"]]
        theta_map[i, j] = row["theta_R_gap"]
        supp_map[i, j] = row["suppression_gap"]
        onset_map[i, j] = 1 if row["memory_on"] else 0

    return pi_0, omega, theta_map, supp_map, onset_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="benchmarks/onset_map_gpu_100x100.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    pi_0, omega, theta_map, supp_map, onset_map = load_grid(csv_path)
    grid_label = f"{len(pi_0)}×{len(omega)}"
    out_dir = csv_path.parent

    # Convert pi_0 to multiples of π for labelling
    pi_0_over_pi = pi_0 / math.pi

    # --- Figure 1: Three-panel overview ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: θ_R gap (log scale, clamped)
    theta_plot = np.log10(np.clip(theta_map, 1e-15, None))
    im0 = axes[0].imshow(
        theta_plot, origin="lower", aspect="auto",
        extent=[omega[0], omega[-1], pi_0_over_pi[0], pi_0_over_pi[-1]],
        cmap="inferno", vmin=-14, vmax=1,
    )
    axes[0].set_xlabel(r"$\omega_{\mathrm{end}}$")
    axes[0].set_ylabel(r"$\pi_0\;/\;\pi$")
    axes[0].set_title(r"(a)  $\log_{10}\,\Delta\theta_R$")
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label(r"$\log_{10}$")

    # Panel B: Suppression gap (log scale)
    supp_plot = np.log10(np.clip(supp_map, 1e-30, None))
    im1 = axes[1].imshow(
        supp_plot, origin="lower", aspect="auto",
        extent=[omega[0], omega[-1], pi_0_over_pi[0], pi_0_over_pi[-1]],
        cmap="viridis", vmin=-28, vmax=0,
    )
    axes[1].set_xlabel(r"$\omega_{\mathrm{end}}$")
    axes[1].set_ylabel(r"$\pi_0\;/\;\pi$")
    axes[1].set_title(r"(b)  $\log_{10}\,\Delta S_{\mathrm{supp}}$")
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label(r"$\log_{10}$")

    # Panel C: Binary onset map with contour
    im2 = axes[2].imshow(
        onset_map, origin="lower", aspect="auto",
        extent=[omega[0], omega[-1], pi_0_over_pi[0], pi_0_over_pi[-1]],
        cmap="RdYlGn", vmin=0, vmax=1,
    )
    # Overlay onset boundary contour
    axes[2].contour(
        omega, pi_0_over_pi, onset_map.astype(float),
        levels=[0.5], colors="black", linewidths=1.5,
    )
    axes[2].set_xlabel(r"$\omega_{\mathrm{end}}$")
    axes[2].set_ylabel(r"$\pi_0\;/\;\pi$")
    axes[2].set_title(r"(c)  Memory ON/OFF boundary")

    on_pct = 100.0 * onset_map.sum() / onset_map.size
    fig.suptitle(
        f"History-Resolved Phase Onset Map  ({grid_label} GPU sweep)  —  "
        f"{on_pct:.1f}% ON",
        fontsize=15, y=1.02,
    )
    fig.tight_layout()

    png_path = out_dir / f"onset_map_{grid_label.replace('×','x')}.png"
    fig.savefig(png_path)
    print(f"Saved: {png_path}")
    plt.close(fig)

    # --- Figure 2: Standalone onset boundary (clean, Twitter-ready) ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.imshow(
        onset_map, origin="lower", aspect="auto",
        extent=[omega[0], omega[-1], pi_0_over_pi[0], pi_0_over_pi[-1]],
        cmap="RdYlGn", vmin=0, vmax=1, alpha=0.6,
    )
    cs = ax2.contour(
        omega, pi_0_over_pi, onset_map.astype(float),
        levels=[0.5], colors="black", linewidths=2,
    )
    ax2.set_xlabel(r"$\omega_{\mathrm{end}}$  (chirp endpoint)", fontsize=14)
    ax2.set_ylabel(r"$\pi_0 / \pi$  (baseline ruler)", fontsize=14)
    ax2.set_title(
        f"Branch-Memory Onset Boundary\n"
        f"({grid_label} GPU sweep, {on_pct:.1f}% memory-on)",
        fontsize=15,
    )
    ax2.text(
        0.98, 0.04, "green = memory ON\nred = memory OFF",
        transform=ax2.transAxes, ha="right", va="bottom",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    png2_path = out_dir / f"onset_boundary_{grid_label.replace('×','x')}.png"
    fig2.savefig(png2_path)
    print(f"Saved: {png2_path}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
