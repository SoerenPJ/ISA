"""
gauge_sweep_plot.py — plot gauge consistency metrics vs chemical potential μ.

Reads the output of gauge_sweep.sh:
    gauge_metrics_mu_<formation>_<Nx>x<Ny>_rot<angle>.txt

Usage:
    python3 ploting/gauge_sweep_plot.py gauge_metrics_mu_*.txt
    python3 ploting/gauge_sweep_plot.py path/to/gauge_metrics_mu_armchair_7x7_rot0.txt

Output:
    gauge_sweep_metrics.png  (saved next to the input file)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

COLUMNS = [
    "mu",
    "corr_flux",
    "alpha_flux",
    "mean_rel_peak",
    "max_rel_peak",
    "mean_ratio_peak",
    "rms_flux_peak",
    "dynamic_range",
    "mean_rel_curl_peak",
]


def load_metrics(path: Path) -> dict:
    data = np.genfromtxt(path, comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < len(COLUMNS):
        raise ValueError(
            f"Expected {len(COLUMNS)} columns, got {data.shape[1]} in {path}"
        )
    # Sort by mu so lines always go left→right
    order = np.argsort(data[:, 0])
    data  = data[order]
    return {col: data[:, i] for i, col in enumerate(COLUMNS)}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_metrics(d: dict, title_suffix: str, out_path: Path) -> None:
    mu = d["mu"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    def _line(ax, y, label, color, lw=1.6, ls="-"):
        mask = np.isfinite(y)
        ax.plot(mu[mask], y[mask], color=color, lw=lw, ls=ls, label=label)

    def _hline(ax, val, color="k", lw=0.9, ls="--", alpha=0.5):
        ax.axhline(val, color=color, lw=lw, ls=ls, alpha=alpha)

    # --- Panel 0: correlation coefficient ---
    ax = axes[0]
    _line(ax, d["corr_flux"], r"$r(\Phi_B,\,\Phi_A)$", "C0")
    _hline(ax, 1.0, label="ideal = 1")
    ax.set_ylabel("Pearson r")
    ax.set_title("Flux correlation coefficient")
    ax.set_ylim(None, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 1: best-fit slope α ---
    ax = axes[1]
    _line(ax, d["alpha_flux"], r"$\alpha\;(\Phi_A = \alpha\,\Phi_B)$", "C1")
    _hline(ax, 1.0, label="ideal = 1")
    ax.set_ylabel(r"Slope $\alpha$")
    ax.set_title("Best-fit slope  (ideal = 1)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: relative flux error (log scale) ---
    ax = axes[2]
    _line(ax, d["mean_rel_peak"], "mean rel. error", "C2")
    _line(ax, d["max_rel_peak"],  "max rel. error",  "C3", ls="--")
    ax.set_ylabel(r"$|\Phi_B - \Phi_A|\,/\,|\Phi_B|$")
    ax.set_title("Relative flux error at peak signal")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # --- Panel 3: mean flux ratio ---
    ax = axes[3]
    _line(ax, d["mean_ratio_peak"], r"$\langle\Phi_A/\Phi_B\rangle$", "C4")
    _hline(ax, 1.0, label="ideal = 1")
    ax.set_ylabel(r"$\langle \Phi_A / \Phi_B \rangle$")
    ax.set_title("Mean flux ratio (ideal = 1)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: dynamic range ---
    ax = axes[4]
    _line(ax, d["dynamic_range"], "signal / noise", "C5")
    ax.set_ylabel("Dynamic range  [×]")
    ax.set_title("Signal-to-noise  (peak / late-time floor)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.grid(True, alpha=0.3, which="both")

    # --- Panel 5: curl / B-field relative error ---
    ax = axes[5]
    _line(ax, d["mean_rel_curl_peak"], r"mean $|\nabla{\times}A - B|/|B|$", "C6")
    ax.set_ylabel(r"mean $|\mathrm{curl}\,A - B_z|\,/\,|B_z|$")
    ax.set_title("Curl vs B-field relative error at peak")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    for ax in axes:
        ax.set_xlabel(r"$\mu$  [eV]")

    fig.suptitle(
        f"Gauge sweep metrics vs $\\mu${title_suffix}",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <gauge_metrics_mu_*.txt>", file=sys.stderr)
        sys.exit(1)

    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            continue

        try:
            d = load_metrics(path)
        except Exception as e:
            print(f"Error loading {path}: {e}", file=sys.stderr)
            continue

        title_suffix = f" — {path.stem}"
        out_path     = path.parent / (path.stem + "_plot.png")
        plot_metrics(d, title_suffix, out_path)


if __name__ == "__main__":
    main()
