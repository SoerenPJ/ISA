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

METRIC_COLUMNS = [
    "corr_flux",
    "alpha_flux",
    "mean_rel_peak",
    "max_rel_peak",
    "mean_ratio_peak",
    "rms_flux_peak",
    "dynamic_range",
    "mean_rel_curl_peak",
]

LEVELS = ["L0", "L1", "L2"]


def load_metrics(path: Path) -> dict[str, dict]:
    """Return {level: {metric: array}} for all levels found in the file.

    Supports both old 9-column format (mu + 8 metrics) and the current
    10-column format (mu + level + 8 metrics).
    """
    raw = path.read_text(encoding="utf-8")
    rows: list[tuple] = []
    for line in raw.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            mu = float(parts[0])
        except ValueError:
            continue
        # Detect whether column 1 is a level label (L0/L1/L2) or a number
        if len(parts) >= 2 and parts[1] in LEVELS:
            level = parts[1]
            vals = [float(v) for v in parts[2:2 + len(METRIC_COLUMNS)]]
        else:
            level = "L2"  # legacy single-level files default to L2
            vals = [float(v) for v in parts[1:1 + len(METRIC_COLUMNS)]]
        rows.append((mu, level, *vals))

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    result: dict[str, dict] = {}
    for lv in LEVELS:
        subset = [(r[0], r[2:]) for r in rows if r[1] == lv]
        if not subset:
            continue
        mus  = np.array([s[0] for s in subset])
        vals = np.array([s[1] for s in subset])
        order = np.argsort(mus)
        result[lv] = {"mu": mus[order]}
        for i, col in enumerate(METRIC_COLUMNS):
            result[lv][col] = vals[order, i]
    return result


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

LEVEL_STYLES = {
    "L0": {"color": "C0", "ls": "-",  "lw": 1.4},
    "L1": {"color": "C1", "ls": "--", "lw": 1.4},
    "L2": {"color": "C2", "ls": "-",  "lw": 1.8},
}


def plot_metrics(data: dict[str, dict], title_suffix: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    def _line(ax, lv, y, label):
        mu   = data[lv]["mu"]
        sty  = LEVEL_STYLES[lv]
        mask = np.isfinite(y)
        ax.plot(mu[mask], y[mask], label=label, **sty)

    def _hline(ax, val, color="k", lw=0.9, ls="--", alpha=0.5, label=None):
        ax.axhline(val, color=color, lw=lw, ls=ls, alpha=alpha, label=label)

    # --- Panel 0: correlation coefficient ---
    ax = axes[0]
    for lv, d in data.items():
        _line(ax, lv, d["corr_flux"], f"{lv}: " + r"$r(\Phi_B,\Phi_A)$")
    _hline(ax, 1.0, label="ideal = 1")
    ax.set_ylabel("Pearson r")
    ax.set_title("Flux correlation coefficient")
    ax.set_ylim(None, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 1: best-fit slope α ---
    ax = axes[1]
    for lv, d in data.items():
        _line(ax, lv, d["alpha_flux"], f"{lv}: " + r"$\alpha$")
    _hline(ax, 1.0, label="ideal = 1")
    ax.set_ylabel(r"Slope $\alpha$")
    ax.set_title("Best-fit slope  (ideal = 1)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: relative flux error (log scale) ---
    ax = axes[2]
    for lv, d in data.items():
        _line(ax, lv, d["mean_rel_peak"], f"{lv}: mean")
        mu   = data[lv]["mu"]
        sty  = LEVEL_STYLES[lv].copy(); sty["ls"] = ":"
        mask = np.isfinite(d["max_rel_peak"])
        ax.plot(mu[mask], d["max_rel_peak"][mask], label=f"{lv}: max", **sty)
    ax.set_ylabel(r"$|\Phi_B - \Phi_A|\,/\,|\Phi_B|$")
    ax.set_title("Relative flux error at peak signal")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # --- Panel 3: mean flux ratio ---
    ax = axes[3]
    for lv, d in data.items():
        _line(ax, lv, d["mean_ratio_peak"], f"{lv}: " + r"$\langle\Phi_A/\Phi_B\rangle$")
    _hline(ax, 1.0, label="ideal = 1")
    ax.set_ylabel(r"$\langle \Phi_A / \Phi_B \rangle$")
    ax.set_title("Mean flux ratio (ideal = 1)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: dynamic range ---
    ax = axes[4]
    for lv, d in data.items():
        _line(ax, lv, d["dynamic_range"], f"{lv}: signal/noise")
    ax.set_ylabel("Dynamic range  [×]")
    ax.set_title("Signal-to-noise  (peak / late-time floor)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # --- Panel 5: curl / B-field relative error ---
    ax = axes[5]
    for lv, d in data.items():
        _line(ax, lv, d["mean_rel_curl_peak"], f"{lv}: " + r"mean $|\nabla\times A - B|/|B|$")
    ax.set_ylabel(r"mean $|\mathrm{curl}\,A - B_z|\,/\,|B_z|$")
    ax.set_title("Curl vs B-field relative error at peak")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
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
