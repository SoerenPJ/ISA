#!/usr/bin/env python3
"""
Q_doping_plots.py — plots for the pure charge-doping sweep produced by
./Q_doping_sweep.sh, where the ONLY thing that varies between runs is Q (the
number of extra electrons relative to neutral).

For every Q in the sweep it shows the three things that respond to doping:

  1. the single-particle spectrum with the OCCUPATION indicated
     (colour = f_j in [0,1], marker = spin channel, dashed line = Fermi level)
  2. the dipole moment time evolution
  3. the extinction cross section sigma_ext(E), plus how its resonance moves with Q

Usage:
    python3 ploting/Q_doping_plots.py data/Qsweep_zigzag_triangle_5x5_rot0
    python3 ploting/Q_doping_plots.py <sweep_dir> --window 8 --emax 12

Output: <sweep_dir>/plots/*.png (+ resonance_vs_Q.txt)

Conventions follow ploting/base_plots.py (occupations from the eigenbasis rho0)
and ploting/sigma_ext.py (sigma in nm^2 vs energy in eV).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    from scipy.signal import find_peaks
except ImportError:          # scipy is optional; fall back to the global maximum
    find_peaks = None

# ----------------------------- constants ----------------------------- #
au_eV = 27.2114
au_nm = 0.0529177
au_s = 2.41888e-17
au_fs = au_s * 1e15

# ----------------------------- styling ------------------------------- #
# Occupation is a magnitude in [0,1] -> single sequential ramp, dark->light,
# readable at both ends and colour-vision-deficiency safe (cividis).
OCC_CMAP = plt.cm.cividis
OCC_NORM = Normalize(vmin=0.0, vmax=1.0)

INK = "#22252a"        # primary text / marks
INK_MUTED = "#7a8089"  # grid, axes, secondary annotation

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": "#d9dce1",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def q_colors(n):
    """Ordered ramp for the Q series (Q is an ordered quantity, not an identity)."""
    if n == 1:
        return [plt.cm.plasma(0.35)]
    return [plt.cm.plasma(x) for x in np.linspace(0.05, 0.82, n)]


def fmt_Q(q):
    s = f"{q:+.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("+0", "-0") else s


# ----------------------------- loading ------------------------------- #
DIR_RE = re.compile(r"^Q_(-?\d+(?:\.\d+)?)$")


def load_complex_matrix(path):
    """File holds one row per line, each element written as 'real imag'."""
    raw = np.loadtxt(path)
    return raw[:, 0::2] + 1j * raw[:, 1::2]


def load_spectrum(run_dir):
    """Return (energies_eV, occupations, spin) sorted by energy.

    Preferred source is hubbard_spectrum.txt (converged UHF spectrum: index,
    energy_eV, spin, occupation). Without the Hubbard feature we rebuild the same
    information from eigenvalues.txt + the diagonal of rho0_j_space.txt, which is
    already written in the eigenbasis (same route as base_plots.py).
    """
    spec = run_dir / "hubbard_spectrum.txt"
    if spec.exists():
        d = np.loadtxt(spec, comments="#")
        if d.ndim == 1:
            d = d[None, :]
        E, spin, occ = d[:, 1], d[:, 2].astype(int), d[:, 3]
    else:
        ev = run_dir / "eigenvalues.txt"
        rho = run_dir / "rho0_j_space.txt"
        if not ev.exists():
            return None
        E = np.atleast_2d(np.loadtxt(ev))[:, 0] * au_eV
        if rho.exists():
            occ = np.real(np.diag(load_complex_matrix(rho)))
        else:
            occ = np.full_like(E, np.nan)
        spin = np.zeros_like(E, dtype=int)   # 0 = spin channel unknown/spinless

    order = np.argsort(E)
    return E[order], occ[order], spin[order]


def frontier_mask(E, E_F, window, min_levels=14):
    """Levels to show: those within `window` of E_F, but never fewer than the
    `min_levels` closest ones — a large magnetic gap must not empty the panel."""
    if not np.isfinite(E_F):
        return np.ones_like(E, dtype=bool)
    m = np.ones_like(E, dtype=bool) if window is None else np.abs(E - E_F) <= window
    if m.sum() < min(min_levels, E.size):
        nearest = np.argsort(np.abs(E - E_F))[:min(min_levels, E.size)]
        m = np.zeros_like(E, dtype=bool)
        m[nearest] = True
    return m


def fermi_level(E, occ):
    """Midpoint between the highest mostly-filled and lowest mostly-empty level."""
    if occ is None or not np.isfinite(occ).any():
        return np.nan, np.nan
    filled = np.where(occ > 0.5)[0]
    empty = np.where(occ <= 0.5)[0]
    if filled.size == 0 or empty.size == 0:
        return float(np.median(E)), np.nan
    i_ho, i_lu = filled.max(), empty.min()
    return 0.5 * (E[i_ho] + E[i_lu]), E[i_lu] - E[i_ho]


def load_dipole(run_dir):
    f = run_dir / "dipole_time_evolution.txt"
    if not f.exists():
        return None
    d = np.loadtxt(f, comments="#")
    if d.ndim != 2 or d.shape[0] < 2:
        return None
    t, dip = d[:, 0], d[:, 1]
    # The adaptive solver can emit repeated / non-monotonic time stamps; keep the
    # first sample of each time so the trace is single valued.
    t, idx = np.unique(t, return_index=True)
    return t * au_fs, dip[idx]


def load_sigma(run_dir):
    f = run_dir / "sigma_ext.txt"
    if not f.exists():
        return None
    d = np.loadtxt(f)
    if d.ndim != 2 or d.shape[0] < 2:
        return None
    return d[:, 0] * au_eV, d[:, 1] * au_nm**2


def resonance(E, sigma):
    """Energy and height of the dominant sigma_ext peak."""
    if sigma.size == 0 or np.nanmax(sigma) <= 0:
        return np.nan, np.nan
    if find_peaks is not None:
        peaks, props = find_peaks(sigma, prominence=np.nanmax(sigma) * 0.1)
        if peaks.size:
            k = peaks[int(np.argmax(props["prominences"]))]
            return E[k], sigma[k]
    k = int(np.nanargmax(sigma))
    return E[k], sigma[k]


def load_runs(sweep_dir):
    runs = []
    for d in sorted(sweep_dir.iterdir()):
        if not d.is_dir():
            continue
        m = DIR_RE.match(d.name)
        if not m:
            continue
        spec = load_spectrum(d)
        E, occ, spin = spec if spec else (None, None, None)
        E_F, gap = fermi_level(E, occ) if spec else (np.nan, np.nan)
        runs.append({
            "Q": float(m.group(1)),
            "dir": d,
            "E": E, "occ": occ, "spin": spin,
            "E_F": E_F, "gap": gap,
            "N_e": float(np.nansum(occ)) if spec and np.isfinite(occ).any() else np.nan,
            "dipole": load_dipole(d),
            "sigma": load_sigma(d),
        })
    runs.sort(key=lambda r: r["Q"])
    return runs


# ----------------------------- panels -------------------------------- #
def draw_level_ladder(ax, runs, window):
    """One column of levels per Q, coloured by occupation.

    Up levels sit on the left half of a column, down levels on the right, so the
    spin channel is encoded by position (never by colour alone).
    """
    x = np.arange(len(runs), dtype=float)
    segs, vals = [], []
    for i, r in enumerate(runs):
        if r["E"] is None:
            continue
        keep = frontier_mask(r["E"], r["E_F"], window)
        E, occ, spin = r["E"][keep], r["occ"][keep], r["spin"][keep]
        spinful = np.any(spin != 0)
        for e, f, s in zip(E, occ, spin):
            if not spinful:
                x0, x1 = x[i] - 0.42, x[i] + 0.42
            elif s > 0:
                x0, x1 = x[i] - 0.44, x[i] - 0.03
            else:
                x0, x1 = x[i] + 0.03, x[i] + 0.44
            segs.append([(x0, e), (x1, e)])
            vals.append(f)
        if np.isfinite(r["E_F"]):
            ax.plot([x[i] - 0.48, x[i] + 0.48], [r["E_F"]] * 2,
                    ls="--", lw=1.0, color="#c0392b", zorder=3)

    if segs:
        lc = LineCollection(segs, array=np.asarray(vals), cmap=OCC_CMAP,
                            norm=OCC_NORM, linewidths=2.0, zorder=2)
        ax.add_collection(lc)

    ax.set_xticks(x)
    ax.set_xticklabels([fmt_Q(r["Q"]) for r in runs])
    ax.set_xlim(-0.6, len(runs) - 0.4)
    ax.set_xlabel("charge doping  Q  (extra electrons)")
    ax.set_ylabel("energy (eV)")
    spinful_any = any(r["spin"] is not None and np.any(r["spin"] != 0) for r in runs)
    sub = "up | down" if spinful_any else "levels"
    scope = "Frontier levels" if window is not None else "Energy levels"
    ax.set_title(f"{scope} ({sub}), colour = occupation\n"
                 r"dashed red = $E_F$")
    ax.grid(axis="y", alpha=0.5)
    ax.autoscale_view()


def draw_spectrum_panels(runs, out_dir, window):
    """Per-Q panel: energy vs level index, occupation as colour, spin as marker."""
    n = len(runs)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.6 * nrow),
                             squeeze=False, sharey=True)
    for ax, r in zip(axes.ravel(), runs):
        if r["E"] is None:
            ax.set_visible(False)
            continue
        idx = np.arange(r["E"].size)
        keep = frontier_mask(r["E"], r["E_F"], window)
        E, occ, spin, idx = r["E"][keep], r["occ"][keep], r["spin"][keep], idx[keep]
        for s, marker, label in ((1, "^", "spin up"), (-1, "v", "spin down"),
                                 (0, "o", "levels")):
            m = spin == s
            if not m.any():
                continue
            ax.scatter(idx[m], E[m], c=occ[m], cmap=OCC_CMAP, norm=OCC_NORM,
                       marker=marker, s=34, edgecolors=INK, linewidths=0.4,
                       label=label, zorder=3)
        if np.isfinite(r["E_F"]):
            ax.axhline(r["E_F"], color="#c0392b", ls="--", lw=1.1,
                       label=fr"$E_F={r['E_F']:.2f}$ eV")
        title = f"Q = {fmt_Q(r['Q'])}"
        if np.isfinite(r["N_e"]):
            title += f"   $N_e$ = {r['N_e']:.2f}"
        if np.isfinite(r["gap"]):
            title += f"   gap = {r['gap']:.2f} eV"
        ax.set_title(title)
        ax.set_xlabel("level index")
        ax.grid(alpha=0.5)
        ax.legend(loc="best", fontsize=8)
    for ax in axes.ravel()[len(runs):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("energy (eV)")

    sm = ScalarMappable(norm=OCC_NORM, cmap=OCC_CMAP)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("occupation  $f_j$")
    fig.suptitle("Eigenvalues and occupations vs charge doping", y=0.995)
    fig.savefig(out_dir / "spectrum_panels.png", bbox_inches="tight")
    plt.close(fig)


def draw_dipole(ax, runs, colors, offset=False):
    traces = [(r, r["dipole"]) for r in runs if r["dipole"] is not None]
    if not traces:
        ax.text(0.5, 0.5, "no dipole_time_evolution.txt", ha="center",
                transform=ax.transAxes, color=INK_MUTED)
        return
    span = max(np.ptp(d[1]) for _, d in traces) or 1.0
    for k, (r, (t, dip)) in enumerate(traces):
        shift = k * 1.15 * span if offset else 0.0
        ax.plot(t, dip + shift, lw=1.3, color=colors[runs.index(r)],
                label=f"Q = {fmt_Q(r['Q'])}")
        if offset:
            ax.annotate(f"Q = {fmt_Q(r['Q'])}", (t[-1], dip[-1] + shift),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8, va="center", color=INK)
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("dipole moment (a.u.)" + (" [offset]" if offset else ""))
    ax.set_title("Dipole moment vs charge doping")
    ax.grid(alpha=0.5)
    if not offset:
        ax.legend(fontsize=8, ncol=2)


def draw_sigma(ax, runs, colors, emax=None):
    any_curve = False
    for r in runs:
        if r["sigma"] is None:
            continue
        E, s = r["sigma"]
        if emax is not None:
            keep = E <= emax
            E, s = E[keep], s[keep]
        ax.plot(E, s, lw=1.4, color=colors[runs.index(r)],
                label=f"Q = {fmt_Q(r['Q'])}")
        Er, sr = resonance(E, s)
        if np.isfinite(Er):
            ax.plot([Er], [sr], marker="o", ms=5, color=colors[runs.index(r)],
                    markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        any_curve = True
    if not any_curve:
        ax.text(0.5, 0.5, "no sigma_ext.txt (run with run_sigma_ext = true)",
                ha="center", transform=ax.transAxes, color=INK_MUTED)
        return
    ax.set_xlabel("energy (eV)")
    ax.set_ylabel(r"$\sigma_{\mathrm{ext}}$ (nm$^2$)")
    ax.set_title("Extinction cross section vs charge doping (dot = dominant peak)")
    ax.grid(alpha=0.5)
    ax.legend(fontsize=8, ncol=2)


def draw_resonance_vs_Q(axes, runs, out_dir, emax=None):
    ax_e, ax_h = axes
    Q, Er, Sr = [], [], []
    for r in runs:
        if r["sigma"] is None:
            continue
        E, s = r["sigma"]
        if emax is not None:
            keep = E <= emax
            E, s = E[keep], s[keep]
        e, h = resonance(E, s)
        Q.append(r["Q"]); Er.append(e); Sr.append(h)
    if not Q:
        for ax in axes:
            ax.set_visible(False)
        return
    # Two measures of different scale -> two panels sharing x, never a second y-axis.
    ax_e.plot(Q, Er, "-o", lw=1.6, ms=6, color=plt.cm.plasma(0.25))
    ax_e.set_ylabel("resonance (eV)")
    ax_e.set_title("Dominant extinction resonance vs Q")
    ax_e.grid(alpha=0.5)
    ax_h.plot(Q, Sr, "-o", lw=1.6, ms=6, color=plt.cm.plasma(0.65))
    ax_h.set_ylabel(r"peak $\sigma_{\mathrm{ext}}$ (nm$^2$)")
    ax_h.set_xlabel("charge doping  Q  (extra electrons)")
    ax_h.grid(alpha=0.5)

    np.savetxt(out_dir / "resonance_vs_Q.txt",
               np.column_stack([Q, Er, Sr]),
               header="Q  E_res_eV  sigma_peak_nm2", fmt="%.6g")


# ------------------------------ main --------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", help="data/Qsweep_<tag> produced by Q_doping_sweep.sh")
    ap.add_argument("--window", type=float, default=0.0,
                    help="zoom the level plots to +/- this many eV around E_F. "
                         "Default 0 = show the full spectrum")
    ap.add_argument("--emax", type=float, default=None,
                    help="upper energy cut for sigma_ext plots, eV")
    ap.add_argument("--dipole-offset", action="store_true",
                    help="stack the dipole traces vertically instead of overlaying")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_dir():
        sys.exit(f"Not a directory: {sweep_dir}")

    runs = load_runs(sweep_dir)
    if not runs:
        sys.exit(f"No Q_<value> run directories found in {sweep_dir}")

    window = args.window if args.window and args.window > 0 else None
    out_dir = sweep_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    colors = q_colors(len(runs))

    print(f"Loaded {len(runs)} doping points: "
          + ", ".join(fmt_Q(r["Q"]) for r in runs))
    for r in runs:
        print(f"  Q = {fmt_Q(r['Q']):>6}  N_e = {r['N_e']:.3f}  "
              f"E_F = {r['E_F']:.3f} eV  gap = {r['gap']:.3f} eV")

    # --- 1. level ladder ------------------------------------------------
    fig, ax = plt.subplots(figsize=(1.6 * len(runs) + 3.5, 5.5))
    draw_level_ladder(ax, runs, window)
    cbar = fig.colorbar(ScalarMappable(norm=OCC_NORM, cmap=OCC_CMAP), ax=ax,
                        fraction=0.035, pad=0.02)
    cbar.set_label("occupation  $f_j$")
    fig.savefig(out_dir / "levels_vs_Q.png", bbox_inches="tight")
    plt.close(fig)

    # --- 2. per-Q spectrum panels ---------------------------------------
    draw_spectrum_panels(runs, out_dir, window)

    # --- 3. dipole ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    draw_dipole(ax, runs, colors, offset=args.dipole_offset)
    fig.savefig(out_dir / "dipole_vs_Q.png", bbox_inches="tight")
    plt.close(fig)

    # --- 4. sigma_ext + resonance trend ---------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(9, 10),
                            gridspec_kw={"height_ratios": [2, 1, 1]})
    draw_sigma(axs[0], runs, colors, emax=args.emax)
    draw_resonance_vs_Q(axs[1:], runs, out_dir, emax=args.emax)
    fig.tight_layout()
    fig.savefig(out_dir / "sigma_ext_vs_Q.png", bbox_inches="tight")
    plt.close(fig)

    # --- 5. one-page overview -------------------------------------------
    fig = plt.figure(figsize=(16, 5.5))
    # dedicated narrow column for the colourbar so it cannot collide with panel 2
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 0.035, 1, 1], wspace=0.55)
    ax0 = fig.add_subplot(gs[0])
    draw_level_ladder(ax0, runs, window)
    cbar = fig.colorbar(ScalarMappable(norm=OCC_NORM, cmap=OCC_CMAP),
                        cax=fig.add_subplot(gs[1]))
    cbar.set_label("occupation  $f_j$")
    draw_dipole(fig.add_subplot(gs[2]), runs, colors, offset=args.dipole_offset)
    draw_sigma(fig.add_subplot(gs[3]), runs, colors, emax=args.emax)
    fig.suptitle(f"Charge-doping sweep — {sweep_dir.name}", y=1.02)
    fig.savefig(out_dir / "Q_overview.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote plots to {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
