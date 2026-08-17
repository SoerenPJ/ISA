#!/usr/bin/env python3
"""
hartree_groundstate.py — look at the self-consistent ground state of ONE run:
where the electrons ended up, and how the SCF loop got there.

Written for [features] hartree_scf = true (the Hubbard-free ground state, U = 0,
full V_rr' including the onsite diagonal — HUBBARD_FEATURE.md §13), but it works
on any run that produced an SCF, including hubbard = true.

Reads, from a single Simulations/<tag>/ directory:
    magnetization.txt        site x y sublattice n_up n_dn m_i  (+ header metadata)
    hubbard_convergence.txt  iter error N_total S_z
    bond_indices.txt         lattice skeleton (optional)

Panels:
  (a) charge map: dn_i = n_i - 1 on the lattice. NOT n_i — the density sits in a
      narrow band around 1 (e.g. 1.004..1.050), so a 0-based sequential scale shows
      a uniform blob. dn_i is also exactly what the Hartree is built on,
      phi_i = sum_j V_ee(i,j)(n_j - 1), and a symmetric diverging scale separates
      excess (edges) from depletion (interior) at a glance.
  (c) SCF residual max_i|dn_i| vs iteration, log scale, against the 1e-8 tolerance.
  (d) population drift N(it) - N(0) (NOT raw N: a 1e-4 leak would be invisible on
      an axis spanning 47) plus net S_z on the twin axis.
  (e,f) spin map and |m_i| vs radius — only when the run is actually magnetic
      (max|m_i| > 1e-6), i.e. skipped for hartree_scf, which is spin-blind.

Usage:
    python3 ploting/hartree_groundstate.py [SIM_DIR]
    (no argument -> the most recent Simulations/* that has a magnetization.txt)
"""

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "figure.dpi": 110,
})

AU_NM = 0.0529177          # bohr -> nm

TOL   = 1e-8               # [features] hubbard_tol default
SPIN_EPS = 1e-6             # below this the run is spin-blind; skip the spin panels

CMAP_CHARGE = "RdBu_r"     # dn_i: red = excess electrons, blue = depletion
CMAP_SPIN   = "bwr_r"      # m_i: blue = up, red = down (same as magnetization.py)
COL_A, COL_B = "#0072B2", "#D55E00"     # sublattice A / B

#DEFAULT_SIM = "Simulations/graphene_zigzag_triangle_bbe0c06e9fc5e063"# Q = 1
         # set a path here to pin one run; None = newest
#DEFAULT_SIM = "Simulations/graphene_zigzag_triangle_2f5fc2d6ff3d0714" #Q = 0
#DEFAULT_SIM = "Simulations/graphene_zigzag_triangle_8fd8de95f4422fd2" # Q = 2
#DEFAULT_SIM = "Simulations/graphene_zigzag_triangle_3a4e73516296203e" #Q = 1 + Hubbard
DEFAULT_SIM = "Simulations/graphene_zigzag_triangle_ba912c570474803d" # no spin check

def find_sim_dir():
    """CLI arg, else DEFAULT_SIM, else the most recently modified Simulations/*
    that actually contains an SCF result. Beats hand-editing a constant every run."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    if DEFAULT_SIM:
        return Path(DEFAULT_SIM)
    root = Path(__file__).resolve().parent.parent / "Simulations"
    cands = [d for d in root.glob("*/") if (d / "magnetization.txt").exists()]
    if not cands:
        sys.exit(f"[groundstate] no run with a magnetization.txt under {root}\n"
                 f"              run with [features] hartree_scf = true (or hubbard = true) first.")
    return max(cands, key=lambda d: (d / "magnetization.txt").stat().st_mtime)


def read_header(path):
    """key=value metadata out of the magnetization.txt comment block
    (U_eV, S_total, sum_abs_m, gap_eV, converged, iters)."""
    meta = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            for k, v in re.findall(r"(\w+)=([\-\d.eE+]+)", line):
                try:
                    meta[k] = float(v)
                except ValueError:
                    pass
    return meta


def colour_range(vals):
    """(vmin, vmax, cmap) for a site field.

    Symmetric diverging ONLY when the field actually changes sign. Doping puts dn_i
    entirely on one side of zero (e.g. +0.004..+0.050), and forcing a symmetric scale
    on that throws away half the colourbar and washes the whole interior out to near
    white — exactly the contrast you need to see where the electrons went. Then a
    sequential scale over the real data range is the honest choice; the panel title
    carries the absolute n_i range so nothing is hidden."""
    lo, hi = float(vals.min()), float(vals.max())
    if lo < -1e-12 and hi > 1e-12:
        v = max(abs(lo), abs(hi))
        return -v, v, CMAP_CHARGE
    if hi > 0:
        return lo, hi, "Reds"
    return lo, hi, "Blues_r"


def draw_map(ax, x, y, vals, cmap, vmin, vmax, bonds, label):
    """Site scatter over the lattice skeleton, marker size ∝ |value|.
    Same construction as hubbard_doping_maps.py's draw(), on one run."""
    if bonds is not None:
        for i, j in bonds:
            ax.plot([x[i], x[j]], [y[i], y[j]], color="0.75", lw=1.0, zorder=1)
    span = np.abs(vals).max() or 1.0
    sizes = 25 + 260 * (np.abs(vals) / span)
    sc = ax.scatter(x, y, c=vals, s=sizes, cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors="k", linewidths=0.3, zorder=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    cb = plt.gcf().colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=9)
    return sc


def main():
    sim_dir = find_sim_dir()
    mag_file  = sim_dir / "magnetization.txt"
    conv_file = sim_dir / "hubbard_convergence.txt"
    if not mag_file.exists():
        sys.exit(f"[groundstate] no magnetization.txt in {sim_dir}\n"
                 f"              run with [features] hartree_scf = true (or hubbard = true) first.")

    d = np.atleast_2d(np.loadtxt(mag_file, comments="#"))
    x, y = d[:, 1] * AU_NM, d[:, 2] * AU_NM
    sub  = d[:, 3].astype(int)
    n_up, n_dn, m = d[:, 4], d[:, 5], d[:, 6]
    n  = n_up + n_dn                      # charge density per site
    dn = n - 1.0                          # deviation from the ionic reference n = 1
    N  = len(n)

    meta = read_header(mag_file)
    U_eV      = meta.get("U_eV", float("nan"))
    converged = bool(meta.get("converged", 1))
    iters     = int(meta.get("iters", 0))
    gap_eV    = meta.get("gap_eV", float("nan"))
    hartree_free = abs(U_eV) < 1e-12

    bonds = None
    bf = sim_dir / "bond_indices.txt"
    if bf.exists():
        bonds = np.atleast_2d(np.loadtxt(bf, comments="#", dtype=int))

    conv = None
    if conv_file.exists():
        c = np.atleast_2d(np.loadtxt(conv_file, comments="#"))
        if c.size and c.shape[1] >= 4:
            conv = {"it": c[:, 0], "err": c[:, 1], "N": c[:, 2], "Sz": c[:, 3]}

    # distance from the flake centroid — the axis edge accumulation shows up on
    r = np.hypot(x - x.mean(), y - y.mean())
    has_spin = np.abs(m).max() > SPIN_EPS

    # ── console summary ──────────────────────────────────────────────────────────
    mode = "Hubbard-FREE Hartree (U = 0)" if hartree_free else f"UHF + Hartree (U = {U_eV:.4g} eV)"
    print("-" * 66)
    print(f"  run                  : {sim_dir.name}")
    print(f"  mode                 : {mode}")
    print(f"  converged            : {'YES' if converged else 'NO'}  ({iters} iterations)")
    if conv is not None:
        print(f"  final residual       : {conv['err'][-1]:.3e}   (tol {TOL:g})")
        print(f"  population drift     : {np.abs(conv['N'] - conv['N'][0]).max():.2e} e")
    print(f"  sites / electrons    : {N} / {n.sum():.4f}")
    print(f"  n_i range            : {n.min():.4f} .. {n.max():.4f}")
    print(f"  max |n_i - 1|        : {np.abs(dn).max():.4f}    charge std = {n.std():.4f}")
    print(f"  net S_z / max |m_i|  : {0.5 * (n_up.sum() - n_dn.sum()):+.3e} / {np.abs(m).max():.3e}"
          f"   -> {'MAGNETIC' if has_spin else 'spin-blind'}")
    print(f"  HOMO-LUMO gap        : {gap_eV:.4g} eV")
    print("-" * 66)
    if not converged:
        print("  !! NOT CONVERGED. The state is still stationary in the dynamics — H_eq and")
        print("     rho0 are built from the SAME occupations, so they commute regardless — but")
        print("     it is NOT self-consistent. Lower hubbard_mix (<= 0.02 for hartree_scf) or")
        print("     raise hubbard_max_iter.")
        print("-" * 66)

    # ── figure ───────────────────────────────────────────────────────────────────
    ncols = 3 if has_spin else 2
    fig, axes = plt.subplots(2, ncols, figsize=(5.6 * ncols, 10.2), squeeze=False)

    # (a) charge map
    vmin, vmax, cmap = colour_range(dn)
    draw_map(axes[0, 0], x, y, dn, cmap, vmin, vmax, bonds,
             r"$\delta n_i = n_i - 1$   (electrons/site)")
    signed = vmin < 0 < vmax
    axes[0, 0].set_title(f"(a) ground-state charge   $n_i \\in$ "
                         f"[{n.min():.4f}, {n.max():.4f}]\n"
                         + ("red = excess, blue = depleted"
                         + f"   ($N = {n.sum():.3f}$ e)"))

    # (b) [removed — radial charge profile] the slot stays in the grid layout so
    # (c)/(d)/(e)/(f) keep their original positions; just hide the empty axis.
    axes[0, 1].axis("off")

    # (c) SCF residual
    axc = axes[1, 0]
    if conv is not None:
        axc.semilogy(conv["it"], conv["err"], "-", color="crimson", lw=1.6)
        axc.axhline(TOL, color="0.5", lw=1.0, ls=":")
        # left-aligned: the residual curve lands in the bottom-RIGHT corner, right on
        # top of where a right-aligned label would sit
        axc.text(0.01, TOL * 1.5, f"tol {TOL:g}", transform=axc.get_yaxis_transform(),
                 ha="left", va="bottom", fontsize=9, color="0.4")
        state = "converged" if converged else "NOT CONVERGED"
        axc.set_title(f"(c) SCF convergence — {state}\n"
                      f"{iters} iterations, final {conv['err'][-1]:.2e}",
                      color="0.1" if converged else "crimson")
    else:
        axc.text(0.5, 0.5, "no hubbard_convergence.txt", transform=axc.transAxes,
                 ha="center", va="center", color="0.5")
        axc.set_title("(c) SCF convergence")
    axc.set_xlabel("self-consistency iteration")
    axc.set_ylabel(r"residual  $\max_i |\Delta n_i|$")
    axc.grid(alpha=0.3, which="both")

    # (d) population drift + S_z
    axd = axes[1, 1]
    if conv is not None:
        drift = conv["N"] - conv["N"][0]
        axd.axhline(0.0, color="0.5", lw=0.8, ls=":", zorder=1)
        axd.plot(conv["it"], drift, "-", color="royalblue", lw=2.0, zorder=3)
        axd.set_ylabel(r"population drift  (electrons)", color="royalblue")
        axd.tick_params(axis="y", labelcolor="royalblue")
        axd.yaxis.get_offset_text().set_fontsize(9)

        # S_z only gets its own axis when it MEANS something. In hartree_scf the field
        # is spin-blind, so S_z is pure round-off (~1e-13) — drawn as a curve it fills
        # the panel with noise that looks like structure. State it as a number instead.
        sz_live = np.abs(conv["Sz"]).max() > SPIN_EPS
        if sz_live:
            axd2 = axd.twinx()
            # dashed: a converged S_z is flat, and so is a conserved drift, so the two
            # curves can land on the same pixel row and read as one line
            axd2.plot(conv["it"], conv["Sz"], "--", color="seagreen", lw=1.6, alpha=0.9)
            axd2.set_ylabel(r"net spin  $S_z$", color="seagreen")
            axd2.tick_params(axis="y", labelcolor="seagreen")
            axd2.yaxis.get_offset_text().set_fontsize(9)
            sz_note = f"final $S_z$ = {conv['Sz'][-1]:+.3f}"
        else:
            axd.text(0.5, 0.86, f"$S_z \\equiv 0$  (spin-blind; "
                                f"$\\max|S_z| = {np.abs(conv['Sz']).max():.0e}$, round-off)",
                     transform=axd.transAxes, ha="center", va="center", fontsize=9.5,
                     color="seagreen",
                     bbox=dict(fc="white", ec="seagreen", alpha=0.85, pad=3.0))
            sz_note = "spin-blind"

        worst = np.abs(drift).max()
        axd.set_title("(d) population & spin through the loop\n"
                      + (f"population exactly conserved, {sz_note}" if worst == 0.0
                         else f"worst drift {worst:.1e} e, {sz_note}"))
    else:
        axd.text(0.5, 0.5, "no hubbard_convergence.txt", transform=axd.transAxes,
                 ha="center", va="center", color="0.5")
        axd.set_title("(d) population & spin")
    axd.set_xlabel("self-consistency iteration")
    axd.grid(alpha=0.3)

    # (e,f) spin — only for a genuinely magnetic run
    if has_spin:
        # spin stays on a symmetric scale whatever the data: up and down must read as
        # opposite, and 0 must be the neutral colour
        mmax = np.abs(m).max()
        draw_map(axes[0, 2], x, y, m, CMAP_SPIN, -mmax, mmax, bonds,
                 r"$m_i = \frac{1}{2}(n_{i\uparrow}-n_{i\downarrow})$")
        axes[0, 2].set_title(f"(e) spin density (blue = up, red = down)\n"
                             f"$S_z = {m.sum():.2f}$,  $\\sum_i|m_i| = {np.abs(m).sum():.2f}$")

        axf = axes[1, 2]
        for s, c, lab in ((+1, COL_A, "sublattice A"), (-1, COL_B, "sublattice B")):
            sel = sub == s
            if sel.any():
                axf.scatter(r[sel], m[sel], s=42, color=c, edgecolors="k",
                            linewidths=0.3, alpha=0.85, label=lab)
        axf.axhline(0.0, color="0.5", lw=0.9, ls="--")
        axf.set_xlabel("distance from flake centre (nm)")
        axf.set_ylabel(r"$m_i$")
        axf.set_title("(f) moment vs radius")
        axf.grid(alpha=0.3)
        axf.legend(frameon=False)

    title = ("Hubbard-free self-consistent Hartree ground state  ($U=0$, full $V_{rr'}$)"
             if hartree_free else
             f"Self-consistent UHF + Hartree ground state  ($U = {U_eV:.3g}$ eV)")
    fig.suptitle(f"{title}\n{sim_dir.name}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    for ext in ("png", "pdf"):
        out = sim_dir / f"hartree_groundstate.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[groundstate] saved {out}")
    plt.show()


if __name__ == "__main__":
    main()