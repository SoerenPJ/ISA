#!/usr/bin/env python3
"""
hubbard_spectrum.py — show how the spins are arranged in the *energy eigenstates*
of the self-consistent (UHF) equilibrium state.

Reads  hubbard_spectrum.txt  (index  energy_eV  spin  occupation) and:
  * draws a two-column level diagram (spin up on the left, spin down on the
    right), filled levels solid and empty levels open,
  * groups near-degenerate levels into single strokes (thicker line =
    more merged states), so a cluster of states doesn't read as noise,
  * shades the HOMO-LUMO gap, and reports the gap size, the exchange
    splitting between the up/down HOMOs, and the occupation summary
    (N_up, N_dn, S_z) in one clean info box.

Usage:
    python3 hubbard_spectrum.py [SIM_DIR]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

#DEFAULT_SIM = "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_fe5e6d629795c05"  # <-- edit
#DEFAULT_SIM = "Simulations/graphene_zigzag_bowtie_1d5ba9cb19481fc2" #bowtie 66, size 4 zigzag
#DEFAULT_SIM = "Simulations/graphene_armchair_triangle_11ffc9cf56cd2124" #armchair triangle 36
#DEFAULT_SIM = "Simulations/graphene_armchair_bowtie_133674db2277fe90" #AC Bowtie 68
DEFAULT_SIM =  "Simulations/graphene_zigzag_triangle_c65149ff485738f1" #ZZ triangle, u 0 3.64

# tweak these to taste — labels/title, tick numbers, and everything else
# scale off these three
FS_LABEL = 22
FS_TITLE = 24
FS_TICK  = 18

# ----------------------------------------------------------------------
# style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS_TICK,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

COL_UP = "#2455A4"      # deep blue
COL_UP_LIGHT = "#8FA9D6"
COL_DN = "#C23B4B"      # deep red
COL_DN_LIGHT = "#E1969F"
COL_GAP = "#F5C453"     # warm amber for the gap band
COL_EF = "#666666"

DEGEN_TOL = 1e-3        # eV; levels closer than this are treated as one


def load(spec_file: Path):
    d = np.atleast_2d(np.loadtxt(spec_file, comments="#"))
    return d[:, 1], d[:, 2].astype(int), d[:, 3]


def group_levels(energy, occ, tol=DEGEN_TOL):
    """Merge near-degenerate levels -> list of (energy, occupied?, multiplicity)."""
    order = np.argsort(energy)
    e_sorted, o_sorted = energy[order], occ[order]
    groups = []
    i = 0
    n = len(e_sorted)
    while i < n:
        j = i + 1
        while j < n and (e_sorted[j] - e_sorted[i]) < tol:
            j += 1
        e_mean = e_sorted[i:j].mean()
        filled = o_sorted[i:j].mean() > 0.5
        groups.append((e_mean, filled, j - i))
        i = j
    return groups


def draw_column(ax, x, groups, color_filled, color_empty):
    # degeneracy is shown by making the stroke a bit thicker/wider per
    # merged level, capped so it never gets silly — no text labels needed.
    for e, filled, mult in groups:
        boost = min(mult - 1, 3)  # cap the visual growth
        half = 0.34 + 0.03 * boost
        if filled:
            ax.hlines(e, x - half, x + half, color=color_filled,
                       lw=2.4 + 0.9 * boost, capstyle="round", zorder=3)
        else:
            ax.hlines(e, x - half, x + half, color=color_empty,
                       lw=1.6 + 0.6 * boost, linestyle=(0, (1, 1.4)), zorder=2)


def main():
    sim_dir = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SIM)
    spec_file = sim_dir / "hubbard_spectrum.txt"
    if not spec_file.exists():
        sys.exit(f"[spectrum] no hubbard_spectrum.txt in {sim_dir}\n"
                  f"           run with [features] hubbard = true first.")

    energy, spin, occ = load(spec_file)
    up, dn = spin > 0, spin < 0

    occupied = occ > 0.5
    homo = energy[occupied].max()
    lumo = energy[~occupied].min()
    e_fermi = 0.5 * (homo + lumo)
    gap = lumo - homo

    homo_up = energy[up & occupied].max()
    homo_dn = energy[dn & occupied].max()
    exch_split = abs(homo_up - homo_dn)

    n_up, n_dn = occ[up].sum(), occ[dn].sum()
    sz = 0.5 * (n_up - n_dn)

    print("-" * 60)
    print(f"  states (up / dn)     : {up.sum()} / {dn.sum()}")
    print(f"  electrons (up / dn)  : {n_up:.3f} / {n_dn:.3f}")
    print(f"  net S_z              : {sz:.4f}")
    print(f"  HOMO / LUMO   [eV]   : {homo:.4f} / {lumo:.4f}")
    print(f"  HOMO-LUMO gap [eV]   : {gap:.4f}")
    print(f"  exchange splitting   : {exch_split:.4f} eV (between up/dn HOMO)")
    print("-" * 60)

    groups_up = group_levels(energy[up], occ[up])
    groups_dn = group_levels(energy[dn], occ[dn])

    fig, ax = plt.subplots(figsize=(8.6, 9.5))

    # shade the HOMO-LUMO gap so it reads as a feature, not a font of dots
    ax.axhspan(homo, lumo, color=COL_GAP, alpha=0.18, zorder=0)
    ax.axhline(e_fermi, color=COL_EF, lw=1.2, linestyle="--", zorder=1)

    # y-span of the whole plot, used to offset labels off the line itself
    y_lo = min(energy.min(), homo) - 0.4
    y_hi = max(energy.max(), lumo) + 0.4
    y_span = y_hi - y_lo
    ef_offset = 0.028 * y_span

    ax.text(1.68, e_fermi + ef_offset, "$E_F$", va="bottom", ha="left",
             color=COL_EF, fontsize=FS_LABEL,
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    draw_column(ax, 0.0, groups_up, COL_UP, COL_UP_LIGHT)
    draw_column(ax, 1.0, groups_dn, COL_DN, COL_DN_LIGHT)

    # single, uncluttered info box: occupations + gap + exchange splitting
    # (kept as short, stacked lines rather than one wide line, so it doesn't
    # run off the figure at large font sizes)
    info_lines = [
        f"$N_\\uparrow$ = {n_up:.2f}   $N_\\downarrow$ = {n_dn:.2f}",
        f"$S_z$ = {sz:.3f}",
        f"gap = {gap:.3f} eV",
    ]
    if exch_split > 1e-4:
        info_lines.append(f"$\\Delta_{{exch}}$ = {exch_split*1000:.1f} meV")
    ax.text(0.02, 0.985, "\n".join(info_lines), transform=ax.transAxes,
             va="top", ha="left", fontsize=FS_TICK, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.5", fc="white",
                        ec="#cccccc", lw=0.8))

    # legend proxies for filled/empty
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#555555", lw=2.6, label="occupied"),
        Line2D([0], [0], color="#999999", lw=1.8, linestyle=(0, (1, 1.4)),
               label="empty"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=FS_TICK)

    ax.set_xlim(-0.9, 1.9)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([r"spin $\uparrow$", r"spin $\downarrow$"], fontsize=FS_LABEL)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylabel("energy (eV)", fontsize=FS_LABEL)
    ax.set_title("UHF equilibrium spectrum", fontsize=FS_TITLE, weight="bold", pad=16)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    out = sim_dir / "hubbard_spectrum.png"
    fig.savefig(out, dpi=200)
    fig.savefig(sim_dir / "hubbard_spectrum.pdf")
    print(f"[spectrum] saved {out}")
    print(f"[spectrum] saved {sim_dir / 'hubbard_spectrum.pdf'}")
    plt.show()


if __name__ == "__main__":
    main()