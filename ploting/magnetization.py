#!/usr/bin/env python3
"""
magnetization.py — quantify the self-consistent Hubbard (UHF) magnetic ground
state written by a run with [features] hubbard = true.

Reads  magnetization.txt  (site x y sublattice n_up n_dn m_i) and:
  * maps the spin-density texture m_i on the lattice (blue up / red down),
  * reports the net moment and the sublattice-resolved moments,
  * checks the net spin against Lieb's theorem S = |N_A - N_B| / 2
    (equivalently the number of zero modes in eigenvalues.txt),
  * shows |m_i| vs distance from the flake centre (edge localization).

Usage:
    python3 magnetization.py [SIM_DIR]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

AU_NM = 0.0529177                       # bohr -> nm
COL_UP = "royalblue"
COL_DN = "crimson"

#DEFAULT_SIM = "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_fe5e6d629795c05"  # zz triangel 46
#DEFAULT_SIM = "Simulations/graphene_zigzag_bowtie_1d5ba9cb19481fc2" #bowtie 66, size 4 zigzag
#DEFAULT_SIM = "Simulations/graphene_armchair_triangle_11ffc9cf56cd2124" #armchair triangle 36
#DEFAULT_SIM = "Simulations/graphene_armchair_bowtie_133674db2277fe90" #AC Bowtie 68
DEFAULT_SIM =  "Simulations/graphene_zigzag_triangle_c65149ff485738f1" #ZZ triangle, u 0 3.64


sim_dir = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SIM)
mag_file = sim_dir / "magnetization.txt"
if not mag_file.exists():
    sys.exit(f"[magnetization] no magnetization.txt in {sim_dir}\n"
             f"                run with [features] hubbard = true first.")

d   = np.loadtxt(mag_file, comments="#")
site = d[:, 0].astype(int)
x, y = d[:, 1] * AU_NM, d[:, 2] * AU_NM
sub  = d[:, 3].astype(int)
n_up, n_dn = d[:, 4], d[:, 5]
m    = d[:, 6]
N    = len(m)

# --- lattice skeleton (optional) ---
bonds = None
bf = sim_dir / "bond_indices.txt"
if bf.exists():
    bonds = np.atleast_2d(np.loadtxt(bf, comments="#", dtype=int))

# --- zero-mode count (Lieb) from the spectrum, if available ---
n_zero = None
ef = sim_dir / "eigenvalues.txt"
if ef.exists():
    ev = np.loadtxt(ef)
    ev = ev[:, 0] if ev.ndim == 2 else ev
    n_zero = int(np.sum(np.abs(ev) < 1e-6)) // 2      # per spin

# ── quantities ────────────────────────────────────────────────────────────────
S_tot  = 0.5 * (n_up.sum() - n_dn.sum())
NA, NB = int(np.sum(sub > 0)), int(np.sum(sub < 0))
mA, mB = m[sub > 0].sum(), m[sub < 0].sum()
lieb_S = abs(NA - NB) / 2

print("-" * 60)
print(f"  sites                : {N}")
print(f"  net spin  S_z        : {S_tot:.4f}")
print(f"  sublattice A (n={NA:3d}): sum m = {mA:+.4f}")
print(f"  sublattice B (n={NB:3d}): sum m = {mB:+.4f}")
print(f"  N_A - N_B            : {NA - NB}")
# zero-mode count is only the Lieb determinant for the *non-interacting*
# spectrum; with hubbard on, U gaps them out (n_zero -> 0), so only report it
# when it is actually informative.
zero_note = (f"   [non-interacting zero modes/spin = {n_zero}]"
             if n_zero not in (None, 0) else "")
print(f"  Lieb S = |N_A-N_B|/2 : {lieb_S:.1f}{zero_note}")
print(f"  net moment == Lieb?  : {'YES' if abs(S_tot - lieb_S) < 1e-2 else 'NO'}")
print(f"  max |m_i|            : {np.abs(m).max():.3f}")
print("-" * 60)

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# panel 1: spin-density map (blue = spin up, red = spin down; same colors as
# the right panel). "bwr_r" reverses blue<->red so that m_i > 0 (n_up > n_dn,
# up-polarized) is blue and m_i < 0 (down-polarized) is red.
if bonds is not None:
    for i, j in bonds:
        ax1.plot([x[i], x[j]], [y[i], y[j]], color="0.0", lw=5, zorder=1)
mmax = np.abs(m).max() or 1.0
sizes = 120 + 900 * (np.abs(m) / mmax)
sc = ax1.scatter(x, y, c=m, s=sizes, cmap="bwr_r", vmin=-mmax, vmax=mmax,
                 edgecolors="k", linewidths=0.5, zorder=2)
# number each site so it can be matched to the right panel — white text with
# a dark outline reads on both the dark-blue/red fills and the near-white
# ones in the middle of the colormap, unlike plain black.
for i in range(N):
    ax1.annotate(str(site[i]), (x[i], y[i]), ha="center", va="center",
                 fontsize=8, color="white", weight="bold", zorder=3,
                 path_effects=[pe.withStroke(linewidth=2, foreground="black")])
cb = fig.colorbar(sc, ax=ax1, shrink=0.85)
cb.set_label(r"$m_i = \frac{1}{2}(n_{i\uparrow}-n_{i\downarrow})$", fontsize=13)
cb.ax.tick_params(labelsize=11)
ax1.set_aspect("equal")
ax1.set_xlabel("x (nm)"); ax1.set_ylabel("y (nm)")
ax1.set_title(f"spin density (blue = up, red = down)\n"
              f"net $S_z={S_tot:.2f}$, Lieb ${lieb_S:.0f}$")

# panel 2: spin occupation per site (n_up and n_dn)
# sort sites by moment so up-polarized sites (n_up > n_dn) sit on the left and
# down-polarized on the right; the vertical stem between the two markers is the
# local moment, and the dashed line is half-filling (n = 0.5).
# panel 2: spin occupation per site (n_up and n_dn)
# sort sites by moment so up-polarized sites (n_up > n_dn) sit on the left and
# down-polarized on the right; the vertical stem between the two markers is the
# local moment, and the dashed line is half-filling (n = 0.5).
order = np.argsort(-m)
idx = np.arange(N)

# faint background shading marks the up- vs down-polarized halves so the
# grouping reads at a glance even before looking at any markers
n_up_pol = int(np.sum(m[order] > 0))
ax2.axvspan(-0.5, n_up_pol - 0.5, color=COL_UP, alpha=0.06, zorder=0)
ax2.axvspan(n_up_pol - 0.5, N - 0.5, color=COL_DN, alpha=0.06, zorder=0)

ax2.vlines(idx, np.minimum(n_up, n_dn)[order], np.maximum(n_up, n_dn)[order],
           color="0.8", lw=1.5, zorder=1)
ax2.scatter(idx, n_up[order], marker="^", c="royalblue", s=30,
            label=r"$n_{i\uparrow}$ (up)", zorder=2)
ax2.scatter(idx, n_dn[order], marker="v", c="crimson", s=30,
            label=r"$n_{i\downarrow}$ (down)", zorder=2)
ax2.axhline(0.5, color="0.5", lw=0.8, ls="--")


max_labels = 20
step = max(1, int(np.ceil(N / max_labels)))
tick_idx = idx[::step]
ax2.set_xticks(tick_idx)
ax2.set_xticklabels(site[order][::step], rotation=90, fontsize=9)
ax2.set_xlabel("site index (sorted by moment)")
ax2.set_ylabel(r"occupation  $n_{i\sigma}$")
ax2.set_title("spin occupation per site")

ax2.legend(loc="center", frameon=True, facecolor="white",
           edgecolor="#cccccc", framealpha=0.9)
ax2.grid(alpha=0.3)

fig.tight_layout()
plt.show()
out = sim_dir / "magnetization.png"
fig.savefig(out, dpi=160)
fig.savefig(sim_dir / "magnetization.pdf")
print(f"[magnetization] saved {out}")
print(f"[magnetization] saved {sim_dir / 'magnetization.pdf'}")