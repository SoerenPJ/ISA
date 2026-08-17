#!/usr/bin/env python3
"""
hubbard_convergence.py — show how the self-consistency (UHF) loop converges.

Reads  hubbard_convergence.txt  (iter  error  N_total  S_z) and:
  * plots the error (max_i |delta n_i|) vs iteration on a log scale
    -> a smooth decrease is healthy; sudden jumps hint at a bad local minimum,
  * plots the total electron count N_total vs iteration
    -> it should stay flat (population conservation).

Usage:
    python3 hubbard_convergence.py [SIM_DIR]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

#DEFAULT_SIM = "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_fe5e6d629795c05"  # <-- edit
#DEFAULT_SIM = "Simulations/graphene_zigzag_bowtie_1d5ba9cb19481fc2" #bowtie 66, size 4 zigzag
#DEFAULT_SIM = "Simulations/graphene_armchair_triangle_11ffc9cf56cd2124" #armchair triangle 36
#DEFAULT_SIM = "Simulations/graphene_armchair_bowtie_133674db2277fe90" #AC Bowtie 68
#DEFAULT_SIM =  "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_498a298cb32e6a77" #ZZ triangle, u 0 3.64
DEFAULT_SIM = "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_7254da71e213a34f"

# tweak these to taste — labels/title, tick numbers, and legend can be sized
# independently
FS_LABEL = 26
FS_TITLE = 28
FS_TICK  = 22

sim_dir = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SIM)
conv_file = sim_dir / "hubbard_convergence.txt"
if not conv_file.exists():
    sys.exit(f"[convergence] no hubbard_convergence.txt in {sim_dir}\n"
              f"              run with [features] hubbard = true first.")

d = np.atleast_2d(np.loadtxt(conv_file, comments="#"))
it   = d[:, 0].astype(int)
err  = d[:, 1]
ntot = d[:, 2]
sz   = d[:, 3]

# population conservation: how much did N_total drift over the run?
drift = ntot.max() - ntot.min()
print("-" * 60)
print(f"  iterations           : {len(it)}")
print(f"  final error          : {err[-1]:.2e}")
print(f"  N_total (first/last) : {ntot[0]:.6f} / {ntot[-1]:.6f}")
print(f"  N_total drift        : {drift:.2e}  (should be ~0)")
print(f"  final net S_z        : {sz[-1]:.4f}")
print("-" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

# panel 1: error vs iteration (log scale)
ax1.semilogy(it, err, "o-", color="crimson", ms=4)
ax1.set_xlabel("iteration", fontsize=FS_LABEL)
ax1.set_ylabel(r"error  $\max_i |\Delta n_i|$", fontsize=FS_LABEL)
ax1.set_title("self-consistency convergence", fontsize=FS_TITLE)
ax1.grid(alpha=0.3, which="both")
ax1.tick_params(axis="both", labelsize=FS_TICK)

# panel 2: total population vs iteration (should be flat)
ax2.plot(it, ntot, "o-", color="royalblue", ms=4)
ax2.set_xlabel("iteration", fontsize=FS_LABEL)
ax2.set_ylabel(r"total electrons  $N_\mathrm{total}$", fontsize=FS_LABEL)
ax2.set_title(f"population conservation (drift = {drift:.1e})", fontsize=FS_TITLE)
ax2.grid(alpha=0.3)
ax2.tick_params(axis="both", labelsize=FS_TICK)
# the offset text (e.g. "1e-8 + 2.4e1") on a tightly-scaled y-axis is a tick
# label too, so bump its size to match rather than leaving it tiny
ax2.yaxis.get_offset_text().set_fontsize(FS_TICK)

fig.tight_layout()
plt.show()
out = sim_dir / "hubbard_convergence.png"
fig.savefig(out, dpi=160)
fig.savefig(sim_dir / "hubbard_convergence.pdf")
print(f"[convergence] saved {out}")
print(f"[convergence] saved {sim_dir / 'hubbard_convergence.pdf'}")