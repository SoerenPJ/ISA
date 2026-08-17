#!/usr/bin/env python3
"""
Spin-resolved current plots — Second goal from SSH__Spin_current.pdf Section 10.

Plots:
  Fig 1: Time domain  Js_alpha  (all three levels)
  Fig 2: |J_alpha(w)|^2 and |Js_alpha(w)|^2  with resonance marked
  Fig 3: Relative corrections  dJ(L0->L1)  and  dJ(L1->L2)
  Fig 4: Spin asymmetry  |Js| / |J|
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────
au_eV = 27.2114
au_s  = 2.41888e-17
au_fs = au_s * 1e15

# ── physics parameters ────────────────────────────────────────────────────────
OMEGA_RES_EV   = 1.676   # drive / plasmon resonance [eV]
XMAX_EV        = 5.0    # plot up to ~6th harmonic
NFFT_PAD       = 8       # zero-padding factor for smoother spectra
FLOOR          = 1e-40   # avoid log(0)

# ── style ─────────────────────────────────────────────────────────────────────
COLORS = {"L0": "steelblue", "L1": "darkorange", "L2": "seagreen"}
LW     = 1.4
ALPHA  = 0.9

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.5,
})
# Accept command-line dirs: spin_current_plots.py <L0_dir> <L1_dir> <L2_dir>
# Falls back to hardcoded paths when not supplied.
if len(sys.argv) == 4:
    runs_input = [
        ("L0", sys.argv[1]),
        ("L1", sys.argv[2]),
        ("L2", sys.argv[3]),
    ]
else:
    runs_input = [
        ("L0", "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_8afd5bafa0042218"),
        ("L1", "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_ac365639ccc7963"),
        ("L2", "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_8afd5bafa0042218"),
    ]

def hann_power_spectrum(x: np.ndarray, dt: float, npad: int = NFFT_PAD):
    """Zero-padded Hann-windowed power spectrum. Returns (omega_eV, power)."""
    w      = np.hanning(len(x))
    w     *= len(x) / w.sum()          # amplitude-preserve normalisation
    xw     = x * w
    n_pad  = len(xw) * npad
    ft     = np.fft.rfft(xw, n=n_pad) * dt
    omega  = 2.0 * np.pi * np.fft.rfftfreq(n_pad, d=dt) * au_eV
    return omega, np.abs(ft) ** 2

def style_log_ax(ax, ylabel, title=None):
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.yaxis.set_minor_locator(LogLocator(subs="auto", numticks=4))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True,  which="major", lw=0.5, alpha=0.5)
    ax.grid(False, which="minor")

def mark_resonance(ax):
    ax.axvline(OMEGA_RES_EV, color="gray", lw=1.0, ls="--", alpha=0.9,
               label=rf"plasmon $\omega_0={OMEGA_RES_EV}$ eV")

# ── load runs ─────────────────────────────────────────────────────────────────
runs = {}

for label, base_path in runs_input:
    base_dir   = Path(base_path)
    spin_file  = base_dir / "spin_current_time_evolution.txt"
    total_file = base_dir / "current_time_evolution.txt"

    if not spin_file.exists():
        print(f"[{label}] spin file missing — skipping.")
        continue

    spin   = np.loadtxt(spin_file)
    t_au   = spin[:, 0]
    J_up_x = spin[:, 1];  J_up_y = spin[:, 2]
    J_dn_x = spin[:, 3];  J_dn_y = spin[:, 4]
    J_s_x  = spin[:, 5];  J_s_y  = spin[:, 6]

    total  = np.loadtxt(total_file)
    J_x    = total[:, 1];  J_y = total[:, 2]

    dt_au  = t_au[1] - t_au[0]

    # build spectra once
    def sp(sig):
        return hann_power_spectrum(sig, dt_au)

    omega_x, _ = sp(J_x)   # same grid for all (same N after padding)

    runs[label] = {
        "color":   COLORS[label],
        "t_fs":    t_au * au_fs,
        "J_x":     J_x,    "J_y":    J_y,
        "J_up_x":  J_up_x, "J_up_y": J_up_y,
        "J_dn_x":  J_dn_x, "J_dn_y": J_dn_y,
        "J_s_x":   J_s_x,  "J_s_y":  J_s_y,
        "dt_au":   dt_au,
        # spectra — (omega_eV array, power array)
        "sp_x":    sp(J_x),    "sp_y":    sp(J_y),
        "sp_s_x":  sp(J_s_x),  "sp_s_y":  sp(J_s_y),
        "sp_up_x": sp(J_up_x), "sp_up_y": sp(J_up_y),
        "sp_dn_x": sp(J_dn_x), "sp_dn_y": sp(J_dn_y),
    }

if not runs:
    print("No data loaded. Exiting.")
    exit(1)

def pos_mask(omega):
    return (omega > 0) & (omega <= XMAX_EV)

out_dir = Path(runs_input[0][1]) / "spin_current_comparison"
out_dir.mkdir(parents=True, exist_ok=True)

xlabel = r"$\omega$ [eV]"

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — time domain spin current  J^s_alpha
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
fig.suptitle(r"Spin current $J^s_\alpha(t)$ — time domain", fontsize=13)

for ax, (key, lbl) in zip(axes, [("J_s_x", "x"), ("J_s_y", "y")]):
    for label, run in runs.items():
        ax.plot(run["t_fs"], run[key], lw=0.8,
                color=run["color"], alpha=ALPHA, label=label)
    ax.set_xlabel("Time [fs]")
    ax.set_ylabel(rf"$J^s_{lbl}$ [a.u.]")
    ax.set_title(rf"$J^s_{lbl}(t)$")
    ax.legend()

plt.tight_layout()
plt.savefig(out_dir / "fig1_time_domain_Js.png", dpi=150, bbox_inches="tight")
print("Saved fig1_time_domain_Js.png")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — |J_alpha(w)|^2  and  |J^s_alpha(w)|^2  with resonance marked
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    rf"Power spectra — resonance at $\omega_0 = {OMEGA_RES_EV}$ eV (dashed)",
    fontsize=13
)

spec_layout = [
    # (row, col, sp_key,   ylabel,                        title)
    (0, 0, "sp_x",   r"$|J_x(\omega)|^2$ [a.u.]",   r"Total $J_x$"),
    (0, 1, "sp_y",   r"$|J_y(\omega)|^2$ [a.u.]",   r"Total $J_y$"),
    (1, 0, "sp_s_x", r"$|J^s_x(\omega)|^2$ [a.u.]", r"Spin current $J^s_x$"),
    (1, 1, "sp_s_y", r"$|J^s_y(\omega)|^2$ [a.u.]", r"Spin current $J^s_y$"),
]

for row, col, sp_key, ylabel, title in spec_layout:
    ax = axes[row, col]
    for label, run in runs.items():
        omega, power = run[sp_key]
        mask = pos_mask(omega)
        ax.semilogy(omega[mask], power[mask], lw=LW,
                    color=run["color"], alpha=ALPHA, label=label)
    mark_resonance(ax)
    style_log_ax(ax, ylabel, title)
    ax.set_xlabel(xlabel)
    ax.legend()

plt.tight_layout()
plt.savefig(out_dir / "fig2_spectra_J_Js.png", dpi=150, bbox_inches="tight")
print("Saved fig2_spectra_J_Js.png")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — relative corrections  dJ(L0->L1)  and  dJ(L1->L2)
# ═══════════════════════════════════════════════════════════════════════════════

if "L0" in runs and "L1" in runs and "L2" in runs:

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        r"Relative corrections $\delta J_\alpha = |J^{B} - J^{A}| \, / \, |J^{A}|$",
        fontsize=13
    )

    corr_layout = [
        (0, 0, "sp_x",   "L0", "L1", r"$\delta J_x$  (L0$\to$L1)", r"$\delta J_x$"),
        (0, 1, "sp_y",   "L0", "L1", r"$\delta J_y$  (L0$\to$L1)", r"$\delta J_y$"),
        (1, 0, "sp_x",   "L1", "L2", r"$\delta J_x$  (L1$\to$L2)", r"$\delta J_x$"),
        (1, 1, "sp_y",   "L1", "L2", r"$\delta J_y$  (L1$\to$L2)", r"$\delta J_y$"),
    ]

    for row, col, sp_key, labA, labB, title, ylabel in corr_layout:
        ax = axes[row, col]

        omA, pwA = runs[labA][sp_key]
        omB, pwB = runs[labB][sp_key]

        # interpolate B onto A's grid if lengths differ
        if len(omA) != len(omB):
            pwB = np.interp(omA, omB, pwB)
            omB = omA

        mask  = pos_mask(omA)
        ampA  = np.sqrt(np.maximum(pwA[mask], FLOOR))
        ampB  = np.sqrt(np.maximum(pwB[mask], FLOOR))
        delta = np.abs(ampB - ampA) / ampA

        color = runs[labB]["color"]
        ax.semilogy(omA[mask], delta, lw=LW, color=color, alpha=ALPHA,
                    label=rf"{labA}$\to${labB}")
        mark_resonance(ax)
        style_log_ax(ax, ylabel, title)
        ax.set_xlabel(xlabel)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_relative_corrections.png", dpi=150, bbox_inches="tight")
    print("Saved fig3_relative_corrections.png")
    plt.show()

else:
    print("Need L0, L1, and L2 to plot corrections — skipping Fig 3.")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — spin asymmetry  |J^s(w)| / |J(w)|
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(r"Spin asymmetry  $|J^s(\omega)| \, / \, |J(\omega)|$", fontsize=13)

for ax, (sp_tot, sp_spin, lbl) in zip(axes, [
    ("sp_x", "sp_s_x", "x"),
    ("sp_y", "sp_s_y", "y"),
]):
    for label, run in runs.items():
        omT, pwT = run[sp_tot]
        omS, pwS = run[sp_spin]
        mask  = pos_mask(omT)
        ratio = np.sqrt(pwS[mask]) / np.sqrt(np.maximum(pwT[mask], FLOOR))
        ax.semilogy(omT[mask], ratio, lw=LW,
                    color=run["color"], alpha=ALPHA, label=label)
    mark_resonance(ax)
    style_log_ax(ax, rf"$|J^s_{lbl}| \, / \, |J_{lbl}|$")
    ax.set_xlabel(xlabel)
    ax.legend()

plt.tight_layout()
plt.savefig(out_dir / "fig4_spin_asymmetry.png", dpi=150, bbox_inches="tight")
print("Saved fig4_spin_asymmetry.png")
plt.show()

print(f"\nAll figures written to {out_dir}/")