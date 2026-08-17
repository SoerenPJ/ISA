#!/usr/bin/env python3
"""
spin_quantify.py — three quantifiable spin figures-of-merit from one run.

Each metric maps onto a fundamental relation in
  Maekawa et al., "Spin and spin current — From fundamentals to recent
  progress" (arXiv:2211.02241), Sec. II.

  (1) Charge -> spin conversion ratio   P(w) = |J_s(w)| / |J_c(w)|
      Eq. (1)-(2):  J_s = j_up - j_dn ,  J_c = j_up + j_dn .
      The review calls the charge-to-spin conversion ratio the fundamental
      figure of merit / obstacle (Sec. III).  We report it at the drive /
      plasmon resonance.

  (2) S_z conservation / relaxation torque   dS_tot/dt
      Eq. (5)-(7):  dM/dt = -gamma div J_s + T .
      For an isolated flake the boundary spin flux vanishes, so any drift of
      the total induced spin S_z measures a relaxation torque T.  With
      Zeeman-only coupling along z, S_z commutes with H and S_z is conserved:
      this panel is a consistency check now, and *measures* T once SOC is on.

  (3) Edge spin accumulation   d(t) = sum_i s_i(t) r_i   (+ relaxation time)
      Eq. (3)-(4) / Fig. 1(b):  up-spin piles on one edge, down on the other,
      i.e. a spatial imbalance analogous to (mu_up - mu_dn).  We track the
      spin-density dipole moment and fit the post-pulse decay -> tau.

Usage:
    python3 spin_quantify.py [SIM_DIR]

Needs a run with spin_on = true and [analysis] save_spin_diag = true.
"""

import sys
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ── unit conversions (match the other ploting/ scripts) ───────────────────────
AU_EV = 27.2114
AU_S  = 2.41888e-17
AU_FS = AU_S * 1e15
AU_NM = 0.0529177                      # bohr -> nm

# ── analysis knobs ────────────────────────────────────────────────────────────
XMAX_EV      = 6.0                     # spectral window for the resonance search
NFFT_PAD     = 8                       # zero-padding factor (smoother spectra)
JC_FLOOR_FRAC = 0.02                   # mask |Jc| below 2% of its peak for P(t)
FLOOR        = 1e-40                   # avoid log(0)

#DEFAULT_SIM = ("/home/soeren/University/masters/2.semester/ISA/scr/"
              # "Simulations/graphene_zigzag_triangle_562f6e0395ef0e7e")
DEFAULT_SIM  = "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_8013b7c09112da08"   # <-- edit: folder with the output files

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.5,
})


# ── helpers ───────────────────────────────────────────────────────────────────
def hann_power_spectrum(x, dt, npad=NFFT_PAD):
    """Zero-padded Hann-windowed power spectrum. Returns (omega_eV, power)."""
    w   = np.hanning(len(x))
    w  *= len(x) / w.sum()             # amplitude-preserving normalisation
    xw  = x * w
    ft  = np.fft.rfft(xw, n=len(xw) * npad) * dt
    om  = 2.0 * np.pi * np.fft.rfftfreq(len(xw) * npad, d=dt) * AU_EV
    return om, np.abs(ft) ** 2


def read_pulse(sim_dir):
    """Return (center_fs, width_fs) from input.toml; fall back to 250 / 100."""
    center, width = 250.0, 100.0
    f = sim_dir / "input.toml"
    if f.exists():
        txt = f.read_text()
        m = re.search(r"^\s*t_shift\s*=\s*([-\d.eE+]+)", txt, re.M)
        if m:
            center = float(m.group(1))
        m = re.search(r"^\s*sigma_gaus\s*=\s*([-\d.eE+]+)", txt, re.M)
        if m:
            width = float(m.group(1))
    return center, width


def dominant_peak(omega, power, lo=0.05, hi=XMAX_EV):
    """omega of the most prominent spectral peak in (lo, hi]; else global max."""
    m = (omega > lo) & (omega <= hi)
    om, pw = omega[m], power[m]
    if pw.size == 0 or np.max(pw) <= 0:
        return np.nan
    peaks, props = find_peaks(pw, prominence=np.max(pw) * 0.1)
    if peaks.size == 0:
        return om[int(np.argmax(pw))]
    return om[peaks[int(np.argmax(props["prominences"]))]]


def eval_at(omega, y, w0):
    """Linear interpolation of y(omega) at w0."""
    return float(np.interp(w0, omega, y))


def uniform_resample(t, sigs):
    """The adaptive solver writes a non-uniform (and duplicated) time axis.
    Drop non-increasing points and interpolate the signals onto a uniform grid
    of the same length, so FFTs and derivatives are well defined.
    Returns (t_uni, dt_uni, [resampled signals])."""
    keep = np.concatenate(([True], np.diff(t) > 0))     # strictly increasing subset
    ts   = t[keep]
    t_uni = np.linspace(ts[0], ts[-1], ts.size)
    dt_uni = t_uni[1] - t_uni[0]
    out = [np.interp(t_uni, ts, s[keep]) for s in sigs]
    return t_uni, dt_uni, out


# ── load ──────────────────────────────────────────────────────────────────────
sim_dir = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SIM)
print(f"[spin_quantify] SIM_DIR = {sim_dir}")

sc  = np.loadtxt(sim_dir / "spin_current_time_evolution.txt", comments="#")
jc  = np.loadtxt(sim_dir / "current_time_evolution.txt",      comments="#")
dg  = np.loadtxt(sim_dir / "spin_diag_time_evolution.txt",    comments="#")
xy  = np.loadtxt(sim_dir / "lattice_points.txt", comments="#")[:, :2] * AU_NM

t_au = sc[:, 0]
dt_au = t_au[1] - t_au[0]
t_fs = t_au * AU_FS

# spin / charge current components
Js_x, Js_y = sc[:, 5], sc[:, 6]
Jc_x, Jc_y = jc[:, 1], jc[:, 2]
Js_mag = np.hypot(Js_x, Js_y)
Jc_mag = np.hypot(Jc_x, Jc_y)

# induced per-site spin  s_i(t) = rho_up,ii - rho_dn,ii   (spin_diag is induced)
diag   = dg[:, 1:]
N      = diag.shape[1] // 2
spin   = diag[:, :N] - diag[:, N:]         # (nt, N)
S_tot  = spin.sum(axis=1)                  # net induced S_z(t)

center_fs, width_fs = read_pulse(sim_dir)
env = np.exp(-((t_fs - center_fs) ** 2) / width_fs ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 1 — charge -> spin conversion ratio  P(w) = |J_s(w)| / |J_c(w)|
# ══════════════════════════════════════════════════════════════════════════════
# Use the driven (x) component for the spectral ratio; report magnitude-based
# time-domain ratios alongside.  Resample to a uniform grid first (adaptive
# solver -> non-uniform t), otherwise the FFT frequency axis is meaningless.
_, dt_uni, (Jc_x_u, Js_x_u) = uniform_resample(t_au, [Jc_x, Js_x])
om, Pc_x = hann_power_spectrum(Jc_x_u, dt_uni)
_,  Ps_x = hann_power_spectrum(Js_x_u, dt_uni)
amp_ratio = np.sqrt((Ps_x + FLOOR) / (Pc_x + FLOOR))     # |Js(w)|/|Jc(w)|

w_res = dominant_peak(om, Pc_x)                          # resonance from charge spectrum
P_res = eval_at(om, amp_ratio, w_res) if np.isfinite(w_res) else np.nan

# time-domain robustness numbers
mask = Jc_mag > JC_FLOOR_FRAC * Jc_mag.max()
P_inst_med = float(np.median(Js_mag[mask] / Jc_mag[mask])) if mask.any() else np.nan
P_rms  = float(np.sqrt(np.mean(Js_mag**2) / np.mean(Jc_mag**2)))
P_peak = float(Js_mag.max() / Jc_mag.max())


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 2 — S_z conservation / relaxation torque  T(t) = dS_tot/dt
# ══════════════════════════════════════════════════════════════════════════════
# derivative on a uniform grid (raw t_fs has duplicate stamps -> divide by zero)
tu, dtu_au, (S_tot_u,) = uniform_resample(t_au, [S_tot])
T_of_t = np.gradient(S_tot_u, dtu_au * AU_FS)            # 1/fs  (net spin torque)
S_peak    = float(np.abs(S_tot).max())
S_final   = float(S_tot[-1])
S_drift   = float(np.abs(S_tot[-1] - S_tot[0]))
# drift relative to how large the signal got: ~1 => not conserved, <<1 => conserved
conservation = S_drift / (S_peak + FLOOR)


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 3 — edge spin accumulation dipole  d(t) = sum_i s_i(t) r_i   (+ tau)
# ══════════════════════════════════════════════════════════════════════════════
dip_x = spin @ xy[:, 0]
dip_y = spin @ xy[:, 1]
dip   = np.hypot(dip_x, dip_y)                           # |dipole|(t)  [nm]
dip_peak = float(dip.max())
t_dip_peak = float(t_fs[int(np.argmax(dip))])

# relaxation time: fit the envelope (peaks) of |d(t)| after the pulse to
# A exp(-(t-t0)/tau)  ->  slope of log(peak) vs t.
t_post = center_fs + 1.5 * width_fs
post = t_fs > t_post
tau_fs = np.nan
fit_t = fit_env = None
if post.sum() > 10 and dip[post].max() > 0:
    tp, dp = t_fs[post], dip[post]
    pk, _ = find_peaks(dp)
    if pk.size >= 3:
        tt, yy = tp[pk], dp[pk]
        good = yy > 0
        tt, yy = tt[good], yy[good]
        if tt.size >= 3:
            slope, intercept = np.polyfit(tt, np.log(yy), 1)
            if slope < 0:
                tau_fs = -1.0 / slope
                fit_t = tt
                fit_env = np.exp(intercept + slope * tt)


# ── report ────────────────────────────────────────────────────────────────────
def line(): print("-" * 66)
print()
line()
print(f"  N_sites = {N}   t_max = {t_fs[-1]:.1f} fs   pulse @ {center_fs:.0f} fs "
      f"(width {width_fs:.0f} fs)")
line()
print("  [1] CHARGE -> SPIN CONVERSION RATIO  P = |Js| / |Jc|   (Eq. 1-2)")
print(f"      resonance omega            : {w_res:.3f} eV")
print(f"      P at resonance             : {P_res:.3e}")
print(f"      P (median, |Jc|>{JC_FLOOR_FRAC:.0%} peak) : {P_inst_med:.3e}")
print(f"      P (rms-integrated)         : {P_rms:.3e}")
print(f"      P (peak/peak)              : {P_peak:.3e}")
line()
print("  [2] S_z CONSERVATION / RELAXATION TORQUE   (Eq. 5-7)")
print(f"      peak |S_tot(t)|            : {S_peak:.3e}")
print(f"      final S_tot                : {S_final:.3e}")
print(f"      drift / peak               : {conservation:.3e}   "
      f"(<<1 => S_z conserved, no spin sink)")
print(f"      peak |T| = |dS/dt|         : {np.abs(T_of_t).max():.3e}  1/fs")
line()
print("  [3] EDGE SPIN ACCUMULATION DIPOLE  d = sum_i s_i r_i   (Eq. 3-4)")
print(f"      peak |dipole|              : {dip_peak:.3e} nm  @ {t_dip_peak:.1f} fs")
print(f"      relaxation time tau        : "
      + (f"{tau_fs:.1f} fs" if np.isfinite(tau_fs)
         else "n/a (no clean decaying envelope)"))
line()


# ── figure ────────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(3, 1, figsize=(8.2, 10.5), constrained_layout=True)

# panel 1: conversion ratio spectrum
ax = axs[0]
band = (om > 0.05) & (om <= XMAX_EV)
ax.semilogy(om[band], amp_ratio[band], color="seagreen", lw=1.3,
            label=r"$P(\omega)=|J_s|/|J_c|$")
if np.isfinite(w_res):
    ax.axvline(w_res, color="gray", ls="--", lw=1.0,
               label=rf"resonance $\omega_0={w_res:.2f}$ eV")
    ax.plot([w_res], [P_res], "o", color="crimson", ms=6,
            label=rf"$P(\omega_0)={P_res:.2e}$")
ax.set_xlim(0, XMAX_EV)
ax.set_xlabel("energy (eV)")
ax.set_ylabel(r"$|J_s|/|J_c|$")
ax.set_title("[1] charge $\\to$ spin conversion ratio")
ax.legend(loc="best")

# panel 2: S_z conservation
ax = axs[1]
ax.fill_between(t_fs, env * S_peak, color="#ffcc88", alpha=0.5,
                label="pulse envelope (arb.)")
ax.plot(t_fs, S_tot, color="#003399", lw=1.3,
        label=r"net induced $S_z=\sum_i(\rho^\uparrow_{ii}-\rho^\downarrow_{ii})$")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("time (fs)")
ax.set_ylabel(r"$S_z^{\rm tot}(t)$")
ax.set_title(f"[2] $S_z$ conservation  (drift/peak = {conservation:.1e})")
ax.legend(loc="best")

# panel 3: edge accumulation dipole + tau fit
ax = axs[2]
ax.plot(t_fs, dip, color="#7a3aab", lw=1.2, label=r"$|\mathbf{d}(t)|=|\sum_i s_i \mathbf{r}_i|$")
if fit_env is not None:
    ax.plot(fit_t, fit_env, "--", color="crimson", lw=1.5,
            label=rf"decay fit  $\tau={tau_fs:.0f}$ fs")
ax.axvline(center_fs, color="gray", ls=":", lw=1.0, label="pulse center")
ax.set_xlabel("time (fs)")
ax.set_ylabel(r"spin dipole $|\mathbf{d}|$ (nm)")
ax.set_title(f"[3] edge spin accumulation  (peak {dip_peak:.2e} nm)")
ax.legend(loc="best")

out = sim_dir / "spin_quantify.png"
fig.savefig(out, dpi=160)
fig.savefig(sim_dir / "spin_quantify.pdf")
plt.show()
print(f"[spin_quantify] saved {out}")
print(f"[spin_quantify] saved {sim_dir / 'spin_quantify.pdf'}")
