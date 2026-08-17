"""
plot_spin_ratio_paper.py
========================
Single-panel paper-ready figure: spin-to-charge ratio
|J_s(omega)|^2 / |J(omega)|^2 for all four graphene nanostructures.

Uses trapezoidal DFT matching the C++ compute_dipole_acceleration,
handling non-uniform (adaptive RK45) time grids correctly.

Style matches biot_savart_zdecay_paper.py exactly.
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  UNITS
# ============================================================
AU_EV = 27.2114
AU_NM = 0.0529177

# ============================================================
#  USER SETTINGS
# ============================================================
STRUCTURES = [
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_bowtie_10x10_rot0",
        "mu": 3.36, "level": "L2",
        "label": r"AC Bowtie",
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_bowtie_15x15_rot90",
        "mu": 3.52, "level": "L2",
        "label": r"ZZ Bowtie",
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_triangle_14x14_rot0",
        "mu": 3.52, "level": "L2",
        "label": r"AC Triangle",
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_triangle_22x22_rot90",
        "mu": 3.52, "level": "L2",
        "label": r"ZZ Triangle",
    },
]

# Match z-decay figure colors: dark blue, mid blue, dark red, bright red
COLORS  = ['#08306b', '#2171b5', '#cb181d', '#a50f15']
MARKERS = ['o', 's', '^', 'D']

# Frequency grid resolution — matches sigma_ext grid from file
N_OMEGA = 500   # points between 0.05 eV and 1.5 * resonance

# ============================================================
#  STYLE — identical to z-decay figure
# ============================================================
plt.rcParams.update({
    "text.usetex":     True,
    "font.family":     "Times new roman",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# ============================================================
#  TRAPEZOIDAL DFT — handles non-uniform time grids
#  Matches C++ compute_dipole_acceleration exactly:
#    integral = trapz( arr(t) * exp(i*omega*t), t )
#    power    = |integral|^2  summed over channels
# ============================================================

def trapz_dft_power(t, arr, omega_eV):
    """
    Vectorised trapezoidal DFT for non-uniform time grid.

    Parameters
    ----------
    t        : (N_t,)       time array in a.u.
    arr      : (N_t, N_chan) signal array
    omega_eV : (N_omega,)   frequencies to evaluate in eV

    Returns
    -------
    power : (N_omega,)  sum of |integral|^2 over channels
    """
    omega_au = omega_eV / AU_EV            # (N_omega,)
    N_t      = len(t)
    N_omega  = len(omega_au)
    N_chan   = arr.shape[1]

    # expo[w, i] = exp(i * omega_au[w] * t[i])
    # shape: (N_omega, N_t)
    expo = np.exp(1j * np.outer(omega_au, t))   # (N_omega, N_t)

    power = np.zeros(N_omega)
    for c in range(N_chan):
        # integrand[w, i] = arr[i, c] * exp(i*omega*t[i])
        integrand = expo * arr[:, c][np.newaxis, :]  # (N_omega, N_t)

        # trapezoidal integration along time axis for each omega
        # np.trapz(integrand, t, axis=1) → (N_omega,)
        integral = np.trapz(integrand, t, axis=1)    # (N_omega,)
        power   += np.abs(integral)**2

    return power


# ============================================================
#  HELPERS
# ============================================================

def find_mu_dir(sweep_dir, level, mu):
    pattern    = os.path.join(sweep_dir, f"{level}_mu_*")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No dirs matching {pattern}")
    def extract_mu(p):
        m = re.search(r"mu_([\d.]+)$", p)
        return float(m.group(1)) if m else np.inf
    return min(candidates, key=lambda p: abs(extract_mu(p) - mu))


def load_sigma(path):
    d        = np.loadtxt(path, comments="#")
    omega_eV = d[:, 0] * AU_EV
    sig      = d[:, 1]
    i_res    = np.argmax(sig)
    return omega_eV, sig, omega_eV[i_res]


# ============================================================
#  MAIN
# ============================================================
fig, ax = plt.subplots(figsize=(3.4, 3.2))

sigma_plotted = False

for struct, color, marker in zip(STRUCTURES, COLORS, MARKERS):
    sweep_dir = struct["sweep_dir"]
    mu        = struct["mu"]
    level     = struct["level"]
    label     = struct["label"]

    print(f"\n{'='*50}\n{label}")

    try:
        path = find_mu_dir(sweep_dir, level, mu)
    except FileNotFoundError as e:
        print(f"  [WARNING] {e}")
        continue

    # sigma_ext
    try:
        omega_sig, sig, res_eV = load_sigma(
            os.path.join(path, "sigma_ext.txt"))
        print(f"  Resonance: {res_eV:.3f} eV")
    except Exception as e:
        print(f"  [WARNING] sigma_ext: {e}")
        continue

    # Charge current [t, Jx, Jy]
    try:
        cur      = np.loadtxt(
            os.path.join(path, "current_time_evolution.txt"),
            comments="#")
        t        = cur[:, 0]          # a.u., non-uniform
        J_charge = cur[:, 1:]         # (N_t, 2)
    except Exception as e:
        print(f"  [WARNING] charge current: {e}")
        continue

    # Spin current [t, J↑x, J↑y, J↓x, J↓y, Js_x, Js_y]
    sc_path    = os.path.join(path, "spin_current_sc_time_evolution.txt")
    plain_path = os.path.join(path, "spin_current_time_evolution.txt")
    sc_file    = sc_path if os.path.isfile(sc_path) else plain_path
    try:
        sc       = np.loadtxt(sc_file, comments="#")
        N_t      = min(len(t), len(sc))
        t        = t[:N_t]
        J_charge = J_charge[:N_t, :]
        J_spin   = sc[:N_t, 5:7]     # Js_x, Js_y
    except Exception as e:
        print(f"  [WARNING] spin current: {e}")
        continue

    # Frequency grid: plasmon region only
    f_min   = 0.5   # eV — below lowest resonance
    f_max   = min(omega_sig.max(), 1.5 * res_eV)
    omega_eV = np.linspace(f_min, f_max, N_OMEGA)

    print(f"  Computing trapezoidal DFT on {N_OMEGA} frequency points ...")
    P_charge = trapz_dft_power(t, J_charge, omega_eV)
    P_spin   = trapz_dft_power(t, J_spin,   omega_eV)

    # Spin-to-charge ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(P_charge > 0, P_spin / P_charge, np.nan)

   

    # Resonance marker
    ax.axvline(res_eV, color=color, lw=2.5, ls=':', alpha=1.0, zorder=1)

    # Plot ratio
    valid = np.isfinite(ratio)
    n_pts = valid.sum()
    every = max(1, n_pts // 8)
    ax.plot(omega_eV[valid], ratio[valid],
            color=color, lw=1.5,
            marker=marker, markevery=every, ms=4,
            label=label, zorder=3)

    print(f"  Ratio at resonance: "
          f"{np.interp(res_eV, omega_eV[valid], ratio[valid]):.3e}")

# ── Styling ───────────────────────────────────────────────────────────────
ax.set_xlabel(r"$\hbar\omega\ (\mathrm{eV})$")

ax.set_xlim(1, 3.0)
#ax.set_yscale('log')
ax.set_ylabel(r"$|J_s(\omega)|^2 / |J(\omega)|^2$")
ax.set_ylim(bottom=0)
ax.legend(framealpha=0.9, loc='upper left',
          handlelength=1.5, handletextpad=0.5)
ax.grid(True, ls=':', alpha=0.4, which='both')
ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

plt.tight_layout()
out = "spin_ratio_paper.pdf"
plt.savefig(out, bbox_inches='tight', dpi=300)
print(f"\nSaved: {out}")
plt.show()