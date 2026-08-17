# =============================================================================
# Figure 3 - The U_spin dial: the collective spin mode is a Stoner/exchange mode
#
# WHAT TO RUN: several LINEAR-kick simulations of the SAME flake, identical
# except for the Stoner coupling:
#   [field] mode = "ddf", spin_on = true, hubbard = true, hubbard_dynamic = true,
#   save_spin_diag = true,  and vary  hubbard_U_spin_eV  (e.g. 0, U/2, U).
# List each output folder + a label below.
#
# Top : charge absorption  ~ w*Im(dipole(w))   (proportional to your sigma_ext,
#       but computed here up to E_MAX so it reaches the resonances; the sigma_ext
#       file itself is capped at omega_cut_off eV). Should be ~invariant in U_spin.
# Bot : spin absorption  ~ w*Im(m_stag(w))  from the induced staggered moment,
#       same Fourier convention. The peak shifts/sharpens with U_spin.
#
# Same kernel e^{i w t} as the simulator; the w-weight kills the DC spike
# (Im -> 0 at w = 0), so the resonances are what you see.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

# (folder, label) - one per U_spin value
runs = [
    ("Simulations/graphene_zigzag_triangle_a9a90ce191f394f5", "U_spin = 0"),
    ("Simulations/graphene_zigzag_triangle_216a93b7dbc67370", "U_spin = U/2"),
    ("Simulations/graphene_zigzag_triangle_c93b3d4fa24f41d5", "U_spin = U"),
]

AU_EV = 27.2114
E_MAX = 14.0                                   # eV, upper edge of the plot
Egrid = np.linspace(0.05, E_MAX, 700)          # skip w=0
Wgrid = Egrid / AU_EV                          # angular frequency [a.u.]

def absorption(t, y):
    # w * Im( integral y(t) e^{i w t} dt )  -- trapezoid, matches the simulator
    t, idx = np.unique(t, return_index=True)   # sort + drop duplicate times
    y = np.asarray(y)[idx]
    y = y - y[0]                               # causal: start from 0
    Y = np.trapz(np.exp(1j * np.outer(Wgrid, t)) * y[None, :], t, axis=1)
    return Wgrid * np.abs(np.imag(Y))

def load_charge(path):
    d = np.loadtxt(path + "/dipole_time_evolution.txt", comments="#")
    return d[:, 0], d[:, 1]

def load_spin_stag(path):
    s = np.loadtxt(path + "/magnetization.txt", comments="#")[:, 3]     # sublattice +/-1
    sd = np.loadtxt(path + "/spin_diag_time_evolution.txt", comments="#")
    t = sd[:, 0]; N = (sd.shape[1] - 1) // 2
    return t, (sd[:, 1:1 + N] - sd[:, 1 + N:]) @ s

# spin gap of the (shared) ground state, for reference
gap = None
try:
    hdr = open(runs[0][0] + "/magnetization.txt").readlines()[1]
    gap = float([w for w in hdr.split() if w.startswith("gap_eV=")][0].split("=")[1])
except Exception:
    pass

colors = plt.cm.viridis(np.linspace(0.0, 0.8, len(runs)))
fig, (ax_c, ax_s) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)

for (path, label), c in zip(runs, colors):
    t, y = load_charge(path);     ax_c.plot(Egrid, absorption(t, y), color=c, lw=1.8, label=label)
    t, y = load_spin_stag(path);  ax_s.plot(Egrid, absorption(t, y), color=c, lw=1.8, label=label)

if gap is not None:
    for ax in (ax_c, ax_s):
        ax.axvline(gap, color="0.5", ls="--", lw=1.0)
    ax_c.text(gap + 0.1, ax_c.get_ylim()[1] * 0.85, f"spin gap {gap:.1f} eV", color="0.4", fontsize=9)

ax_c.set_ylabel(r"charge abs.  $\omega\,$Im$\,d(\omega)$")
ax_c.set_title(r"Charge channel - invariant under $U_{spin}$")
ax_s.set_ylabel(r"spin abs.  $\omega\,$Im$\,m_{stag}(\omega)$")
ax_s.set_title(r"Spin channel - collective mode shifts with $U_{spin}$")
ax_s.set_xlabel("photon energy [eV]")
ax_s.set_xlim(0, E_MAX)
for ax in (ax_c, ax_s):
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("uspin_dial.png", dpi=150)
print("saved: uspin_dial.png")
plt.show()
