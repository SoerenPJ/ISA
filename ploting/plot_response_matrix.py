# =============================================================================
# Figure 1 - Response matrix:  (linear | nonlinear) x (charge | spin)
#
# WHAT TO RUN: two simulations of the SAME flake, both with
#   spin_on = true, hubbard = true, hubbard_dynamic = true, save_spin_diag = true
#   - one with [field] mode = "ddf"          (weak kick -> linear response)  -> path_linear
#   - one with [field] mode = "time_impulse" (strong pulse -> HHG/nonlinear) -> path_nonlinear
# Put the two output folders below.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

path_linear    = "Simulations/graphene_zigzag_triangle_c93b3d4fa24f41d5"
#path_nonlinear = "Simulations/graphene_zigzag_triangle_dff5c85b6318cbe7" off reso
path_nonlinear = "Simulations/graphene_zigzag_triangle_fff7329ac02be929" # on reso

HARTREE_eV = 27.2114

def spectrum(t, y):
    # resample the non-uniform (and possibly duplicated) time grid, then FFT
    t, idx = np.unique(t, return_index=True)      # sort + drop duplicate times
    y = np.asarray(y)[idx]
    y = y - y.mean()                              # remove DC
    n = len(t)
    tu = np.linspace(t[0], t[-1], n)
    yu = np.interp(tu, t, y)
    Y = np.fft.rfft(yu * np.hanning(n))
    f = np.fft.rfftfreq(n, d=(tu[1] - tu[0]))
    E = 2 * np.pi * f * HARTREE_eV                # photon energy [eV]
    return E, np.abs(Y)

def load_charge(path):
    d = np.loadtxt(path + "/dipole_time_evolution.txt", comments="#")
    return d[:, 0], d[:, 1]                        # t, dipole

def load_spin_stag(path):
    # induced staggered (Neel) moment:  sum_i s_i ( up_i(t) - dn_i(t) )
    s = np.loadtxt(path + "/magnetization.txt", comments="#")[:, 3]   # sublattice +/-1
    sd = np.loadtxt(path + "/spin_diag_time_evolution.txt", comments="#")
    t = sd[:, 0]; ind = sd[:, 1:]
    N = ind.shape[1] // 2
    m = ind[:, :N] - ind[:, N:]                    # induced local moment per site
    return t, m @ s                                # staggered moment vs time

fig, ax = plt.subplots(2, 2, figsize=(11, 8))

# --- charge, linear (top-left) ---
t, y = load_charge(path_linear);      E, A = spectrum(t, y)
ax[0, 0].plot(E, A);                  ax[0, 0].set_title("charge - linear (ddf)")
ax[0, 0].set_ylabel("|dipole(w)|  (absorption)")

# --- charge, nonlinear / HHG (top-right) ---
t, y = load_charge(path_nonlinear);   E, A = spectrum(t, y)
ax[0, 1].semilogy(E, (E**2 * A)**2);  ax[0, 1].set_title("charge - nonlinear (time_impulse, HHG)")
ax[0, 1].set_ylabel("|acc(w)|^2")

# --- spin, linear (bottom-left) ---
t, y = load_spin_stag(path_linear);   E, A = spectrum(t, y)
ax[1, 0].plot(E, A);                  ax[1, 0].set_title("spin - linear (ddf)")
ax[1, 0].set_ylabel("|m_stag(w)|  (spin susc.)")

# --- spin, nonlinear (bottom-right) ---
t, y = load_spin_stag(path_nonlinear); E, A = spectrum(t, y)
ax[1, 1].semilogy(E, A**2);           ax[1, 1].set_title("spin - nonlinear (time_impulse)")
ax[1, 1].set_ylabel("|m_stag(w)|^2")

for a in ax.ravel():
    a.set_xlim(0, 20)                 # eV; widen for HHG if needed
    a.set_xlabel("photon energy [eV]")

plt.tight_layout()
plt.savefig("response_matrix.png", dpi=150)
print("saved: response_matrix.png")
plt.show()
