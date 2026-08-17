import numpy as np
import matplotlib.pyplot as plt
import sys

# ============================================================
# Put the path to your simulation output folder here
# (or pass it on the command line: python plot_hubbard_dynamics.py <folder>)
# ============================================================
path = "Simulations/graphene_zigzag_triangle_4b55d0757ea677a"
if len(sys.argv) > 1:
    path = sys.argv[1]

# ---- load equilibrium magnetization (n_up, n_dn per site) ----
mag = np.loadtxt(path + "/magnetization.txt", comments="#")
n_up_eq = mag[:, 4]
n_dn_eq = mag[:, 5]
N = len(n_up_eq)

# ---- load induced diagonal rho_ii(t) - rho0_ii  (up block then dn block) ----
d = np.loadtxt(path + "/spin_diag_time_evolution.txt", comments="#")
t = d[:, 0]
ind_up = d[:, 1:1 + N]        # induced up occupations
ind_dn = d[:, 1 + N:1 + 2 * N]  # induced dn occupations

# ---- absolute occupations and magnetization at each time ----
n_up = n_up_eq + ind_up
n_dn = n_dn_eq + ind_dn
m = 0.5 * (n_up - n_dn)          # per-site moment, shape (time, site)
Sz = m.sum(axis=1)              # net spin S_z(t)
m_abs = np.abs(m).sum(axis=1)    # sum |m_i|(t)

# time in fs (1 a.u. of time = 0.0241888 fs)
t_fs = t * 0.0241888

# ---- plot ----
fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

ax[0].plot(t_fs, Sz)
ax[0].set_ylabel("S_z(t)")
ax[0].set_title("Net spin")

ax[1].plot(t_fs, m_abs)
ax[1].set_ylabel("sum |m_i|(t)")
ax[1].set_title("Total moment")

for i in range(N):
    ax[2].plot(t_fs, m[:, i])
ax[2].set_ylabel("m_i(t)")
ax[2].set_xlabel("time (fs)")
ax[2].set_title("Per-site moment")

plt.tight_layout()
plt.savefig(path + "/hubbard_dynamics.png", dpi=150)
print("saved:", path + "/hubbard_dynamics.png")
plt.show()
