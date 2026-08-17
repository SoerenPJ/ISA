import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# The "spin block difference" as a matrix (companion to Spin_propagation.py).
#   M = rho_up - rho_dn      (N_sites x N_sites, complex, induced)
# where rho_up = rho[0:N, 0:N] and rho_dn = rho[N:2N, N:2N].
#
# The structure plot shows only the diagonal of M (per-site spin density);
# here we show the full matrix so the off-diagonal coherences are visible.
#
# This is the ONLY spin plot that needs off-diagonals, so it requires the big
# full-matrix file: run with spin_on = true and [analysis] save_rho_full = true
# (the other spin plots only need the lean save_spin_diag file).
# =====================================================================

SIM_DIR = "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_abdfdc535a1b3acc"
AU_FS   = 2.41888e-17 * 1e15            # atomic time unit -> fs
TIME_FS = 250.0                         # snapshot time to show (nearest frame)

# --- induced density matrix rho(t) - rho0 ---
raw     = np.loadtxt(f"{SIM_DIR}/rho_full_induced_time_evolution.txt", comments="#")
t_fs    = raw[:, 0] * AU_FS
N_mat   = int(round(np.sqrt((raw.shape[1] - 1) / 2)))
N       = N_mat // 2

flat    = raw[:, 1:].reshape(len(t_fs), N_mat, N_mat, 2)
rho     = flat[..., 0] + 1j * flat[..., 1]

# --- pick the frame closest to TIME_FS and build M = rho_up - rho_dn ---
k       = int(np.argmin(np.abs(t_fs - TIME_FS)))
rho_up  = rho[k, :N, :N]
rho_dn  = rho[k, N:, N:]
M       = rho_up - rho_dn

vmax = np.abs(M).max() or 1e-12

fig, (axr, axi) = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
fig.suptitle(rf"Spin block difference $M=\rho^\uparrow-\rho^\downarrow$  at t = {t_fs[k]:.0f} fs")

imr = axr.imshow(M.real, cmap="bwr", vmin=-vmax, vmax=vmax)
axr.set_title("Re(M)")
axr.set_xlabel("site j"); axr.set_ylabel("site i")
fig.colorbar(imr, ax=axr, shrink=0.8)

imi = axi.imshow(M.imag, cmap="bwr", vmin=-vmax, vmax=vmax)
axi.set_title("Im(M)")
axi.set_xlabel("site j"); axi.set_ylabel("site i")
fig.colorbar(imi, ax=axi, shrink=0.8)

plt.savefig("spin_matrix.png", dpi=150, bbox_inches="tight")
print("Saved: spin_matrix.png")
plt.show()
