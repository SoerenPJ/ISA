import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# Spin dynamics over time (companion to Spin_propagation.py).
#   top:    net induced spin  sum_i [ rho_up(i,i) - rho_dn(i,i) ]  vs time
#   bottom: kymograph  -  per-site spin density vs time (propagation)
#
# Needs a run with spin_on = true and [analysis] save_spin_diag = true.
# =====================================================================

SIM_DIR  = "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_8013b7c09112da08"   # <-- edit: folder with the output files
AU_FS   = 2.41888e-17 * 1e15            # atomic time unit -> fs

# pulse (time impulse): env = exp(-((t - center)^2) / width^2), all in fs
PULSE_CENTER_FS = 250.0
PULSE_WIDTH_FS  = 100.0

# --- spin-resolved induced diagonal rho_ii(t)-rho0_ii (lean: N_mat reals/row) ---
raw     = np.loadtxt(f"{SIM_DIR}/spin_diag_time_evolution.txt", comments="#")
t_fs    = raw[:, 0] * AU_FS
diag    = raw[:, 1:]                                     # (nt, N_mat)  [up..., dn...]
N_sites = diag.shape[1] // 2

# --- per-site induced spin density, then net (summed over sites) ---
spin    = diag[:, :N_sites] - diag[:, N_sites:]          # (nt, N_sites) up - down
net     = spin.sum(axis=1)                               # (nt,)

# this is the *induced* spin only (rho(t)-rho0); it has no equilibrium value.
# The signal is small (self-induced field ~ 1/c^2), so let it set its own scale.
peak = np.abs(spin).max() or 1e-12

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True,
                               sharex=True)

# --- top: net spin vs time, pulse envelope shown as a reference shape only ---
env = np.exp(-((t_fs - PULSE_CENTER_FS) ** 2) / PULSE_WIDTH_FS ** 2)
ax1.fill_between(t_fs, env * np.abs(net).max(), color="#ffcc88", alpha=0.5,
                 label="pulse envelope (arb.)")
ax1.plot(t_fs, net, color="#003399", lw=1.5, label=r"net spin $\sum_i(\rho^\uparrow_{ii}-\rho^\downarrow_{ii})$")
ax1.axhline(0, color="k", lw=0.6)
ax1.set_ylabel("net induced spin")
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))   # show the true scale
ax1.legend(loc="upper right", fontsize=9)

# --- bottom: kymograph, per-site induced spin density vs time (own scale) ---
im = ax2.imshow(spin.T, origin="lower", aspect="auto", cmap="bwr",
                vmin=-peak, vmax=peak,
                extent=[t_fs.min(), t_fs.max(), 0, N_sites])
ax2.set_xlabel("Time [fs]")
ax2.set_ylabel("site index")
fig.colorbar(im, ax=ax2, label=r"$\rho^\uparrow_{ii}-\rho^\downarrow_{ii}$")

fig.suptitle(rf"Induced spin dynamics ($\rho^\uparrow-\rho^\downarrow$)   "
             rf"peak per-site $|s|$ = {peak:.1e}")
plt.savefig("spin_vs_time.png", dpi=150, bbox_inches="tight")
print("Saved: spin_vs_time.png")
plt.show()
