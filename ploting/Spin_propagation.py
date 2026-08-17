import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# Plot the induced spin density  (spin-up block - spin-down block)
# on top of the lattice structure, at a few time snapshots.
#
# Per site i:  s_i(t) = Re[ rho_up(i,i) ] - Re[ rho_dn(i,i) ]   (induced)
#   blue  -> net spin up ,  red -> net spin down
#
# Needs a run with spin_on = true and [analysis] save_spin_diag = true.
# Just point SIM_DIR at the simulation output folder.
# =====================================================================

SIM_DIR  = "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_8013b7c09112da08"   # <-- edit: folder with the output files
#SIM_DIR   = "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_562f6e0395ef0e7e"
#SIM_DIR = "Simulations/graphene_zigzag_triangle_fe5e6d629795c05"

N_SNAP   = 5                            # number of time snapshots to show
AU_NM    = 0.0529177                    # bohr -> nm (just for nicer axes)
AU_FS    = 2.41888e-17 * 1e15           # atomic time unit -> fs

# pulse (time impulse): env = exp(-((t - center)^2) / width^2), all in fs
PULSE_CENTER_FS = 250.0
PULSE_WIDTH_FS  = 100.0

# --- structure: site coordinates ---
coords  = np.loadtxt(f"{SIM_DIR}/lattice_points.txt", comments="#")[:, :2] * AU_NM
x, y    = coords[:, 0], coords[:, 1]
N_sites = len(coords)

# --- bonds (for drawing the lattice skeleton) ---
bonds = np.loadtxt(f"{SIM_DIR}/bond_indices.txt", comments="#", dtype=int)
bonds = np.atleast_2d(bonds)

# --- spin-resolved induced diagonal rho_ii(t) - rho0_ii (lean: N_mat reals/row) ---
# each line: t  then N_mat reals, ordered [up_0..up_{N-1}, dn_0..dn_{N-1}]
raw    = np.loadtxt(f"{SIM_DIR}/spin_diag_time_evolution.txt", comments="#")
t_fs   = raw[:, 0] * AU_FS                             # time axis in fs
diag   = raw[:, 1:]                                    # (nt, N_mat) induced diagonal
assert diag.shape[1] == 2 * N_sites, "expected spin_on run: N_mat should be 2 * N_sites"

# --- per-site induced spin density for every time step ---
spin   = diag[:, :N_sites] - diag[:, N_sites:]          # up block - down block

# --- pick evenly spaced snapshots and a shared symmetric colour scale ---
snaps  = np.linspace(0, len(t_fs) - 1, N_SNAP, dtype=int)
vmax   = np.abs(spin[snaps]).max()
if vmax == 0:
    vmax = 1e-12

# top row = structure snapshots, bottom row = pulse envelope timeline
mosaic = [[f"s{i}" for i in range(N_SNAP)],
          ["pulse"] * N_SNAP]
fig, axd = plt.subplot_mosaic(mosaic, figsize=(3.2 * N_SNAP, 4.4),
                              height_ratios=[3, 1], constrained_layout=True)
fig.suptitle(r"Induced spin density  (spin $\uparrow$ - spin $\downarrow$)")

for i, k in enumerate(snaps):
    ax = axd[f"s{i}"]
    # lattice skeleton
    for a, b in bonds:
        ax.plot([x[a], x[b]], [y[a], y[b]], color="#cccccc", lw=0.6, zorder=1)
    # spin density on top
    sc = ax.scatter(x, y, c=spin[k], cmap="bwr", vmin=-vmax, vmax=vmax,
                    s=60, edgecolors="k", linewidths=0.3, zorder=2)
    ax.set_title(f"t = {t_fs[k]:.0f} fs")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

fig.colorbar(sc, ax=[axd[f"s{i}"] for i in range(N_SNAP)], shrink=0.7,
             label=r"$\rho^{\uparrow}_{ii} - \rho^{\downarrow}_{ii}$")

# --- pulse timeline: envelope vs fs with the snapshot times marked ---
tt  = np.linspace(t_fs.min(), t_fs.max(), 400)
env = np.exp(-((tt - PULSE_CENTER_FS) ** 2) / PULSE_WIDTH_FS ** 2)
axp = axd["pulse"]
axp.fill_between(tt, env, color="#ffcc88", alpha=0.7)
axp.plot(tt, env, color="#cc7a00", lw=1.0)
for k in snaps:
    axp.axvline(t_fs[k], color="k", ls="--", lw=0.8)
axp.set_xlim(t_fs.min(), t_fs.max())
axp.set_yticks([])
axp.set_xlabel("Time [fs]")
axp.set_title("pulse envelope", fontsize=9)

plt.savefig("spin_propagation.png", dpi=150, bbox_inches="tight")
print("Saved: spin_propagation.png")
plt.show()
