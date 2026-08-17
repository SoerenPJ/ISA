import numpy as np
import matplotlib.pyplot as plt



SIM_DIR = "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_abdfdc535a1b3acc"
AU_FS   = 2.41888e-17 * 1e15            # atomic time unit -> fs
AU_NM   = 0.0529177                     # bohr -> nm

# pulse (time impulse): env = exp(-((t - center)^2) / width^2), all in fs
PULSE_CENTER_FS = 250.0
PULSE_WIDTH_FS  = 100.0

# --- structure ---
coords  = np.loadtxt(f"{SIM_DIR}/lattice_points.txt", comments="#")[:, :2] * AU_NM
x, y    = coords[:, 0], coords[:, 1]
bonds   = np.atleast_2d(np.loadtxt(f"{SIM_DIR}/bond_indices.txt", comments="#", dtype=int))

# --- equilibrium density matrix rho0 (site basis, l-space) ---
raw0    = np.loadtxt(f"{SIM_DIR}/rho0_l_space.txt", comments="#")
N_mat   = raw0.shape[0]
N       = N_mat // 2
rho0    = (raw0[:, 0::2] + 1j * raw0[:, 1::2])            # (N_mat, N_mat)
d0      = np.real(np.diag(rho0))
s_eq    = d0[:N] - d0[N:]                                # equilibrium spin per site

# --- spin-resolved induced diagonal rho_ii(t)-rho0_ii (lean: N_mat reals/row) ---
raw     = np.loadtxt(f"{SIM_DIR}/spin_diag_time_evolution.txt", comments="#")
t_fs    = raw[:, 0] * AU_FS
diag    = raw[:, 1:]                                     # (nt, N_mat)  [up..., dn...]
s_ind   = diag[:, :N] - diag[:, N:]                      # induced spin per site

# --- total spin ---
s_tot   = s_eq[None, :] + s_ind                          # (nt, N)

# --- flip report: a site "flips" if its total spin ever changes sign vs equilibrium ---
# (a pulse-driven flip is often transient, so we check all times, not just the end).
tol         = 0.05 * np.abs(s_eq).max() if np.abs(s_eq).max() > 0 else 1e-9
eq_sites    = np.abs(s_eq) > tol
sign_diff   = (np.sign(s_tot) != np.sign(s_eq)[None, :]) & eq_sites[None, :]  # (nt, N)
flip_any    = sign_diff.any(axis=0)                     # flipped at some time
flip_end    = sign_diff[-1]                             # still flipped at end
print(f"equilibrium spin-polarised sites : {eq_sites.sum()}")
print(f"sites flipped at some time        : {flip_any.sum()}  -> {np.where(flip_any)[0].tolist()}")
print(f"sites still flipped at end of run : {flip_end.sum()}  (transient flips recover)")

# =====================================================================
# figure: equilibrium (before) + final (after) on the structure,
#         and total-spin kymograph so sign-flips are visible over time
# =====================================================================
vmax = max(np.abs(s_eq).max(), np.abs(s_tot).max()) or 1e-12

mosaic = [["eq", "post"],
          ["kymo", "kymo"]]
fig, axd = plt.subplot_mosaic(mosaic, figsize=(11, 8), constrained_layout=True)
fig.suptitle(r"Total spin  (spin $\uparrow$ - spin $\downarrow$):  where it sits and whether the pulse flips it")

for name, svals, title in [("eq", s_eq, f"equilibrium  (t = {t_fs[0]:.0f} fs)"),
                           ("post", s_tot[-1], f"after pulse  (t = {t_fs[-1]:.0f} fs)")]:
    ax = axd[name]
    for a, b in bonds:
        ax.plot([x[a], x[b]], [y[a], y[b]], color="#cccccc", lw=0.6, zorder=1)
    sc = ax.scatter(x, y, c=svals, cmap="bwr", vmin=-vmax, vmax=vmax,
                    s=70, edgecolors="k", linewidths=0.3, zorder=2)
    ax.set_title(title)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(sc, ax=[axd["eq"], axd["post"]], shrink=0.7,
             label=r"$\rho^\uparrow_{ii}-\rho^\downarrow_{ii}$")

axk = axd["kymo"]
im  = axk.imshow(s_tot.T, origin="lower", aspect="auto", cmap="bwr",
                 vmin=-vmax, vmax=vmax,
                 extent=[t_fs.min(), t_fs.max(), 0, N])
for tt in (PULSE_CENTER_FS - PULSE_WIDTH_FS, PULSE_CENTER_FS, PULSE_CENTER_FS + PULSE_WIDTH_FS):
    axk.axvline(tt, color="k", ls="--", lw=0.8)
axk.set_xlabel("Time [fs]  (dashed = pulse center +/- width)")
axk.set_ylabel("site index")
axk.set_title("total spin per site vs time  (colour crossing white = flip)")
fig.colorbar(im, ax=axk, label=r"$\rho^\uparrow_{ii}-\rho^\downarrow_{ii}$")

plt.savefig("spin_flip.png", dpi=150, bbox_inches="tight")
print("Saved: spin_flip.png")
plt.show()
