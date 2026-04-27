import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.spatial import cKDTree

au_eV  = 27.2114
au_nm  = 0.0529177

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
})

# ----------------------------------------------------------------
# Load data 
#graphene_armchair_1c2ccdd8d66aba19
# ----------------------------------------------------------------
#path = "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_14f441d4fe6c1d6b" #dipole_acc here
path = "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_726df5e8c662a269"
#path = "/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_armchair_1c2ccdd8d66aba19"
lattice  = np.loadtxt(f"{path}/lattice_points.txt")

sigma    = np.loadtxt(f"{path}/sigma_ext.txt")
#sigma = np.loadtxt(f"{path}/dipole_acc.txt") #dipole acc
J_bond_t = np.loadtxt(f"{path}/J_bond_time_evolution.txt")
B_ind_t  = np.loadtxt(f"{path}/B_ind_z_time_evolution.txt")

# Build bonds from lattice points
tree = cKDTree(lattice)
dists, _ = tree.query(lattice, k=2)
a_nn = dists[:, 1].min()
bonds_raw = np.array(sorted(tree.query_pairs(r=1.0005 * a_nn)), dtype=int)
print(f"Found {len(bonds_raw)} bonds, a_nn = {a_nn:.4f} a.u.")

time_au = J_bond_t[:, 0]
J_bond  = J_bond_t[:, 1:]
B_z     = B_ind_t[:, 1:]

# ----------------------------------------------------------------
# Fourier transforms
# ----------------------------------------------------------------
dt  = time_au[1] - time_au[0]
N_t = len(time_au)

J_fft   = np.fft.rfft(J_bond, axis=0)
freq_au = np.fft.rfftfreq(N_t, d=dt)
freq_eV = freq_au *au_eV

B_fft   = np.fft.rfft(B_z, axis=0) 

# Find plasmon resonance from sigma_ext
omega_au     = sigma[:, 0]
sig_vals     = sigma[:, 1]
i_res        = np.argmax(sig_vals)
omega_res_au = omega_au[i_res]
omega_res_eV = omega_res_au * au_eV
print(f"Plasmon resonance: {omega_res_eV:.3f} eV")

i_freq_res = np.argmin(np.abs(freq_eV - omega_res_eV))
print(f"Using frequency bin: {freq_eV[i_freq_res]:.3f} eV")

J_res        = np.abs(J_fft[i_freq_res, :])
B_res_signed = B_fft[i_freq_res, :].real

B_max_omega = np.max(np.abs(B_fft), axis=1)
B_rms_omega = np.sqrt(np.mean(np.abs(B_fft)**2, axis=1))

# ----------------------------------------------------------------
# Build figure
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

x = lattice[:, 0] * au_nm
y = lattice[:, 1] * au_nm

# ---- Panel A one: Bond current distribution ----
'''
ax = axes[0]
ax.set_title(r" Bond currents $|J_{ll'}(\omega_\mathrm{res})|$")

J_norm = J_res / J_res.max() if J_res.max() > 0 else J_res
lw_max = 4.0
lw_min = 0.2
cmap_J = cm.plasma

segments   = []
linewidths = []
colors     = []

for b, (i, j) in enumerate(bonds_raw):
    segments.append([(x[i], y[i]), (x[j], y[j])])
    linewidths.append(lw_min + (lw_max - lw_min) * J_norm[b])
    colors.append(cmap_J(J_norm[b]))

lc = mc.LineCollection(segments, linewidths=linewidths, colors=colors)
ax.add_collection(lc)
ax.scatter(x, y, s=5, c='gray', zorder=2, alpha=0.5)
ax.set_xlim(x.min() - 0.2, x.max() + 0.2)
ax.set_ylim(y.min() - 0.2, y.max() + 0.2)
ax.set_aspect('equal')
ax.set_xlabel(r"$x$ (nm)")
ax.set_ylabel(r"$y$ (nm)")

sm_J = cm.ScalarMappable(cmap=cmap_J, norm=Normalize(0, J_res.max()))
sm_J.set_array([])
fig.colorbar(sm_J, ax=ax, label=r"$|J_{ll'}(\omega_\mathrm{res})|$ (a.u.)", shrink=0.8)
'''
# ---- Panel A: Bond current as quiver ----
ax = axes[0]
ax.set_title(r" Bond currents $J_{ll'}(\omega_\mathrm{res})$")

# Compute current vector at each bond midpoint
mid_x = np.array([0.5*(x[i]+x[j]) for i,j in bonds_raw])
mid_y = np.array([0.5*(y[i]+y[j]) for i,j in bonds_raw])

# Direction along bond, scaled by current magnitude
dx = np.array([(x[j]-x[i]) for i,j in bonds_raw])
dy = np.array([(y[j]-y[i]) for i,j in bonds_raw])
bond_len = np.sqrt(dx**2 + dy**2)

# J_res is signed here — use real part to get direction
J_res_signed = J_fft[i_freq_res, :].real
u = J_res_signed * dx / bond_len
v = J_res_signed * dy / bond_len

ax.quiver(mid_x, mid_y, u, v, J_res_signed,
          cmap='seismic', scale=None, angles='xy')
ax.scatter(x, y, s=40, c='gray', zorder=2, alpha=0.5)
ax.set_xlim(x.min() - 0.2, x.max() + 0.2)
ax.set_ylim(y.min() - 0.2, y.max() + 0.2)
ax.set_aspect('equal')
ax.set_xlabel(r"$x$ (nm)")
ax.set_ylabel(r"$y$ (nm)")

sm_J = cm.ScalarMappable(cmap='seismic',
       norm=Normalize(-np.abs(J_res_signed).max(), np.abs(J_res_signed).max()))
sm_J.set_array([])
fig.colorbar(sm_J, ax=ax, label=r"$J_{ll'}(\omega_\mathrm{res})$ (a.u.)", fraction=0.046, pad=0.04)

# ---- Panel B: Site-resolved B_ind_z at resonance (signed) ----

ax = axes[1]
ax.set_title(r"$B_{\mathrm{ind},z}(r_l, \omega_\mathrm{res})$")

bmax = np.abs(B_res_signed).max()
sc = ax.scatter(x, y, c=B_res_signed, cmap='seismic', s=40,
                norm=Normalize(-bmax, bmax))
ax.set_aspect('equal')
ax.set_xlabel(r"$x$ (nm)")
ax.set_ylabel(r"$y$ (nm)")
fig.colorbar(sc, ax=ax, label=r"$B_{\mathrm{ind},z}$ (a.u.)", fraction=0.046, pad=0.04)


# ---- Panel C: B_max(omega) and B_rms(omega) vs sigma_ext ----
ax  = axes[2]
ax2 = ax.twinx()
ax.set_title(r"Magnetic response vs.\ extinction")

omega_eV = omega_au * au_eV
ax.plot(omega_eV, sig_vals* au_nm**2, 'b-', lw=1.5, label=r"$\sigma_\mathrm{ext}(\omega)$")
ax.set_xlabel(r"Energy $\hbar\omega$ (eV)")
ax.set_ylabel(r"$\sigma_\mathrm{ext}$", color='b')
ax.tick_params(axis='y', labelcolor='b')

mask = freq_eV <= omega_eV.max()
ax2.plot(freq_eV[mask], B_max_omega[mask], 'r-',  lw=1.5, label=r"$B_\mathrm{max}(\omega)$")
ax2.plot(freq_eV[mask], B_rms_omega[mask], 'r--', lw=1.5, label=r"$B_\mathrm{rms}(\omega)$")
ax2.set_ylabel(r"$B_\mathrm{ind}$ (a.u.)", color='r')
ax2.tick_params(axis='y', labelcolor='r')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig("three_panel_figure.pdf", bbox_inches='tight')
plt.show()
print("Saved three_panel_figure.pdf")