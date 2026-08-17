
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.spatial import cKDTree

AU_EV = 27.2114     # atomic units -> eV
AU_NM = 0.0529177   # atomic units -> nm

# path to the sigma sweep folder (relative to repo root, works from anywhere)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_DIR = os.path.join(REPO, "data_LLM", "sigma_sweep_mu_armchair_triangle_3x3_rot0")

# ---- area of the flake (nm^2), same definition as the paper: A = N_C * hex area ----
lattice = np.loadtxt(os.path.join(SWEEP_DIR, "lattice_points.txt"), comments="#")[:, :2]
tree = cKDTree(lattice)
d, _ = tree.query(lattice, k=2)
a_cc_au = d[:, 1][d[:, 1] > 0.1].min()          # C-C bond length in a.u.
N_atoms = len(lattice)
area_nm2 = (N_atoms / 2.0) * (3.0 * np.sqrt(3.0) / 2.0) * a_cc_au**2 * AU_NM**2
print(f"N_atoms={N_atoms}, a_cc={a_cc_au:.4f} a.u., area={area_nm2:.3f} nm^2")

# ---- load the sweep ----
files = sorted(glob.glob(os.path.join(SWEEP_DIR, "mu_*", "sigma_ext.txt")))
print("found", len(files), "files in", SWEEP_DIR)

mu_vals = []
columns = []
omega = None
for path in files:
    mu = float(os.path.basename(os.path.dirname(path)).replace("mu_", ""))
    data = np.loadtxt(path)
    if omega is None:
        omega = data[:, 0] * AU_EV
    mu_vals.append(mu)
    columns.append(data[:, 1] * AU_NM**2 / area_nm2)   # sigma_ext / A  (normalized)

mu_vals = np.array(mu_vals)
grid = np.column_stack(columns)   # shape: (n_omega, n_mu)

extent = [mu_vals.min(), mu_vals.max(), omega.min(), omega.max()]

plt.imshow(grid, extent=extent, origin="lower", aspect="auto", cmap="hot", vmin=0)
plt.plot(mu_vals, 2 * mu_vals, "--", color="cornflowerblue", lw=1)  # interband threshold hw = 2E_F
plt.ylim(omega.min(), omega.max())
plt.colorbar(label=r"$\sigma^\mathrm{ext}/A$")
plt.xlabel("Fermi energy (eV)")
plt.ylabel("Photon energy (eV)")
plt.show()
