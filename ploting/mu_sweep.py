import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

au_nm = 0.0529177

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

data = np.loadtxt("sigma_mu_zigzag_2x2_rot0.txt")
data = data[np.lexsort((data[:,1], data[:,0]))]

mu = data[:,0]
omega = data[:,1]*27.2114

sigma_base = data[:,2] * au_nm**2
sigma_full = data[:,3] * au_nm**2

# ---------- grid ----------
mu_unique = np.unique(mu)
omega_unique = np.unique(omega)

n_mu = len(mu_unique)
n_omega = len(omega_unique)

base_grid = sigma_base.reshape(n_mu, n_omega).T
full_grid = sigma_full.reshape(n_mu, n_omega).T
diff_grid = full_grid - base_grid

extent = [
    mu_unique.min(), mu_unique.max(),
    omega_unique.min(), omega_unique.max()
]

# ---------- plotting ----------
fig, axes = plt.subplots(1,3, figsize=(16,6))

# dynamic range
vmax = max(base_grid.max(), full_grid.max())
vmin = 0

# σ_base
im0 = axes[0].imshow(
    base_grid,
    extent=extent,
    origin="lower",
    aspect="auto",
    cmap="jet",
    vmin=vmin,
    vmax=vmax, interpolation="bicubic"
)

axes[0].set_title(r"$\sigma_{\mathrm{base}}$")
axes[0].set_ylabel(r"Energy $\hbar\omega$ (eV)")

fig.colorbar(im0, ax=axes[0], label=r"$\sigma$")

# σ_full
im1 = axes[1].imshow(
    full_grid,
    extent=extent,
    origin="lower",
    aspect="auto",
    cmap="jet",
    vmin=vmin,
    vmax=vmax,     interpolation="bicubic"
)

axes[1].set_title(r"$\sigma_{\mathrm{full}}$")
axes[1].set_xlabel(r"Chemical potential $\mu$ (eV)")

fig.colorbar(im1, ax=axes[1], label=r"$\sigma$")

# σ_diff
#max_abs = np.percentile(np.abs(diff_grid), 80)
max_abs = np.max(np.abs(diff_grid))
norm = TwoSlopeNorm(
    vmin=-max_abs,
    vcenter=0,
    vmax=max_abs
)

im2 = axes[2].imshow(
    diff_grid,
    extent=extent,
    origin="lower",
    aspect="auto",
    cmap="seismic",
    norm=norm, interpolation="bicubic"
)

axes[2].set_title(r"$\sigma_{\mathrm{diff}}$")

fig.colorbar(im2, ax=axes[2], label=r"$\sigma_{\mathrm{full}}-\sigma_{\mathrm{base}}$")

plt.tight_layout()
plt.show()