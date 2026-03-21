import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

au_nm = 0.0529177
au_eV = 27.2114

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# ---- Load datasets ----
data5 = np.loadtxt("sigma_mu_armchair_4x4_rot0.txt")
data10 = np.loadtxt("sigma_mu_armchair_8x8_rot0.txt")

datasets = [data5, data10]
labels = ["4x4", "8x8"]

fig, axes = plt.subplots(2, 3, figsize=(16,10))

for row, data in enumerate(datasets):

    # sort so reshape works
    data = data[np.lexsort((data[:,1], data[:,0]))]

    mu = data[:,0]
    omega = data[:,1] * au_eV

    sigma_base = data[:,2] * au_nm**2
    sigma_full = data[:,3] * au_nm**2

    mu_unique = np.unique(mu)
    omega_unique = np.unique(omega)

    n_mu = len(mu_unique)
    n_omega = len(omega_unique)

    # reshape grids
    base_grid = sigma_base.reshape(n_mu, n_omega).T
    full_grid = sigma_full.reshape(n_mu, n_omega).T
    diff_grid = full_grid - base_grid

    extent = [
        mu_unique.min(), mu_unique.max(),
        omega_unique.min(), omega_unique.max()
    ]

    # same dynamic range as single plot
    vmax = max(base_grid.max(), full_grid.max())
    vmin = 0

    # ---- σ_base ----
    im0 = axes[row,0].imshow(
        base_grid,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic"
    )

    # ---- σ_full ----
    im1 = axes[row,1].imshow(
        full_grid,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic"
    )

    # ---- σ_diff ----
    #max_abs = np.percentile(np.abs(diff_grid), 98)
    max_abs = np.max(np.abs(diff_grid))

    norm = TwoSlopeNorm(
        vmin=-max_abs,
        vcenter=0,
        vmax=max_abs
    )

    im2 = axes[row,2].imshow(
        diff_grid,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="seismic",
        norm=norm,
        interpolation="bicubic"
    )

    axes[row,0].set_title(rf"$\sigma_{{base}}$ ({labels[row]})")
    axes[row,1].set_title(rf"$\sigma_{{full}}$ ({labels[row]})")
    axes[row,2].set_title(rf"$\sigma_{{diff}}$ ({labels[row]})")

    fig.colorbar(im0, ax=axes[row,0], label=r"$\sigma$")
    fig.colorbar(im1, ax=axes[row,1], label=r"$\sigma$")
    fig.colorbar(im2, ax=axes[row,2], label=r"$\sigma_{\mathrm{full}}-\sigma_{\mathrm{base}}$")

# ---- axis labels ----
axes[0,0].set_ylabel(r"Energy $\hbar\omega$ (eV)")
axes[1,0].set_ylabel(r"Energy $\hbar\omega$ (eV)")

axes[1,1].set_xlabel(r"Chemical potential $\mu$ (eV)")

plt.tight_layout()
plt.show()