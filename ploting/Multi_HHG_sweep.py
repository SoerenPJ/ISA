import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter

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
data1 = np.loadtxt("dipole_acc_mu_zigzag_5x5_rot90.txt")
data2 = np.loadtxt("dipole_acc_mu_zigzag_10x10_rot90.txt")

datasets = [data1, data2]
labels = ["5x5", "10x10"]
mu = [data1]
fig, axes = plt.subplots(2, 3, figsize=(16,10))

for row, data in enumerate(datasets):

    # ---- sort for reshape ----
    data = data[np.lexsort((data[:,1], data[:,0]))]

    mu = data[:,0]
    omega = data[:,1] * au_eV
    I_base = data[:,2]
    I_full = data[:,3]

    mu_unique = np.unique(mu)
    omega_unique = np.unique(omega)

    n_mu = len(mu_unique)
    n_omega = len(omega_unique)

    # ---- reshape ----
    base_grid = I_base.reshape(n_mu, n_omega).T
    full_grid = I_full.reshape(n_mu, n_omega).T

    # ---- log scale ----
    eps = 1e-30
    base_grid = np.log10(base_grid + eps)
    full_grid = np.log10(full_grid + eps)

    # ---- smooth (same as your HHG script) ----
    base_grid = gaussian_filter(base_grid, sigma=0.4)
    full_grid = gaussian_filter(full_grid, sigma=0.4)

    diff_grid = full_grid - base_grid

    # ---- dynamic range (same idea as HHG) ----
    vmax = max(base_grid.max(), full_grid.max())
    vmin = vmax - 16

    # ---- remove noise in diff ----
    signal_threshold = vmax - 10
    mask = (base_grid < signal_threshold) & (full_grid < signal_threshold)
    diff_grid = np.ma.masked_where(mask, diff_grid)

    # ---- harmonic axis (IMPORTANT) ----
    harmonic = np.zeros_like(omega)

    for m in mu_unique:
        mask_mu = mu == m
        omega_slice = omega[mask_mu]
        I_slice = I_full[mask_mu]

        omega0 = omega_slice[np.argmax(I_slice)]
        harmonic[mask_mu] = omega_slice / omega0

    harmonic_axis = harmonic[mu == mu_unique[0]]

    extent = [
        mu_unique.min(), mu_unique.max(),
        harmonic_axis.min(), harmonic_axis.max()
    ]

    # ---- HHG base ----
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

    # ---- HHG full ----
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

    # ---- HHG diff ----
    max_abs = np.percentile(np.abs(diff_grid.compressed()), 20)

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
        cmap="RdBu_r",
        norm=norm,
        interpolation="bicubic"
    )

    # ---- titles ----
    axes[row,0].set_title(rf"HHG base ({labels[row]})")
    axes[row,1].set_title(rf"HHG full ({labels[row]})")
    axes[row,2].set_title(rf"HHG diff ({labels[row]})")

    # ---- colorbars ----
    fig.colorbar(im0, ax=axes[row,0], label=r"$\log_{10}|\ddot{p}(\omega)|^2$")
    fig.colorbar(im1, ax=axes[row,1], label=r"$\log_{10}|\ddot{p}(\omega)|^2$")
    fig.colorbar(im2, ax=axes[row,2], label=r"$\log_{10}(I_{\mathrm{full}}/I_{\mathrm{base}})$")


# ---- axis labels ----
axes[0,0].set_ylabel(r"Harmonic order $\omega/\omega_0$")
axes[1,0].set_ylabel(r"Harmonic order $\omega/\omega_0$")

axes[1,1].set_xlabel(r"Chemical potential $\mu$ (eV)")

plt.tight_layout()
plt.show()