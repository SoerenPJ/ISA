import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import FixedLocator

au_eV = 27.2114

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# ---------------- load data ----------------
data = np.loadtxt("dipole_acc_mu_zigzag_5x5_rot90.txt")
data = data[np.lexsort((data[:,1], data[:,0]))]

mu     = data[:,0]
omega  = data[:,1] * au_eV
I_base = data[:,2]
I_full = data[:,3]

# ---------------- grid shape ----------------
mu_unique = np.unique(mu)
n_mu      = len(mu_unique)
n_omega   = len(data) // n_mu

# ---------------- compute harmonic order ----------------
harmonic = np.zeros_like(omega)

for m in mu_unique:
    mask        = mu == m
    omega_slice = omega[mask]
    I_slice     = I_full[mask]
    omega0      = omega_slice[np.argmax(I_slice)]
    harmonic[mask] = omega_slice / omega0

# ================================================================
# PLOT 1: single-mu HHG spectrum (choose mu here)
# ================================================================
target_mu  = 2.3
mu_closest = mu_unique[np.argmin(np.abs(mu_unique - target_mu))]
mask_single = mu == mu_closest

omega_slice = omega[mask_single]
base_slice  = I_base[mask_single]
full_slice  = I_full[mask_single]
index       = np.argmax(base_slice)
omega0      = omega_slice[index]

x_val      = omega_slice / omega0

plt.figure(figsize=(6, 4))
plt.plot(x_val, base_slice, linewidth=2, label="base")
plt.plot(x_val, full_slice, linewidth=2, label="full")
plt.yscale("log")
plt.xlim(0, 11)
ax = plt.gca()
ax.xaxis.set_major_locator(FixedLocator(np.arange(0, 11, 1)))
ax.xaxis.set_minor_locator(FixedLocator([]))
plt.xlabel(r"$\omega/\omega_0$")
plt.ylabel(r"$|\ddot{p}(\omega)|^2$")
plt.title(rf"HHG spectrum at $\mu={mu_closest:.3f}$ eV")
plt.legend()
plt.grid(which='major', linestyle='-', linewidth=0.8)
plt.tight_layout()
plt.show()

# ================================================================
# PLOT 2: peak intensity ratio at harmonic orders 1, 3, 5, 7
# ================================================================
TARGET_ORDERS = [1, 3, 5, 7]
HALF_WIDTH    = 0.4   # ± window in harmonic order units

base_peaks = np.zeros((n_mu, len(TARGET_ORDERS)))
full_peaks = np.zeros((n_mu, len(TARGET_ORDERS)))

for i, m in enumerate(mu_unique):
    mask = mu == m
    h    = harmonic[mask]
    ib   = I_base[mask]
    if_  = I_full[mask]

    for j, order in enumerate(TARGET_ORDERS):
        in_band = np.abs(h - order) <= HALF_WIDTH
        if in_band.any():
            base_peaks[i, j] = ib[in_band].max()
            full_peaks[i, j] = if_[in_band].max()
        else:
            base_peaks[i, j] = np.nan
            full_peaks[i, j] = np.nan

# --- raw ratio (no normalization) ---
ratio = full_peaks / base_peaks

# --- mask where base signal is below noise floor ---
noise_floor = 1e-10
ratio = np.where(base_peaks > noise_floor, ratio, np.nan)

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
labels = [r"$\omega/\omega_0 = 1$",
          r"$\omega/\omega_0 = 3$",
          r"$\omega/\omega_0 = 5$",
          r"$\omega/\omega_0 = 7$"]

fig, ax = plt.subplots(figsize=(8, 5))
for j, (color, label) in enumerate(zip(colors, labels)):
    ax.plot(mu_unique, ratio[:, j], color=color, linewidth=2, label=label)
ax.set_xlabel(r"Chemical potential $\mu$ (eV)")
ax.set_ylabel(r"$I_{\mathrm{full}}(\omega_n) \,/\, I_{\mathrm{base}}(\omega_n)$")
ax.set_title(r"Peak intensity ratio at harmonic orders 1, 3, 5, 7")
ax.legend()
ax.grid(which="major", linestyle="-", linewidth=0.6, alpha=0.5)
plt.tight_layout()
plt.show()

# ================================================================
# PLOT 3: 2D colormaps — base, full, difference
# ================================================================

# ---------------- reshape grids ----------------
base_grid = I_base.reshape(n_mu, n_omega).T
full_grid = I_full.reshape(n_mu, n_omega).T

# ---------------- ratio grid (raw, before log) ----------------
eps        = 1e-30
ratio_grid = full_grid / (base_grid + eps)

# ---------------- log for base/full display ----------------
base_log = np.log10(base_grid + eps)
full_log = np.log10(full_grid + eps)

# ---------------- smooth ----------------
base_log   = gaussian_filter(base_log,   sigma=0.4)
full_log   = gaussian_filter(full_log,   sigma=0.4)
ratio_grid = gaussian_filter(ratio_grid, sigma=0.4)

# ---------------- log of ratio for difference panel ----------------
log_ratio = np.log10(ratio_grid + eps)

# ---------------- dynamic range ----------------
vmax = max(base_log.max(), full_log.max())
vmin = vmax - 12

# ---------------- mask only truly zero-signal pixels ----------------
noise_mask   = (base_log < vmin) & (full_log < vmin)
ratio_masked = np.ma.masked_where(noise_mask, log_ratio)

# ---------------- harmonic axis ----------------
first_mask    = mu == mu_unique[0]
harmonic_axis = harmonic[first_mask]

extent = [
    mu_unique.min(), mu_unique.max(),
    harmonic_axis.min(), harmonic_axis.max()
]

# ---------------- plot ----------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

im0 = axes[0].imshow(
    base_log, extent=extent, origin="lower", aspect="auto",
    cmap="inferno", vmin=vmin, vmax=vmax, interpolation="bicubic"
)
axes[0].set_title("HHG base")
axes[0].set_ylabel(r"Harmonic order $\omega/\omega_0$")
axes[0].set_xlabel(r"Chemical potential $\mu$ (eV)")
fig.colorbar(im0, ax=axes[0], pad=0.02, shrink=0.9,
             label=r"$\log_{10}|\ddot{p}(\omega)|^2$")

im1 = axes[1].imshow(
    full_log, extent=extent, origin="lower", aspect="auto",
    cmap="inferno", vmin=vmin, vmax=vmax, interpolation="bicubic"
)
axes[1].set_title("HHG full")
axes[1].set_xlabel(r"Chemical potential $\mu$ (eV)")
fig.colorbar(im1, ax=axes[1], pad=0.02, shrink=0.9,
             label=r"$\log_{10}|\ddot{p}(\omega)|^2$")

max_abs   = np.percentile(np.abs(ratio_masked.compressed()), 95)
norm_diff = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

im2 = axes[2].imshow(
    ratio_masked, extent=extent, origin="lower", aspect="auto",
    cmap="RdBu_r", norm=norm_diff, interpolation="bicubic"
)
axes[2].set_title("HHG difference")
axes[2].set_xlabel(r"Chemical potential $\mu$ (eV)")
fig.colorbar(im2, ax=axes[2], pad=0.02, shrink=0.9,
             label=r"$\log_{10}(I_{\mathrm{full}}/I_{\mathrm{base}})$")

plt.tight_layout()
plt.show()