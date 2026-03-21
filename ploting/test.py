import numpy as np
import matplotlib.pyplot as plt
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

# ---------------- compute harmonic order per mu ----------------
harmonic = np.zeros_like(omega)

for m in mu_unique:
    mask        = mu == m
    omega_slice = omega[mask]
    I_slice     = I_full[mask]
    omega0      = omega_slice[np.argmax(I_slice)]
    harmonic[mask] = omega_slice / omega0

# ---------------- extract peak intensity at orders 1, 3 ----------------
TARGET_ORDERS = [1, 3, 5 , 7]
HALF_WIDTH    = 0.04   # ± window 

# results: shape (n_mu, n_orders)
base_peaks = np.zeros((n_mu, len(TARGET_ORDERS)))
full_peaks = np.zeros((n_mu, len(TARGET_ORDERS)))

for i, m in enumerate(mu_unique):
    mask   = mu == m
    h      = harmonic[mask]
    ib     = I_base[mask]
    if_    = I_full[mask]

    for j, order in enumerate(TARGET_ORDERS):
        in_band = np.abs(h - order) <= HALF_WIDTH
        if in_band.any():
            base_peaks[i, j] = ib[in_band].max()
            full_peaks[i, j] = if_[in_band].max()
        else:
            base_peaks[i, j] = np.nan
            full_peaks[i, j] = np.nan

# ----------------  full - base ----------------
ratio = full_peaks / base_peaks   

# ---------------- line plot ----------------
import numpy as np
import matplotlib.pyplot as plt
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

# ---------------- compute harmonic order per mu ----------------
harmonic = np.zeros_like(omega)

for m in mu_unique:
    mask        = mu == m
    omega_slice = omega[mask]
    I_slice     = I_full[mask]
    omega0      = omega_slice[np.argmax(I_slice)]
    harmonic[mask] = omega_slice / omega0

# ---------------- extract peak intensity at orders 1, 3 ----------------
TARGET_ORDERS = [1, 3,5,7]
HALF_WIDTH    = 0.04   # ± window 

# results: shape (n_mu, n_orders)
base_peaks = np.zeros((n_mu, len(TARGET_ORDERS)))
full_peaks = np.zeros((n_mu, len(TARGET_ORDERS)))

for i, m in enumerate(mu_unique):
    mask   = mu == m
    h      = harmonic[mask]
    ib     = I_base[mask]
    if_    = I_full[mask]

    for j, order in enumerate(TARGET_ORDERS):
        in_band = np.abs(h - order) <= HALF_WIDTH
        if in_band.any():
            base_peaks[i, j] = ib[in_band].max()
            full_peaks[i, j] = if_[in_band].max()
        else:
            base_peaks[i, j] = np.nan
            full_peaks[i, j] = np.nan

# ----------------  full - base ----------------
#ratio = full_peaks - base_peaks   
# ---------------- normalize intensity to 1st harmonic (comment out to disable) ----------------
#base_peaks = base_peaks / base_peaks[:, 0:1]   # divide each column by order-1 column
#full_peaks = full_peaks / full_peaks[:, 0:1]   # same for full
ratio = full_peaks / base_peaks

# ---------------- line plot ----------------
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
labels = [r"$\omega/\omega_0 = 1$",
          r"$\omega/\omega_0 = 3$",
          r"$\omega/\omega_0 = 5$",
          r"$\omega/\omega_0 = 7$"]

fig, ax = plt.subplots(figsize=(8, 5))

for j, (color, label) in enumerate(zip(colors, labels)):
    ax.plot(mu_unique, ratio[:, j], color=color, linewidth=2, label=label)

ax.set_xlabel(r"Chemical potential $\mu$ (eV)")
ax.set_ylabel(r"$I_{\mathrm{full}}(\omega_n) \,/\, I_{\mathrm{base}}(\omega_n)$")  # fixed
ax.set_title(r"Peak intensity ratio at harmonic orders 1, 3, 5, 7")
ax.legend()
ax.grid(which="major", linestyle="-", linewidth=0.6, alpha=0.5)
plt.tight_layout()
plt.show()