import numpy as np
from scipy.signal import find_peaks

# constants
au_nm = 0.0529177

# ---------- load sweep ----------
input_file = "sigma_mu_armchair_8x8_rot0.txt"
output_file = f"resonance_vs_mu_{input_file}"
data = np.loadtxt(input_file)

# sort (important)
data = data[np.lexsort((data[:,1], data[:,0]))]

mu = data[:,0]
omega = data[:,1]*27.2114
sigma_base = data[:,2]*au_nm**2
sigma_full = data[:,3]*au_nm**2

mu_unique = np.unique(mu)

omega_res_base = []
omega_res_full = []

for m in mu_unique:

    mask = mu == m

    omega_slice = omega[mask]
    sigma_base_slice = sigma_base[mask]
    sigma_full_slice = sigma_full[mask]

    peaks_base,_ = find_peaks(
        sigma_base_slice,
        prominence=np.max(sigma_base_slice)*0.1
    )

    peaks_full,_ = find_peaks(
        sigma_full_slice,
        prominence=np.max(sigma_full_slice)*0.1
    )

    omega_res_base.append(omega_slice[peaks_base[0]])
    omega_res_full.append(omega_slice[peaks_full[0]])

omega_res_base = np.array(omega_res_base)
omega_res_full = np.array(omega_res_full)

# ---------- save result ----------
out = np.column_stack((mu_unique, omega_res_base, omega_res_full))

np.savetxt(
    output_file,
    out,
    header="mu omega_base omega_full",
    fmt="%.6f"
)

print("Saved resonance table:")