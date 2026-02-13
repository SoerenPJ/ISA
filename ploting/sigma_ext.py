import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# ----------------- utilities ----------------- #
def fnv1a_64(data: bytes) -> int:
    """Match the simulator's FNV-1a 64-bit hash for folder naming."""
    h = 14695981039346656037
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def resolve_config_path(arg: str) -> Path:
    p = Path(arg)
    if p.exists() and p.is_file():
        return p
    # allow calling with "configs/SSH" (no extension)
    if not p.suffix:
        p2 = Path(str(p) + ".toml")
        if p2.exists() and p2.is_file():
            return p2
    raise FileNotFoundError(f"Could not find config file: {arg} (tried '{p}' and '{p}.toml')")

def simulation_dir_from_config(cfg_path: Path) -> Path:
    cfg_bytes = cfg_path.read_bytes()
    h = fnv1a_64(cfg_bytes)
    # Match C++: std::hex with no setw/setfill → unpadded hex
    folder = f"{cfg_path.stem}_{h:x}"
    return Path("Simulations") / folder
# constants
au_eV = 27.2114
au_nm = 0.0529177
au_s = 2.41888e-17
au_fs = au_s * 1e15
au_c = 137.036
au_kg = 9.10938e-31
au_kB = 3.16681e-6
au_hbar = 1.0
au_m = 1.0
au_J = 4.35974e-18
au_w = 4.13414e16
au_I = 3.50945e16
au_me = 9.10938e-31
alpha = 1.0 / au_c
e = 1
au_mu_0 = 4 * np.pi * 1e-7 * (au_eV * au_s) / (au_J * au_m)


a = 0.142 / au_nm           #Lattice constant 

mu = 0.0/au_eV                     # Chemical potential
#Physical parameters
N = 50
T = 100
t1 = -2.8/au_eV
t2 = t1;#// + 0.5

#spin_on = false
#two_dim = false


# external field parameters
Intensity = (1e15/au_kg) * (au_s*au_s*au_s)  #// time pules = 1e13 ddf = 1e15
E0 = np.sqrt((2 *np.pi* Intensity) / (au_c)) #// Electric field amplitude

t_shift = 200 / au_fs
sigma_gaus = 60.07/au_fs 
sigma_ddf = 0.01/au_fs

gamma = 0.01/au_eV
au_omega = 0.2/au_eV




out_dir = "Ploting/sigma_ext"

base_dir = Path(".")
if len(sys.argv) >= 2:
    arg = Path(sys.argv[1])
    if arg.exists() and arg.is_dir():
        base_dir = arg
    else:
        cfg = resolve_config_path(sys.argv[1])
        base_dir = simulation_dir_from_config(cfg)
        if not base_dir.exists():
            raise FileNotFoundError(
                f"Simulation output folder not found: {base_dir}\n"
                f"Run the simulator first with: ./sim_mkl {cfg}"
            )

out_dir = base_dir / "sigma_ext"
out_dir.mkdir(parents=True, exist_ok=True)

# ----------------- load complex matrix (file: one row per line, each element as "real imag") ----------------- #
def load_complex_matrix(path):
    raw = np.loadtxt(path)   # shape (N, 2*N)
    n = raw.shape[0]
    pairs = raw.reshape(n, n, 2)
    return pairs[..., 0] + 1j * pairs[..., 1]   # shape (N, N)

#=================debug code=============

#================Hamiltonian===============
HTB_raw = np.loadtxt(base_dir / "HTB.txt")  # shape (N, 2N)
N = HTB_raw.shape[0]
print("This is raw HTB", HTB_raw)
# Interpret columns as [Re, Im] pairs
HTB_pairs = HTB_raw.reshape(N, N, 2)
HTB_real = HTB_pairs[..., 0]                # shape (N, N), real part

print("HTB_real shape", HTB_real.shape)
plt.matshow(HTB_real, cmap='viridis')
plt.show()

#===============Vll matrix=================

V_ll = np.loadtxt(base_dir / "V_ee_spin.txt")
print("V_ee shape", V_ll.shape)
plt.matshow(np.real(V_ll), cmap='viridis')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pathlib import Path

# Constants (same as elsewhere)
au_nm = 0.0529177

phases_path = base_dir / "peierls_phases.txt"

if phases_path.exists():
    # Load: i j xa ya xb yb phi   (skipping comment lines)
    data = np.loadtxt(phases_path, comments="#")
    i, j, xa, ya, xb, yb, phi = data.T
    print("this is phi max", phi.max())
    print("this is phi min", phi.min())
    # Convert to nm
    xa_nm = xa * au_nm
    ya_nm = ya * au_nm
    xb_nm = xb * au_nm
    yb_nm = yb * au_nm

    # Build segments: [[(xa,ya),(xb,yb)], ...]
    segments = np.stack(
        [np.stack([xa_nm, ya_nm], axis=1), np.stack([xb_nm, yb_nm], axis=1)],
        axis=1
    )  # shape (n_bonds, 2, 2)

    values = phi
    vmin = values.min()
    vmax = values.max()
    if vmin == vmax:
        eps = 1e-3
        vmin -= eps
        vmax += eps

    cmap = plt.cm.plasma
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(LineCollection(segments, colors='lightgray', linewidths=2, zorder=1))
    lc = LineCollection(segments, array=values, cmap=cmap, norm=norm,
                        linewidths=3, zorder=2)
    ax.add_collection(lc)
    ax.set_aspect('equal', adjustable='box')
    pad = 0.1
    ax.set_xlim(xa_nm.min() - pad, xa_nm.max() + pad)
    ax.set_ylim(ya_nm.min() - pad, ya_nm.max() + pad)
    ax.set_xlabel("Position x (nm)")
    ax.set_ylabel("Position y (nm)")
    ax.set_title("Peierls Phase Distribution Over Graphene Bonds")
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Peierls Phase")
    plt.grid(False)
    plt.show()
else:
    print("peierls_phases.txt not found (B_ext was false); skipping Peierls phase plot.")


#================ diagonal values (rho files are complex: real imag pairs per element) ================
rho_j_space = load_complex_matrix(base_dir / "rho_j_space.txt")
rho0_j_space = load_complex_matrix(base_dir / "rho0_j_space.txt")
rho0_l_space = load_complex_matrix(base_dir / "rho0_l_space.txt")
print(rho0_l_space.shape)
print("this is the diagonal of the rho_j_space", np.diag(rho_j_space))
print("this is the diagonal of the rho0_j_space", np.diag(rho0_j_space))
print("this is the diagonal of the rho0_l_space", np.diag(rho0_l_space))



sigma_ext = np.loadtxt(base_dir / 'sigma_ext.txt')
alpha_ext = np.loadtxt(base_dir / 'alpha_ext.txt')


alpha_real = alpha_ext[:, 0]
alpha_imag = alpha_ext[:, 1]

plt.plot(sigma_ext[:,0] * au_eV, alpha_real, label='Re(alpha)', color='blue')
plt.plot(sigma_ext[:,0] * au_eV, alpha_imag, label='Im(alpha)', color='orange')
plt.legend()
plt.xlabel('Energy (eV)', fontsize=12)
plt.show()



plt.figure(figsize=(15, 9))
plt.plot(sigma_ext[:,0]*au_eV, sigma_ext[:,1] * au_nm**2) 
#plt.yscale('log')
plt.xlabel('Energy (eV)', fontsize=12)
plt.ylabel('Extinction Cross-Section', fontsize=12)
plt.xlim(0, (sigma_ext[:,0][-1] * au_eV))  # automated to always go to the last element of au_omega_fourier
plt.legend()
plt.grid(True)
plt.show()