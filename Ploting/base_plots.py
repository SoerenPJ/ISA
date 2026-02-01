import numpy as np
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
    folder = f"{cfg_path.stem}_{h:016x}"
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
Intensity = (1e13/au_kg) * (au_s*au_s*au_s)  #// time pules = 1e13 ddf = 1e15
E0 = np.sqrt((2 *np.pi* Intensity) / (au_c)) #// Electric field amplitude

t_shift = 200 / au_fs
sigma_gaus = 60.07/au_fs 
sigma_ddf = 0.01/au_fs

gamma = 0.01/au_eV
au_omega = 0.2/au_eV
au_omega_fourier = 0.0 




out_dir = "Ploting/base_plots"

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

out_dir = base_dir / "base_plots"
out_dir.mkdir(parents=True, exist_ok=True)

eigenvalues = np.loadtxt(base_dir / 'eigenvalues.txt')
rho_j_space = np.loadtxt(base_dir / 'rho0_j_space.txt')
#reshapet the rho_j_space to a NxN matrix
rho_j_NxN = rho_j_space[:,0::2] + 1j*rho_j_space[:,1::2]
#generate the number of sites array
N_sites = np.arange(rho_j_NxN.shape[1])


rho_l_space = np.loadtxt(base_dir / 'rho0_l_space.txt')
#reshape the rho_l_space to a NxN matrix
rho_l_NxN = rho_l_space[:,0::2] + 1j*rho_l_space[:,1::2]

dipole_moments = np.loadtxt(base_dir / 'dipole_time_evolution.txt')
dipole = np.real(dipole_moments[:,1])
time = np.real(dipole_moments[:,0]*au_fs)


print("shape of dipolemoments", dipole_moments.shape)

#exit()
def baseplots(N_sites, t, eigenvalues, rho_j_NxN, rho_l_NxN, dipole):#, rho_l_space, dipole_moments):
    
    # Plot of eigenvalues
    plt.figure(figsize=(8, 6))
    plt.plot(eigenvalues[:,0] *au_eV, 'o')
    plt.title('Eigenvalues ')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue [eV]')
    plt.grid()
    plt.savefig(out_dir / "eigenvalues_plot.png", dpi=300, bbox_inches="tight")

    #plot of rho_j_space
    plt.figure(figsize = (8,6))
    # extrar onyl the diagonal of the rho_j_space
    diag_rho_j = np.diag(rho_j_NxN)
    plt.plot(N_sites,diag_rho_j, '-o')
    plt.title('Density matrix diagonal elements in site space')
    plt.xlabel('Site index')
    plt.ylabel(r'$\rho_{jj}$')
    plt.grid()
    plt.savefig(out_dir / "rho_j_space_diagonal_plot.png", dpi=300, bbox_inches="tight")

    #plot of the rho_l_space
    rho_l_diag = np.diag(rho_l_NxN)
    plt.figure(figsize = (8,6))
    plt.plot(N_sites,rho_l_diag, '-o')
    plt.title('Density matrix diagonal elements in lattice space')
    plt.xlabel('Site index')
    plt.ylabel(r'$\rho_{jj}$')
    plt.grid()
    plt.savefig(out_dir / "rho_l_space_diagonal_plot.png", dpi=300, bbox_inches="tight")
    # Plot of dipole moment time evolution
    plt.figure(figsize=(8,6))
    plt.plot(time, dipole)
    plt.xlabel('Time [fs]')
    plt.ylabel("Dipole moment [au]")
    plt.title("Time evolution of dipole moment")
    plt.savefig(out_dir / "Dipole_time_evolution_plot.png", dpi= 300, bbox_inches="tight")
    plt.show()
    
    
    plt.close()

baseplots(N_sites, time, eigenvalues, rho_j_NxN, rho_l_NxN, dipole)