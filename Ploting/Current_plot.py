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




out_dir = "Ploting/current_time_evolution"

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

# Create subfolder for saving plots (like base_plots); C++ writes .txt in simulation root
out_dir = base_dir / "current_time_evolution"
out_dir.mkdir(parents=True, exist_ok=True)

data = np.loadtxt(base_dir / "current_time_evolution.txt")
t = data[:,0]
J_x = data[:,1]
J_y = data[:,2]

plt.plot(t, J_x)
plt.show()

plt.plot(t, J_y)
plt.show()