"""
Single panel: Time-averaged B_z from B_ind_z file (direct)

Usage:
    python plot_B_single.py                        # looks in current directory
    python plot_B_single.py path/to/config.toml    # resolves simulation folder
    python plot_B_single.py path/to/sim/folder     # use folder directly

Output:
    <base_dir>/B_ind_z_time_avg.png
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from pathlib import Path

# ── Path resolution (same logic as simulator) ────────────────────────────────
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
    if not p.suffix:
        p2 = Path(str(p) + ".toml")
        if p2.exists() and p2.is_file():
            return p2
    raise FileNotFoundError(f"Could not find config file: {arg}")

def simulation_dir_from_config(cfg_path: Path) -> Path:
    cfg_bytes = cfg_path.read_bytes()
    h = fnv1a_64(cfg_bytes)
    folder = f"{cfg_path.stem}_{h:x}"
    return Path("Simulations") / folder

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

# ── File paths ────────────────────────────────────────────────────────────────
LATTICE_FILE = base_dir / "lattice_points.txt"
B_FILE       = base_dir / "B_ind_z_time_evolution.txt"
OUTPUT_FILE  = base_dir / "B_ind_z_time_avg.png"

# ── Load lattice ──────────────────────────────────────────────────────────────
coords = np.loadtxt(LATTICE_FILE, comments="#")
x, y   = coords[:, 0], coords[:, 1]
N      = len(x)

# ── Panel 1: time-averaged B_ind_z ───────────────────────────────────────────
B_data   = np.loadtxt(B_FILE, comments="#")
B_z_raw  = B_data[:, 1:]                    # (n_times, N)
B_z_avg  = np.mean(B_z_raw[1:], axis=0)    # skip t=0

# ── Plotting ──────────────────────────────────────────────────────────────────
DARK_BG  = "white"
SPINE_C  = "#cccccc"
TICK_C   = "black"
LABEL_C  = "black"

panels = [
    (B_z_avg,  r"$\langle B_z^{\,\mathrm{ind}} \rangle$  (direct)",  "Method I — B_ind_z file"),
]

fig, ax = plt.subplots(1, 1, figsize=(7, 6.5), facecolor=DARK_BG)
fig.suptitle(
    "Induced magnetic field  $B_z$ time-averaged",
    color=LABEL_C, fontsize=15, y=1.01, fontweight="bold"
)

# Grid for interpolation
xi = np.linspace(x.min() - 3, x.max() + 3, 450)
yi = np.linspace(y.min() - 3, y.max() + 3, 450)
Xi, Yi = np.meshgrid(xi, yi)

for field, cbar_label, title in panels:
    ax.set_facecolor(DARK_BG)

    vmax = np.abs(field).max()
    if vmax == 0:
        vmax = 1e-12
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    # Interpolated background
    Zi = griddata((x, y), field, (Xi, Yi), method="cubic")
    ax.pcolormesh(Xi, Yi, Zi, cmap=cmap, norm=norm,
                  shading="auto", alpha=0.72, zorder=1)

    # Site markers
    sc = ax.scatter(x, y, c=field, cmap=cmap, norm=norm,
                    s=110, edgecolors="white", linewidths=0.45, zorder=3)

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.03, fraction=0.038, aspect=28)
    cbar.set_label(cbar_label, color=LABEL_C, fontsize=10, labelpad=8)
    cbar.ax.yaxis.set_tick_params(color=TICK_C, labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TICK_C)
    cbar.outline.set_edgecolor(SPINE_C)

    # Axes formatting
    ax.set_title(title, color=LABEL_C, fontsize=11, pad=10, fontweight="bold")
    ax.set_xlabel("x (a.u)", color=LABEL_C, fontsize=10)
    ax.set_ylabel("y (a.u)", color=LABEL_C, fontsize=10)
    ax.tick_params(colors=TICK_C, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)
    ax.set_aspect("equal")

plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved → {OUTPUT_FILE}")
plt.show()