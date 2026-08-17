"""
biot_savart_contour.py
======================
Compute |B_ind,z| at z = Z_OBS above the structure plane using the
vector-potential curl approach from the paper (Eq. 14):

    A_ind(r) = α² Σ_{bonds} J_b * (r - r̄_b) / |r - r̄_b|

    B_z = (∇ × A_ind)_z = ∂A_y/∂x - ∂A_x/∂y

The curl is evaluated numerically on a fine grid using central differences.
This matches the L2 implementation in the simulation code.

Units: atomic units internally, nm on axes.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ============================================================
#  USER SETTINGS
# ============================================================

STRUCTURES = [
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_bowtie_10x10_rot0",
        "mu":    5.0,
        "level": "L2",
        "label": r"Armchair bowtie $10{\times}10$",
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_bowtie_15x15_rot90",
        "mu":    5.0,
        "level": "L2",
        "label": r"Zigzag bowtie $15{\times}15$",
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_triangle_14x14_rot0",
        "mu":    5.0,
        "level": "L2",
        "label": r"Armchair triangle $14{\times}14$",
    },
     {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_triangle_22x22_rot90",
        "mu":    5.0,
        "level": "L2",
        "label": r"Zigzag triangle $22{\times}$",
    },
]

Z_OBS     = 1.0    # observation height above structure plane [nm]
GRID_PTS  = 300    # grid points along each axis
MARGIN_NM = 1.0    # extra margin beyond atom bounding box [nm]
CMAP      = "inferno"

# ============================================================
#  CONSTANTS
# ============================================================
AU_EV   = 27.2114
AU_NM   = 0.0529177
NM_AU   = 1.0 / AU_NM
ALPHA   = 1.0 / 137.036
MU0_4PI = ALPHA**2    # μ₀/4π in atomic units

plt.rcParams.update({
    "text.usetex":     True,
    "font.family":     "serif",
    "font.size":       18,
    "axes.titlesize":  20,
    "axes.labelsize":  18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# ============================================================
#  HELPERS
# ============================================================

def find_lattice(start_dir):
    start_dir = os.path.abspath(start_dir)
    for root in [start_dir] + [
            os.path.join(start_dir, n) for n in os.listdir(start_dir)
            if os.path.isdir(os.path.join(start_dir, n))]:
        c = os.path.join(root, "lattice_points.txt")
        if os.path.isfile(c):
            return c
    d = os.path.dirname(start_dir)
    while True:
        c = os.path.join(d, "lattice_points.txt")
        if os.path.isfile(c):
            return c
        p = os.path.dirname(d)
        if p == d:
            break
        d = p
    raise FileNotFoundError(f"lattice_points.txt not found from {start_dir}")


def find_mu_dir(sweep_dir, level, mu):
    for fmt in [f"{mu:.2f}", f"{mu:.1f}", f"{mu}", f"{mu:.4f}", str(mu)]:
        p = os.path.join(sweep_dir, f"{level}_mu_{fmt}")
        if os.path.isdir(p):
            return p
    pattern = re.compile(rf"^{level}_mu_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
    best = None
    for name in os.listdir(sweep_dir):
        m = pattern.match(name)
        if m:
            diff = abs(float(m.group(1)) - mu)
            if best is None or diff < best[0]:
                best = (diff, os.path.join(sweep_dir, name))
    if best:
        print(f"  [INFO] Fuzzy match for mu={mu}: {os.path.basename(best[1])}")
        return best[1]
    return None


def load_ts(base, sc_stem, fb_stem):
    sc = os.path.join(base, sc_stem)
    if os.path.isfile(sc):
        return np.loadtxt(sc)
    return np.loadtxt(os.path.join(base, fb_stem))


def build_bonds(lattice_au):
    tree  = cKDTree(lattice_au)
    dists, _ = tree.query(lattice_au, k=2)
    a_nn  = dists[:, 1][dists[:, 1] > 0.1].min()
    bonds = np.array(sorted(tree.query_pairs(r=1.0005 * a_nn)), dtype=int)
    print(f"  Built {len(bonds)} bonds, a_nn = {a_nn:.4f} a.u.")
    return bonds


# ============================================================
#  VECTOR POTENTIAL + CURL ON GRID  (Eq. 14 of paper)
# ============================================================

def compute_Bz_grid(x_au, y_au, bonds, J_res, GX, GY, gz_au):
    """
    Compute B_z = ∂A_y/∂x - ∂A_x/∂y on a 2-D grid at height gz_au [Bohr].

    A_ind(r) = α² Σ_b J_b * (r - r̄_b) / |r - r̄_b|   [Eq. 14]

    The curl is taken by numerical central differences on the grid.

    Parameters
    ----------
    GX, GY : (GRID_PTS, GRID_PTS) arrays in Bohr
    gz_au  : scalar observation height in Bohr

    Returns
    -------
    Bz : complex array, shape = GX.shape
    """
    bi = bonds[:, 0]
    bj = bonds[:, 1]

    # Bond midpoints and bond vectors ℓ = r_i - r_j  (Eq. 14 uses r_i - r_j)
    xc = 0.5 * (x_au[bi] + x_au[bj])   # (N_bonds,)
    yc = 0.5 * (y_au[bi] + y_au[bj])
    lx = x_au[bi] - x_au[bj]            # bond vector x-component
    ly = y_au[bi] - y_au[bj]            # bond vector y-component

    # Flatten grid for vectorised computation
    gx_flat = GX.ravel()   # (N_grid,)
    gy_flat = GY.ravel()

    # Displacement from each bond midpoint to each grid point
    # dx[m, b] = gx[m] - xc[b],  shape (N_grid, N_bonds)
    dx = gx_flat[:, None] - xc[None, :]
    dy = gy_flat[:, None] - yc[None, :]
    dz = gz_au  # scalar

    r = np.sqrt(dx**2 + dy**2 + dz**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_r = np.where(r < 1e-12, 0.0, 1.0 / r)

    # A_x[m] = α² Σ_b J_b * lx[b] / |r_m - r̄_b|
    # A_y[m] = α² Σ_b J_b * ly[b] / |r_m - r̄_b|
    # J_res is complex; keep complex throughout
    weighted = (inv_r * J_res[None, :])           # (N_grid, N_bonds) complex
    A_x_flat = MU0_4PI * (weighted @ lx)          # (N_grid,) complex
    A_y_flat = MU0_4PI * (weighted @ ly)

    A_x = A_x_flat.reshape(GX.shape)
    A_y = A_y_flat.reshape(GX.shape)

    # Numerical curl: B_z = ∂A_y/∂x - ∂A_x/∂y  (central differences)
    dx_grid = GX[0, 1] - GX[0, 0]   # uniform spacing in x [Bohr]
    dy_grid = GY[1, 0] - GY[0, 0]   # uniform spacing in y [Bohr]

    dAy_dx = np.gradient(A_y, dx_grid, axis=1)
    dAx_dy = np.gradient(A_x, dy_grid, axis=0)

    return dAy_dx - dAx_dy   # complex B_z


# ============================================================
#  MAIN LOOP
# ============================================================
N     = len(STRUCTURES)
fig_w = 5.5 * N
fig_h = 5.0
fig, axes = plt.subplots(1, N, figsize=(fig_w, fig_h), squeeze=False)

Z_OBS_AU  = Z_OBS     * NM_AU
MARGIN_AU = MARGIN_NM * NM_AU

for col, struct in enumerate(STRUCTURES):
    sweep_dir = struct["sweep_dir"]
    mu        = struct["mu"]
    level     = struct["level"]
    label     = struct["label"]

    print(f"\n{'='*50}\n{label}")

    path = find_mu_dir(sweep_dir, level, mu)
    if path is None:
        axes[0, col].text(0.5, 0.5, "no data", ha='center', va='center',
                          transform=axes[0, col].transAxes, color='grey')
        continue

    lattice_path = find_lattice(path)
    lattice      = np.loadtxt(lattice_path, comments="#")
    x_au = lattice[:, 0]
    y_au = lattice[:, 1]
    x_nm = x_au * AU_NM
    y_nm = y_au * AU_NM

    # ── Bond indices ──────────────────────────────────────────────────────
    _bond_candidates = [
        os.path.join(os.path.dirname(lattice_path), "bond_indices.txt"),
        os.path.join(path, "bond_indices.txt"),
        os.path.join(os.path.dirname(path), "bond_indices.txt"),
    ]
    bond_idx_path = next((p for p in _bond_candidates if os.path.isfile(p)), None)
    if bond_idx_path:
        bonds = np.loadtxt(bond_idx_path, dtype=int, comments='#')
        print(f"  Loaded {len(bonds)} bonds from bond_indices.txt")
    else:
        bonds = build_bonds(lattice)

    # ── Bond currents FFT ─────────────────────────────────────────────────
    J_bond_t    = load_ts(path, "J_bond_sc_time_evolution.txt",
                                 "J_bond_time_evolution.txt")
    time_au_arr = J_bond_t[:, 0]
    J_bond      = J_bond_t[:, 1:]
    dt          = time_au_arr[1] - time_au_arr[0]
    N_t         = len(time_au_arr)
    N_pad       = 8 * N_t

    J_fft   = np.fft.rfft(J_bond, n=N_pad, axis=0)   # no /N_t — matches Paper_plot_multi.py
    freq_eV = np.fft.rfftfreq(N_pad, d=dt) * AU_EV

    # ── Resonance ─────────────────────────────────────────────────────────
    sigma     = np.loadtxt(os.path.join(path, "sigma_ext.txt"))
    omega_eV  = sigma[:, 0] * AU_EV
    i_res     = np.argmax(sigma[:, 1])
    omega_res = omega_eV[i_res]
    i_freq    = np.argmin(np.abs(freq_eV - omega_res))
    print(f"  Resonance: {omega_res:.3f} eV  (bin: {freq_eV[i_freq]:.3f} eV)")

    J_res = J_fft[i_freq, :]

    if J_res.shape[0] != len(bonds):
        print(f"  [WARNING] J cols ({J_res.shape[0]}) != bonds ({len(bonds)}), skipping.")
        continue

    # ── Observation grid ──────────────────────────────────────────────────
    xlo_au = x_au.min() - MARGIN_AU;  xhi_au = x_au.max() + MARGIN_AU
    ylo_au = y_au.min() - MARGIN_AU;  yhi_au = y_au.max() + MARGIN_AU

    gx1d = np.linspace(xlo_au, xhi_au, GRID_PTS)
    gy1d = np.linspace(ylo_au, yhi_au, GRID_PTS)
    GX, GY = np.meshgrid(gx1d, gy1d)

    # ── Compute B_z via curl(A_ind) ───────────────────────────────────────
    print(f"  Computing curl(A_ind) on {GRID_PTS}x{GRID_PTS} grid "
          f"(z = {Z_OBS} nm = {Z_OBS_AU:.2f} Bohr) ...")

    Bz_complex = compute_Bz_grid(x_au, y_au, bonds, J_res, GX, GY, Z_OBS_AU)
    Bz_abs     = np.abs(Bz_complex)

    GX_nm = GX * AU_NM
    GY_nm = GY * AU_NM

    # ── Plot ──────────────────────────────────────────────────────────────
    ax = axes[0, col]

    cf = ax.contourf(GX_nm, GY_nm, Bz_abs, levels=50, cmap=CMAP)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$|B_{\mathrm{ind},z}|$ (a.u.)", fontsize=11)

    ax.scatter(x_nm, y_nm, s=4, c='white', alpha=0.5, zorder=3, linewidths=0)

    ax.set_xlim(GX_nm.min(), GX_nm.max())
    ax.set_ylim(GY_nm.min(), GY_nm.max())
    ax.set_aspect('equal')
    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$y$ (nm)")

plt.tight_layout()
out = "biot_savart_contour_rot90.png"
plt.savefig(out, bbox_inches='tight', dpi=600)
print(f"\nSaved: {out}")
plt.show()