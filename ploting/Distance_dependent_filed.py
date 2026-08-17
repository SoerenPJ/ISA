"""
biot_savart_zdecay_paper.py
===========================
Single linear panel, paper-ready.
Labels show fitted power law (∝ z^x).
AC/ZZ shorthand, no z^-3 reference.
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
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_bowtie_10x10_rot90",
        "mu":    3.52,
        "level": "L2",
        "label": r"AC Bowtie ",
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_bowtie_15x15_rot0",
        "mu":    3.52,
        "level": "L2",
        "label": r"ZZ Bowtie",
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_triangle_14x14_rot90",
        "mu":    3.52,
        "level": "L2",
        "label": r"AC Triangle",
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_triangle_22x22_rot0",
        "mu":    3.52,
        "level": "L2",
        "label": r"ZZ Triangle",
    },
]

Z_HEIGHTS_NM = [12, 24, 30, 36, 42, 48, 54]
#Z_HEIGHTS_NM = [0.1,0.4,0.8,1.6,3.2,6.4,12.8,]

GRID_PTS  = 150
MARGIN_NM = 0.5

COLORS  = ['#08306b', '#2171b5', '#cb181d', '#a50f15']
MARKERS = ['o', 's', '^', 'D']

# ============================================================
#  CONSTANTS
# ============================================================
AU_EV   = 27.2114
AU_NM   = 0.0529177
NM_AU   = 1.0 / AU_NM
ALPHA   = 1.0 / 137.036
MU0_4PI = ALPHA**2

plt.rcParams.update({
    "text.usetex":     True,
    "font.family":     "Times new roman",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "lines.linewidth": 1.5,
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
    tree = cKDTree(lattice_au)
    dists, _ = tree.query(lattice_au, k=2)
    a_nn = dists[:, 1][dists[:, 1] > 0.1].min()
    bonds = np.array(sorted(tree.query_pairs(r=1.0005 * a_nn)), dtype=int)
    print(f"  Built {len(bonds)} bonds, a_nn = {a_nn:.4f} a.u.")
    return bonds


def max_Bz_at_height(x_au, y_au, bonds, J_res, gz_au,
                     grid_pts=GRID_PTS, margin_au=None):
    if margin_au is None:
        margin_au = MARGIN_NM * NM_AU

    bi = bonds[:, 0];  bj = bonds[:, 1]
    xc = 0.5 * (x_au[bi] + x_au[bj])
    yc = 0.5 * (y_au[bi] + y_au[bj])
    lx = x_au[bi] - x_au[bj]
    ly = y_au[bi] - y_au[bj]

    GX, GY = np.meshgrid(
        np.linspace(x_au.min() - margin_au, x_au.max() + margin_au, grid_pts),
        np.linspace(y_au.min() - margin_au, y_au.max() + margin_au, grid_pts),
    )

    gx_f = GX.ravel();  gy_f = GY.ravel()
    dx   = gx_f[:, None] - xc[None, :]
    dy   = gy_f[:, None] - yc[None, :]
    r    = np.sqrt(dx**2 + dy**2 + gz_au**2)

    with np.errstate(divide='ignore', invalid='ignore'):
        inv_r = np.where(r < 1e-12, 0.0, 1.0 / r)

    weighted = inv_r * J_res[None, :]
    A_x = (MU0_4PI * (weighted @ lx)).reshape(GX.shape)
    A_y = (MU0_4PI * (weighted @ ly)).reshape(GX.shape)

    dx_g = GX[0, 1] - GX[0, 0]
    dy_g = GY[1, 0] - GY[0, 0]
    Bz   = np.gradient(A_y, dx_g, axis=1) - np.gradient(A_x, dy_g, axis=0)

    return np.abs(Bz).max()


# ============================================================
#  MAIN — single log-log panel
# ============================================================
fig, ax = plt.subplots(figsize=(3.4, 3.2))

Z_AU = np.array(Z_HEIGHTS_NM) * NM_AU

for struct, color, marker in zip(STRUCTURES, COLORS, MARKERS):
    sweep_dir  = struct["sweep_dir"]
    mu         = struct["mu"]
    level      = struct["level"]
    base_label = struct["label"]

    print(f"\n{'='*50}\n{base_label}")

    path = find_mu_dir(sweep_dir, level, mu)
    if path is None:
        print(f"  [WARNING] mu={mu} not found, skipping.")
        continue

    lattice_path = find_lattice(path)
    lattice      = np.loadtxt(lattice_path, comments="#")
    x_au = lattice[:, 0]
    y_au = lattice[:, 1]

    _cands = [
        os.path.join(os.path.dirname(lattice_path), "bond_indices.txt"),
        os.path.join(path, "bond_indices.txt"),
        os.path.join(os.path.dirname(path), "bond_indices.txt"),
    ]
    bond_path = next((p for p in _cands if os.path.isfile(p)), None)
    bonds = (np.loadtxt(bond_path, dtype=int, comments='#')
             if bond_path else build_bonds(lattice))

    J_bond_t = load_ts(path, "J_bond_sc_time_evolution.txt",
                              "J_bond_time_evolution.txt")
    time_au  = J_bond_t[:, 0]
    J_bond   = J_bond_t[:, 1:]
    dt       = time_au[1] - time_au[0]
    N_pad    = 8 * len(time_au)
    J_fft    = np.fft.rfft(J_bond, n=N_pad, axis=0)
    freq_eV  = np.fft.rfftfreq(N_pad, d=dt) * AU_EV

    sigma     = np.loadtxt(os.path.join(path, "sigma_ext.txt"))
    omega_eV  = sigma[:, 0] * AU_EV
    i_res     = np.argmax(sigma[:, 1])
    omega_res = omega_eV[i_res]
    i_freq    = np.argmin(np.abs(freq_eV - omega_res))
    print(f"  Resonance: {omega_res:.3f} eV")

    J_res = J_fft[i_freq, :]

    if J_res.shape[0] != len(bonds):
        print(f"  [WARNING] bond count mismatch, skipping.")
        continue

    bmax_vals = []
    for z_nm, z_au in zip(Z_HEIGHTS_NM, Z_AU):
        bmax = max_Bz_at_height(x_au, y_au, bonds, J_res, z_au)
        bmax_vals.append(bmax)
        print(f"    z = {z_nm:.2f} nm  ->  max|Bz| = {bmax:.4e} a.u.")

    bmax_vals = np.array(bmax_vals)

    # Power-law fit
    log_z   = np.log(np.array(Z_HEIGHTS_NM))
    log_b   = np.log(bmax_vals)
    n, logA = np.polyfit(log_z, log_b, 1)
    A_fit   = np.exp(logA)
    print(f"  Power-law fit: B ~ z^{n:.2f}")

    z_fine    = np.linspace(Z_HEIGHTS_NM[0], Z_HEIGHTS_NM[-1], 300)
    b_fine    = A_fit * z_fine**n
    bmax_plot = bmax_vals

    legend_label = base_label + rf",\ $\propto z^{{{n:.2f}}}$"

    ax.plot(Z_HEIGHTS_NM, bmax_plot,
            color=color, marker=marker, lw=0, ms=5,
            label=legend_label, zorder=3)
    ax.plot(z_fine, b_fine,
            color=color, lw=1.5, ls='--', zorder=2)

# ── Styling ───────────────────────────────────────────────────────────────
ax.set_xlabel(r"$z$ (nm)")
ax.set_ylabel(r"$\max|B^{\mathrm{ind}}_z|\ (\mathrm{a.u.})$")
ax.set_xlim(Z_HEIGHTS_NM[0], Z_HEIGHTS_NM[-1])
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(framealpha=0.9, loc='upper right',
          handlelength=1.5, handletextpad=0.5)
ax.grid(True, ls=':', alpha=0.4, which='both')

plt.tight_layout()
out = "biot_savart_zdecay_paper.pdf"
plt.savefig(out, bbox_inches='tight', dpi=300)
print(f"\nSaved: {out}")
plt.show()