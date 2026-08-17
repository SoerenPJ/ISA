"""
plot_sweep_sigma_ext.py
=======================
Layout:
  rows = structures
  cols = [L2 | (L1-L0) | (L2-L0)]
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree

# ============================================================
#  USER SETTINGS
# ============================================================

STRUCTURES = [
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_bowtie_10x10_rot0",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_bowtie_15x15_rot0",
]

A_CC_AU_PER_STRUCTURE = {
    "sweep_data_mu_armchair_bowtie_10x10_rot0": 2.6825,
    "sweep_data_mu_zigzag_bowtie_15x15_rot0":  2.6825,
}

BASE_DIR = "."

AU_NM = 0.0529177
AU_EV = 27.2114

MU_UNIT    = "eV"
OMEGA_UNIT = "eV"

CMAP_ABS = "jet"
CMAP_REL = "seismic"

DIFF_CLIP = 99        # percentile cap on |diff| for colorscale (per row)
L0_MASK_FRAC  = 0.10

L0_MASK_FRAC_PER_STRUCTURE = {
    "sweep_data_mu_armchair_bowtie_10x10_rot0": 0.30,
    "sweep_data_mu_zigzag_bowtie_15x15_rot0": 0.60,
    "sweep_data_mu_armchair_triangle_20x20_rot0": 0.10,
    "sweep_data_mu_zigzag_triangle_30x30_rot0": 0.10,
}

INTERPOLATION = "None"

# ============================================================
#  TYPOGRAPHY
# ============================================================
FONT_SIZE_GLOBAL    = 16
FONT_SIZE_TITLE     = 13
FONT_SIZE_COL_TITLE = 16
FONT_SIZE_ROW_LABEL = 16
FONT_SIZE_AXIS_LABEL= 16
FONT_SIZE_TICK      = 14
FONT_SIZE_CBAR      = 16
FONT_SIZE_CBAR_LABEL= 16
FONT_FAMILY         = "times new roman"
USE_LATEX           = False

# ============================================================
#  LAYOUT
# ============================================================
PANEL_W   = 3.6
PANEL_H   = 2.8
CBAR_W    = 0.35
GAP_W     = 0.7
LEFT_PAD  = 1.0
RIGHT_PAD = 0.8
TOP_PAD   = 0.5
BOT_PAD   = 0.6
HSPACE    = 0.08
WSPACE    = 0.05

CBAR_TITLE_X   = 1.4
CBAR_TITLE_Y   = 0.8
CBAR_TITLE_PAD = 8

# ============================================================
#  HELPERS
# ============================================================

def get_acc_au(lattice):
    tree = cKDTree(lattice)
    dists, _ = tree.query(lattice, k=2)
    positive = dists[:, 1][dists[:, 1] > 0.1]
    return positive.min()


def graphene_hex_area_nm2(N_atoms, a_cc_au):
    hex_area_au2 = (3.0 * np.sqrt(3.0) / 2.0) * a_cc_au**2
    total_area_au2 = (N_atoms / 2.0) * hex_area_au2
    return total_area_au2 * AU_NM**2


def find_lattice(structure_dir):
    candidate = os.path.join(structure_dir, "lattice_points.txt")
    if os.path.isfile(candidate):
        return candidate
    try:
        for name in os.listdir(structure_dir):
            child = os.path.join(structure_dir, name)
            if os.path.isdir(child):
                candidate = os.path.join(child, "lattice_points.txt")
                if os.path.isfile(candidate):
                    return candidate
    except PermissionError:
        pass
    raise FileNotFoundError(f"lattice_points.txt not found in {structure_dir}")


def load_total_area_nm2(structure_dir):
    struct_key = os.path.basename(structure_dir.rstrip("/"))
    try:
        lattice_path = find_lattice(structure_dir)
    except FileNotFoundError:
        print(f"[WARNING] lattice_points.txt not found for {struct_key}, sigma will NOT be normalised.")
        return None
    lattice = np.loadtxt(lattice_path, comments="#")
    a_cc_au = A_CC_AU_PER_STRUCTURE.get(struct_key, get_acc_au(lattice))
    N_atoms = len(lattice)
    area    = graphene_hex_area_nm2(N_atoms, a_cc_au)
    print(f"  {struct_key}: a_cc = {a_cc_au:.4f} a.u., N = {N_atoms}, A = {area:.2f} nm2")
    return area


def find_mu_folders(structure_dir, level):
    pattern = re.compile(rf"^{level}_mu_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
    entries = []
    try:
        names = os.listdir(structure_dir)
    except FileNotFoundError:
        return []
    for name in names:
        m = pattern.match(name)
        if m:
            entries.append((float(m.group(1)), os.path.join(structure_dir, name)))
    entries.sort(key=lambda x: x[0])
    return entries


def load_level_grid(structure_dir, level, total_area_nm2):
    mu_folders = find_mu_folders(structure_dir, level)
    if not mu_folders:
        return None, None, None
    mu_vals, omega_ref, columns = [], None, []
    for mu_val, folder in mu_folders:
        fpath = os.path.join(folder, "sigma_ext.txt")
        if not os.path.isfile(fpath):
            continue
        raw = np.loadtxt(fpath)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        omega = raw[:, 0] * AU_EV
        sigma = raw[:, 1] * AU_NM**2
        if total_area_nm2 is not None:
            sigma = sigma / total_area_nm2
        if omega_ref is None:
            omega_ref = omega
        elif not np.allclose(omega, omega_ref, atol=1e-10):
            sigma = np.interp(omega_ref, omega, sigma)
        mu_vals.append(mu_val)
        columns.append(sigma)
    if not columns:
        return None, None, None
    return np.array(mu_vals), omega_ref, np.column_stack(columns)


def make_extent(mu, omega):
    dmu = (mu[-1]    - mu[0])    / max(len(mu)    - 1, 1) if len(mu)    > 1 else 1
    dw  = (omega[-1] - omega[0]) / max(len(omega) - 1, 1) if len(omega) > 1 else 1
    return [mu[0]-dmu/2, mu[-1]+dmu/2, omega[0]-dw/2, omega[-1]+dw/2]


def abs_diff(num, base, mask_frac=L0_MASK_FRAC):
    """Compute (num - base), masking pixels where base is below mask_frac * col peak."""
    if num is None or base is None:
        return None
    col_peak = np.nanmax(np.abs(base), axis=0, keepdims=True)
    mask = np.abs(base) < mask_frac * col_peak
    return np.where(mask, np.nan, num - base)


def short_label(path):
    return (os.path.basename(path)
            .replace("sweep_data_mu_", "")
            .replace("_rot0", "")
            .replace("_", " "))


# ============================================================
#  APPLY STYLE
# ============================================================
plt.rcParams.update({
    "text.usetex":     USE_LATEX,
    "font.family":     FONT_FAMILY,
    "font.size":       FONT_SIZE_GLOBAL,
    "axes.titlesize":  FONT_SIZE_COL_TITLE,
    "axes.labelsize":  FONT_SIZE_AXIS_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
})

# ============================================================
#  LOAD ALL DATA
# ============================================================
LEVELS = ["L0", "L1", "L2"]
all_grids, all_extents = [], []

for structure in STRUCTURES:
    struct_dir = structure if os.path.isabs(structure) \
                 else os.path.join(BASE_DIR, structure)
    total_area_nm2 = load_total_area_nm2(struct_dir)
    grids = {}
    mu_ref = omega_ref = None
    for lvl in LEVELS:
        mu, omega, grid = load_level_grid(struct_dir, lvl, total_area_nm2)
        if grid is None:
            print(f"[WARNING] No data for {short_label(structure)}/{lvl}")
            grids[lvl] = None
            continue
        if mu_ref is None:
            mu_ref, omega_ref = mu, omega
        grids[lvl] = grid
    all_grids.append(grids)
    all_extents.append(make_extent(mu_ref, omega_ref) if mu_ref is not None else None)

# ============================================================
#  FIGURE LAYOUT
# ============================================================
N_ROWS = len(STRUCTURES)
fig_w = LEFT_PAD + PANEL_W + CBAR_W + GAP_W + 2*PANEL_W + CBAR_W + RIGHT_PAD
fig_h = TOP_PAD + N_ROWS * PANEL_H + BOT_PAD
fig = plt.figure(figsize=(fig_w, fig_h))

l = LEFT_PAD   / fig_w
r = 1 - RIGHT_PAD / fig_w
t = 1 - TOP_PAD   / fig_h
b = BOT_PAD    / fig_h

wr = [PANEL_W, CBAR_W, GAP_W, PANEL_W, PANEL_W, CBAR_W]
gs = gridspec.GridSpec(
    N_ROWS, 6, figure=fig, width_ratios=wr,
    hspace=HSPACE, wspace=WSPACE,
    left=l, right=r, top=t, bottom=b,
)

COL_TITLES = ["L2", "", "", r"$L1 - L0$", r"$L2 - L0$", ""]

# ============================================================
#  PLOT
# ============================================================
for row, (structure, grids, extent) in enumerate(
        zip(STRUCTURES, all_grids, all_extents)):

    if extent is None:
        continue

    g0 = grids.get("L0")
    g1 = grids.get("L1")
    g2 = grids.get("L2")

    struct_key = os.path.basename(structure.rstrip("/"))
    mask_frac  = L0_MASK_FRAC_PER_STRUCTURE.get(struct_key, L0_MASK_FRAC)

    d1 = abs_diff(g1, g0, mask_frac=mask_frac)
    d2 = abs_diff(g2, g0, mask_frac=mask_frac)

    diff_finite = [d[np.isfinite(d)] for d in [d1, d2] if d is not None]
    if diff_finite:
        combined  = np.concatenate(diff_finite)
        diff_max  = np.percentile(np.abs(combined), DIFF_CLIP) if combined.size else 1.0
    else:
        diff_max  = 1.0
    diff_max = max(diff_max, 1e-10)

    def clip_diff(arr, cap):
        return np.clip(arr, -cap, cap) if arr is not None else None

    d1_plot   = clip_diff(d1, diff_max)
    d2_plot   = clip_diff(d2, diff_max)
    diff_norm = TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max)

    abs_im_ref = diff_im_ref = None
    abs_max = g2.max() if g2 is not None else 1.0

    # ── L2 abs panel: col 0 ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[row, 0])
    if row == 0:
        ax.set_title(COL_TITLES[0], pad=5)
    if g2 is not None:
        im = ax.imshow(g2, extent=extent, origin="lower",
                       aspect="auto", cmap=CMAP_ABS,
                       vmin=0, vmax=abs_max,
                       interpolation=INTERPOLATION)
        abs_im_ref = im
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    ax.set_ylabel(
        short_label(structure) + "\n" + rf"$\hbar\omega$ ({OMEGA_UNIT})",
        fontsize=FONT_SIZE_ROW_LABEL,
    )
    if row == N_ROWS - 1:
        ax.set_xlabel(rf"$\mu$ ({MU_UNIT})")
    else:
        ax.set_xticklabels([])
    ax.xaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))

    # ── abs colorbar: col 1 ──────────────────────────────────────────────
    ax_cbar_a = fig.add_subplot(gs[row, 1])
    if abs_im_ref is not None:
        cb_a = fig.colorbar(abs_im_ref, cax=ax_cbar_a)
        if row == 0:
            ax_cbar_a.set_title(r"$\sigma_\mathrm{ext}/A$",
                                fontsize=FONT_SIZE_CBAR_LABEL, pad=CBAR_TITLE_PAD)
            ax_cbar_a.title.set_position((CBAR_TITLE_X, CBAR_TITLE_Y))
        cb_a.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        cb_a.ax.tick_params(labelsize=FONT_SIZE_CBAR)

    # col 2 = gap

    # ── diff panels: cols 3, 4 ───────────────────────────────────────────
    for col_idx, data in enumerate([d1_plot, d2_plot]):
        ax = fig.add_subplot(gs[row, col_idx + 3])
        if row == 0:
            ax.set_title(COL_TITLES[col_idx + 3], pad=5)
        if data is not None:
            cmap_obj = plt.get_cmap(CMAP_REL).copy()
            cmap_obj.set_bad(color="white")
            im = ax.imshow(data, extent=extent, origin="lower",
                           aspect="auto", cmap=cmap_obj,
                           norm=diff_norm,
                           interpolation=INTERPOLATION)
            diff_im_ref = im
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
        ax.set_yticklabels([])
        if row == N_ROWS - 1:
            ax.set_xlabel(rf"$\mu$ ({MU_UNIT})")
        else:
            ax.set_xticklabels([])
        ax.xaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))

    # ── diff colorbar: col 5 ─────────────────────────────────────────────
    ax_cbar_r = fig.add_subplot(gs[row, 5])
    if diff_im_ref is not None:
        cb_r = fig.colorbar(diff_im_ref, cax=ax_cbar_r)
        if row == 0:
            ax_cbar_r.set_title(r"$\Delta(\sigma/A)$",
                                fontsize=FONT_SIZE_CBAR_LABEL, pad=CBAR_TITLE_PAD)
            ax_cbar_r.title.set_position((CBAR_TITLE_X, CBAR_TITLE_Y))
        cb_r.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        cb_r.ax.tick_params(labelsize=FONT_SIZE_CBAR)
    else:
        ax_cbar_r.set_visible(False)

plt.savefig("sigma_ext_sweep_diff.png", bbox_inches="tight", dpi=600)
print("Saved: sigma_ext_sweep_diff.png")
plt.show()