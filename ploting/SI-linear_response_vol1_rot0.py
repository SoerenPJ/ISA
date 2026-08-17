"""
plot_supplementary.py
=====================
Supplementary figure: 5-panel layout (A–E) × N_STRUCTURES rows.

  A  L0 σ_ext sweep
  B  L1 σ_ext sweep
  C  L2 σ_ext sweep
  D  Relative error L0 vs L1  [(L1-L0)/|L0|] × 100 %
  E  Relative error L0 vs L2  [(L2-L0)/|L0|] × 100 %

Per-row colorbars (abs right of C, rel right of E).
Panel labels A–E appear in the top row only as column headers.
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
'''
STRUCTURES = [
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_bowtie_10x10_rot90",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_bowtie_15x15_rot0",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_triangle_14x14_rot90",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_triangle_22x22_rot0",

]
'''

STRUCTURES = [
    "/home/soeren/University/masters/2.semester/ISA/scr/data_LLM/sweep_data_mu_armchair_bowtie_2x2_rot90",
    "/home/soeren/University/masters/2.semester/ISA/scr/data_LLM/sweep_data_mu_zigzag_bowtie_2x2_rot0",
    "/home/soeren/University/masters/2.semester/ISA/scr/data_LLM/sweep_data_mu_armchair_triangle_2x2_rot90",
    "/home/soeren/University/masters/2.semester/ISA/scr/data_LLM/sweep_data_mu_zigzag_triangle_2x2_rot0",

]

A_CC_AU_PER_STRUCTURE = {
    "sweep_data_mu_armchair_bowtie_10x10_rot0": 2.6825,
    "sweep_data_mu_zigzag_bowtie_15x15_rot90":  2.6825,
}

BASE_DIR   = "."
AU_NM      = 0.0529177
AU_EV      = 27.2114
MU_UNIT    = "eV"
CMAP_ABS   = "jet"
CMAP_REL   = "seismic"
REL_ERR_CLIP = 99          # percentile used to set rel-error colour scale
INTERPOLATION = "None"



# ── masking for rel-error (suppress near-zero baseline artefacts) ─────────────
L0_MASK_FRAC = 0.10
L0_MASK_FRAC_PER_STRUCTURE = {
    "sweep_data_mu_armchair_bowtie_10x10_rot0": 0.30,
    "sweep_data_mu_zigzag_bowtie_15x15_rot0":  0.30,
}

EDGE_TOL_EV = 0.03

# ── row labels (leave None to use auto-generated names) ──────────────────────
ROW_LABELS = None          # e.g. ["Armchair", "Zigzag"]

# ============================================================
#  TYPOGRAPHY
# ============================================================
FONT_SIZE_GLOBAL     = 20
FONT_SIZE_AXIS_LABEL = 18
FONT_SIZE_TICK       = 16
FONT_SIZE_CBAR       = 14
FONT_SIZE_CBAR_LABEL = 14
FONT_SIZE_PANEL_LABEL= 16
FONT_FAMILY          = "times new roman"
USE_LATEX            = False

# ============================================================
#  LAYOUT
# ============================================================
PANEL_W   = 2.60
PANEL_H   = 2.20    # matches working figure exactly
CBAR_W    = 0.28
GAP_W     = 0.40
SMALL_GAP = 0.12
LEFT_PAD  = 0.80
RIGHT_PAD = 0.55
TOP_PAD   = 0.40
BOT_PAD   = 0.50
HSPACE    = 0.04
WSPACE    = 0.05

PANEL_LABELS = ["A", "B", "C", "D", "E"]

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
        return None
    lattice = np.loadtxt(lattice_path, comments="#")
    a_cc_au = A_CC_AU_PER_STRUCTURE.get(struct_key, get_acc_au(lattice))
    N_atoms = len(lattice)
    area    = graphene_hex_area_nm2(N_atoms, a_cc_au)
    print(f"  {struct_key}: a_cc={a_cc_au:.4f} a.u., N={N_atoms}, A={area:.2f} nm²")
    return area


def find_bulk_gap(energies_eV):
    e = np.sort(energies_eV)
    bulk = e[np.abs(e) > EDGE_TOL_EV]
    bulk_neg = bulk[bulk < 0]
    bulk_pos = bulk[bulk > 0]
    if len(bulk_neg) > 0 and len(bulk_pos) > 0:
        gap_bot = bulk_neg.max()
        gap_top = bulk_pos.min()
    else:
        N = len(e)
        gap_bot = e[N // 2 - 1]
        gap_top = e[N // 2]
    return gap_bot, gap_top, gap_top - gap_bot


def find_mu_folders(structure_dir, level):
    pattern = re.compile(
        rf"^{level}_mu_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
    entries = []
    try:
        for name in os.listdir(structure_dir):
            m = pattern.match(name)
            if m:
                entries.append(
                    (float(m.group(1)), os.path.join(structure_dir, name)))
    except FileNotFoundError:
        return []
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
    dmu = (mu[-1] - mu[0]) / max(len(mu) - 1, 1) if len(mu) > 1 else 1
    dw  = (omega[-1] - omega[0]) / max(len(omega) - 1, 1) if len(omega) > 1 else 1
    return [mu[0]-dmu/2, mu[-1]+dmu/2, omega[0]-dw/2, omega[-1]+dw/2]


def rel_err(num, base, mask_frac=L0_MASK_FRAC):
    if num is None or base is None:
        return None
    col_peak = np.nanmax(np.abs(base), axis=0, keepdims=True)
    mask = np.abs(base) < mask_frac * col_peak
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(mask, np.nan, (num - base) / np.abs(base)) * 100
    return r


def short_label(path):
    return (os.path.basename(path)
            .replace("sweep_data_mu_", "")
            .replace("_rot0", "")
            .replace("_", " "))


def style_heatmap(ax, is_last_row, show_ylabel):
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))
    if show_ylabel:
        ax.set_ylabel("Photon energy (eV)", fontsize=FONT_SIZE_AXIS_LABEL)
    else:
        ax.set_yticklabels([])
    if is_last_row:
        ax.set_xlabel(rf"$\mu$ ({MU_UNIT})", fontsize=FONT_SIZE_AXIS_LABEL)
    else:
        ax.set_xticklabels([])


# ============================================================
#  APPLY STYLE
# ============================================================
plt.rcParams.update({
    "text.usetex":     USE_LATEX,
    "font.family":     FONT_FAMILY,
    "font.size":       FONT_SIZE_GLOBAL,
    "axes.labelsize":  FONT_SIZE_AXIS_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
})

# ============================================================
#  LOAD ALL DATA
# ============================================================
LEVELS = ["L0", "L1", "L2"]
all_data = []   # list of dicts, one per structure

for structure in STRUCTURES:
    struct_dir = structure if os.path.isabs(structure) \
                 else os.path.join(BASE_DIR, structure)
    struct_key = os.path.basename(struct_dir.rstrip("/"))
    print(f"\nLoading {struct_key}")

    total_area_nm2 = load_total_area_nm2(struct_dir)

    grids = {}
    mu_ref = omega_ref = None
    mu_sweep_vals = []
    for lvl in LEVELS:
        mu, omega, grid = load_level_grid(struct_dir, lvl, total_area_nm2)
        if grid is None:
            print(f"  [WARNING] No data for {lvl}")
            grids[lvl] = None
            continue
        if mu_ref is None:
            mu_ref, omega_ref = mu, omega
            mu_sweep_vals = list(mu)
        grids[lvl] = grid

    extent = make_extent(mu_ref, omega_ref) if mu_ref is not None else None

    mask_frac = L0_MASK_FRAC_PER_STRUCTURE.get(struct_key, L0_MASK_FRAC)
    r_D = rel_err(grids.get("L1"), grids.get("L0"), mask_frac=mask_frac)
    r_E = rel_err(grids.get("L2"), grids.get("L0"), mask_frac=mask_frac)

    all_data.append(dict(
        struct_key=struct_key,
        grids=grids,
        extent=extent,
        r_D=r_D,
        r_E=r_E,
    ))

# ============================================================
#  FIGURE LAYOUT
#
#  GridSpec columns (indices):
#    0  panel A (L0)
#    1  small gap
#    2  panel B (L1)
#    3  small gap
#    4  panel C (L2)
#    5  abs colorbar
#    6  gap
#    7  panel D (rel L0vL1)
#    8  small gap
#    9  panel E (rel L0vL2)
#   10  rel colorbar
# ============================================================
N_ROWS = len(STRUCTURES)

wr = [
    PANEL_W, SMALL_GAP,   # A
    PANEL_W, SMALL_GAP,   # B
    PANEL_W, CBAR_W,      # C + abs cbar
    GAP_W,                # gap between groups
    PANEL_W, SMALL_GAP,   # D
    PANEL_W, CBAR_W,      # E + rel cbar
]

fig_w = LEFT_PAD + sum(wr) + RIGHT_PAD
fig_h = TOP_PAD + N_ROWS * PANEL_H + (N_ROWS - 1) * PANEL_H * HSPACE + BOT_PAD
fig = plt.figure(figsize=(fig_w, fig_h))

l = LEFT_PAD  / fig_w
r = 1 - RIGHT_PAD / fig_w
t = 1 - TOP_PAD   / fig_h
b = BOT_PAD   / fig_h

gs = gridspec.GridSpec(
    N_ROWS, 11, figure=fig, width_ratios=wr,
    hspace=HSPACE, wspace=WSPACE,
    left=l, right=r, top=t, bottom=b,
)

# Column indices for each panel
COL = dict(A=0, B=2, C=4, CBAR_ABS=5, D=7, E=9, CBAR_REL=10)

# ============================================================
#  PANEL HEADER LABELS  (A–E, top row only, above first data row)
# ============================================================
PANEL_TITLES = {
    "A": "Hartree",
    "B": "Zeeman",
    "C": "Peierls phase",
    "D": r"$\delta\sigma/\sigma_\mathrm{L0}$ (%)" + "\nZeeman",
    "E": r"$\delta\sigma/\sigma_\mathrm{L0}$ (%)" + "\nPeierls phase",
}

# ============================================================
#  PLOT ROWS  — per-row colour scales and colorbars
# ============================================================
cmap_rel = plt.get_cmap(CMAP_REL).copy()
cmap_rel.set_bad(color="white")

for row, d in enumerate(all_data):
    struct_key = d["struct_key"]
    extent     = d["extent"]
    grids      = d["grids"]
    r_D        = d["r_D"]
    r_E        = d["r_E"]
    is_last    = (row == N_ROWS - 1)

    if extent is None:
        continue

    g0 = grids.get("L0")
    g1 = grids.get("L1")
    g2 = grids.get("L2")

    # ── per-row colour scales ──────────────────────────────────────────────
    abs_max = max(
        g.max() for g in [g0, g1, g2] if g is not None
    ) if any(g is not None for g in [g0, g1, g2]) else 1.0

    row_rels = [r for r in [r_D, r_E] if r is not None]
    row_rel_finite = np.concatenate([r[np.isfinite(r)] for r in row_rels]) \
                     if row_rels else np.array([1.0])
    rel_max = max(
        np.percentile(np.abs(row_rel_finite), REL_ERR_CLIP)
        if row_rel_finite.size else 1.0,
        1e-10,
    )
    rel_norm = TwoSlopeNorm(vmin=-rel_max, vcenter=0, vmax=rel_max)

    # ── row label (left y-axis of panel A) ────────────────────────────────
    row_label = (ROW_LABELS[row] if ROW_LABELS else
                 short_label(STRUCTURES[row]))

    # ── panels A, B, C  (absolute σ_ext) ──────────────────────────────────
    abs_im_ref = None
    for panel, grid in [("A", g0), ("B", g1), ("C", g2)]:
        ax = fig.add_subplot(gs[row, COL[panel]])
        show_y = (panel == "A")

        if row == 0:
            ax.set_title(PANEL_TITLES[panel],
                         fontsize=FONT_SIZE_PANEL_LABEL, pad=4)

        if grid is not None:
            im = ax.imshow(grid, extent=extent, origin="lower",
                           aspect="auto", cmap=CMAP_ABS,
                           vmin=0, vmax=abs_max,
                           interpolation=INTERPOLATION, rasterized=True)
            if panel == "C":
                abs_im_ref = im
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")

        style_heatmap(ax, is_last, show_ylabel=show_y)

        # row label on the left edge of panel A
        if show_y:
            ax.set_ylabel("Photon energy (eV)",
                          fontsize=FONT_SIZE_AXIS_LABEL)

    # ── abs colorbar — per row ─────────────────────────────────────────────
    ax_cbar_a = fig.add_subplot(gs[row, COL["CBAR_ABS"]])
    if abs_im_ref is not None:
        cb = fig.colorbar(abs_im_ref, cax=ax_cbar_a)
        cb.ax.tick_params(labelsize=FONT_SIZE_CBAR)
        cb.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        if row == 0:
            ax_cbar_a.set_title(r"$\sigma^\mathrm{ext}/A$",
                                 fontsize=FONT_SIZE_CBAR_LABEL, pad=5)
    else:
        ax_cbar_a.set_visible(False)

    # ── panels D, E  (relative error) ─────────────────────────────────────
    rel_im_ref = None
    for panel, data in [("D", r_D), ("E", r_E)]:
        ax = fig.add_subplot(gs[row, COL[panel]])

        if row == 0:
            ax.set_title(PANEL_TITLES[panel],
                         fontsize=FONT_SIZE_PANEL_LABEL, pad=4)

        if data is not None:
            clipped = np.clip(data, -rel_max, rel_max)
            im = ax.imshow(clipped, extent=extent, origin="lower",
                           aspect="auto", cmap=cmap_rel,
                           norm=rel_norm, interpolation=INTERPOLATION, rasterized=True)
            if panel == "E":
                rel_im_ref = im
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")

        style_heatmap(ax, is_last, show_ylabel=False)

    # ── rel colorbar — per row ─────────────────────────────────────────────
    ax_cbar_r = fig.add_subplot(gs[row, COL["CBAR_REL"]])
    if rel_im_ref is not None:
        cb = fig.colorbar(rel_im_ref, cax=ax_cbar_r)
        cb.ax.tick_params(labelsize=FONT_SIZE_CBAR)
        cb.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        if row == 0:
            ax_cbar_r.set_title(r"$\delta\sigma/\sigma_0$ (%)",
                                 fontsize=FONT_SIZE_CBAR_LABEL, pad=5)
    else:
        ax_cbar_r.set_visible(False)

# ============================================================
#  SAVE
# ============================================================
out = "supplementary_sigma_ext_rot_0vol1_size2x2.pdf"
plt.savefig(out, bbox_inches="tight", dpi=600, format="pdf")
plt.show()