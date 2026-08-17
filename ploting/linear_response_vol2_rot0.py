"""
plot_sweep_sigma_ext.py  —  v3: 4-row layout for A4 portrait
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import cKDTree

# ============================================================
#  USER SETTINGS
# ============================================================
'''
STRUCTURES = [
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_bowtie_10x10_rot90",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_bowtie_15x15_rot0",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_triangle_14x14_rot90",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_triangle_22x22_rot0",  # ← replace
]
'''

STRUCTURES = [
    "/home/soeren/University/masters/2.semester/ISA/scr/data_LLM/sweep_data_mu_armchair_bowtie_2x2_rot90",
    "/home/soeren/University/masters/2.semester/ISA/scr/data_LLM/sweep_data_mu_zigzag_bowtie_2x2_rot0",
    "/home/soeren/University/masters/2.semester/ISA/scr/data_LLM/sweep_data_mu_armchair_triangle_2x2_rot90",
    "/home/soeren/University/masters/2.semester/ISA/scr/data_LLM/sweep_data_mu_zigzag_triangle_2x2_rot0",

]

A_CC_AU_PER_STRUCTURE = {
    "sweep_data_mu_armchair_bowtie_10x10_rot90":   2.6825,
    "sweep_data_mu_zigzag_bowtie_15x15_rot0":    2.6825,
    "sweep_data_mu_armchair_triangle_14x14_rot90": 2.6825,
    "sweep_data_mu_zigzag_triangle_22x22_rot0":  2.6825,  # ← replace key & value
}

BASE_DIR = "."
AU_NM = 0.0529177
AU_EV = 27.2114
MU_UNIT    = "eV"
OMEGA_UNIT = "eV"
CMAP_ABS = "jet"

INTERPOLATION = "None"

L0_MASK_FRAC = 0.10

L0_MASK_FRAC_PER_STRUCTURE = {
    "sweep_data_mu_armchair_bowtie_10x10_rot0": 0.30,
    "sweep_data_mu_zigzag_bowtie_15x15_rot0":  0.30,
    "sweep_data_mu_armchair_triangle_20x20_rot0": 0.10,
    "sweep_data_mu_zigzag_triangle_30x30_rot0":   0.10,
}

MU_CASES = "auto"   # global fallback
'''
MU_CASES_PER_STRUCTURE = {
    "sweep_data_mu_armchair_bowtie_10x10_rot90":   [0.5, 2.0, 3.5],
    "sweep_data_mu_zigzag_bowtie_15x15_rot0":    [0.5, 2.0, 3.5],
    "sweep_data_mu_armchair_triangle_14x14_rot90": [0.5, 2.0, 3.5],
    "sweep_data_mu_zigzag_triangle_22x22_rot0":   [0.5, 2.0, 3.5],  
}
'''

MU_CASES_PER_STRUCTURE = {
    "sweep_data_mu_armchair_bowtie_2x2_rot90":   [0.5, 2.0, 3.5],
    "sweep_data_mu_zigzag_bowtie_2x2_rot0":    [0.5, 2.0, 3.5],
    "sweep_data_mu_armchair_triangle_2x2_rot90": [0.5, 2.0, 3.5],
    "sweep_data_mu_zigzag_triangle_2x2_rot0":   [0.5, 2.0, 3.5],  
}

MU_COLORS = ['#F033C0', '#F06030', '#E02800']
MU_LABELS = None

Y_FLOOR = 1e-5

EDGE_TOL_EV = 0.03   # eV

CIRCLE_SIZE = 6      # reduced from 12
CIRCLE_LW   = 0.5

# ── Structure plot settings ──────────────────────────────────────────────────
STRUCT_DOT_SIZE   = 8        # reduced from 15
STRUCT_DOT_COLOR  = '#111111'
STRUCT_DOT_ALPHA  = 1.0
DRAW_BONDS        = True
BOND_MAX_FACTOR   = 1.20
BOND_COLOR        = '#555555'
BOND_LW           = 1.0      # reduced from 1.5
BOND_ALPHA        = 1.0

# ── Scale bar & E-field arrow ─────────────────────────────────────────────────
SCALEBAR_DEFAULTS = {
    "arrow":       "horizontal",   # default: horizontal along longest side
    "E_angle_deg": 0,
    "show_efield": False,
    "show_scale":  True,
}
SCALEBAR_PER_STRUCTURE = {
    # bowties: full-width horizontal (tip-to-tip) + vertical (height)
    "sweep_data_mu_armchair_bowtie_10x10_rot90": {"arrow": "bowtie_h", "E_angle_deg": 0},
    "sweep_data_mu_zigzag_bowtie_15x15_rot0":    {"arrow": "bowtie_h", "E_angle_deg": 0},
    # triangles: horizontal along base (longest side)
    "sweep_data_mu_armchair_triangle_14x14_rot90": {"arrow": "horizontal"},
    "sweep_data_mu_zigzag_triangle_22x22_rot0":    {"arrow": "horizontal"},
}
SCALEBAR_COLOR = "#E65C00"
EFIELD_COLOR   = "#1a7a1a"
FONT_SCALEBAR  = 18   # reduced from 13
FONT_EFIELD    = 9   # reduced from 13

# ============================================================
#  TYPOGRAPHY  — scaled down for A4 portrait
# ============================================================
FONT_SIZE_GLOBAL    = 20   # was 14
FONT_SIZE_COL_TITLE = 18   # was 11
FONT_SIZE_AXIS_LABEL= 18   # was 11
FONT_SIZE_TICK      = 16   # was 10
FONT_SIZE_CBAR      = 14   # was 10
FONT_SIZE_CBAR_LABEL= 14   # was 10
FONT_SIZE_LEGEND    = 13   # was 8
FONT_SCALEBAR       = 14   # was 9
FONT_FAMILY         = "times new roman"
USE_LATEX           = False

# ============================================================
#  LAYOUT  — A4 portrait, 4 rows
# ============================================================
FIG_W     = 16.0
STRUCT_W  = 3.2    # was 2.5 — more room for structure
EIG_W     = 3.5    # was 2.5 — more room for eigenvalue plot
PANEL_W   = 3.5    # was 2.2 — more room for heatmap
PANEL_H   = 2.2    # was 1.3 — taller rows so fonts don't feel cramped
CBAR_W    = 0.30   # was 0.25
GAP_W     = 0.20   # was 0.30 — tighter gaps
LEFT_PAD  = 0.15
RIGHT_PAD = 0.80   # was 0.60 — colorbar label needs room
TOP_PAD   = 0.20
BOT_PAD   = 0.50
HSPACE    = 0.15   # was 0.10
WSPACE    = 0.25   # was 0.20


CBAR_TITLE_X   = 0.8
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
        return None
    lattice = np.loadtxt(lattice_path, comments="#")
    a_cc_au = A_CC_AU_PER_STRUCTURE.get(struct_key, get_acc_au(lattice))
    N_atoms = len(lattice)
    area    = graphene_hex_area_nm2(N_atoms, a_cc_au)
    print(f"  {struct_key}: a_cc={a_cc_au:.4f} a.u., N={N_atoms}, A={area:.2f} nm2")
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


def load_eigenvalues_eV(structure_dir):
    try:
        lattice_path = find_lattice(structure_dir)
    except FileNotFoundError:
        return None, None, None, None
    eig_path = os.path.join(os.path.dirname(lattice_path), "eigenvalues.txt")
    if not os.path.isfile(eig_path):
        print(f"  [WARNING] eigenvalues.txt not found")
        return None, None, None, None
    raw = np.loadtxt(eig_path)
    energies_au = raw if raw.ndim == 1 else raw[:, 0]
    energies_eV = np.sort(energies_au * AU_EV)

    gap_bot_abs, gap_top_abs, gap = find_bulk_gap(energies_eV)
    E_fermi = 0.5 * (gap_bot_abs + gap_top_abs)
    energies_eV -= E_fermi
    gap_bot = gap_bot_abs - E_fermi
    gap_top = gap_top_abs - E_fermi
    gap     = gap_top - gap_bot

    print(f"  {len(energies_eV)} states, bulk gap={gap:.4f} eV "
          f"[{gap_bot:.4f} -> {gap_top:.4f}], E_fermi(abs)={E_fermi:.4f} eV")
    return energies_eV, gap_bot, gap_top, gap


def load_lattice_coords(structure_dir):
    try:
        lattice_path = find_lattice(structure_dir)
    except FileNotFoundError:
        return None, None
    lattice = np.loadtxt(lattice_path, comments="#")
    struct_key = os.path.basename(structure_dir.rstrip("/"))
    a_cc_au = A_CC_AU_PER_STRUCTURE.get(struct_key, get_acc_au(lattice))
    coords_nm = lattice[:, :2] * AU_NM
    return coords_nm, a_cc_au


def find_mu_folders(structure_dir, level):
    pattern = re.compile(rf"^{level}_mu_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
    entries = []
    try:
        for name in os.listdir(structure_dir):
            m = pattern.match(name)
            if m:
                entries.append((float(m.group(1)), os.path.join(structure_dir, name)))
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


def short_label(path):
    return (os.path.basename(path)
            .replace("sweep_data_mu_", "")
            .replace("_rot0", "")
            .replace("_", " "))


def resolve_mu_cases(struct_key, mu_sweep):
    if struct_key in MU_CASES_PER_STRUCTURE:
        return list(MU_CASES_PER_STRUCTURE[struct_key])
    if MU_CASES != "auto":
        return list(MU_CASES)
    if len(mu_sweep) >= 3:
        return [mu_sweep[0], mu_sweep[len(mu_sweep) // 2], mu_sweep[-1]]
    return list(mu_sweep)


def nearest_mu_index(mu_array, mu_target):
    return int(np.argmin(np.abs(mu_array - mu_target)))


# ============================================================
#  DRAW: structure (lattice)
# ============================================================

def _get_scalebar_opts(struct_key):
    opts = dict(SCALEBAR_DEFAULTS)
    opts.update(SCALEBAR_PER_STRUCTURE.get(struct_key, {}))
    return opts


def _draw_scalebar(ax, x, y, xpad, ypad, arrow_style):
    ap = dict(arrowstyle="<->", color=SCALEBAR_COLOR, lw=2.0)
    true_w = x.max() - x.min()
    true_h = y.max() - y.min()

    if arrow_style in ("horizontal", "bowtie_h"):
        # ── horizontal arrow spanning full width, placed below the structure ──
        bar_len = round(true_w,1) 
        bar_y   = y.min() - ypad * 0.55
        bar_xlo = x.min()
        bar_xhi = x.max()
        ax.annotate("", xy=(bar_xhi, bar_y), xytext=(bar_xlo, bar_y),
                    arrowprops=ap)
        ax.text((bar_xlo + bar_xhi) / 2, bar_y - ypad * 0.22,
                rf"{bar_len:.1f} nm",
                ha='center', va='top', fontsize=FONT_SCALEBAR,
                color=SCALEBAR_COLOR, fontweight='bold')

        if arrow_style == "bowtie_h":
            # ── second arrow: vertical span on the left side ──────────────
            bar2_len  = round(true_h,1) 
            bar2_x    = x.min() - xpad * 0.55
            bar2_ymid = (y.min() + y.max()) / 2
            bar2_ylo  = bar2_ymid - bar2_len / 2
            bar2_yhi  = bar2_ymid + bar2_len / 2
            ax.annotate("", xy=(bar2_x, bar2_yhi), xytext=(bar2_x, bar2_ylo),
                        arrowprops=ap)
            ax.text(bar2_x - xpad * 0.15, bar2_ymid,
                    rf"{bar2_len:.1f} nm",
                    ha='right', va='center', fontsize=FONT_SCALEBAR,
                    color=SCALEBAR_COLOR, rotation=90, fontweight='bold')

    elif arrow_style == "diagonal":
        bar_len = round(true_h,1) 
        bar_x   = x.min() - xpad * 0.40
        bar_ylo = y.min()
        bar_yhi = bar_ylo + bar_len
        ax.annotate("", xy=(bar_x, bar_yhi), xytext=(bar_x, bar_ylo),
                    arrowprops=ap)
        ax.text(bar_x - xpad * 0.10, (bar_ylo + bar_yhi) / 2,
                rf"{bar_len:.1f} nm",
                ha='right', va='center', fontsize=FONT_SCALEBAR,
                color=SCALEBAR_COLOR, rotation=90, fontweight='bold')

    else:  # vertical (legacy fallback)
        bar_len  = round(true_h,1) 
        bar_x    = x.min() - xpad * 0.55
        bar_ymid = (y.min() + y.max()) / 2
        bar_ylo  = bar_ymid - bar_len / 2
        bar_yhi  = bar_ymid + bar_len / 2
        ax.annotate("", xy=(bar_x, bar_yhi), xytext=(bar_x, bar_ylo),
                    arrowprops=ap)
        ax.text(bar_x - xpad * 0.15, bar_ymid,
                rf"{bar_len:.1f} nm",
                ha='right', va='center', fontsize=FONT_SCALEBAR,
                color=SCALEBAR_COLOR, rotation=90, fontweight='bold')


def _draw_efield_arrow(ax, x, y, xpad, ypad, angle_deg):
    x_span    = x.max() - x.min()
    y_span    = y.max() - y.min()
    arrow_len = 0.40 * min(x_span, y_span)
    theta = np.deg2rad(angle_deg)
    dx = arrow_len * np.cos(theta)
    dy = arrow_len * np.sin(theta)
    x_tail = (x.min() + x.max()) / 2 - dx / 2
    y_tail = y.max() + ypad * 0.55 - dy / 2
    ax.annotate(
        "", xy=(x_tail + dx, y_tail + dy), xytext=(x_tail, y_tail),
        arrowprops=dict(arrowstyle="-|>", color=EFIELD_COLOR,
                        lw=2.2, mutation_scale=16),
    )
    ax.text(x_tail + dx / 2, y_tail + dy / 2 + ypad * 0.22,
            r"$\mathbf{E}$",
            ha='center', va='bottom', fontsize=FONT_EFIELD,
            color=EFIELD_COLOR, fontweight='bold')


def draw_structure(ax, coords_nm, a_cc_au, struct_key, is_last_row):
    if coords_nm is None:
        ax.text(0.5, 0.5, "no lattice", ha='center', va='center',
                transform=ax.transAxes, color='grey')
        return

    x, y = coords_nm[:, 0], coords_nm[:, 1]

    if DRAW_BONDS:
        bond_cutoff_nm = BOND_MAX_FACTOR * a_cc_au * AU_NM
        tree = cKDTree(coords_nm)
        pairs = tree.query_pairs(bond_cutoff_nm)
        from matplotlib.collections import LineCollection
        segments = [[(x[i], y[i]), (x[j], y[j])] for i, j in pairs]
        lc = LineCollection(segments, colors=BOND_COLOR, linewidths=BOND_LW,
                            alpha=BOND_ALPHA, zorder=1)
        ax.add_collection(lc)

    ax.scatter(x, y, s=STRUCT_DOT_SIZE, color=STRUCT_DOT_COLOR,
               alpha=STRUCT_DOT_ALPHA, zorder=2, linewidths=0, )

    opts        = _get_scalebar_opts(struct_key)
    arrow_style = opts["arrow"]
    xpad = (x.max() - x.min()) * 0.08

    # ── ypad: give more room below for horizontal arrows ──────────────────
    ypad = (y.max() - y.min()) * (0.1 if arrow_style in ("horizontal", "bowtie_h") else 0.26)

    if opts["show_scale"]:
        _draw_scalebar(ax, x, y, xpad, ypad, arrow_style)

    if opts["show_efield"]:
        _draw_efield_arrow(ax, x, y, xpad, ypad, opts["E_angle_deg"])

    ax.set_xlim(x.min() - xpad, x.max() + xpad)
    ax.set_ylim(y.min() - ypad, y.max() + ypad)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ============================================================
#  DRAW: eigenvalue spectrum
# ============================================================

def draw_eigenvalue_spectrum(ax, energies_eV, gap_bot, gap_top, gap,
                             mu_cases_eV, mu_labels, is_last_row):
    N = len(energies_eV)
    indices = np.arange(N)

    e_span   = energies_eV[-1] - energies_eV[0]
    e_margin = e_span * 0.04
    ax.set_ylim(energies_eV[0] - e_margin, energies_eV[-1] + e_margin)
    ax.set_xlim(-1, N)

    ax.scatter(indices, energies_eV,
               s=CIRCLE_SIZE, facecolors='none',
               edgecolors='#888888', linewidths=CIRCLE_LW,
               zorder=2, label='_nolegend_')

    for mu_val, color, label in sorted(
            zip(mu_cases_eV, MU_COLORS, mu_labels), key=lambda x: -x[0]):
        filled = energies_eV <= mu_val
        if filled.any():
            ax.scatter(indices[filled], energies_eV[filled],
                       s=CIRCLE_SIZE, facecolors=color, edgecolors=color,
                       linewidths=CIRCLE_LW, zorder=3, alpha=0.9, label=label, rasterized=True)
        ax.axhline(mu_val, color=color, lw=1.5, ls='--', alpha=0.8, zorder=4)

    if gap > 0.05:
        gap_mid = (gap_bot + gap_top) / 2
        ax.text(N * 0.98, gap_mid -  5.5,
                rf'$E_g = {gap:.2f}$ eV',
                ha='right', va='center', fontsize=12, color='black',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='#aaaaaa', lw=0.8, alpha=0.95),
                zorder=6)

    ax.set_ylabel('Energy (eV)', fontsize=FONT_SIZE_AXIS_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, lw=0.3, alpha=0.4)
    if is_last_row:
        ax.set_xlabel('State index', fontsize=FONT_SIZE_AXIS_LABEL)
    else:
        ax.set_xticklabels([])

    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper left',
              framealpha=0.8, markerscale=2.5,
              handletextpad=0.3, borderpad=0.4, labelspacing=0.3)


def draw_mu_vlines(ax, mu_cases_eV):
    for mu_val, color in zip(mu_cases_eV, MU_COLORS):
        ax.axvline(mu_val, color=color, lw=2.0, ls='--', alpha=0.85, zorder=5)


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
all_grids, all_extents, all_eig, all_mu_sweeps, all_mu_arrays, all_omega = \
    [], [], [], [], [], []
all_lattice_coords = []

for structure in STRUCTURES:
    struct_dir = structure if os.path.isabs(structure) \
                 else os.path.join(BASE_DIR, structure)
    total_area_nm2 = load_total_area_nm2(struct_dir)
    all_eig.append(load_eigenvalues_eV(struct_dir))
    all_lattice_coords.append(load_lattice_coords(struct_dir))

    grids = {}
    mu_ref = omega_ref = None
    mu_sweep_vals = []
    for lvl in LEVELS:
        mu, omega, grid = load_level_grid(struct_dir, lvl, total_area_nm2)
        if grid is None:
            print(f"[WARNING] No data for {short_label(structure)}/{lvl}")
            grids[lvl] = None
            continue
        if mu_ref is None:
            mu_ref, omega_ref = mu, omega
            mu_sweep_vals = list(mu)
        grids[lvl] = grid

    all_grids.append(grids)
    all_extents.append(make_extent(mu_ref, omega_ref) if mu_ref is not None else None)
    all_mu_sweeps.append(mu_sweep_vals)
    all_mu_arrays.append(mu_ref)
    all_omega.append(omega_ref)

# ============================================================
#  FIGURE LAYOUT  (A4 portrait, 4 rows)
# ============================================================
N_ROWS = len(STRUCTURES)
wr = [STRUCT_W, GAP_W, EIG_W, GAP_W, PANEL_W, CBAR_W]

inner_w = sum(wr)
total_pad_w = LEFT_PAD + RIGHT_PAD
scale = (FIG_W - total_pad_w) / inner_w
wr_scaled = [w * scale for w in wr]

fig_h = TOP_PAD + N_ROWS * PANEL_H + (N_ROWS - 1) * PANEL_H * HSPACE + BOT_PAD
fig = plt.figure(figsize=(FIG_W, fig_h))

l = LEFT_PAD / FIG_W
r = 1 - RIGHT_PAD / FIG_W
t = 1 - TOP_PAD  / fig_h
b = BOT_PAD      / fig_h

gs = gridspec.GridSpec(
    N_ROWS, 6, figure=fig, width_ratios=wr_scaled,
    hspace=HSPACE, wspace=WSPACE,
    left=l, right=r, top=t, bottom=b,
)

# ============================================================
#  PLOT
# ============================================================
for row, (structure, grids, extent, eig_data, mu_sweep, mu_array, omega,
          lattice_data) in enumerate(
        zip(STRUCTURES, all_grids, all_extents, all_eig,
            all_mu_sweeps, all_mu_arrays, all_omega,
            all_lattice_coords)):

    if extent is None:
        continue

    energies_eV, gap_bot, gap_top, gap = eig_data
    g2 = grids.get("L2")
    is_last_row = (row == N_ROWS - 1)

    struct_key = os.path.basename(structure.rstrip("/"))
    mu_cases   = resolve_mu_cases(struct_key, mu_sweep)
    labels = MU_LABELS if MU_LABELS is not None else \
             [rf'$\mu={m:.2f}$ eV' for m in mu_cases]

    coords_nm, a_cc_au = lattice_data
    abs_max = g2.max() if g2 is not None else 1.0

    # ── col 0: structure plot ──────────────────────────────────────────────
    ax_struct = fig.add_subplot(gs[row, 0])
    draw_structure(ax_struct, coords_nm, a_cc_au if a_cc_au is not None else 1.0,
                   struct_key, is_last_row)

    # ── col 2: eigenvalue spectrum ─────────────────────────────────────────
    ax_eig = fig.add_subplot(gs[row, 2])
    if energies_eV is not None:
        draw_eigenvalue_spectrum(ax_eig, energies_eV, gap_bot, gap_top, gap,
                                 mu_cases, labels, is_last_row)
    else:
        ax_eig.text(0.5, 0.5, "no eigenvalues", ha='center', va='center',
                    transform=ax_eig.transAxes, color='grey')

    # ── col 4: L2 σ_ext heatmap ────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[row, 4])
    abs_im_ref = None
    if g2 is not None:
        im = ax_heat.imshow(g2, extent=extent, origin="lower",
                            aspect="auto", cmap=CMAP_ABS,
                            vmin=0, vmax=abs_max,
                            interpolation=INTERPOLATION, rasterized=True)
        abs_im_ref = im
        draw_mu_vlines(ax_heat, mu_cases)
    else:
        ax_heat.text(0.5, 0.5, "no data", ha="center", va="center",
                     transform=ax_heat.transAxes, color="grey")
    ax_heat.tick_params(axis="y", which="both", labelleft=True, left=True)
    ax_heat.set_ylabel("Photon energy (eV)", fontsize=FONT_SIZE_AXIS_LABEL)
    if is_last_row:
        ax_heat.set_xlabel(rf"$\mu$ ({MU_UNIT})", fontsize=FONT_SIZE_AXIS_LABEL)
    else:
        ax_heat.set_xticklabels([])
    ax_heat.xaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))
    ax_heat.yaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))

    # ── col 5: abs colorbar ─────────────────────────────────────────────────
    ax_cbar_a = fig.add_subplot(gs[row, 5])
    if abs_im_ref is not None:
        cb_a = fig.colorbar(abs_im_ref, cax=ax_cbar_a)
        if row == 0:
            ax_cbar_a.set_title(r"$\sigma^\mathrm{ext}/A$",
                                fontsize=FONT_SIZE_CBAR_LABEL, pad=CBAR_TITLE_PAD)
            ax_cbar_a.title.set_position((CBAR_TITLE_X, CBAR_TITLE_Y))
        cb_a.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        cb_a.ax.tick_params(labelsize=FONT_SIZE_CBAR)

out = "linear_response_rot_0_vol2_size2.png"
plt.savefig(out, bbox_inches="tight", dpi=600)
plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight', dpi=600)
print(f"Saved: {out} and {out.replace('.png', '.pdf')}")
plt.show()