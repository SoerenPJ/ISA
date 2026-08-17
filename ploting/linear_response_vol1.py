"""
plot_sweep_sigma_ext.py
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

INTERPOLATION = "None"

L0_MASK_FRAC = 0.10

L0_MASK_FRAC_PER_STRUCTURE = {
    "sweep_data_mu_armchair_bowtie_10x10_rot0": 0.30,
    "sweep_data_mu_zigzag_bowtie_15x15_rot0":  0.30,
    "sweep_data_mu_armchair_triangle_20x20_rot0": 0.10,
    "sweep_data_mu_zigzag_triangle_30x30_rot0":   0.10,
}

# ── μ cases ──────────────────────────────────────────────────────────────────
# Three physically motivated μ values (in eV) per structure:
#   [below gap, near plasmon onset, well-established plasmon]
# Shown as horizontal dashed lines in the eigenvalue spectrum,
# vertical dashed lines in the σ_ext heatmap, and as the three
# line colours in the L1-vs-L0 / L2-vs-L0 comparison panels.
MU_CASES = "auto"   # global fallback

MU_CASES_PER_STRUCTURE = {
    # Edit these to your physically motivated values:
    "sweep_data_mu_armchair_bowtie_10x10_rot0": [0.5, 2.0, 3.5],
    "sweep_data_mu_zigzag_bowtie_15x15_rot0":   [0.5, 2.0, 3.5],
}

MU_COLORS = ['#2166ac', '#f4a582', '#d6604d']
MU_LABELS = None

# Floor for log-scale comparison panels — values below this are not shown
Y_FLOOR = 1e-5

# Gap detection settings.
EDGE_TOL_EV = 0.03   # eV

CIRCLE_SIZE = 12
CIRCLE_LW   = 0.5

# ============================================================
#  TYPOGRAPHY
# ============================================================
FONT_SIZE_GLOBAL    = 20
FONT_SIZE_COL_TITLE = 16
FONT_SIZE_AXIS_LABEL= 16
FONT_SIZE_TICK      = 16
FONT_SIZE_CBAR      = 16
FONT_SIZE_CBAR_LABEL= 16
FONT_SIZE_LEGEND    = 8
FONT_FAMILY         = "times new roman"
USE_LATEX           = False

# ============================================================
#  LAYOUT
# ============================================================
EIG_W     = 2.5
PANEL_W   = 3.6
PANEL_H   = 3.2
CBAR_W    = 0.35
GAP_W     = 1.3
LEFT_PAD  = 0.76
RIGHT_PAD = 0.1
TOP_PAD   = 0.5
BOT_PAD   = 0.6
HSPACE    = 0.10
WSPACE    = 0.05

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
    """Return the column index in the grid closest to mu_target."""
    return int(np.argmin(np.abs(mu_array - mu_target)))


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
                       linewidths=CIRCLE_LW, zorder=3, alpha=0.9, label=label)
        ax.axhline(mu_val, color=color, lw=1.0, ls='--', alpha=0.8, zorder=4)

    if gap > 0.05:
        gap_mid = (gap_bot + gap_top) / 2
        ax.text(N * 0.02, gap_mid,
                rf'$E_g = {gap:.2f}$ eV',
                ha='left', va='center', fontsize=11, color='black',
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
              framealpha=0.8, markerscale=1.2,
              handletextpad=0.3, borderpad=0.4, labelspacing=0.3)


def draw_mu_vlines(ax, mu_cases_eV):
    for mu_val, color in zip(mu_cases_eV, MU_COLORS):
        ax.axvline(mu_val, color=color, lw=1.2, ls='--', alpha=0.85, zorder=5)


def _darken(hex_color, factor=0.45):
    """Return a darkened version of a hex colour (factor < 1 = darker)."""
    import matplotlib.colors as mcolors
    r, g, b, _ = mcolors.to_rgba(hex_color)
    return (r * factor, g * factor, b * factor, 1.0)


def _brighten(hex_color, alpha=0.45):
    """Return the colour with reduced alpha for a washed-out look."""
    import matplotlib.colors as mcolors
    r, g, b, _ = mcolors.to_rgba(hex_color)
    return (r, g, b, alpha)


def draw_comparison_lines(ax, omega, mu_array, g_base, g_comp,
                          mu_cases_eV, mu_labels,
                          comp_label, is_last_row, is_leftmost, show_title=True):
    """
    L0 = solid line (ground truth).
    L1/L2 = dashed, slightly transparent (approximation).
    Color encodes μ; linestyle encodes level.
    """

    for mu_val, color, label in zip(mu_cases_eV, MU_COLORS, mu_labels):
        idx = nearest_mu_index(mu_array, mu_val)

        # ── L0: solid line ───────────────────────────────────────────────
        y0 = g_base[:, idx]
        ax.plot(omega, y0, color=color, lw=1.8, ls='-', zorder=4)

        # ── L1 or L2: dashed, slightly transparent ───────────────────────
        if g_comp is not None:
            bright_col = _brighten(color, alpha=0.60)
            ax.plot(omega, g_comp[:, idx],
                    color=bright_col, lw=1.4, ls='--', zorder=3)

    ax.set_xlim(omega[0], omega[-1])
    ax.set_yscale('log')
    ax.set_ylim(bottom=Y_FLOOR)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, lw=0.3, alpha=0.4, which='both')

    if is_last_row:
        ax.set_xlabel("Photon energy (eV)", fontsize=FONT_SIZE_AXIS_LABEL)
    else:
        ax.set_xticklabels([])

    if is_leftmost:
        ax.set_ylabel(r"$\log(\sigma^\mathrm{ext}/A)$", fontsize=FONT_SIZE_AXIS_LABEL)
    else:
        ax.set_yticklabels([])

    # Legend: linestyle only — μ colours readable from eigenvalue panel
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='k', lw=1.8, ls='-',  label='L0'),
        Line2D([0], [0], color='k', lw=1.4, ls='--', label=comp_label),
    ]
    ax.legend(handles=handles, fontsize=FONT_SIZE_LEGEND, loc='lower right',
              framealpha=0.8, handletextpad=0.3,
              borderpad=0.4, labelspacing=0.3)

    if show_title:
        ax.set_title(f'L0 vs {comp_label}', fontsize=FONT_SIZE_COL_TITLE)


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

for structure in STRUCTURES:
    struct_dir = structure if os.path.isabs(structure) \
                 else os.path.join(BASE_DIR, structure)
    total_area_nm2 = load_total_area_nm2(struct_dir)
    all_eig.append(load_eigenvalues_eV(struct_dir))

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
#  FIGURE LAYOUT
#
#  Columns per row:
#    0   eigenvalue spectrum
#    1   (gap)
#    2   L2 σ_ext heatmap
#    3   abs colorbar
#    4   (gap)
#    5   line comparison: L0 vs L1
#    6   line comparison: L0 vs L2
#    7   hidden (keeps figure width unchanged)
# ============================================================
N_ROWS = len(STRUCTURES)
EIG_GAP_W = 0.3
wr = [EIG_W, EIG_GAP_W, PANEL_W, CBAR_W, GAP_W, PANEL_W, PANEL_W, CBAR_W]
fig_w = LEFT_PAD + sum(wr) + RIGHT_PAD
fig_h = TOP_PAD + N_ROWS * PANEL_H + BOT_PAD
fig = plt.figure(figsize=(fig_w, fig_h))

l = LEFT_PAD   / fig_w
r = 1 - RIGHT_PAD / fig_w
t = 1 - TOP_PAD   / fig_h
b = BOT_PAD    / fig_h

gs = gridspec.GridSpec(
    N_ROWS, 8, figure=fig, width_ratios=wr,
    hspace=HSPACE, wspace=WSPACE,
    left=l, right=r, top=t, bottom=b,
)

# ============================================================
#  PLOT
# ============================================================
for row, (structure, grids, extent, eig_data, mu_sweep, mu_array, omega) in enumerate(
        zip(STRUCTURES, all_grids, all_extents, all_eig,
            all_mu_sweeps, all_mu_arrays, all_omega)):

    if extent is None:
        continue

    energies_eV, gap_bot, gap_top, gap = eig_data
    g0 = grids.get("L0")
    g1 = grids.get("L1")
    g2 = grids.get("L2")
    is_last_row = (row == N_ROWS - 1)

    struct_key = os.path.basename(structure.rstrip("/"))
    mu_cases   = resolve_mu_cases(struct_key, mu_sweep)

    labels = MU_LABELS if MU_LABELS is not None else \
             [rf'$\mu={m:.2f}$ eV' for m in mu_cases]

    abs_im_ref = None
    abs_max = g2.max() if g2 is not None else (g0.max() if g0 is not None else 1.0)

    # ── col 0: eigenvalue spectrum ───────────────────────────────────────
    ax_eig = fig.add_subplot(gs[row, 0])
    if energies_eV is not None:
        draw_eigenvalue_spectrum(ax_eig, energies_eV, gap_bot, gap_top, gap,
                                 mu_cases, labels, is_last_row)
    else:
        ax_eig.text(0.5, 0.5, "no eigenvalues", ha='center', va='center',
                    transform=ax_eig.transAxes, color='grey')

    # ── col 2: L2 σ_ext heatmap ──────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[row, 2])
    if g2 is not None:
        im = ax_heat.imshow(g2, extent=extent, origin="lower",
                            aspect="auto", cmap=CMAP_ABS,
                            vmin=0, vmax=abs_max,
                            interpolation=INTERPOLATION)
        abs_im_ref = im
        draw_mu_vlines(ax_heat, mu_cases)
    else:
        ax_heat.text(0.5, 0.5, "no data", ha="center", va="center",
                     transform=ax_heat.transAxes, color="grey")
    ax_heat.tick_params(axis="y", which="both", labelleft=True, left=True)
    if is_last_row:
        ax_heat.set_xlabel(rf"$\mu$ ({MU_UNIT})", fontsize=FONT_SIZE_AXIS_LABEL)
    else:
        ax_heat.set_xticklabels([])
    ax_heat.xaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))
    ax_heat.yaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))

    # ── col 3: abs colorbar ──────────────────────────────────────────────
    ax_cbar_a = fig.add_subplot(gs[row, 3])
    if abs_im_ref is not None:
        cb_a = fig.colorbar(abs_im_ref, cax=ax_cbar_a)
        if row == 0:
            ax_cbar_a.set_title(r"$\sigma^\mathrm{ext}/A$",
                                fontsize=FONT_SIZE_CBAR_LABEL, pad=CBAR_TITLE_PAD)
            ax_cbar_a.title.set_position((CBAR_TITLE_X, CBAR_TITLE_Y))
        cb_a.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        cb_a.ax.tick_params(labelsize=FONT_SIZE_CBAR)

    # ── col 5 & 6: comparison panels — share y-limits across both ───────
    # Pre-compute the global y range for this row across all mu values and both grids
    y_min_row, y_max_row = np.inf, -np.inf
    for g in [g0, g1, g2]:
        if g is not None:
            for mu_val in mu_cases:
                idx = nearest_mu_index(mu_array, mu_val)
                col = g[:, idx]
                finite = col[col > Y_FLOOR]
                if finite.size:
                    y_min_row = min(y_min_row, finite.min())
                    y_max_row = max(y_max_row, finite.max())
    if not np.isfinite(y_min_row):
        y_min_row, y_max_row = Y_FLOOR, 1.0
    # add a little headroom
    y_lim = (Y_FLOOR, y_max_row * 3.0)

    ax_c1 = fig.add_subplot(gs[row, 5])
    if g0 is not None and mu_array is not None and omega is not None:
        draw_comparison_lines(ax_c1, omega, mu_array, g0, g1,
                              mu_cases, labels,
                              comp_label='L1',
                              is_last_row=is_last_row,
                              is_leftmost=True,
                              show_title=(row == 0))
        ax_c1.set_ylim(y_lim)
    else:
        ax_c1.text(0.5, 0.5, "no data", ha="center", va="center",
                   transform=ax_c1.transAxes, color="grey")

    # ── col 6: L0 vs L2 line comparison ─────────────────────────────────
    ax_c2 = fig.add_subplot(gs[row, 6])
    if g0 is not None and mu_array is not None and omega is not None:
        draw_comparison_lines(ax_c2, omega, mu_array, g0, g2,
                              mu_cases, labels,
                              comp_label='L2',
                              is_last_row=is_last_row,
                              is_leftmost=False,
                              show_title=(row == 0))
        ax_c2.set_ylim(y_lim)
    else:
        ax_c2.text(0.5, 0.5, "no data", ha="center", va="center",
                   transform=ax_c2.transAxes, color="grey")

    # col 7 — hidden (keeps figure width unchanged)
    fig.add_subplot(gs[row, 7]).set_visible(False)

plt.savefig("linear_response_vol1.png", bbox_inches="tight", dpi=600)
print("Saved: linear_response_vol1.png")
plt.show()

# ============================================================
#  RATIO FIGURE  —  σ_L1/σ_L0  and  σ_L2/σ_L0  at each μ
#
#  Layout: N_ROWS rows × 2 columns
#    col 0 = L1/L0 ratio
#    col 1 = L2/L0 ratio
#  A horizontal dashed line at y=1 marks perfect agreement.
#  Color encodes μ (same MU_COLORS as everywhere else).
# ============================================================
RATIO_FLOOR = 1e-2   # smallest ratio shown (clips unphysical blow-ups near zero)
RATIO_CAP   = 1e+2   # largest ratio shown

fig2_w = 2 * PANEL_W + LEFT_PAD + RIGHT_PAD + GAP_W
fig2_h = TOP_PAD + N_ROWS * PANEL_H + BOT_PAD
fig2 = plt.figure(figsize=(fig2_w, fig2_h))

gs2 = gridspec.GridSpec(
    N_ROWS, 2, figure=fig2,
    hspace=HSPACE * 2, wspace=0.30,
    left=LEFT_PAD / fig2_w,
    right=1 - RIGHT_PAD / fig2_w,
    top=1 - TOP_PAD / fig2_h,
    bottom=BOT_PAD / fig2_h,
)

for row, (structure, grids, eig_data, mu_sweep, mu_array, omega) in enumerate(
        zip(STRUCTURES, all_grids, all_eig,
            all_mu_sweeps, all_mu_arrays, all_omega)):

    if mu_array is None or omega is None:
        continue

    g0 = grids.get("L0")
    g1 = grids.get("L1")
    g2 = grids.get("L2")
    is_last_row = (row == N_ROWS - 1)

    struct_key = os.path.basename(structure.rstrip("/"))
    mu_cases   = resolve_mu_cases(struct_key, mu_sweep)
    labels     = MU_LABELS if MU_LABELS is not None else \
                 [rf'$\mu={m:.2f}$ eV' for m in mu_cases]

    for col_idx, (g_comp, comp_label) in enumerate([(g1, 'L1'), (g2, 'L2')]):
        ax = fig2.add_subplot(gs2[row, col_idx])

        if g0 is not None and g_comp is not None:
            for mu_val, color, lbl in zip(mu_cases, MU_COLORS, labels):
                idx  = nearest_mu_index(mu_array, mu_val)
                y0   = g0[:, idx]
                yc   = g_comp[:, idx]
                # avoid division by zero
                safe = np.where(np.abs(y0) > 0, y0, np.nan)
                ratio = yc / safe
                ratio = np.clip(ratio, RATIO_FLOOR, RATIO_CAP)
                ax.plot(omega, ratio, color=color,
                        lw=1.4, label=lbl)

            ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.5, zorder=5)
            ax.set_yscale('log')
            ax.set_ylim(RATIO_FLOOR, RATIO_CAP)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")

        ax.set_xlim(omega[0], omega[-1])
        ax.grid(True, lw=0.3, alpha=0.4, which='both')
        ax.tick_params(labelsize=FONT_SIZE_TICK)

        if is_last_row:
            ax.set_xlabel("Photon energy (eV)", fontsize=FONT_SIZE_AXIS_LABEL)
        else:
            ax.set_xticklabels([])

        if col_idx == 0:
            ax.set_ylabel(rf"$\sigma_{{\mathrm{{{comp_label}}}}}/\sigma_{{\mathrm{{L0}}}}$",
                          fontsize=FONT_SIZE_AXIS_LABEL)
        else:
            ax.set_ylabel(rf"$\sigma_{{\mathrm{{{comp_label}}}}}/\sigma_{{\mathrm{{L0}}}}$",
                          fontsize=FONT_SIZE_AXIS_LABEL)

        if row == 0:
            ax.set_title(f'L0 vs {comp_label} ratio', fontsize=FONT_SIZE_COL_TITLE)

        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right',
                  framealpha=0.8, handletextpad=0.3,
                  borderpad=0.4, labelspacing=0.3)

plt.savefig("linear_response_ratio.png", bbox_inches="tight", dpi=600)
print("Saved: linear_response_ratio.png")
plt.show()