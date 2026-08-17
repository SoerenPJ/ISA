"""
Paper_plot_multi.py
Panel A: bond currents |J_ll'(ω_res)|
Panel B: on-site B_ind,z(r_l, ω_res)
Panel C: Biot-Savart |B_z| contour at z = Z_OBS above structure (signed, seismic)
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.spatial import cKDTree

# ============================================================
#  USER SETTINGS
# ============================================================

STRUCTURES = [
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_bowtie_10x10_rot0",
        "mu":        3.36,
        "level":     "L2",
        "label":     "armchair bowtie 10x10",
        "panel_a_xfrac": 0.50, "panel_a_crop": "y", "panel_a_yside": "top" ,
        "E_angle_deg": 0,
        "a_cc_au":   2.6825,
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_bowtie_15x15_rot90",
        "mu":        3.52,
        "level":     "L2",
        "label":     "zigzag bowtie 15x15",
        "panel_a_xfrac": 0.50, "panel_a_crop": "y", "panel_a_yside": "top",
        "E_angle_deg": 0,
        "a_cc_au":   2.6825,
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_triangle_14x14_rot0",
        "mu":        3.52,
        "level":     "L2",
        "label":     "Armchair triangle 14x14",
      "panel_a_xfrac": 0.50, "panel_a_xside": "left" ,
        "E_angle_deg": 0,
        "a_cc_au":   2.6825,
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_triangle_22x22_rot90",
        "mu":        3.52,
        "level":     "L2",
        "label":     "zigzag triangle 22x22",
      "panel_a_xfrac": 0.50, "panel_a_xside": "left" ,
        "E_angle_deg": 0,
        "a_cc_au":   2.6825,
    },
]

# Biot-Savart settings
Z_OBS      = 1.0    # observation height above structure [nm]
GRID_PTS   = 150    # grid resolution
MARGIN_NM  = 0.5    # padding beyond atom bounding box [nm]

DISPLAY_SCALE = 2.0

# ============================================================
#  LAYOUT
# ============================================================
PANEL_H   = 2.20        # locked — matches linear response script
HSPACE    = 0.15        # locked
TOP_PAD   = 0.45        # enough room for colorbar titles
BOT_PAD   = 0.50        # locked

LEFT_PAD  = 0.10
RIGHT_PAD = 0.90        # room for cbar C tick labels
CBAR_W    = 0.35        # only used for panel C GridSpec column
GAP_W     = 0.20        # gap between panel groups

PANEL_W_A = 4.70        # largest — current arrows
PANEL_W_B = 4.40        # on-site B field
PANEL_W_C = 4.20        # Biot-Savart contour

FONT_BASE   = 20
FONT_TITLE  = 20
FONT_EFIELD = 18
FONT_LEGEND = 16
WSPACE      = 0.0

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
    "font.family":     "serif",
    "font.size":       FONT_BASE,
    "axes.titlesize":  FONT_TITLE,
    "axes.labelsize":  FONT_BASE,
    "xtick.labelsize": FONT_BASE,
    "ytick.labelsize": FONT_BASE,
})

# ============================================================
#  HELPERS
# ============================================================

def get_acc_au(lattice):
    tree = cKDTree(lattice)
    dists, _ = tree.query(lattice, k=2)
    return dists[:, 1][dists[:, 1] > 0.1].min()


def graphene_hex_area_nm2(N_atoms, a_cc_au):
    hex_area_au2   = (3.0 * np.sqrt(3.0) / 2.0) * a_cc_au**2
    total_area_au2 = (N_atoms / 2.0) * hex_area_au2
    return total_area_au2 * AU_NM**2


def find_lattice(start_dir):
    start_dir = os.path.abspath(start_dir)
    candidate = os.path.join(start_dir, "lattice_points.txt")
    if os.path.isfile(candidate):
        return candidate
    try:
        for name in os.listdir(start_dir):
            child = os.path.join(start_dir, name)
            if os.path.isdir(child):
                candidate = os.path.join(child, "lattice_points.txt")
                if os.path.isfile(candidate):
                    return candidate
    except PermissionError:
        pass
    d = os.path.dirname(start_dir)
    while True:
        candidate = os.path.join(d, "lattice_points.txt")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(f"lattice_points.txt not found from {start_dir}")


def load_mu_dir(sweep_dir, level, mu):
    for fmt in [f"{mu:.2f}", f"{mu:.1f}", f"{mu}", f"{mu:.4f}", str(mu)]:
        p = os.path.join(sweep_dir, f"{level}_mu_{fmt}")
        if os.path.isdir(p):
            return p
    pattern = re.compile(rf"^{level}_mu_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
    candidates = []
    for name in os.listdir(sweep_dir):
        m = pattern.match(name)
        if m:
            candidates.append((abs(float(m.group(1)) - mu),
                               os.path.join(sweep_dir, name)))
    if candidates:
        candidates.sort()
        p = candidates[0][1]
        print(f"[INFO] Fuzzy match for mu={mu}: {os.path.basename(p)}")
        return p
    return None


def build_bonds(lattice):
    tree = cKDTree(lattice)
    dists, _ = tree.query(lattice, k=2)
    a_nn  = dists[:, 1][dists[:, 1] > 0.1].min()
    bonds = np.array(sorted(tree.query_pairs(r=1.0005 * a_nn)), dtype=int)
    print(f"  Bonds: {len(bonds)}, a_nn = {a_nn:.4f} a.u.")
    return bonds


def _load_ts(base, sc_stem, fb_stem):
    sc = os.path.join(base, sc_stem)
    if os.path.isfile(sc):
        return np.loadtxt(sc)
    return np.loadtxt(os.path.join(base, fb_stem))


def draw_efield_arrow(ax, x_start, x_end, y_vis_min, y_vis_max, angle_deg,
                      color="#1a7a1a", length_frac=0.40):
    x_span    = x_end - x_start
    y_span    = y_vis_max - y_vis_min
    arrow_len = length_frac * min(x_span, y_span)
    theta = np.deg2rad(angle_deg)
    dx = arrow_len * np.cos(theta)
    dy = arrow_len * np.sin(theta)
    x_tail = (x_start + x_end) / 2 - dx / 2
    y_tail  = y_vis_max + (y_span * 0.08) * 0.55 - dy / 2
    ax.annotate(
        "", xy=(x_tail + dx, y_tail + dy),
        xytext=(x_tail, y_tail),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16),
    )
    ax.text(x_tail + dx / 2, y_tail + dy / 2 + (y_span * 0.08) * 0.22,
            r"$\mathbf{E}$",
            ha='center', va='bottom', fontsize=FONT_EFIELD,
            color=color, fontweight='bold')


# ============================================================
#  FIGURE — locked height, max 16" wide
#  GridSpec: A=0, gap=1, B=2, gap=3, C=4, cbarC=5
#  Colorbars for A and B are inset_axes (stick to data edge)
# ============================================================
N_ROWS = len(STRUCTURES)
fig_h  = TOP_PAD + N_ROWS * PANEL_H + (N_ROWS - 1) * PANEL_H * HSPACE + BOT_PAD
fig_w  = LEFT_PAD + PANEL_W_A + GAP_W + PANEL_W_B + GAP_W + PANEL_W_C + CBAR_W + RIGHT_PAD

fig = plt.figure(figsize=(fig_w, fig_h))

l = LEFT_PAD / fig_w
r = 1 - RIGHT_PAD / fig_w
t = 1 - TOP_PAD   / fig_h
b = BOT_PAD        / fig_h

gs = gridspec.GridSpec(
    N_ROWS, 6,
    figure=fig,
    width_ratios=[PANEL_W_A, GAP_W, PANEL_W_B, GAP_W, PANEL_W_C, CBAR_W],
    hspace=HSPACE, wspace=WSPACE,
    left=l, right=r, top=t, bottom=b,
)
# col indices: A=0, gap=1, B=2, gap=3, C=4, cbarC=5

Z_OBS_AU  = Z_OBS * NM_AU
MARGIN_AU = MARGIN_NM * NM_AU

# ============================================================
#  PLOT ROWS
# ============================================================
for row, struct in enumerate(STRUCTURES):
    sweep_dir   = struct["sweep_dir"]
    mu          = struct["mu"]
    level       = struct["level"]
    label       = struct["label"]
    E_angle_deg = struct.get("E_angle_deg", 0)
    skip_ab     = struct.get("skip_ab", False)
    print(f"\n{'='*50}\n{label}")

    path = load_mu_dir(sweep_dir, level, mu)
    if path is None:
        print(f"  [WARNING] mu={mu} not found, skipping row.")
        continue

    try:
        lattice_path = find_lattice(path)
    except FileNotFoundError as e:
        print(f"  [WARNING] {e}")
        continue

    lattice = np.loadtxt(lattice_path, comments="#")
    x_au = lattice[:, 0]
    y_au = lattice[:, 1]
    x    = x_au * AU_NM * DISPLAY_SCALE
    y    = y_au * AU_NM * DISPLAY_SCALE

    a_cc_au        = struct.get("a_cc_au", get_acc_au(lattice))
    N_atoms        = len(lattice)
    total_area_nm2 = graphene_hex_area_nm2(N_atoms, a_cc_au)

    try:
        J_bond_t = _load_ts(path, "J_bond_sc_time_evolution.txt",
                                   "J_bond_time_evolution.txt")
        B_ind_t  = _load_ts(path, "B_ind_z_curl_time_evolution.txt",
                                   "B_ind_z_curl_time_evolution.txt")
    except FileNotFoundError as e:
        print(f"  [WARNING] Missing time-series: {e}")
        continue

    sigma = np.loadtxt(os.path.join(path, "sigma_ext.txt"))

    _bond_candidates = [
        os.path.join(os.path.dirname(lattice_path), "bond_indices.txt"),
        os.path.join(path, "bond_indices.txt"),
        os.path.join(os.path.dirname(path), "bond_indices.txt"),
    ]
    bond_idx_path = next((p for p in _bond_candidates if os.path.isfile(p)), None)
    bonds_raw = np.loadtxt(bond_idx_path, dtype=int, comments='#') \
                if bond_idx_path else build_bonds(lattice)

    time_au = J_bond_t[:, 0]
    J_bond  = J_bond_t[:, 1:]
    B_z_t   = B_ind_t[:, 1:]
    dt      = time_au[1] - time_au[0]
    N_t     = len(time_au)
    N_pad   = 8 * N_t
    J_fft   = np.fft.rfft(J_bond, n=N_pad, axis=0)
    B_fft   = np.fft.rfft(B_z_t,  n=N_pad, axis=0)
    freq_eV = np.fft.rfftfreq(N_pad, d=dt) * AU_EV

    omega_eV  = sigma[:, 0] * AU_EV
    i_res     = np.argmax(sigma[:, 1])
    omega_res = omega_eV[i_res]
    i_freq    = np.argmin(np.abs(freq_eV - omega_res))

    if skip_ab or J_fft.shape[1] != len(bonds_raw):
        bonds_raw = None
    if skip_ab or B_fft.shape[1] != len(lattice):
        B_fft = None

    xpad = (x.max() - x.min()) * 0.04
    ypad = (y.max() - y.min()) * 0.08

    # ── Panel A: bond currents (col 0) ───────────────────────────────────
    ax_a = fig.add_subplot(gs[row, 0])

    if bonds_raw is not None:
        mid_x = np.array([0.5*(x[i]+x[j]) for i,j in bonds_raw])
        mid_y = np.array([0.5*(y[i]+y[j]) for i,j in bonds_raw])
        dx_b  = np.array([x[j]-x[i] for i,j in bonds_raw])
        dy_b  = np.array([y[j]-y[i] for i,j in bonds_raw])
        blen  = np.sqrt(dx_b**2 + dy_b**2)
        safe  = np.where(blen > 0, blen, 1.0)
        J_sig = J_fft[i_freq, :].real
        J_abs = np.abs(J_sig)
        jmax  = J_abs.max() or 1.0
        u = np.where(blen > 0, (J_sig / jmax) * dx_b / safe, 0.0)
        v = np.where(blen > 0, (J_sig / jmax) * dy_b / safe, 0.0)
        ax_a.quiver(mid_x, mid_y, u, v, J_abs,
                    cmap='viridis', norm=Normalize(0, jmax),
                    scale=4.0, scale_units='inches',
                    angles='xy', width=0.01, rasterized=True)
        sm = cm.ScalarMappable(cmap='viridis', norm=Normalize(0, jmax))
        sm.set_array([])
        cbar_ax_a = ax_a.inset_axes([1.02, 0.0, 0.10, 1.0])
        cb = fig.colorbar(sm, cax=cbar_ax_a)
        ticks_raw = np.linspace(0, jmax, 5)
        cb.set_ticks(ticks_raw)
        cb.set_ticklabels([f"{v*1e3:.3g}" for v in ticks_raw])
        cb.ax.tick_params(labelsize=FONT_BASE - 2)
        if row == 0:\
            cbar_ax_a.set_title(r"$|J_{ll'}|$" +r"$(\times 10^{-3}\; \mathrm{a.u.})$",
                                fontsize=FONT_LEGEND, pad=12)
    xfrac     = struct.get("panel_a_xfrac", 1.0)
    crop_axis = struct.get("panel_a_crop", "x")
    xside     = struct.get("panel_a_xside", "left")
    yside     = struct.get("panel_a_yside", "top")

    if crop_axis == "y":
        if yside == "top":
            y_start = y.max() - xfrac * (y.max() - y.min())
            y_end   = y.max()
        else:
            y_start = y.min()
            y_end   = y.min() + xfrac * (y.max() - y.min())
        visible   = (y >= y_start) & (y <= y_end)
        x_vis_min = x[visible].min() if visible.any() else x.min()
        x_vis_max = x[visible].max() if visible.any() else x.max()
        x_c  = (x_vis_min + x_vis_max) / 2
        y_c  = (y_start   + y_end)     / 2
        x_half = (x_vis_max - x_vis_min) / 2 + xpad
        y_half = (y_end - y_start)       / 2 + ypad
        # use x_start/x_end for the E arrow
        x_start_arrow, x_end_arrow = x_vis_min, x_vis_max
        y_start_arrow, y_end_arrow = y_start,   y_end
    else:
        if xside == "right":
            x_start = x.min() + (1.0 - xfrac) * (x.max() - x.min())
            x_end   = x.max()
        else:
            x_start = x.min()
            x_end   = x.min() + xfrac * (x.max() - x.min())
        visible   = (x >= x_start) & (x <= x_end)
        y_vis_min = y[visible].min() if visible.any() else y.min()
        y_vis_max = y[visible].max() if visible.any() else y.max()
        x_c  = (x_start + x_end) / 2
        y_c  = (y_vis_min + y_vis_max) / 2
        x_half = (x_end - x_start)       / 2 + xpad
        y_half = (y_vis_max - y_vis_min)  / 2 + ypad
        x_start_arrow, x_end_arrow = x_start, x_end
        y_start_arrow, y_end_arrow = y_vis_min, y_vis_max

    panel_aspect = PANEL_W_A / PANEL_H
    y_half = max(y_half, x_half / panel_aspect)

    ax_a.set_xlim(x_c - x_half, x_c + x_half)
    ax_a.set_ylim(y_c - y_half, y_c + y_half)
    ax_a.set_aspect('equal')
    ax_a.axis('off')
    draw_efield_arrow(ax_a, x_start_arrow, x_end_arrow,
                      y_start_arrow, y_end_arrow, angle_deg=E_angle_deg)
      # ── Panel B: on-site B_ind,z (col 2) ─────────────────────────────────
    
    ax_b = fig.add_subplot(gs[row, 2])

    if B_fft is not None:
        B_res = B_fft[i_freq, :].real
        bmax  = np.abs(B_res).max() or 1.0
        vmax  = np.percentile(np.abs(B_res), 95)
        sc = ax_b.scatter(x, y, c=B_res, cmap='coolwarm', s=12,
                          norm=Normalize(-vmax, vmax),  rasterized=True)
        cbar_ax_b = ax_b.inset_axes([0.98, 0.0, 0.06, 1.0])
        cb = fig.colorbar(sc, cax=cbar_ax_b)
        ticks_b = np.linspace(-vmax, vmax, 5)
        cb.set_ticks(ticks_b)
        cb.set_ticklabels([f"{v*1e8:.3g}" for v in ticks_b])
        cb.ax.tick_params(labelsize=FONT_BASE - 2)
        if row == 0:
            cbar_ax_b.set_title(r"$B^{\mathrm{ind}}_z \; (\times 10^{-8} \; \mathrm{a.u.})$",
                                fontsize=FONT_LEGEND, pad=12)

    ax_b.set_aspect('equal', adjustable='datalim')
    ax_b.set_xlim(x.min() - xpad, x.max() + xpad)
    ax_b.set_ylim(y.min() - ypad, y.max() + ypad)
    ax_b.axis('off')
    # replace the current crop indicator block with:
    line_kwargs = dict(color='black', linewidth=2.2, linestyle='--', alpha=0.6)
    if crop_axis == "y":
        # horizontal line — span only the x-width of the visible atoms at that y
        vis_at_cut = (y >= y_start - ypad*0.5) & (y <= y_start + ypad*0.5)
        x_lo = x.min() - xpad
        x_hi = x.max() + xpad
        ax_b.plot([x_lo, x_hi], [y_start, y_start], **line_kwargs)
    else:
        # vertical line — span only the y-height of the structure
        y_lo = y.min() - ypad
        y_hi = y.max() + ypad
        cut_x = x_end if xside == "left" else x_start
        ax_b.plot([cut_x, cut_x], [y_lo, y_hi], **line_kwargs)

    # ── Panel C: Biot-Savart |B_z| (col 4, cbar col 5) ──────────────────
    ax_c    = fig.add_subplot(gs[row, 4])
    ax_cb_c = fig.add_subplot(gs[row, 5])

    Z_OBS_AU  = 1.0 * NM_AU
    MARGIN_AU = 2 * NM_AU
    bonds_c   = bonds_raw if bonds_raw is not None else build_bonds(lattice)
    J_res     = J_fft[i_freq, :]

    if J_res.shape[0] == len(bonds_c):
        xlo = x_au.min() - MARGIN_AU;  xhi = x_au.max() + MARGIN_AU
        ylo = y_au.min() - MARGIN_AU;  yhi = y_au.max() + MARGIN_AU
        GX, GY = np.meshgrid(np.linspace(xlo, xhi, GRID_PTS),
                              np.linspace(ylo, yhi, GRID_PTS))
        bi = bonds_c[:, 0]; bj = bonds_c[:, 1]
        xc = 0.5*(x_au[bi]+x_au[bj]); yc = 0.5*(y_au[bi]+y_au[bj])
        lx = x_au[bi]-x_au[bj];       ly = y_au[bi]-y_au[bj]
        gx_f = GX.ravel(); gy_f = GY.ravel()
        dx   = gx_f[:,None] - xc[None,:]
        dy   = gy_f[:,None] - yc[None,:]
        r    = np.sqrt(dx**2 + dy**2 + Z_OBS_AU**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_r = np.where(r < 1e-12, 0.0, 1.0/r)
        weighted = inv_r * J_res[None,:]
        A_x = (MU0_4PI*(weighted@lx)).reshape(GX.shape)
        A_y = (MU0_4PI*(weighted@ly)).reshape(GX.shape)
        dx_g = GX[0,1]-GX[0,0]; dy_g = GY[1,0]-GY[0,0]
        Bz     = np.gradient(A_y,dx_g,axis=1) - np.gradient(A_x,dy_g,axis=0)
        Bz_abs = np.abs(Bz)
        GX_nm  = GX*AU_NM; GY_nm = GY*AU_NM

        cf = ax_c.contourf(GX_nm, GY_nm, Bz_abs, levels=50, cmap='inferno',  rasterized=True)
        cb = fig.colorbar(cf, cax=ax_cb_c)
        ticks_c = np.linspace(Bz_abs.min(), Bz_abs.max(), 5)
        cb.set_ticks(ticks_c)
        cb.set_ticklabels([f"{v*1e8:.3g}" for v in ticks_c])
        cb.ax.tick_params(labelsize=FONT_BASE - 2)
        if row == 0:
            ax_cb_c.set_title(r"$|B^{\mathrm{ind}}_z|$" +  r"$(\times 10^{-8} \; \mathrm{a.u.})$",
                              fontsize=FONT_LEGEND, pad=12)
        ax_c.scatter(x_au*AU_NM, y_au*AU_NM, s=10, c='white',
                     alpha=0.35, zorder=3, linewidths=0)
        ax_c.set_xlim(GX_nm.min(), GX_nm.max())
        ax_c.set_ylim(GY_nm.min(), GY_nm.max())
    else:
        ax_c.text(0.5, 0.5, "bond mismatch", ha='center', va='center',
                  transform=ax_c.transAxes, color='grey')
        ax_cb_c.set_visible(False)

    ax_c.set_aspect('equal')
    ax_c.set_xticks([]); ax_c.set_yticks([])
    for sp in ax_c.spines.values(): sp.set_visible(False)

# ============================================================
#  SAVE
# ============================================================
out = "multi_structure_panels_biot_savart_rot_90_vol1.png"
plt.savefig(out.replace('.png', '.pdf'), 
            bbox_inches='tight', dpi=600,
            backend='pdf')
print(f"\nSaved: {out}")
plt.show()