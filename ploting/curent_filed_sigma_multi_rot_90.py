"""
Paper_plot_multi.py
"""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.spatial import cKDTree

# ============================================================
#  USER SETTINGS — edit these
# ============================================================

STRUCTURES = [
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_bowtie_10x10_rot0",
        "mu":        3.36,
        "level":     "L2",
        "label":     "armchair bowtie 10x10",
        "E_angle_deg": 0,
        "a_cc_au":   2.6825,
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_bowtie_15x15_rot90",
        "mu":        3.36,
        "level":     "L2",
        "label":     "zigzag bowtie 15x15",
        "E_angle_deg": 0,
        "a_cc_au":   2.6825,
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_triangle_14x14_rot0",
        "mu":        3.36,
        "level":     "L2",
        "label":     "Armchair triangle 10x10",
        "E_angle_deg": 0,
        "a_cc_au":   2.6825,
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_triangle_22x22_rot90",
        "mu":        3.36,
        "level":     "L2",
        "label":     "zigzag triangle 22x22",
        "E_angle_deg": 0,
        "a_cc_au":   2.6825,
    },
]

SIGMA_COLORS = ['#08519c', '#2171b5', '#6baed6']
B_COLORS     = ['#a50f15', '#de2d26', '#fb6a4a']
SIGMA_ALPHA_FILL = 0.18
DISPLAY_SCALE = 2.0

# ============================================================
#  LAYOUT
# ============================================================
PANEL_W_AB = 5.00
PANEL_W_C  = 4.80
PANEL_H    = 3.8
LEFT_PAD   = 0.2
RIGHT_PAD  = 0.8
TOP_PAD    = 0.6
BOT_PAD    = 0.7
HSPACE     = 0.12
WSPACE     = 0.38

FONT_BASE      = 16
FONT_TITLE     = 20
FONT_SCALEBAR  = 16
FONT_EFIELD    = 13
FONT_LEGEND    = 9

AU_EV = 27.2114
AU_NM = 0.0529177

plt.rcParams.update({
    "text.usetex":     True,
    "font.family":     "serif",
    "font.size":       FONT_BASE,
    "axes.titlesize":  FONT_TITLE,
    "axes.labelsize":  FONT_BASE,
    "xtick.labelsize": FONT_BASE - 2,
    "ytick.labelsize": FONT_BASE - 2,
})

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


def load_trio(sweep_dir, level, total_area_nm2):
    pattern = re.compile(rf"^{level}_mu_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
    all_dirs = sorted(
        [d for d in os.listdir(sweep_dir) if pattern.match(d)],
        key=lambda d: float(pattern.match(d).group(1))
    )
    if not all_dirs:
        return []
    if len(all_dirs) >= 3:
        trio = [all_dirs[0], all_dirs[len(all_dirs)//2], all_dirs[-1]]
    else:
        trio = all_dirs
    result = []
    for d in trio:
        mu_val = float(pattern.match(d).group(1))
        s_path = os.path.join(sweep_dir, d, "sigma_ext.txt")
        b_path = os.path.join(sweep_dir, d, "B_ind_z_curl_time_evolution.txt")
        if not os.path.isfile(s_path):
            continue
        s = np.loadtxt(s_path)
        if os.path.isfile(b_path):
            bt      = np.loadtxt(b_path)
            N_t_b   = len(bt)
            N_pad_b = 8 * N_t_b
            b_fft   = np.fft.rfft(bt[:, 1:], n=N_pad_b, axis=0)
            dt      = bt[1, 0] - bt[0, 0]
            freq    = np.fft.rfftfreq(N_pad_b, d=dt) * AU_EV
            b_abs   = np.abs(b_fft)
            b_max   = np.max(b_abs, axis=1)
        else:
            freq = b_max = None
        result.append((mu_val, s, freq, b_max))
    return result


def build_bonds(lattice):
    tree = cKDTree(lattice)
    dists, _ = tree.query(lattice, k=2)
    positive = dists[:, 1][dists[:, 1] > 0.1]
    if positive.size == 0:
        raise ValueError("No physical nn distances found.")
    a_nn  = positive.min()
    bonds = np.array(sorted(tree.query_pairs(r=1.0005 * a_nn)), dtype=int)
    print(f"  Bonds: {len(bonds)}, a_nn = {a_nn:.4f} a.u.")
    return bonds


def _load_ts(base, sc_stem, fb_stem):
    sc = os.path.join(base, sc_stem)
    if os.path.isfile(sc):
        return np.loadtxt(sc)
    return np.loadtxt(os.path.join(base, fb_stem))


def draw_efield_arrow(ax, x, y, xpad, ypad, angle_deg,
                      color="#1a7a1a", length_frac=0.40):
    x_span = x.max() - x.min()
    y_span = y.max() - y.min()
    arrow_len = length_frac * min(x_span, y_span)
    theta = np.deg2rad(angle_deg)
    dx = arrow_len * np.cos(theta)
    dy = arrow_len * np.sin(theta)
    x_tail = (x.min() + x.max()) / 2 - dx / 2
    y_tail = y.max() + ypad * 0.55 - dy / 2
    ax.annotate(
        "", xy=(x_tail + dx, y_tail + dy),
        xytext=(x_tail, y_tail),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16),
    )
    ax.text(x_tail + dx / 2, y_tail + dy / 2 + ypad * 0.22,
            r"$\mathbf{E}$",
            ha='center', va='bottom', fontsize=FONT_EFIELD,
            color=color, fontweight='bold')


# ============================================================
#  FIGURE
# ============================================================
N_ROWS = len(STRUCTURES)
fig_w = LEFT_PAD + 2*PANEL_W_AB + PANEL_W_C + RIGHT_PAD
fig_h = TOP_PAD + N_ROWS * PANEL_H + BOT_PAD
fig = plt.figure(figsize=(fig_w, fig_h))

l = LEFT_PAD  / fig_w
r = 1 - RIGHT_PAD / fig_w
t = 1 - TOP_PAD   / fig_h
b = BOT_PAD   / fig_h

gs = gridspec.GridSpec(
    N_ROWS, 3,
    figure=fig,
    width_ratios=[PANEL_W_AB, PANEL_W_AB, PANEL_W_C],
    hspace=HSPACE, wspace=WSPACE,
    left=l, right=r, top=t, bottom=b,
)

COL_TITLES = [
    r"Bond currents $|J_{ll'}(\omega_\mathrm{res})|$",
    r"$B_{\mathrm{ind},z}(r_l,\,\omega_\mathrm{res})$",
    r"$\sigma_\mathrm{ext}$ \& magnetic response",
]

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
        print(f"  [WARNING] mu={mu} not found, skipping panels A/B.")
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            ax.text(0.5, 0.5, "no data", ha='center', va='center',
                    transform=ax.transAxes, color='grey')
        continue

    try:
        lattice_path = find_lattice(path)
    except FileNotFoundError as e:
        print(f"  [WARNING] {e}")
        continue

    lattice = np.loadtxt(lattice_path, comments="#")
    x = lattice[:, 0] * AU_NM * DISPLAY_SCALE
    y = lattice[:, 1] * AU_NM * DISPLAY_SCALE

    # ── Area normalisation: use hardcoded a_cc_au if given, else from lattice
    a_cc_au        = struct.get("a_cc_au", get_acc_au(lattice))
    N_atoms        = len(lattice)
    total_area_nm2 = graphene_hex_area_nm2(N_atoms, a_cc_au)
    print(f"  a_cc = {a_cc_au:.4f} a.u.,  A = {total_area_nm2:.2f} nm2")

    try:
        J_bond_t = _load_ts(path, "J_bond_sc_time_evolution.txt",
                                   "J_bond_time_evolution.txt")
        B_ind_t  = _load_ts(path, "B_ind_z_curl_time_evolution.txt",
                                   "B_ind_z_time_evolution.txt")
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
    if bond_idx_path:
        bonds_raw = np.loadtxt(bond_idx_path, dtype=int, comments='#')
        print(f"  Loaded {len(bonds_raw)} bonds from bond_indices.txt")
    else:
        bonds_raw = build_bonds(lattice)

    time_au = J_bond_t[:, 0]
    J_bond  = J_bond_t[:, 1:]
    B_z     = B_ind_t[:, 1:]
    dt      = time_au[1] - time_au[0]
    N_t     = len(time_au)
    N_pad   = 8 * N_t
    J_fft   = np.fft.rfft(J_bond, n=N_pad, axis=0)
    B_fft   = np.fft.rfft(B_z,    n=N_pad, axis=0)
    freq_eV = np.fft.rfftfreq(N_pad, d=dt) * AU_EV

    if skip_ab:
        bonds_raw = None
        B_fft     = None
        print("  [INFO] skip_ab=True")
    else:
        if J_fft.shape[1] != len(bonds_raw):
            print(f"  [WARNING] J_bond cols ({J_fft.shape[1]}) != bonds ({len(bonds_raw)}), skipping A.")
            bonds_raw = None
        if B_fft.shape[1] != len(lattice):
            print(f"  [WARNING] B_ind_z cols ({B_fft.shape[1]}) != sites ({len(lattice)}), skipping B.")
            B_fft = None

    omega_eV  = sigma[:, 0] * AU_EV
    sig_vals  = sigma[:, 1]
    i_res     = np.argmax(sig_vals)
    omega_res = omega_eV[i_res]
    i_freq    = np.argmin(np.abs(freq_eV - omega_res))
    print(f"  Resonance: {omega_res:.3f} eV  ->  bin {freq_eV[i_freq]:.3f} eV")

    arrow_style = struct.get("arrow", "vertical")
    xpad = (x.max() - x.min()) * 0.18
    ypad = (y.max() - y.min()) * (0.18 if arrow_style == "horizontal" else 0.22)

    # ── Panel A ──────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[row, 0])
    if row == 0:
        ax.set_title(COL_TITLES[0], pad=5)

    if bonds_raw is not None:
        mid_x = np.array([0.5*(x[i]+x[j]) for i,j in bonds_raw])
        mid_y = np.array([0.5*(y[i]+y[j]) for i,j in bonds_raw])
        dx_b  = np.array([(x[j]-x[i]) for i,j in bonds_raw])
        dy_b  = np.array([(y[j]-y[i]) for i,j in bonds_raw])
        blen  = np.sqrt(dx_b**2 + dy_b**2)
        safe  = np.where(blen > 0, blen, 1.0)
        J_sig = J_fft[i_freq, :].real
        J_abs = np.abs(J_sig)
        jmax  = J_abs.max() or 1.0
        u = np.where(blen > 0, (J_sig / jmax) * dx_b / safe, 0.0)
        v = np.where(blen > 0, (J_sig / jmax) * dy_b / safe, 0.0)
        ax.quiver(mid_x, mid_y, u, v, J_abs,
                  cmap='YlOrRd', norm=Normalize(0, jmax),
                  scale=4.5, scale_units='inches',
                  angles='xy', width=0.005)
        sm = cm.ScalarMappable(cmap='YlOrRd', norm=Normalize(0, jmax))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=r"$|J_{ll'}|$ (a.u.)",
                     fraction=0.046, pad=0.04)

    ax.scatter(x, y, s=15, c='grey', zorder=2, alpha=0.4)
    ax.set_xlim(x.min() - xpad, x.max() + xpad)
    ax.set_ylim(y.min() - ypad, y.max() + ypad)
    ax.set_aspect('equal')
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    

    draw_efield_arrow(ax, x, y, xpad, ypad, angle_deg=E_angle_deg)

    # ── Panel B ──────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[row, 1])
    if row == 0:
        ax.set_title(COL_TITLES[1], pad=5)

    if B_fft is not None:
        B_res = B_fft[i_freq, :].real
        bmax  = np.abs(B_res).max() or 1.0
        sc = ax.scatter(x, y, c=B_res, cmap='seismic', s=15,
                        norm=Normalize(-bmax, bmax))
        fig.colorbar(sc, ax=ax, label=r"$B_{\mathrm{ind},z}$ (a.u.)",
                     fraction=0.046, pad=0.04)
    else:
        ax.scatter(x, y, s=15, c='grey', alpha=0.4)

    ax.set_aspect('equal')
    ax.set_xlim(x.min() - xpad, x.max() + xpad)
    ax.set_ylim(y.min() - ypad, y.max() + ypad)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([]); ax.set_yticks([])

    # ── Panel C ──────────────────────────────────────────────────────────────
    ax  = fig.add_subplot(gs[row, 2])
    ax2 = ax.twinx()
    if row == 0:
        ax.set_title(COL_TITLES[2], pad=5)

    trio      = load_trio(sweep_dir, level, total_area_nm2)
    omega_max = omega_eV.max()

    for idx, (mu_val, s, freq_t, b_max_t) in enumerate(trio):
        sc_col = SIGMA_COLORS[idx % len(SIGMA_COLORS)]
        bc_col = B_COLORS[idx % len(B_COLORS)]
        o_eV   = s[:, 0] * AU_EV
        sv     = (s[:, 1] * AU_NM**2) / total_area_nm2
        mu_lbl = rf"\mu={mu_val:.1f}"

        ax.plot(o_eV, sv, '-', color=sc_col, lw=1.6,
                label=rf"$\sigma_{{\mathrm{{ext}}}},\;{mu_lbl}$")
        ax.fill_between(o_eV, sv, alpha=SIGMA_ALPHA_FILL, color=sc_col)

        if freq_t is not None:
            mask = freq_t <= omega_max
            ax2.plot(freq_t[mask], b_max_t[mask], '-', color=bc_col, lw=1.6,
                     label=rf"$\max_l |B_{{\mathrm{{ind}},z}}(r_l,\omega)|,\;{mu_lbl}$")

    ax.set_xlabel(r"$\hbar\omega$ (eV)")
    ax.set_ylabel(r"$\sigma_\mathrm{ext}/A$", color=SIGMA_COLORS[0])
    ax.tick_params(axis='y', labelcolor=SIGMA_COLORS[0])
    ax2.set_ylabel(r"$\max_l |B_{\mathrm{ind},z}(r_l,\omega)|$ (a.u.)", color=B_COLORS[0])
    ax2.tick_params(axis='y', labelcolor=B_COLORS[0])

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2,
              fontsize=FONT_LEGEND, loc='upper left', framealpha=0.7)

    if row < N_ROWS - 1:
        ax.set_xticklabels([])
        ax.set_xlabel("")

# ============================================================
#  SAVE
# ============================================================
out = "multi_structure_panels_rot90_vol1.png"
plt.savefig(out, bbox_inches='tight', dpi=200)
print(f"\nSaved: {out}")
plt.show()