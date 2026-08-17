"""
biot_savart_comparison.py
=========================
Side-by-side comparison of:
  Left  — TDDFT B_ind,z at each atom site (signed real part, from time-domain FFT)
  Right — Biot-Savart B_ind,z at each atom site (signed real part, z ~ 0)

Both panels share the same seismic colormap with symmetric vmin/vmax per row.
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
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_bowtie_10x10_rot0",
        "mu":    3.36,
        "level": "L2",
        "label": r"Armchair bowtie $10{\times}10$",
        "a_cc_au": 2.6825,
    },
    {
        "sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_bowtie_15x15_rot0",
        "mu":    3.36,
        "level": "L2",
        "label": r"Zigzag bowtie $15{\times}15$",
        "a_cc_au": 2.6825,
    },
]

# z height for Biot-Savart evaluation in Bohr.
# Essentially zero — just large enough to avoid wire singularities.
Z_OBS_AU = 1e-6

SCATTER_S = 18     # dot size
CMAP      = "seismic"

# ============================================================
#  CONSTANTS
# ============================================================
AU_EV   = 27.2114
AU_NM   = 0.0529177
ALPHA   = 1.0 / 137.036
MU0_4PI = ALPHA**2    # Biot-Savart prefactor mu_0/4pi in atomic units

# ============================================================
#  LAYOUT
# ============================================================
PANEL_W  = 4.2
PANEL_H  = 4.2
CBAR_PAD = 0.04
CBAR_FRAC= 0.046
LEFT_PAD = 0.3
RIGHT_PAD= 0.3
TOP_PAD  = 0.6
BOT_PAD  = 0.7
HSPACE   = 0.25
WSPACE   = 0.35

FONT_BASE  = 16
FONT_TITLE = 14

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
    a_nn  = dists[:, 1][dists[:, 1] > 0.1].min()
    bonds = np.array(sorted(tree.query_pairs(r=1.0005 * a_nn)), dtype=int)
    print(f"  Built {len(bonds)} bonds, a_nn = {a_nn:.4f} a.u.")
    return bonds, a_nn


def compute_Bz_curl(x_au, y_au, bonds, a_nn, J_res):
    """
    Compute B_z = (∇ × A_ind)_z at each atom site using Eq. (14) from the paper,
    then the discrete curl of Eq. (27) from the post-processing script.

    A_ind(r) = α² Σ_{bonds} J_b * (r - r̄_b) / |r - r̄_b|
    B_z(i)   = sum_curl(i) / (n_neighbors(i) * a_bond²)

    This exactly mirrors what B_ind_z_curl_time_evolution.txt contains.
    """
    N = len(x_au)
    N_bonds = len(bonds)

    # Bond midpoints and bond vectors ℓ = r_i - r_j
    bi = bonds[:, 0]
    bj = bonds[:, 1]
    xc = 0.5 * (x_au[bi] + x_au[bj])
    yc = 0.5 * (y_au[bi] + y_au[bj])
    lx = x_au[bi] - x_au[bj]
    ly = y_au[bi] - y_au[bj]

    # A_ind at each site: (N_sites, N_bonds) inverse distance matrix
    dx_mat = x_au[:, None] - xc[None, :]   # (N, N_bonds)
    dy_mat = y_au[:, None] - yc[None, :]
    r_mat  = np.sqrt(dx_mat**2 + dy_mat**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_r = np.where(r_mat < 1e-10, 0.0, 1.0 / r_mat)

    # A_x[m] = α² Σ_b J_b * lx[b] / |r_m - r̄_b|
    # using complex J_res to keep phase information
    A_x = MU0_4PI * (inv_r * lx[None, :]) @ J_res   # (N,) complex
    A_y = MU0_4PI * (inv_r * ly[None, :]) @ J_res

    # Discrete curl (Eq. 27): accumulate over bonds where bond_i is the lower index
    dx_curl = x_au[bj] - x_au[bi]
    dy_curl = y_au[bj] - y_au[bi]

    sum_curl = np.zeros(N, dtype=complex)
    for k in range(N_bonds):
        i, j = bi[k], bj[k]
        circ = (A_y[j] - A_y[i]) * dx_curl[k] - (A_x[j] - A_x[i]) * dy_curl[k]
        sum_curl[i] += circ

    # Normalise by n_neighbors * a_bond²
    n_neighbors = np.bincount(bi, minlength=N).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        norm = np.where(n_neighbors > 0, 1.0 / (n_neighbors * a_nn**2), 0.0)

    return (sum_curl * norm).real


# ============================================================
#  FIGURE
# ============================================================
N_ROWS = len(STRUCTURES)
fig_w  = LEFT_PAD + 2 * PANEL_W + RIGHT_PAD
fig_h  = TOP_PAD  + N_ROWS * PANEL_H + BOT_PAD

fig = plt.figure(figsize=(fig_w, fig_h))
gs  = gridspec.GridSpec(
    N_ROWS, 2, figure=fig,
    hspace=HSPACE, wspace=WSPACE,
    left=LEFT_PAD / fig_w,
    right=1 - RIGHT_PAD / fig_w,
    top=1 - TOP_PAD / fig_h,
    bottom=BOT_PAD / fig_h,
)

COL_TITLES = [
    r"TDDFT $B_{\mathrm{ind},z}(r_l,\,\omega_\mathrm{res})$",
    r"Biot-Savart $B_{\mathrm{ind},z}(r_l,\,\omega_\mathrm{res})$",
]

# ============================================================
#  PLOT ROWS
# ============================================================
for row, struct in enumerate(STRUCTURES):
    sweep_dir = struct["sweep_dir"]
    mu        = struct["mu"]
    level     = struct["level"]
    label     = struct["label"]

    print(f"\n{'='*50}\n{label}")

    path = find_mu_dir(sweep_dir, level, mu)
    if path is None:
        for col in range(2):
            fig.add_subplot(gs[row, col]).text(
                0.5, 0.5, "no data", ha='center', va='center', color='grey')
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
        bonds_raw = np.loadtxt(bond_idx_path, dtype=int, comments='#')
        tree = cKDTree(lattice)
        dists, _ = tree.query(lattice, k=2)
        a_nn = dists[:, 1][dists[:, 1] > 0.1].min()
        print(f"  Loaded {len(bonds_raw)} bonds from bond_indices.txt, a_nn = {a_nn:.4f} a.u.")
    else:
        bonds_raw, a_nn = build_bonds(lattice)

    # ── FFT of time-domain data ───────────────────────────────────────────
    J_bond_t    = load_ts(path, "J_bond_sc_time_evolution.txt",
                                 "J_bond_time_evolution.txt")

    # Try all known B_ind_z filename variants
    b_stems = [
        "B_ind_z_curl_time_evolution.txt",
        "B_ind_z_time_evolution.txt",
        "B_ind_z_sc_time_evolution.txt",
    ]
    B_ind_t = None
    for stem in b_stems:
        candidate = os.path.join(path, stem)
        if os.path.isfile(candidate):
            B_ind_t = np.loadtxt(candidate)
            print(f"  Loaded B_ind_z from: {stem}")
            break
    if B_ind_t is None:
        print(f"  [WARNING] No B_ind_z file found in {path}, skipping TDDFT panel.")

    time_au_arr = J_bond_t[:, 0]
    J_bond      = J_bond_t[:, 1:]
    dt_J        = time_au_arr[1] - time_au_arr[0]
    N_t_J       = len(time_au_arr)
    N_pad_J     = 8 * N_t_J

    J_fft    = np.fft.rfft(J_bond, n=N_pad_J, axis=0)   # no /N_t — matches Paper_plot_multi.py
    freq_J   = np.fft.rfftfreq(N_pad_J, d=dt_J) * AU_EV

    B_fft   = None
    freq_B  = None
    if B_ind_t is not None:
        B_z_td  = B_ind_t[:, 1:]
        time_B  = B_ind_t[:, 0]
        dt_B    = time_B[1] - time_B[0]
        N_t_B   = len(time_B)
        N_pad_B = 8 * N_t_B
        print(f"  B_ind_t shape: {B_ind_t.shape}, B_z_td cols: {B_z_td.shape[1]}, atoms: {len(x_au)}")
        if B_z_td.shape[1] == len(x_au):
            B_fft  = np.fft.rfft(B_z_td, n=N_pad_B, axis=0)   # no /N_t — same convention
            freq_B = np.fft.rfftfreq(N_pad_B, d=dt_B) * AU_EV
        else:
            print(f"  [WARNING] B_z_td cols ({B_z_td.shape[1]}) != atoms ({len(x_au)}), skipping TDDFT panel.")

    # ── Resonance: find best bin in each FFT grid independently ───────────
    sigma     = np.loadtxt(os.path.join(path, "sigma_ext.txt"))
    omega_eV  = sigma[:, 0] * AU_EV
    i_res     = np.argmax(sigma[:, 1])
    omega_res = omega_eV[i_res]

    i_freq_J = np.argmin(np.abs(freq_J - omega_res))
    print(f"  Resonance: {omega_res:.3f} eV  (J bin: {freq_J[i_freq_J]:.3f} eV)")

    # ── TDDFT B_ind,z: signed real part at resonance ──────────────────────
    B_tddft = None
    if B_fft is not None:
        i_freq_B = np.argmin(np.abs(freq_B - omega_res))
        print(f"  B bin: {freq_B[i_freq_B]:.3f} eV")
        B_tddft = B_fft[i_freq_B, :].real
        print(f"  B_tddft range: [{B_tddft.min():.3e}, {B_tddft.max():.3e}]")

    # ── B_z from curl(A_ind) at atom sites — matches Eq.14+27 of paper ──
    J_res = J_fft[i_freq_J, :]     # complex bond currents at resonance
    B_curl = compute_Bz_curl(x_au, y_au, bonds_raw, a_nn, J_res)
    print(f"  B_curl range: [{B_curl.min():.3e}, {B_curl.max():.3e}]")

    # ── Per-panel symmetric colorscales ──────────────────────────────────
    vmax_tddft = np.abs(B_tddft).max() if B_tddft is not None else 1.0
    vmax_curl  = np.abs(B_curl).max()
    norm_tddft = TwoSlopeNorm(vmin=-vmax_tddft, vcenter=0, vmax=vmax_tddft)
    norm_curl  = TwoSlopeNorm(vmin=-vmax_curl,  vcenter=0, vmax=vmax_curl)

    xpad = (x_nm.max() - x_nm.min()) * 0.08
    ypad = (y_nm.max() - y_nm.min()) * 0.08

    # ── Left panel: TDDFT ─────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[row, 0])
    if B_tddft is not None:
        sc0 = ax0.scatter(x_nm, y_nm, c=B_tddft, cmap=CMAP, norm=norm_tddft,
                          s=SCATTER_S, linewidths=0)
        cb0 = fig.colorbar(sc0, ax=ax0, fraction=CBAR_FRAC, pad=CBAR_PAD)
        cb0.set_label(r"$B_{\mathrm{ind},z}$ (a.u.)", fontsize=FONT_BASE - 2)
    else:
        ax0.text(0.5, 0.5, "no TDDFT data", ha='center', va='center',
                 transform=ax0.transAxes, color='grey')
    ax0.set_aspect('equal')
    ax0.set_xlim(x_nm.min() - xpad, x_nm.max() + xpad)
    ax0.set_ylim(y_nm.min() - ypad, y_nm.max() + ypad)
    ax0.set_xlabel(r"$x$ (nm)")
    ax0.set_ylabel(r"$y$ (nm)")
    ax0.set_title(
        label + "\n" + COL_TITLES[0] +
        rf", $\hbar\omega = {omega_res:.2f}$ eV",
        fontsize=FONT_TITLE)

    # ── Right panel: curl(A_ind) ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[row, 1])
    sc1 = ax1.scatter(x_nm, y_nm, c=B_curl, cmap=CMAP, norm=norm_curl,
                      s=SCATTER_S, linewidths=0)
    cb1 = fig.colorbar(sc1, ax=ax1, fraction=CBAR_FRAC, pad=CBAR_PAD)
    cb1.set_label(r"$B_{\mathrm{ind},z}$ (a.u.)", fontsize=FONT_BASE - 2)
    ax1.set_aspect('equal')
    ax1.set_xlim(x_nm.min() - xpad, x_nm.max() + xpad)
    ax1.set_ylim(y_nm.min() - ypad, y_nm.max() + ypad)
    ax1.set_xlabel(r"$x$ (nm)")
    ax1.set_ylabel(r"$y$ (nm)")
    ax1.set_title(
        label + "\n" + r"$\nabla\times A_\mathrm{ind}$, Eq.\,(14)" +
        rf", $\hbar\omega = {omega_res:.2f}$ eV",
        fontsize=FONT_TITLE)

# ============================================================
#  SAVE
# ============================================================
plt.savefig("biot_savart_vs_tddft.png", bbox_inches='tight', dpi=300)
print("\nSaved: biot_savart_vs_tddft.png")
plt.show()