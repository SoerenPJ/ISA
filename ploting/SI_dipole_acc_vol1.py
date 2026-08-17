
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.ndimage import median_filter

# ============================================================
#  USER SETTINGS
# ============================================================

DATA_ROOT = "/home/soeren/University/masters/2.semester/ISA/scr/data_LLM"

STRUCTURES = [
    os.path.join(DATA_ROOT, "dipole_sweep_data_mu_zigzag_triangle_2x2_rot0"),
    # add more dipole_sweep_data_mu_* directories here
]


PLOT_SINGLE         = True
SINGLE_STRUCTURE    = os.path.join(DATA_ROOT, "dipole_sweep_data_mu_zigzag_triangle_2x2_rot0")
SINGLE_MU           = 1.52               # nearest available mu is used
SINGLE_LEVELS       = ["L0"]#, "L1", "L2"]   # any subset of L0/L1/L2 to overlay
SINGLE_OMEGA_MAX_EV = None                 # e.g. 20.0 to zoom; None = full range

AU_EV   = 27.2114
MU_UNIT = "eV"

LEVEL_LABELS = {
    "L0": "Hartree (L0)",
    "L1": "Zeeman (L1)",
    "L2": "Peierls phase (L2)",
}

CMAP_ABS = "turbo"     # log HHG intensity seismic
CMAP_REL = "seismic"     # log-ratio comparison

# ── absolute (log) colour scale ──────────────────────────────────────────────
# vmax = shared peak over L0/L1/L2 of the row; vmin = vmax / 10**ABS_DECADES.
# ~6 decades shows the harmonic comb cleanly; larger values let the numerical
# noise plateau (which sits ~6-7 decades below the peak) creep back into view.
ABS_DECADES = 6

# ── noise handling for the comparison panels ─────────────────────────────────
# A pixel is "signal" if its intensity exceeds  peak / 10**FLOOR_DECADES.
# The log-ratio is masked (white) where NEITHER L0 nor Lx is signal.
FLOOR_DECADES = 6
# Symmetric colour limit for log10(ratio): percentile of the finite, masked data
# (robust to the few extreme pixels), with a hard cap so it stays readable.
RATIO_CLIP_PCTILE = 99
RATIO_CLIP_MAX    = 5     # never exceed +/- this many decades


PEAK_SIGNIFICANCE = True
PEAK_PROM_FACTOR  = 30.0    # peak must be this many x the local baseline
PEAK_BG_WINDOW    = 0.5     # harmonic-order width of the local-baseline window
GATE_ABSOLUTE     = True    # also blank non-peak pixels in panels A/B/C

# ── optional overlay: the per-mu drive (fundamental) frequency ───────────────
SHOW_DRIVE_LINE = True

# ── restrict the photon-energy (y) range; None = full range in the files ─────
# (only used when NORMALIZE = False)
OMEGA_MAX_EV = None        # e.g. 30.0 to zoom into the low harmonics


NORMALIZE         = True
HARMONIC_N_MAX    = None    # highest harmonic order shown; None -> auto from data
HARMONIC_N_POINTS = 1000    # samples along the resampled harmonic-order axis

YLABEL_SWEEP   = (r"Harmonic order $\omega/\omega_0$" if NORMALIZE
                  else "Photon energy (eV)")
CBAR_ABS_TITLE = (r"$I/I_{\omega_0}$" if NORMALIZE else r"$|a(\omega)|^2$")

INTERPOLATION = "none"

ROW_LABELS = None          # e.g. ["Zigzag triangle"]; None -> auto from dir name

# ============================================================
#  TYPOGRAPHY  (matched to SI_linear_response_vol1)
# ============================================================
FONT_SIZE_GLOBAL      = 20
FONT_SIZE_AXIS_LABEL  = 18
FONT_SIZE_TICK        = 16
FONT_SIZE_CBAR        = 14
FONT_SIZE_CBAR_LABEL  = 14
FONT_SIZE_PANEL_LABEL = 16
FONT_FAMILY           = "times new roman"
USE_LATEX             = False

# ============================================================
#  LAYOUT  (matched to SI_linear_response_vol1)
# ============================================================
PANEL_W   = 2.60
PANEL_H   = 2.20
CBAR_W    = 0.28
GAP_W     = 0.40
SMALL_GAP = 0.12
LEFT_PAD  = 0.80
RIGHT_PAD = 0.55
TOP_PAD   = 0.40
BOT_PAD   = 0.50
HSPACE    = 0.04
WSPACE    = 0.05

# ============================================================
#  HELPERS
# ============================================================

def find_mu_folders(structure_dir, level):
    pattern = re.compile(rf"^{level}_mu_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
    entries = []
    try:
        for name in os.listdir(structure_dir):
            m = pattern.match(name)
            if m:
                entries.append((float(m.group(1)),
                                os.path.join(structure_dir, name)))
    except FileNotFoundError:
        return []
    entries.sort(key=lambda x: x[0])
    return entries


def load_level_grid(structure_dir, level):
    """Return (mu_vals, omega_eV, grid) with grid shape (N_omega, N_mu).

    grid holds the dipole-acceleration intensity I(omega) = |a(omega)|^2.
    """
    mu_folders = find_mu_folders(structure_dir, level)
    if not mu_folders:
        return None, None, None

    mu_vals, omega_ref, columns = [], None, []
    for mu_val, folder in mu_folders:
        fpath = os.path.join(folder, "dipole_acc.txt")
        if not os.path.isfile(fpath):
            continue
        raw = np.loadtxt(fpath)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        omega = raw[:, 0] * AU_EV
        intensity = raw[:, 1] ** 2 + raw[:, 2] ** 2     # |a|^2
        if omega_ref is None:
            omega_ref = omega
        elif not np.allclose(omega, omega_ref, atol=1e-10):
            intensity = np.interp(omega_ref, omega, intensity)
        mu_vals.append(mu_val)
        columns.append(intensity)

    if not columns:
        return None, None, None
    return np.array(mu_vals), omega_ref, np.column_stack(columns)


def load_single_spectrum(structure_dir, level, mu_target):
    """Load |a(omega)|^2 for the mu folder closest to mu_target.

    Returns (mu_actual, omega_eV, intensity) or (None, None, None).
    """
    folders = find_mu_folders(structure_dir, level)
    if not folders:
        return None, None, None
    mu_avail = np.array([m for m, _ in folders])
    idx = int(np.argmin(np.abs(mu_avail - mu_target)))
    mu_actual, folder = folders[idx]
    fpath = os.path.join(folder, "dipole_acc.txt")
    if not os.path.isfile(fpath):
        return None, None, None
    raw = np.loadtxt(fpath)
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    omega = raw[:, 0] * AU_EV
    intensity = raw[:, 1] ** 2 + raw[:, 2] ** 2
    return mu_actual, omega, intensity


def plot_single_dipole(structure_dir, mu_target, levels, omega_max=None):
    """Single dipole-acceleration spectrum, levels overlaid, log y-scale.

    With NORMALIZE the axes follow the HHG.py convention:
        x = omega / omega_0   (harmonic order)
        y = |a(omega)|^2 / |a(omega_0)|^2
    where omega_0 is the fundamental (drive) frequency at this mu.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = {"L0": "k", "L1": "tab:blue", "L2": "tab:red"}
    drive = load_drive(structure_dir)

    specs = []
    mu_actual = None
    for lvl in levels:
        mu_a, omega, intensity = load_single_spectrum(structure_dir, lvl, mu_target)
        if omega is None:
            print(f"  [single] no data for {lvl}")
            continue
        mu_actual = mu_a
        specs.append((lvl, omega, intensity))

    if not specs:
        print("  [single] nothing to plot")
        plt.close(fig)
        return

    # fundamental omega_0 (shared across levels): drive freq, else argmax
    if drive is not None:
        w0 = float(np.interp(mu_actual, drive[0], drive[1]))
    else:
        _, om0, I0 = specs[0]
        w0 = om0[int(np.argmax(I0))]
    I_ref = max(I.max() for _, _, I in specs)

    for lvl, omega, intensity in specs:
        if NORMALIZE:
            x, y = omega / w0, intensity / I_ref
        else:
            x, y = omega, intensity
            if omega_max is not None:
                keep = omega <= omega_max
                x, y = x[keep], y[keep]
        ax.plot(x, y, lw=1.3, color=colors.get(lvl), label=LEVEL_LABELS.get(lvl, lvl))

    ax.set_yscale("log")
    if NORMALIZE:
        n_max = HARMONIC_N_MAX if HARMONIC_N_MAX else float(specs[0][1].max() / w0)
        ax.set_xlim(0, n_max)
        ax.axvline(1.0, color="grey", ls="--", lw=0.9,
                   label=rf"fundamental $\omega_0$ = {w0:.2f} eV")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(which="major", ls=":", lw=0.6, alpha=0.6)
        ax.set_xlabel(r"$\omega/\omega_0$", fontsize=FONT_SIZE_AXIS_LABEL)
        ax.set_ylabel(r"$|a(\omega)|^2/|a(\omega_0)|^2$", fontsize=FONT_SIZE_AXIS_LABEL)
    else:
        if drive is not None:
            w_drive = float(np.interp(mu_actual, drive[0], drive[1]))
            ax.axvline(w_drive, color="grey", ls="--", lw=0.9,
                       label=rf"drive $\omega_0$ = {w_drive:.2f} eV")
        ax.set_xlabel("Photon energy (eV)", fontsize=FONT_SIZE_AXIS_LABEL)
        ax.set_ylabel(r"$|a(\omega)|^2$", fontsize=FONT_SIZE_AXIS_LABEL)

    ax.set_title(rf"{short_label(structure_dir)}  —  $\mu$ = {mu_actual:.2f} {MU_UNIT}",
                 fontsize=FONT_SIZE_PANEL_LABEL)
    ax.legend(fontsize=FONT_SIZE_CBAR, frameon=False)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    fig.tight_layout()
    out = f"dipole_acc_single_mu{mu_actual:.2f}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"  [single] saved {out}")


def load_drive(structure_dir, level="L0"):
    """mu -> drive (fundamental) photon energy in eV, from resonance_used.txt."""
    path = os.path.join(structure_dir, "resonance_used.txt")
    if not os.path.isfile(path):
        return None
    mus, drives = [], []
    with open(path) as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3 or parts[1] != level:
                continue
            try:
                mus.append(float(parts[0]))
                drives.append(float(parts[2]))
            except ValueError:
                continue
    if not mus:
        return None
    order = np.argsort(mus)
    return np.array(mus)[order], np.array(drives)[order]


def crop_omega(omega, grids, omega_max):
    if omega_max is None:
        return omega, grids
    keep = omega <= omega_max
    return omega[keep], {k: (v[keep] if v is not None else None)
                         for k, v in grids.items()}


def make_extent(mu, omega):
    dmu = (mu[-1] - mu[0]) / max(len(mu) - 1, 1) if len(mu) > 1 else 1
    dw  = (omega[-1] - omega[0]) / max(len(omega) - 1, 1) if len(omega) > 1 else 1
    return [mu[0] - dmu / 2, mu[-1] + dmu / 2,
            omega[0] - dw / 2, omega[-1] + dw / 2]


def log_ratio(num, base, floor):
    """log10(num/base), masked (NaN) where BOTH are below the noise floor."""
    if num is None or base is None:
        return None
    tiny = 1e-300
    both_noise = (base < floor) & (num < floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log10((num + tiny) / (base + tiny))
    r[both_noise] = np.nan
    return r


def log_ratio_frac(num, base, floor_frac):
    """log10(num/base) with a PER-COLUMN noise floor (fraction of column peak).

    Used for the normalised harmonic grids, where each mu-column has its own
    fundamental peak. Masked (NaN) where both levels are below their column floor
    or where either grid is undefined (out of harmonic range)."""
    if num is None or base is None:
        return None
    tiny = 1e-300
    base_floor = floor_frac * np.nanmax(base, axis=0, keepdims=True)
    num_floor = floor_frac * np.nanmax(num, axis=0, keepdims=True)
    both_noise = (base < base_floor) & (num < num_floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log10((num + tiny) / (base + tiny))
    r[both_noise] = np.nan
    r[~np.isfinite(base) | ~np.isfinite(num)] = np.nan
    return r


def to_harmonic_grid(omega_eV, grid, w0_arr, n_grid):
    """Resample each mu-column of grid onto the common harmonic-order axis n_grid.

    Column j (driven at omega_0 = w0_arr[j]) is interpolated at photon energies
    n_grid * omega_0. Orders that fall outside the available omega range -> NaN.
    """
    n_mu = grid.shape[1]
    out = np.full((n_grid.size, n_mu), np.nan)
    for j in range(n_mu):
        w0 = w0_arr[j]
        if not np.isfinite(w0) or w0 <= 0:
            continue
        out[:, j] = np.interp(n_grid * w0, omega_eV, grid[:, j],
                              left=np.nan, right=np.nan)
    return out


def signal_mask_harmonic(grid, n_grid, prom_factor, bg_window, floor_frac):
    """Boolean mask: True where grid is a genuine harmonic peak.

    Two conditions must BOTH hold (per mu-column):
      1. Local prominence: the value exceeds prom_factor x the local
         inter-harmonic baseline (rolling median over bg_window orders) -- i.e.
         its neighbours are significantly smaller.
      2. Absolute floor: the value exceeds floor_frac x the column's own
         fundamental peak -- so a locally-prominent but absolutely-tiny noise
         bump (e.g. the forbidden 2nd harmonic at mu=0, ~1e-9) is rejected.
    Out-of-range (NaN) pixels are never signal.
    """
    if grid is None:
        return None
    g = np.where(np.isfinite(grid), grid, 0.0)
    dn = (n_grid[-1] - n_grid[0]) / max(len(n_grid) - 1, 1)
    win = max(3, int(round(bg_window / dn)))
    if win % 2 == 0:
        win += 1
    base = median_filter(g, size=(win, 1), mode="nearest")
    gmax = np.nanmax(g)
    tiny = gmax * 1e-12 if gmax > 0 else 1e-300
    prominent = g / np.maximum(base, tiny) > prom_factor
    col_peak = np.nanmax(g, axis=0, keepdims=True)
    above_floor = g > floor_frac * col_peak
    return prominent & above_floor & np.isfinite(grid)


def log_ratio_masked(num, base, sig_num, sig_base):
    """log10(num/base), kept only where EITHER level has a genuine peak.

    Used with the peak-significance masks so the comparison shows a pixel only
    where at least one level carries real harmonic signal (a suppressed peak or
    a newly switched-on harmonic), and is blank everywhere else."""
    if num is None or base is None:
        return None
    tiny = 1e-300
    keep = np.zeros(num.shape, dtype=bool)
    if sig_base is not None:
        keep |= sig_base
    if sig_num is not None:
        keep |= sig_num
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log10((num + tiny) / (base + tiny))
    r[~keep] = np.nan
    r[~np.isfinite(base) | ~np.isfinite(num)] = np.nan
    return r


def short_label(path):
    return (os.path.basename(path.rstrip("/"))
            .replace("dipole_sweep_data_mu_", "")
            .replace("_rot0", "")
            .replace("_", " "))


def style_heatmap(ax, is_last_row, show_ylabel):
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))
    if show_ylabel:
        ax.set_ylabel(YLABEL_SWEEP, fontsize=FONT_SIZE_AXIS_LABEL)
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
#  SINGLE-SPECTRUM PREVIEW  (before the sweep figure)
# ============================================================
if PLOT_SINGLE:
    print(f"\nSingle spectrum: {os.path.basename(SINGLE_STRUCTURE)}  "
          f"mu~{SINGLE_MU}  levels={SINGLE_LEVELS}")
    plot_single_dipole(SINGLE_STRUCTURE, SINGLE_MU, SINGLE_LEVELS,
                        omega_max=SINGLE_OMEGA_MAX_EV)

# ============================================================
#  LOAD ALL DATA
# ============================================================
LEVELS = ["L0", "L1", "L2"]
all_data = []

for structure in STRUCTURES:
    struct_key = os.path.basename(structure.rstrip("/"))
    print(f"\nLoading {struct_key}")

    grids = {}
    mu_ref = omega_ref = None
    for lvl in LEVELS:
        mu, omega, grid = load_level_grid(structure, lvl)
        if grid is None:
            print(f"  [WARNING] No data for {lvl}")
            grids[lvl] = None
            continue
        if mu_ref is None:
            mu_ref, omega_ref = mu, omega
        grids[lvl] = grid

    if mu_ref is None:
        print(f"  [WARNING] No usable data, skipping {struct_key}")
        continue

    drive = load_drive(structure)

    if NORMALIZE:
        # fundamental omega_0(mu): the drive frequency, else per-column argmax of L0
        if drive is not None:
            w0_arr = np.interp(mu_ref, drive[0], drive[1])
        else:
            base_g = grids.get("L0") or next(g for g in grids.values() if g is not None)
            w0_arr = omega_ref[np.argmax(base_g, axis=0)]

        # common harmonic-order axis (capped so every column is covered)
        w0_max = np.nanmax(w0_arr[w0_arr > 0])
        n_max = omega_ref.max() / w0_max * 0.999
        if HARMONIC_N_MAX is not None:
            n_max = min(n_max, HARMONIC_N_MAX)
        n_grid = np.linspace(0.0, n_max, HARMONIC_N_POINTS)

        harm = {lvl: (to_harmonic_grid(omega_ref, g, w0_arr, n_grid)
                      if g is not None else None)
                for lvl, g in grids.items()}

        # absolute panels: normalise each column to its own fundamental peak
        norm = {lvl: (g / np.nanmax(g, axis=0, keepdims=True) if g is not None else None)
                for lvl, g in harm.items()}

        if PEAK_SIGNIFICANCE:
            # keep only genuine local peaks (neighbours significantly smaller)
            floor_frac = 10 ** (-FLOOR_DECADES)
            sig = {lvl: signal_mask_harmonic(harm.get(lvl), n_grid,
                                             PEAK_PROM_FACTOR, PEAK_BG_WINDOW,
                                             floor_frac)
                   for lvl in LEVELS}
            r_D = log_ratio_masked(harm.get("L1"), harm.get("L0"),
                                   sig.get("L1"), sig.get("L0"))
            r_E = log_ratio_masked(harm.get("L2"), harm.get("L0"),
                                   sig.get("L2"), sig.get("L0"))
            if GATE_ABSOLUTE:
                for lvl in LEVELS:
                    if norm.get(lvl) is not None and sig.get(lvl) is not None:
                        norm[lvl] = np.where(sig[lvl], norm[lvl], np.nan)
        else:
            # comparison from the raw harmonic grids, per-column absolute floor
            floor_frac = 10 ** (-FLOOR_DECADES)
            r_D = log_ratio_frac(harm.get("L1"), harm.get("L0"), floor_frac)
            r_E = log_ratio_frac(harm.get("L2"), harm.get("L0"), floor_frac)

        grids = norm
        peak = 1.0
        extent = make_extent(mu_ref, n_grid)
    else:
        omega_ref, grids = crop_omega(omega_ref, grids, OMEGA_MAX_EV)
        extent = make_extent(mu_ref, omega_ref)
        valid = [g for g in grids.values() if g is not None]
        peak = max(g.max() for g in valid)
        floor = peak / 10 ** FLOOR_DECADES
        r_D = log_ratio(grids.get("L1"), grids.get("L0"), floor)
        r_E = log_ratio(grids.get("L2"), grids.get("L0"), floor)

    all_data.append(dict(
        struct_key=struct_key, grids=grids, extent=extent,
        peak=peak, r_D=r_D, r_E=r_E, drive=drive,
    ))

if not all_data:
    raise SystemExit("No structures loaded — check STRUCTURES / DATA_ROOT.")

# ============================================================
#  FIGURE LAYOUT  (identical column scheme to SI_linear_response_vol1)
# ============================================================
N_ROWS = len(all_data)

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

COL = dict(A=0, B=2, C=4, CBAR_ABS=5, D=7, E=9, CBAR_REL=10)

PANEL_TITLES = {
    "A": "Hartree",
    "B": "Zeeman",
    "C": "Peierls phase",
    "D": r"$\log_{10}(I/I_\mathrm{L0})$" + "\nZeeman",
    "E": r"$\log_{10}(I/I_\mathrm{L0})$" + "\nPeierls phase",
}

cmap_rel = plt.get_cmap(CMAP_REL).copy()
cmap_rel.set_bad(color="white")

# ============================================================
#  PLOT ROWS
# ============================================================
for row, d in enumerate(all_data):
    extent  = d["extent"]
    grids   = d["grids"]
    peak    = d["peak"]
    r_D     = d["r_D"]
    r_E     = d["r_E"]
    drive   = d["drive"]
    is_last = (row == N_ROWS - 1)

    g0, g1, g2 = grids.get("L0"), grids.get("L1"), grids.get("L2")

    # ── absolute (log) colour scale, shared across A/B/C in this row ─────────
    vmax = peak
    vmin = vmax / 10 ** ABS_DECADES
    abs_norm = LogNorm(vmin=vmin, vmax=vmax)

    # ── symmetric log-ratio colour scale for D/E ────────────────────────────
    row_rels = [r for r in [r_D, r_E] if r is not None]
    finite = (np.concatenate([r[np.isfinite(r)] for r in row_rels])
              if row_rels else np.array([1.0]))
    rel_max = min(
        max(np.percentile(np.abs(finite), RATIO_CLIP_PCTILE)
            if finite.size else 1.0, 1e-3),
        RATIO_CLIP_MAX,
    )
    rel_norm = TwoSlopeNorm(vmin=-rel_max, vcenter=0, vmax=rel_max)

 

    # ── panels A, B, C ──────────────────────────────────────────────────────
    abs_im_ref = None
    for panel, grid in [("A", g0), ("B", g1), ("C", g2)]:
        ax = fig.add_subplot(gs[row, COL[panel]])
        if row == 0:
            ax.set_title(PANEL_TITLES[panel], fontsize=FONT_SIZE_PANEL_LABEL, pad=4)
        if grid is not None:
            disp = np.clip(grid, vmin, vmax)
            disp = np.where(np.isfinite(disp), disp, vmin)  # out-of-range -> floor
            im = ax.imshow(disp, extent=extent,
                           origin="lower", aspect="auto", cmap=CMAP_ABS,
                           norm=abs_norm, interpolation=INTERPOLATION,
                           rasterized=True)
            
            if panel == "C":
                abs_im_ref = im
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
        style_heatmap(ax, is_last, show_ylabel=(panel == "A"))

    ax_cbar_a = fig.add_subplot(gs[row, COL["CBAR_ABS"]])
    if abs_im_ref is not None:
        cb = fig.colorbar(abs_im_ref, cax=ax_cbar_a)
        cb.ax.tick_params(labelsize=FONT_SIZE_CBAR)
        if row == 0:
            ax_cbar_a.set_title(CBAR_ABS_TITLE, fontsize=FONT_SIZE_CBAR_LABEL, pad=5)
    else:
        ax_cbar_a.set_visible(False)

    # ── panels D, E ─────────────────────────────────────────────────────────
    rel_im_ref = None
    for panel, data in [("D", r_D), ("E", r_E)]:
        ax = fig.add_subplot(gs[row, COL[panel]])
        if row == 0:
            ax.set_title(PANEL_TITLES[panel], fontsize=FONT_SIZE_PANEL_LABEL, pad=4)
        if data is not None:
            im = ax.imshow(np.clip(data, -rel_max, rel_max), extent=extent,
                           origin="lower", aspect="auto", cmap=cmap_rel,
                           norm=rel_norm, interpolation=INTERPOLATION,
                           rasterized=True)
            if panel == "E":
                rel_im_ref = im
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
        style_heatmap(ax, is_last, show_ylabel=False)

    ax_cbar_r = fig.add_subplot(gs[row, COL["CBAR_REL"]])
    if rel_im_ref is not None:
        cb = fig.colorbar(rel_im_ref, cax=ax_cbar_r)
        cb.ax.tick_params(labelsize=FONT_SIZE_CBAR)
        cb.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        if row == 0:
            ax_cbar_r.set_title(r"$\log_{10}\frac{I}{I_\mathrm{L0}}$",
                                fontsize=FONT_SIZE_CBAR_LABEL, pad=5)
    else:
        ax_cbar_r.set_visible(False)

# ============================================================
#  SAVE
# ============================================================
out = "supplementary_dipole_acc_vol1.pdf"
plt.savefig(out, bbox_inches="tight", dpi=600, format="pdf")
print(f"\nSaved {out}")
plt.show()
