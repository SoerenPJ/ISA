"""
diagnose_sweep_params.py
========================
Run this before plot_sweep_sigma_ext.py to find the optimal masking and
clipping parameters for each structure automatically.

For each structure it shows:
  Left:  Histogram of L0 values — the natural gap between signal and noise
         floor tells you where to set L0_MASK_FRAC.
  Right: CDF of |rel-err| values (unmasked) — the knee of the curve tells
         you where to set REL_ERR_CLIP.

Prints recommended parameter values at the end.

Usage:
    python3 diagnose_sweep_params.py
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── copy these from plot_sweep_sigma_ext.py ──────────────────────────────────
STRUCTURES = [
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_bowtie_10x10_rot0",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_bowtie_15x15_rot0",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_triangle_20x20_rot0",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_triangle_30x30_rot0",
]
AU_NM = 0.0529177
AU_EV = 27.2114
# ─────────────────────────────────────────────────────────────────────────────


def find_mu_folders(structure_dir, level):
    pattern = re.compile(
        rf"^{level}_mu_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$"
    )
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


def load_grid(structure_dir, level):
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
        if omega_ref is None:
            omega_ref = omega
        elif not np.allclose(omega, omega_ref, atol=1e-10):
            sigma = np.interp(omega_ref, omega, sigma)
        mu_vals.append(mu_val)
        columns.append(sigma)
    if not columns:
        return None, None, None
    return np.array(mu_vals), omega_ref, np.column_stack(columns)


def short_label(path):
    return (os.path.basename(path)
            .replace("sweep_data_mu_", "")
            .replace("_rot0", "")
            .replace("_", " "))


def find_signal_gap(values, n_bins=200):
    """
    Find the natural gap between noise floor and signal in a value distribution.
    Returns the fraction of col_peak at which the gap occurs.
    Strategy: look for the largest gap in the histogram below the 50th percentile
    — that gap separates background from signal.
    """
    v = values[values > 0]
    if v.size == 0:
        return None
    counts, edges = np.histogram(v, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Only look below 50th percentile (noise floor region)
    p50 = np.percentile(v, 50)
    mask = centers < p50
    if not mask.any():
        return None
    # Find bin with largest gap (zero or near-zero count followed by signal)
    sub_counts = counts[mask]
    sub_centers = centers[mask]
    # Look for the rightmost zero-count bin
    zero_bins = np.where(sub_counts == 0)[0]
    if len(zero_bins) == 0:
        # No clean gap — use the valley (local minimum)
        gap_idx = np.argmin(sub_counts)
    else:
        gap_idx = zero_bins[-1]
    gap_value = sub_centers[gap_idx]
    col_peak_median = np.median(np.max(values, axis=0)) if values.ndim == 2 else v.max()
    return gap_value / col_peak_median if col_peak_median > 0 else None


def find_clip_percentile(rel_err_vals):
    """
    Find the knee of the |rel-err| CDF using the maximum-curvature method.

    The CDF is plotted as percentile (x) vs |rel-err| value (y).  The knee
    is where the curve bends most sharply — i.e. where a small increase in
    percentile causes a large jump in value (outliers start dominating).
    We find this as the maximum of d(value)/d(percentile) after the bulk.

    Returns the recommended clip percentile (integer, 70-99).
    """
    finite = rel_err_vals[np.isfinite(rel_err_vals)]
    if finite.size == 0:
        return None
    abs_vals = np.abs(finite)
    # Work in percentile space: x = percentile 0-100, y = |rel-err| value
    pcts = np.linspace(50, 99.9, 500)   # only look in upper half — bulk is below 50
    vals = np.percentile(abs_vals, pcts)
    # Derivative of value w.r.t. percentile — large = outlier cliff
    dv = np.diff(vals)
    dp = np.diff(pcts)
    deriv = dv / dp
    # The knee is the percentile just before the derivative jumps largest
    # Use a smoothed derivative to avoid noise spikes
    kernel = np.ones(10) / 10
    smooth_deriv = np.convolve(deriv, kernel, mode='same')
    knee_rel_idx = np.argmax(smooth_deriv)
    knee_pct = pcts[knee_rel_idx]
    # Clamp to sensible range
    knee_pct = float(np.clip(knee_pct, 70, 99))
    return knee_pct


# ── Main ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 4 * len(STRUCTURES)))
gs  = gridspec.GridSpec(len(STRUCTURES), 3, figure=fig,
                        hspace=0.45, wspace=0.35,
                        left=0.07, right=0.97, top=0.95, bottom=0.05)

recommendations = {}

for row, structure in enumerate(STRUCTURES):
    label = short_label(structure)
    print(f"\n{'='*60}")
    print(f"Structure: {label}")

    g0 = load_grid(structure, "L0")[2]
    g1 = load_grid(structure, "L1")[2]
    g2 = load_grid(structure, "L2")[2]

    if g0 is None:
        print("  No L0 data — skipping.")
        recommendations[label] = {"mask_frac": "N/A", "clip_pct": "N/A"}
        continue

    # ── Panel 1: L0 value histogram ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[row, 0])
    flat = g0.ravel()
    flat_pos = flat[flat > 0]
    ax1.hist(flat_pos, bins=200, color='steelblue', alpha=0.7, log=True)

    # Mark several candidate mask fractions based on per-column normalisation
    col_peak = g0.max(axis=0, keepdims=True)
    col_norm = np.where(col_peak > 0, g0 / col_peak, 0)
    flat_norm = col_norm.ravel()
    flat_norm_pos = flat_norm[flat_norm > 0]

    ax2 = fig.add_subplot(gs[row, 1])
    ax2.hist(flat_norm_pos, bins=200, color='darkorange', alpha=0.7, log=True)
    for frac in [0.05, 0.10, 0.20, 0.30, 0.50]:
        ax2.axvline(frac, color='red', lw=0.8, linestyle='--', alpha=0.7,
                    label=f"{frac:.2f}")
    ax2.set_xlabel("L0 / col_peak")
    ax2.set_ylabel("count (log)")
    ax2.set_title(f"{label}\nL0/col_peak histogram")
    ax2.legend(fontsize=7, title="frac candidates", loc='upper right')

    # Estimate recommended mask frac: fraction below which < 1% of signal lives
    # i.e. find the value where the cumulative count from below = 1% of total
    sorted_norm = np.sort(flat_norm_pos)
    n = len(sorted_norm)
    # Find the value at the 5th percentile — pixels below this are noise floor
    p5  = np.percentile(sorted_norm, 5)
    p10 = np.percentile(sorted_norm, 10)
    # Find natural gap: largest empty bin below median
    counts, edges = np.histogram(sorted_norm, bins=200, range=(0, 0.5))
    centers = 0.5 * (edges[:-1] + edges[1:])
    zero_mask = counts == 0
    if zero_mask.any():
        # Rightmost zero bin below 0.5 = natural noise/signal boundary
        gap_frac = centers[np.where(zero_mask)[0][-1]]
    else:
        # No clean gap: use the valley (minimum count bin) below 0.3
        sub = counts[centers < 0.3]
        if sub.size:
            gap_frac = centers[np.argmin(sub)]
        else:
            gap_frac = p10

    recommended_mask = round(float(gap_frac), 2)
    ax2.axvline(recommended_mask, color='green', lw=2, linestyle='-',
                label=f"recommended: {recommended_mask:.2f}")
    ax2.legend(fontsize=7, title="frac candidates", loc='upper right')

    print(f"  Recommended L0_MASK_FRAC: {recommended_mask:.2f}")
    print(f"    (gap in L0/col_peak histogram at {gap_frac:.3f})")

    # ── Panel 3: |rel-err| CDF ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[row, 2])

    rel_vals_all = []
    for g_num in [g1, g2]:
        if g_num is None:
            continue
        col_peak2 = np.nanmax(np.abs(g0), axis=0, keepdims=True)
        noise_mask = np.abs(g0) < recommended_mask * col_peak2
        with np.errstate(invalid='ignore', divide='ignore'):
            r = np.where(noise_mask, np.nan,
                         (g_num - g0) / np.abs(g0)) * 100
        rel_vals_all.append(r)

    if rel_vals_all:
        combined = np.concatenate([v.ravel() for v in rel_vals_all])
        finite   = combined[np.isfinite(combined)]
        abs_finite = np.abs(finite)
        sorted_v = np.sort(abs_finite)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax3.plot(sorted_v, cdf * 100, color='purple', lw=1.5)

        # Mark common percentiles
        for pct in [80, 90, 95, 99]:
            val = np.percentile(abs_finite, pct)
            ax3.axvline(val, lw=0.8, linestyle='--', alpha=0.7,
                        label=f"p{pct}={val:.2f}%")
            ax3.axhline(pct, lw=0.4, linestyle=':', color='grey', alpha=0.5)

        # Knee detection
        knee_pct = find_clip_percentile(finite)
        if knee_pct:
            knee_val = np.percentile(abs_finite, knee_pct)
            ax3.axvline(knee_val, color='green', lw=2,
                        label=f"knee p{knee_pct:.0f}={knee_val:.2f}%")
            recommended_clip = round(float(knee_pct))
        else:
            recommended_clip = 95

        ax3.set_xlabel(r"|rel-err| (%)")
        ax3.set_ylabel("CDF (%)")
        ax3.set_title(f"{label}\n|rel-err| CDF  →  recommended clip: p{recommended_clip}")
        ax3.set_xlim(left=0)
        ax3.set_ylim(0, 100)
        ax3.legend(fontsize=7, loc='lower right')
        print(f"  Recommended REL_ERR_CLIP: {recommended_clip}")
    else:
        ax3.text(0.5, 0.5, "no L1/L2 data", ha='center', va='center',
                 transform=ax3.transAxes)
        recommended_clip = 95

    ax1.set_xlabel("L0 (nm²)")
    ax1.set_ylabel("count (log)")
    ax1.set_title(f"{label}\nL0 absolute histogram")

    recommendations[os.path.basename(structure)] = {
        "mask_frac":  recommended_mask,
        "clip_pct":   recommended_clip,
    }

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RECOMMENDED SETTINGS FOR plot_sweep_sigma_ext.py:")
print(f"{'='*60}")
clip_vals = [v['clip_pct'] for v in recommendations.values() if isinstance(v['clip_pct'], float)]
suggested_clip = int(round(np.median(clip_vals) / 5) * 5) if clip_vals else 95
print(f"REL_ERR_CLIP = {suggested_clip}  # (median of per-structure knees, rounded to 5)")
print()
print("L0_MASK_FRAC_PER_STRUCTURE = {")
for k, v in recommendations.items():
    print(f'    "{k}": {v["mask_frac"]},')
print("}")

plt.savefig("sweep_param_diagnostics.png", dpi=150, bbox_inches="tight")
print("\nSaved: sweep_param_diagnostics.png")
plt.show()