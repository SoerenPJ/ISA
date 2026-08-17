"""
violin_flux_plot.py
===================
Load cache from violin_flux_compute.py and produce a two-panel violin figure:

  Top panel:    symmetric relative discrepancy eps [%]
                eps(t) = mean_hex(|Phi_L1-Phi_L2|) / mean_hex(0.5*(|Phi_L1|+|Phi_L2|))
                → How much do the two approximations disagree?

  Bottom panel: absolute signal magnitude phi_mean [a.u.]
                phi_mean(t) = mean_hex(0.5*(|Phi_L1|+|Phi_L2|))
                → How large is the field they are both trying to describe?

Together: if eps << 1 and phi_mean is significant, both methods agree well.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CACHE_FILE = "violin_flux_cache.npz"

STRUCT_LABELS = [
    "AC bowtie 10x10",
    "ZZ bowtie 15x15",
    "AC triangle 14x14",
    "ZZ triangle 22x22",
]
STRUCT_DISPLAY = [
    r"AC bowtie",
    r"ZZ bowtie",
    r"AC triangle",
    r"ZZ triangle",
]

MU_COLORS = ['#08306b', '#2171b5', '#6baed6']
GROUP_GAP  = 1.0
VIOLIN_W   = 0.30

plt.rcParams.update({
    "text.usetex":     True,
    "font.family":     "serif",
    "font.serif":      ["Times New Roman"],
    "font.size":       14,
    "axes.labelsize":  14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

# ── load cache ────────────────────────────────────────────────────────────
cache      = np.load(CACHE_FILE, allow_pickle=True)
keys       = list(cache["keys"])
eps_arrays = list(cache["eps_arrays"])
phi_arrays = list(cache["phi_arrays"])
data_eps   = {k: a for k, a in zip(keys, eps_arrays)}
data_phi   = {k: a for k, a in zip(keys, phi_arrays)}
print(f"Loaded {len(data_eps)} entries from {CACHE_FILE}")

# ── parse keys → organised by structure ──────────────────────────────────
pat = re.compile(r"^(.+)__mu_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")

def organise(data_dict):
    out = {s: [] for s in STRUCT_LABELS}
    for key, arr in data_dict.items():
        m = pat.match(key)
        if not m: continue
        slabel = m.group(1); mu_val = float(m.group(2))
        if slabel in out:
            out[slabel].append((mu_val, arr))
    for slabel in STRUCT_LABELS:
        entries = sorted(out[slabel], key=lambda x: x[0])
        if len(entries) >= 3:
            out[slabel] = [entries[0], entries[len(entries)//2], entries[-1]]
        else:
            out[slabel] = entries
    return out

all_eps = organise(data_eps)
all_phi = organise(data_phi)

# ── collect actual mu values for legend ───────────────────────────────────
mu_vals = sorted(set(
    mu for entries in all_eps.values() for mu, _ in entries
))

# ── violin positions ──────────────────────────────────────────────────────
offsets       = np.array([-1, 0, 1]) * VIOLIN_W * 1.5
group_centers = [g * (3*VIOLIN_W*1.5 + GROUP_GAP) for g in range(len(STRUCT_LABELS))]

def draw_violin_panel(ax, all_data, y_label):
    for g, slabel in enumerate(STRUCT_LABELS):
        center  = group_centers[g]
        entries = all_data[slabel]
        for k, color in enumerate(MU_COLORS):
            if k >= len(entries): continue
            mu_val, arr = entries[k]
            if arr is None or len(arr) == 0: continue
            pos      = center + offsets[k]
            plot_arr = arr

            vp = ax.violinplot([plot_arr], positions=[pos],
                               widths=VIOLIN_W,
                               showmedians=False, showextrema=False)
            for body in vp['bodies']:
                body.set_facecolor(color)
                body.set_alpha(0.65)
                body.set_edgecolor('none')
                # Change rasterized to False for PDF output:
                body.set_rasterized(False)

            ax.boxplot([plot_arr], positions=[pos],
                       widths=VIOLIN_W*0.38, patch_artist=False,
                       showfliers=False,
                       medianprops=dict(color='black', linewidth=2.5),
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))

    ax.set_xticks(group_centers)
    ax.set_xticklabels(STRUCT_DISPLAY, fontsize=13)
    for g in range(len(STRUCT_LABELS) - 1):
        sep = 0.5 * (group_centers[g] + group_centers[g+1])
        ax.axvline(sep, color='grey', lw=0.8, ls='--', alpha=0.4)
    ax.set_ylabel(y_label, fontsize=13)
    ax.grid(True, axis='y', ls=':', alpha=0.4)

# ── figure ────────────────────────────────────────────────────────────────
fig, (ax_eps, ax_phi) = plt.subplots(2, 1, figsize=(16, 7),
                                      gridspec_kw={'hspace': 0.40})

# ── Top: discrepancy ──────────────────────────────────────────────────────
all_eps_flat = np.concatenate([arr for entries in all_eps.values()
                                for _, arr in entries
                                if arr is not None and len(arr) > 0])
y_top_eps = np.percentile(all_eps_flat, 95) * 1.1
ax_eps.set_ylim(0, y_top_eps)

draw_violin_panel(
    ax_eps, all_eps,
    y_label=r"$\varepsilon(t)$ (\%)",
)

# ... ax_eps title stuff ...



# ── Bottom: signal magnitude ──────────────────────────────────────────────
all_phi_flat = np.concatenate([arr for entries in all_phi.values()
                                for _, arr in entries
                                if arr is not None and len(arr) > 0])
y_top_phi = np.percentile(all_phi_flat, 95) * 1.1
ax_phi.set_ylim(0, y_top_phi)


draw_violin_panel(
    ax_phi, all_phi,
    y_label=r"$\bar{\Phi}$ (a.u.)",
)

# ── Shared legend with actual mu values ───────────────────────────────────
mu_labels = [rf'$\mu = {mu_vals[i]:.2f}$ eV' if i < len(mu_vals) else ''
             for i in range(3)]
legend_handles = [
    mpatches.Patch(facecolor=MU_COLORS[0], alpha=0.7, label=mu_labels[0]),
    mpatches.Patch(facecolor=MU_COLORS[1], alpha=0.7, label=mu_labels[1]),
    mpatches.Patch(facecolor=MU_COLORS[2], alpha=0.7, label=mu_labels[2]),
]
ax_eps.legend(handles=legend_handles, loc='upper right', framealpha=0.8)

# ── Save ──────────────────────────────────────────────────────────────────
out = "violin_flux_L1_vs_L2.pdf"
plt.savefig(out, bbox_inches='tight', dpi=600, format='pdf')
print(f"Saved: {out}")
plt.show()