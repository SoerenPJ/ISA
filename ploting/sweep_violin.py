import sys
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

if len(sys.argv) < 2:
    print("Usage: python3 ploting/sweep_violin.py <gauge_ts_mu_.../>")
    print("       python3 ploting/sweep_violin.py <gauge_metrics_mu_...txt>")
    sys.exit(1)

arg = Path(sys.argv[1])

if arg.is_file() and arg.suffix == ".txt":
    ts_dir = arg.parent / arg.name.replace("gauge_metrics_mu_", "gauge_ts_mu_").replace(".txt", "")
elif arg.is_dir():
    ts_dir = arg
else:
    ts_dir = arg

if not ts_dir.is_dir():
    print(f"Error: timeseries directory not found: {ts_dir}")
    print("Run gauge_sweep.sh or combined_sweep.sh first to generate it.")
    sys.exit(1)

# ── load all per-mu timeseries files, sorted by mu ──────────────────────────
files = sorted(ts_dir.glob("mu_*.txt"),
               key=lambda p: float(re.search(r"mu_([0-9]+\.[0-9]+)", p.name).group(1)))

if not files:
    print(f"No mu_*.txt files found in {ts_dir}")
    sys.exit(1)

mu_labels     = []
datasets_flux = []
datasets_curl = []

for f in files:
    mu_val = float(re.search(r"mu_([0-9]+\.[0-9]+)", f.name).group(1))
    try:
        data = np.loadtxt(f)
    except Exception:
        continue
    if data.ndim < 2 or data.shape[1] < 8:
        continue
    flux = data[2:, 5] * 100
    curl = data[2:, 7] * 100
    flux = flux[np.isfinite(flux) & (flux < 10000)]
    curl = curl[np.isfinite(curl) & (curl < 10000)]
    if flux.size == 0 or curl.size == 0:
        continue
    mu_labels.append(f"{mu_val:.2f}")
    datasets_flux.append(flux)
    datasets_curl.append(curl)

if not datasets_flux:
    print("No valid data found.")
    sys.exit(1)

plot_positions = list(range(len(datasets_flux)))

# ── violin panel ─────────────────────────────────────────────────────────────
def violin_panel(ax, datasets, title):
    parts = ax.violinplot(datasets, positions=plot_positions,
                          showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.7)
    ax.boxplot(datasets, positions=plot_positions,
               widths=0.15,
               patch_artist=False,
               showfliers=False,
               medianprops=dict(color='black', linewidth=2.5),
               boxprops=dict(linewidth=1.5),
               whiskerprops=dict(linewidth=1.5),
               capprops=dict(linewidth=1.5))
    ax.set_ylabel("Mean relative error (%)")
    ax.set_title(title)
    n = len(mu_labels)
    step = max(1, n // 20)
    tick_pos = plot_positions[::step]
    tick_lab = mu_labels[::step]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([fr"$\mu$={l}" for l in tick_lab], rotation=30, ha="right")
    ax.set_xlabel(r"Chemical potential $\mu$ [eV]")
    ax.set_ylim(0, np.percentile(np.concatenate(datasets), 90))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
violin_panel(ax1, datasets_flux,
             r"Flux error:  $|\Phi_A - \Phi_B|\,/\,|\Phi_B|$")
violin_panel(ax2, datasets_curl,
             r"Curl error:  $|(\nabla\times A)_z - B_z|\,/\,|B_z|$")

title_str = ts_dir.name.replace("gauge_ts_mu_", "").replace("_", " ")
plt.suptitle(fr"Gauge check — $\mu$ sweep   ({title_str})", fontsize=12)
plt.tight_layout()
plt.show()

# ── statistics table ─────────────────────────────────────────────────────────
col_w = [8, 12, 10, 12, 10]
header = (f"{'mu':>{col_w[0]}}"
          f"{'flux med%':>{col_w[1]}}"
          f"{'flux IQR%':>{col_w[2]}}"
          f"{'curl med%':>{col_w[3]}}"
          f"{'curl IQR%':>{col_w[4]}}")
separator = "-" * sum(col_w)

print(f"\n--- Statistics: {title_str} ---")
print(separator)
print(header)
print(separator)

for label, d_flux, d_curl in zip(mu_labels, datasets_flux, datasets_curl):
    iqr_flux = np.percentile(d_flux, 75) - np.percentile(d_flux, 25)
    iqr_curl = np.percentile(d_curl, 75) - np.percentile(d_curl, 25)
    print(f"{label:>{col_w[0]}}"
          f"{np.median(d_flux):>{col_w[1]}.3f}"
          f"{iqr_flux:>{col_w[2]}.3f}"
          f"{np.median(d_curl):>{col_w[3]}.3f}"
          f"{iqr_curl:>{col_w[4]}.3f}")

print(separator)