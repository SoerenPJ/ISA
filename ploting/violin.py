import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch

structures = [
    (90,  "AC-90",  "AC", "/home/soeren/University/masters/2.semester/ISA/Simulations/graphene_armchair_f90f0c2605e6284c/gauge_plots/flux_comparison_timeseries.txt"),
    (330, "AC-330", "AC", "/home/soeren/University/masters/2.semester/ISA//Simulations/graphene_armchair_30890db2abe065aa/gauge_plots/flux_comparison_timeseries.txt"),
    (1260, "AC-1260", "AC", "/home/soeren/University/masters/2.semester/ISA/Simulations/graphene_armchair_61e62e3e5f2c9346/gauge_plots/flux_comparison_timeseries.txt"),
    (46,  "ZZ-46",  "ZZ", "/home/soeren/University/masters/2.semester/ISA/Simulations/graphene_zigzag_73344dae63c56cd3/gauge_plots/flux_comparison_timeseries.txt"),
    (141, "ZZ-141", "ZZ", "/home/soeren/University/masters/2.semester/ISA/Simulations/graphene_zigzag_c7ce1f2e5c889dd5/gauge_plots/flux_comparison_timeseries.txt"),
    (481, "ZZ-481", "ZZ", "/home/soeren/University/masters/2.semester/ISA/Simulations/graphene_zigzag_8105eddd77565c49/gauge_plots/flux_comparison_timeseries.txt"),
    (1761, "ZZ-1761", "ZZ", "/home/soeren/University/masters/2.semester/ISA/Simulations/graphene_zigzag_af45107a502d0af9/gauge_plots/flux_comparison_timeseries.txt"),
]
 

color_map = {"AC": "steelblue", "ZZ": "tomato", "Hexagon": "green"}


datasets_flux = []
datasets_curl = []
labels        = []
positions     = []
edge_types    = []

for n_atoms, label, etype, path in structures:
    data = np.loadtxt(path)

    # Flux relative error — column 5
    mean_rel_flux = data[2:, 5] * 100
    filtered_flux = mean_rel_flux[np.isfinite(mean_rel_flux) & (mean_rel_flux < 10000)]
    datasets_flux.append(filtered_flux)

    # Curl relative error — column 7 (added by gauge_comparison.py)
    mean_rel_curl = data[2:, 7] * 100
    filtered_curl = mean_rel_curl[np.isfinite(mean_rel_curl) & (mean_rel_curl < 10000)]
    datasets_curl.append(filtered_curl)

    labels.append(label)
    positions.append(n_atoms)
    edge_types.append(etype)

plot_positions = list(range(len(datasets_flux)))

def violin_panel(ax, datasets, title):
    parts = ax.violinplot(datasets, positions=plot_positions, showmedians=False, showextrema=False)
    for pc, etype in zip(parts["bodies"], edge_types):
        pc.set_facecolor(color_map[etype])
        pc.set_alpha(0.7)
    ax.boxplot(datasets, positions=plot_positions,
               widths=0.15,
               patch_artist=False,
               showfliers=False,
               medianprops=dict(color='black', linewidth=2.5),
               boxprops=dict(linewidth=1.5),
               whiskerprops=dict(linewidth=1.5),
               capprops=dict(linewidth=1.5))
    legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Armchair'),
                       Patch(facecolor='tomato',    alpha=0.7, label='Zigzag'), 
                       Patch(facecolor = 'green', alpha = 0.7, label ='Hexagon')]
    
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_ylabel("Mean relative error (%)")
    ax.set_title(title)
    ax.set_xticks(plot_positions)
    ax.set_xticklabels([f"{l}\n({p} atoms)" for l, p in zip(labels, positions)], rotation=15)
    ax.set_ylim(0, np.percentile(np.concatenate(datasets), 90))

# -----------------------------------------------------------------------
# Plot 1: flux and curl side by side
# -----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
violin_panel(ax1, datasets_flux, "Flux error:  $|\\Phi_A - \\Phi_B| / |\\Phi_B|$")
violin_panel(ax2, datasets_curl, "Curl error:  $| (\\nabla\\times A)_z| - B_z / |B_z|$")
plt.suptitle("Gauge check — flux vs curl comparison", fontsize=12)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------
for etype_filter in ["AC", "ZZ", "Hexagon"]:

    print(f"\n--- {etype_filter} ---")
    for d_flux, d_curl, label, etype in zip(datasets_flux, datasets_curl, labels, edge_types):
        if etype != etype_filter:
            continue
        iqr_flux = np.percentile(d_flux, 75) - np.percentile(d_flux, 25)
        iqr_curl = np.percentile(d_curl, 75) - np.percentile(d_curl, 25)
        print(f"  {label}:")
        print(f"    flux: median={np.median(d_flux):.2f}%  IQR={iqr_flux:.2f}%")
        print(f"    curl: median={np.median(d_curl):.2f}%  IQR={iqr_curl:.2f}%")

# -----------------------------------------------------------------------
# Plot 2: percentile band — AC and ZZ as separate lines, flux and curl
# -----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for ax, datasets, title in [
    (ax1, datasets_flux, "Flux error — AC vs ZZ"),
    (ax2, datasets_curl, "Curl error — AC vs ZZ"),
]:
    for etype_filter, color in color_map.items():
        mask    = [e == etype_filter for e in edge_types]
        d_sub   = [d for d, m in zip(datasets, mask) if m]
        pos_sub = [p for p, m in zip(positions, mask) if m]

        medians = [np.median(d) for d in d_sub]
        p25     = [np.percentile(d, 25) for d in d_sub]
        p75     = [np.percentile(d, 75) for d in d_sub]
        p95     = [np.percentile(d, 95) for d in d_sub]

        ax.fill_between(pos_sub, p25, p75, alpha=0.3, color=color)
        ax.fill_between(pos_sub, p75, p95, alpha=0.15, color=color)
        ax.plot(pos_sub, medians, 'o-', color=color, label=f'{etype_filter} median', lw=2)

    legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Armchair'),
                       Patch(facecolor='tomato',    alpha=0.7, label='Zigzag')]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Mean relative error (%)")
    ax.set_title(title)

plt.tight_layout()
plt.show()