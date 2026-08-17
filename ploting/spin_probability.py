import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

AU_EV       = 27.2114
AU_NM       = 0.0529177
EDGE_TOL_EV = 0.3
MU_EV       = 3.52

STRUCTURES = [
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_bowtie_10x10_rot90",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_bowtie_15x15_rot0",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_triangle_14x14_rot90",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_triangle_22x22_rot0",
]
TITLES = ["AC Bowtie", "ZZ Bowtie", "AC Triangle", "ZZ Triangle"]

def find_lattice(d):
    c = os.path.join(d, "lattice_points.txt")
    if os.path.isfile(c): return c
    for n in os.listdir(d):
        ch = os.path.join(d, n)
        if os.path.isdir(ch):
            c = os.path.join(ch, "lattice_points.txt")
            if os.path.isfile(c): return c
    raise FileNotFoundError

def find_bulk_gap(e):
    e = np.sort(e)
    bulk = e[np.abs(e) > EDGE_TOL_EV]
    bn = bulk[bulk < 0]; bp = bulk[bulk > 0]
    if len(bn) > 0 and len(bp) > 0: return bn.max(), bp.min()
    N = len(e); return e[N//2-1], e[N//2]

fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
fig.suptitle(f"Probability amplitude — spin $\\uparrow$ (blue) vs spin $\\downarrow$ (red)  |  $\\mu={MU_EV}$ eV", fontsize=13)

for ax, structure, title in zip(axes, STRUCTURES, TITLES):
    lattice_path = find_lattice(structure)
    base_dir     = os.path.dirname(lattice_path)
    coords       = np.loadtxt(lattice_path, comments="#")[:, :2] * AU_NM
    N_atoms      = len(coords)
    N_sites      = N_atoms // 2
    x, y         = coords[:N_sites, 0], coords[:N_sites, 1]

    raw     = np.loadtxt(os.path.join(base_dir, "eigenvalues.txt"))
    eigs_eV = (raw if raw.ndim == 1 else raw[:, 0]) * AU_EV
    gb, gt  = find_bulk_gap(np.sort(eigs_eV))
    eigs_eV -= 0.5*(gb+gt)

    raw_vecs = np.loadtxt(os.path.join(base_dir, "eigenvectors.txt"))
    flat     = raw_vecs[:,0] + 1j*raw_vecs[:,1]
    vec_len  = N_atoms
    n_vecs   = len(flat) // vec_len
    vecs     = flat[:n_vecs*vec_len].reshape(n_vecs, vec_len).T

    idx     = np.argsort(eigs_eV)
    eigs_eV = eigs_eV[idx]
    vecs    = vecs[:, idx]

    occ_mask  = eigs_eV <= MU_EV
    prob_up   = np.sum(np.abs(vecs[:N_sites, occ_mask])**2, axis=1)
    prob_down = np.sum(np.abs(vecs[N_sites:, occ_mask])**2, axis=1)

    vmax  = max(prob_up.max(), prob_down.max())
    a_up   = np.clip(prob_up   / vmax, 0, 1)
    a_down = np.clip(prob_down / vmax, 0, 1)

    ax.scatter(x, y, c='#cccccc', s=15, linewidths=0, zorder=1)
    ax.scatter(x, y, c='#cc0000', s=25, alpha=a_down, linewidths=0, zorder=2)
    ax.scatter(x, y, c='#003399', s=25, alpha=a_up,   linewidths=0, zorder=3)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12)

handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor='#003399',
                  markersize=10, label=r'spin $\uparrow$'),
           Line2D([0],[0], marker='o', color='w', markerfacecolor='#cc0000',
                  markersize=10, label=r'spin $\downarrow$')]
axes[-1].legend(handles=handles, loc='lower right', fontsize=11, framealpha=0.9)

plt.savefig("prob_amplitude.png", dpi=150, bbox_inches='tight')
print("Saved: prob_amplitude.png")
plt.show()