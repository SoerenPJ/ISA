import os
import numpy as np

AU_EV = 27.2114
EDGE_TOL_EV = 0.3

STRUCTURES = [
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_bowtie_10x10_rot90",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_bowtie_15x15_rot0",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_triangle_14x14_rot90",
    "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_triangle_22x22_rot0",
]

MU_CASES = [ 3.52]

def find_lattice(structure_dir):
    candidate = os.path.join(structure_dir, "lattice_points.txt")
    if os.path.isfile(candidate):
        return candidate
    for name in os.listdir(structure_dir):
        child = os.path.join(structure_dir, name)
        if os.path.isdir(child):
            candidate = os.path.join(child, "lattice_points.txt")
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError

def find_bulk_gap(energies_eV):
    e = np.sort(energies_eV)
    bulk = e[np.abs(e) > EDGE_TOL_EV]
    bulk_neg = bulk[bulk < 0]
    bulk_pos = bulk[bulk > 0]
    if len(bulk_neg) > 0 and len(bulk_pos) > 0:
        return bulk_neg.max(), bulk_pos.min()
    N = len(e)
    return e[N // 2 - 1], e[N // 2]

def load_eigenvalues(structure_dir):
    lattice_path = find_lattice(structure_dir)
    eig_path = os.path.join(os.path.dirname(lattice_path), "eigenvalues.txt")
    raw = np.loadtxt(eig_path)
    energies_au = raw if raw.ndim == 1 else raw[:, 0]
    energies_eV = np.sort(energies_au * AU_EV)
    gap_bot, gap_top = find_bulk_gap(energies_eV)
    E_fermi = 0.5 * (gap_bot + gap_top)
    energies_eV -= E_fermi
    n_edge = np.sum(np.abs(energies_eV) <= EDGE_TOL_EV)
    return energies_eV, n_edge

data = {}
for structure in STRUCTURES:
    label = os.path.basename(structure).replace("sweep_data_mu_", "")
    energies_eV, n_edge = load_eigenvalues(structure)
    data[label] = (energies_eV, n_edge)

pairs = [
    ("armchair_bowtie_10x10_rot90",   "zigzag_bowtie_15x15_rot0",   "Bowtie"),
    ("armchair_triangle_14x14_rot90", "zigzag_triangle_22x22_rot0", "Triangle"),
]

for ac_key, zz_key, geom in pairs:
    ac_eig, ac_edge = data[ac_key]
    zz_eig, zz_edge = data[zz_key]
    print("=" * 72)
    print(f"{geom}  (AC total={len(ac_eig)}, ZZ total={len(zz_eig)}, ZZ edge states={zz_edge})")
    print("=" * 72)
    print(f"  {'':35} {'Armchair':>10} {'Zigzag':>10} {'ZZ-AC':>10}")
    print(f"  {'Edge states':35} {ac_edge:>10} {zz_edge:>10} {zz_edge-ac_edge:>+10}")
    print()
    for mu in MU_CASES:
        ac_occ  = np.sum(ac_eig <= mu)
        zz_occ  = np.sum(zz_eig <= mu)
        ac_pct  = 100 * ac_occ / len(ac_eig)
        zz_pct  = 100 * zz_occ / len(zz_eig)
        dpct    = zz_pct - ac_pct

        # highest occupied energy level (state just at or below mu)
        ac_homo = ac_eig[ac_eig <= mu].max() if np.any(ac_eig <= mu) else float('nan')
        zz_homo = zz_eig[zz_eig <= mu].max() if np.any(zz_eig <= mu) else float('nan')

        # lowest unoccupied energy level (state just above mu)
        ac_lumo = ac_eig[ac_eig > mu].min() if np.any(ac_eig > mu) else float('nan')
        zz_lumo = zz_eig[zz_eig > mu].min() if np.any(zz_eig > mu) else float('nan')

        print(f"  mu = {mu:.1f} eV")
        print(f"    {'% occupied':<31} {ac_pct:>9.2f}% {zz_pct:>9.2f}% {dpct:>+9.2f}%")
        print(f"    {'Highest occ. level (HOMO) eV':<31} {ac_homo:>10.4f} {zz_homo:>10.4f} {zz_homo-ac_homo:>+10.4f}")
        print(f"    {'Lowest unocc. level (LUMO) eV':<31} {ac_lumo:>10.4f} {zz_lumo:>10.4f} {zz_lumo-ac_lumo:>+10.4f}")
        print(f"    {'HOMO-LUMO gap eV':<31} {ac_lumo-ac_homo:>10.4f} {zz_lumo-zz_homo:>10.4f} {(zz_lumo-zz_homo)-(ac_lumo-ac_homo):>+10.4f}")
        print()