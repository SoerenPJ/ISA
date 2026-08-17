"""
violin_flux_compute.py  —  vectorised version
==============================================
All time steps processed in a single batched matrix multiply per quantity.
No subsampling — full time resolution retained.

Speedup vs loop version: ~20-50x depending on N_t and N_hex.

Quantities saved to cache:
  eps(t)      [%]    symmetric relative discrepancy between L1 and L2
  phi_mean(t) [a.u.] mean absolute flux magnitude (signal both methods describe)

Definitions:
  Phi_L1(t) = <B_ind_z^L1>_corners * S_hex          (Eq. 19, Biot-Savart)
  Phi_L2(t) = sum_hex_edges 0.5*(Ax_i+Ax_j)*dx_ij   (Eq. 24+27, loop integral)
              with A reconstructed from L2 J_bond currents via Eq. 24

  eps(t)      = mean_hex(|Phi_L1-Phi_L2|) / mean_hex(0.5*(|Phi_L1|+|Phi_L2|))
  phi_mean(t) = mean_hex(0.5*(|Phi_L1|+|Phi_L2|))
"""

import os, re
import numpy as np
from scipy.spatial import cKDTree

# ============================================================
#  USER SETTINGS
# ============================================================

STRUCTURES = [
    {"sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_bowtie_10x10_rot90",   "label": "AC bowtie 10x10"},
    {"sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_bowtie_15x15_rot0",     "label": "ZZ bowtie 15x15"},
    {"sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_armchair_triangle_14x14_rot90","label": "AC triangle 14x14"},
    {"sweep_dir": "/home/soeren/University/masters/2.semester/ISA/data_LLM/sweep_data_mu_zigzag_triangle_22x22_rot0",   "label": "ZZ triangle 22x22"},
]

CACHE_FILE = "violin_flux_cache.npz"
A_CC_AU    = 2.6825
CUTOFF_FAC = 1.05
ALPHA      = 1.0 / 137.036
MU0_4PI    = ALPHA**2

# ============================================================
#  LATTICE / HEXAGON HELPERS
# ============================================================

def find_lattice(start_dir):
    start_dir = os.path.abspath(start_dir)
    for root in [start_dir] + [os.path.join(start_dir, n)
                                for n in os.listdir(start_dir)
                                if os.path.isdir(os.path.join(start_dir, n))]:
        c = os.path.join(root, "lattice_points.txt")
        if os.path.isfile(c): return c
    d = os.path.dirname(start_dir)
    while True:
        c = os.path.join(d, "lattice_points.txt")
        if os.path.isfile(c): return c
        p = os.path.dirname(d)
        if p == d: break
        d = p
    raise FileNotFoundError(f"lattice_points.txt not found from {start_dir}")


def find_mu_dirs_pair(sweep_dir):
    pat1 = re.compile(r"^L1_mu_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
    pat2 = re.compile(r"^L2_mu_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
    mu_L1, mu_L2 = {}, {}
    for name in os.listdir(sweep_dir):
        m1 = pat1.match(name); m2 = pat2.match(name)
        if m1: mu_L1[float(m1.group(1))] = os.path.join(sweep_dir, name)
        if m2: mu_L2[float(m2.group(1))] = os.path.join(sweep_dir, name)
    common = sorted(set(mu_L1) & set(mu_L2))
    return [(mu, mu_L1[mu], mu_L2[mu]) for mu in common]


def build_nn_bonds(x, y, cutoff):
    tree = cKDTree(np.column_stack([x, y]))
    return np.array(sorted(tree.query_pairs(r=cutoff)), dtype=int)


def neighbor_lists(n, bonds):
    neigh = [[] for _ in range(n)]
    for i, j in bonds:
        neigh[i].append(j); neigh[j].append(i)
    return neigh


def find_hexagons(bonds, neigh):
    bond_set = {(min(a,b), max(a,b)) for a,b in bonds}
    hexagons, seen = [], set()
    def dfs(start, current, path):
        depth = len(path)
        if depth == 6:
            if (min(current,start), max(current,start)) in bond_set:
                key = tuple(sorted(path))
                if key not in seen:
                    seen.add(key); hexagons.append(list(path))
            return
        prev = path[-1] if depth >= 1 else -1
        for nb in neigh[current]:
            if nb == prev: continue
            if nb == start and depth < 5: continue
            if nb != start and nb in path: continue
            dfs(start, nb, path + [nb])
    for start in range(len(neigh)):
        dfs(start, start, [start])
    return hexagons


def order_ccw(verts, x, y):
    cx, cy = np.mean(x[verts]), np.mean(y[verts])
    return [verts[i] for i in np.argsort(np.arctan2(y[verts]-cy, x[verts]-cx))]


def hexagon_area(verts_ccw, x, y):
    xs, ys = x[verts_ccw], y[verts_ccw]
    return 0.5*float(np.sum(xs*np.roll(ys,-1) - np.roll(xs,-1)*ys))


# ============================================================
#  PRECOMPUTE GEOMETRY MATRICES  (done once per structure)
# ============================================================

def build_geometry_matrices(hexagons, x, y, bonds):
    """
    Build three sparse-dense matrices that turn time series into fluxes
    via a single matmul — no per-time-step Python loop needed.

    Returns
    -------
    M_B   : (n_hex, n_sites)   M_B @ B_z[t]   = Phi_L1[t]
    M_Ax  : (n_hex, n_sites)   }  combined:
    M_Ay  : (n_hex, n_sites)   }  M_Ax @ Ax[t] + M_Ay @ Ay[t] = Phi_L2[t]
    W_A   : (n_sites, n_bonds) W_A @ J[t] = Ax[t]  (and similarly for Ay)
    W_Ay  : (n_sites, n_bonds)
    """
    n_hex   = len(hexagons)
    n_sites = len(x)
    n_bonds = len(bonds)

    # ── M_B: flux from B  (Phi_L1 = M_B @ B_z) ──────────────────────────
    # Each hexagon h gets weight area_h / 6 on each of its 6 corner sites.
    M_B = np.zeros((n_hex, n_sites))
    for h, verts in enumerate(hexagons):
        vcc  = order_ccw(verts, x, y)
        area = hexagon_area(vcc, x, y)
        for v in verts:
            M_B[h, v] += area / 6.0   # mean over 6 corners × area

    # ── Loop-integral weights: Phi_L2 = sum_edges 0.5*(Ax_i+Ax_j)*dx_ij ─
    # Phi_L2[h] = M_Ax[h,:] @ Ax + M_Ay[h,:] @ Ay
    # where M_Ax[h, i] = sum over edges of hex h that touch site i of
    #                     0.5 * (x[j] - x[i])   (trapezoidal rule contribution)
    M_Ax = np.zeros((n_hex, n_sites))
    M_Ay = np.zeros((n_hex, n_sites))
    for h, verts in enumerate(hexagons):
        vcc = order_ccw(verts, x, y)
        n_v = len(vcc)
        for e in range(n_v):
            i = vcc[e]; j = vcc[(e+1) % n_v]
            dx = x[j] - x[i]; dy = y[j] - y[i]
            # contribution from Ax[i] and Ax[j]:  0.5*(Ax_i + Ax_j)*dx
            M_Ax[h, i] += 0.5 * dx
            M_Ax[h, j] += 0.5 * dx
            # contribution from Ay[i] and Ay[j]:  0.5*(Ay_i + Ay_j)*dy
            M_Ay[h, i] += 0.5 * dy
            M_Ay[h, j] += 0.5 * dy

    # ── W_Ax, W_Ay: reconstruct A from J  (Eq. 24) ───────────────────────
    # Ax[m] = MU0_4PI * sum_b J_b * lx_b / |r_m - r_bar_b|
    # So Ax = W_Ax @ J_bond,  W_Ax[m, b] = MU0_4PI * lx_b / |r_m - r_bar_b|
    bi = bonds[:, 0]; bj = bonds[:, 1]
    xc = 0.5*(x[bi]+x[bj]); yc = 0.5*(y[bi]+y[bj])
    lx = x[bi]-x[bj];       ly = y[bi]-y[bj]

    dx_mat = x[:, None] - xc[None, :]   # (n_sites, n_bonds)
    dy_mat = y[:, None] - yc[None, :]
    r_mat  = np.sqrt(dx_mat**2 + dy_mat**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_r = np.where(r_mat < 1e-10, 0.0, 1.0/r_mat)

    W_Ax = MU0_4PI * inv_r * lx[None, :]   # (n_sites, n_bonds)
    W_Ay = MU0_4PI * inv_r * ly[None, :]

    # ── Combine: Phi_L2 = (M_Ax @ W_Ax + M_Ay @ W_Ay) @ J_bond ──────────
    # Precompute the combined (n_hex, n_bonds) matrix once:
    M_J = M_Ax @ W_Ax + M_Ay @ W_Ay   # (n_hex, n_bonds)

    return M_B, M_J


# ============================================================
#  VECTORISED CORE COMPUTATION
# ============================================================

def compute_quantities(L1_dir, L2_dir, M_B, M_J):
    """
    Parameters
    ----------
    M_B : (n_hex, n_sites)  maps B_z[t] -> Phi_L1[t]
    M_J : (n_hex, n_bonds)  maps J[t]   -> Phi_L2[t]

    Returns
    -------
    eps      [%]    symmetric relative discrepancy, one value per time step
    phi_mean [a.u.] mean absolute flux magnitude,   one value per time step
    """
    # ── Load data ─────────────────────────────────────────────────────────
    p_B = os.path.join(L1_dir, "B_ind_z_time_evolution.txt")
    if not os.path.isfile(p_B):
        raise FileNotFoundError(f"Missing: {p_B}")
    B_L1 = np.loadtxt(p_B)[:, 1:]   # (N_t, n_sites)

    p_J = os.path.join(L2_dir, "J_bond_sc_time_evolution.txt")
    if not os.path.isfile(p_J):
        p_J = os.path.join(L2_dir, "J_bond_time_evolution.txt")
    if not os.path.isfile(p_J):
        raise FileNotFoundError(f"Missing J_bond in {L2_dir}")
    J_L2 = np.loadtxt(p_J)[:, 1:]   # (N_t, n_bonds)

    N_t  = min(B_L1.shape[0], J_L2.shape[0])
    B_L1 = B_L1[:N_t]
    J_L2 = J_L2[:N_t]
    print(f"    {N_t} steps — computing fluxes (vectorised) ...", flush=True)

    # ── Batched matmul: (N_t, n_sites) @ M_B.T = (N_t, n_hex) ───────────
    # Phi_L1[t, h] = sum_m M_B[h, m] * B_L1[t, m]
    Phi_L1 = B_L1 @ M_B.T   # (N_t, n_hex)
    Phi_L2 = J_L2 @ M_J.T   # (N_t, n_hex)  — full pipeline in one shot
    print(f"    Flux matrices computed.", flush=True)

    # ── Symmetric discrepancy ─────────────────────────────────────────────
    sym_denom = 0.5 * (np.abs(Phi_L1) + np.abs(Phi_L2))   # (N_t, n_hex)

    peak_signal  = sym_denom.max()
    threshold    = 0.01 * peak_signal
    print(f"    Phi_L1[100, :5] = {Phi_L1[100, :5]}")
    print(f"    Phi_L2[100, :5] = {Phi_L2[100, :5]}")
    print(f"    ratio sample    = {Phi_L1[100, :5] / (Phi_L2[100, :5] + 1e-30)}")
    # Mask near-zero hexagons; compute element-wise relative discrepancy
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(sym_denom > threshold,
                       np.abs(Phi_L1 - Phi_L2) / sym_denom,
                       np.nan)

    # Mean over hexagons per time step; skip first 2 transient rows
    eps_raw = np.nanmean(rel,       axis=1)[2:] * 100   # [%]
    phi_raw = np.nanmean(sym_denom, axis=1)[2:]          # [a.u.]

    valid = np.isfinite(eps_raw) & np.isfinite(phi_raw) & (eps_raw < 10000)
    return eps_raw[valid], phi_raw[valid]


# ============================================================
#  MAIN LOOP
# ============================================================
results_eps = {}
results_phi = {}

for struct in STRUCTURES:
    sweep_dir = struct["sweep_dir"]
    label     = struct["label"]
    print(f"\n{'='*60}\n{label}")

    mu_list = find_mu_dirs_pair(sweep_dir)
    if len(mu_list) < 2:
        print("  [WARNING] < 2 paired mu dirs, skipping."); continue

    trio = [mu_list[0], mu_list[len(mu_list)//2], mu_list[-1]]

    # ── Build lattice + hexagons once per structure ───────────────────────
    try:    lat_path = find_lattice(trio[0][2])
    except: lat_path = find_lattice(trio[0][1])

    lattice  = np.loadtxt(lat_path, comments="#")
    x, y     = lattice[:, 0], lattice[:, 1]
    bonds    = build_nn_bonds(x, y, A_CC_AU * CUTOFF_FAC)
    neigh    = neighbor_lists(len(x), bonds)
    hexagons = find_hexagons(bonds, neigh)
    print(f"  {len(x)} sites | {len(bonds)} bonds | {len(hexagons)} hexagons")

    # ── Precompute geometry matrices once ─────────────────────────────────
    print("  Building geometry matrices ...", flush=True)
    M_B, M_J = build_geometry_matrices(hexagons, x, y, bonds)
    print(f"  M_B: {M_B.shape}   M_J: {M_J.shape}", flush=True)

    # ── Loop over mu values ───────────────────────────────────────────────
    for mu_val, L1_dir, L2_dir in trio:
        key = f"{label}__mu_{mu_val:.4f}"
        print(f"\n  mu = {mu_val:.2f} eV", flush=True)
        try:
            eps, phi = compute_quantities(L1_dir, L2_dir, M_B, M_J)
            results_eps[key] = eps
            results_phi[key] = phi
            print(f"    eps  median={np.median(eps):.2f}%   "
                  f"phi  median={np.median(phi):.4e} a.u.")
        except FileNotFoundError as e:
            print(f"    SKIP: {e}")

# ============================================================
#  SAVE CACHE
# ============================================================
keys = list(results_eps.keys())
ea   = np.empty(len(keys), dtype=object)
pa   = np.empty(len(keys), dtype=object)
for i, k in enumerate(keys):
    ea[i] = results_eps[k]
    pa[i] = results_phi[k]

np.savez(CACHE_FILE, keys=np.array(keys), eps_arrays=ea, phi_arrays=pa)
print(f"\n{'='*60}")
print(f"Saved {CACHE_FILE}  ({len(keys)} entries)")