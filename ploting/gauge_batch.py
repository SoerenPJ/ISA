"""
gauge_batch.py — single-simulation gauge consistency metrics, stdout only.

Usage:
    python3 ploting/gauge_batch.py <sim_dir_or_config.toml>

Prints one space-separated line:
    corr_flux  alpha_flux  mean_rel_peak  max_rel_peak  mean_ratio_peak
    rms_flux_peak  dynamic_range  mean_rel_curl_peak

On any error prints:  nan nan nan nan nan nan nan nan
Intended to be called from gauge_sweep.sh for each mu step.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

AU_NM = 0.0529177  # Bohr radius in nm

NAN_LINE = "nan nan nan nan nan nan nan nan"


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _fnv1a_64(data: bytes) -> int:
    h = 14695981039346656037
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _resolve_sim_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p
    # Try treating it as a config file → derive hash-based sim dir
    if not p.exists():
        p2 = Path(str(p) + ".toml")
        if p2.exists():
            p = p2
    if p.is_file():
        h = _fnv1a_64(p.read_bytes())
        d = Path("Simulations") / f"{p.stem}_{h:x}"
        return d
    raise FileNotFoundError(f"Cannot resolve sim dir from: {arg}")


def _load_toml_lattice_const_au(sim_dir: Path) -> float:
    p = sim_dir / "input.toml"
    val_nm = 0.1421
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.split("#", 1)[0].strip()
            if s.startswith("lattice_const"):
                parts = s.split("=", 1)
                if len(parts) == 2:
                    try:
                        val_nm = float(parts[1].strip())
                    except ValueError:
                        pass
                    break
    return val_nm / AU_NM


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_lattice(sim_dir: Path):
    path = sim_dir / "lattice_points.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        return data, np.zeros_like(data)
    return data[:, 0], data[:, 1]


def _load_B_ind(sim_dir: Path):
    path = sim_dir / "B_ind_z_time_evolution.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} — run with zeeman_induced=true, spin_on=true."
        )
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:]


def _load_A_ind(sim_dir: Path):
    path = sim_dir / "A_ind_time_evolution.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} — run with self_consistent_phase=true."
        )
    data = np.loadtxt(path)
    t    = data[:, 0]
    rest = data[:, 1:]
    Ax   = rest[:, 0::2]   # shape (n_t, n_sites)
    Ay   = rest[:, 1::2]
    return t, Ax, Ay


# ---------------------------------------------------------------------------
# Lattice topology
# ---------------------------------------------------------------------------

def _build_nn_bonds(x: np.ndarray, y: np.ndarray, cutoff: float) -> List[Tuple[int, int]]:
    n  = x.size
    c2 = cutoff * cutoff
    bonds: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx * dx + dy * dy <= c2:
                bonds.append((i, j))
    return bonds


def _neighbor_lists(n: int, bonds: List[Tuple[int, int]]) -> List[List[int]]:
    neigh: List[List[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        neigh[i].append(j)
        neigh[j].append(i)
    return neigh


def _find_hexagons(bonds, neigh) -> List[List[int]]:
    bond_set = {(min(a, b), max(a, b)) for a, b in bonds}
    hexagons: List[List[int]] = []
    seen: set = set()

    def dfs(start: int, current: int, path: List[int]) -> None:
        depth = len(path)
        if depth == 6:
            if (min(current, start), max(current, start)) in bond_set:
                key = tuple(sorted(path))
                if key not in seen:
                    seen.add(key)
                    hexagons.append(list(path))
            return
        prev = path[-1] if depth >= 1 else -1
        for nb in neigh[current]:
            if nb == prev:
                continue
            if nb == start and depth < 5:
                continue
            if nb != start and nb in path:
                continue
            dfs(start, nb, path + [nb])

    for start in range(len(neigh)):
        dfs(start, start, [start])
    return hexagons


def _order_ccw(verts: List[int], x: np.ndarray, y: np.ndarray) -> List[int]:
    cx = np.mean(x[verts])
    cy = np.mean(y[verts])
    angles = np.arctan2(y[verts] - cy, x[verts] - cx)
    return [verts[i] for i in np.argsort(angles)]


def _hexagon_area(verts_ccw: List[int], x: np.ndarray, y: np.ndarray) -> float:
    xs = x[verts_ccw]
    ys = y[verts_ccw]
    return 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))


# ---------------------------------------------------------------------------
# Flux computation
# ---------------------------------------------------------------------------

def _flux_from_B(hexagons, x, y, B_z) -> np.ndarray:
    phi = np.zeros(len(hexagons))
    for h, verts in enumerate(hexagons):
        verts_ccw = _order_ccw(verts, x, y)
        area = _hexagon_area(verts_ccw, x, y)
        phi[h] = np.mean(B_z[verts]) * area
    return phi


def _flux_from_A(hexagons, x, y, Ax, Ay) -> np.ndarray:
    phi = np.zeros(len(hexagons))
    for h, verts in enumerate(hexagons):
        verts_ccw = _order_ccw(verts, x, y)
        n_v   = len(verts_ccw)
        flux  = 0.0
        for e in range(n_v):
            i   = verts_ccw[e]
            j   = verts_ccw[(e + 1) % n_v]
            amx = 0.5 * (Ax[i] + Ax[j])
            amy = 0.5 * (Ay[i] + Ay[j])
            flux += amx * (x[j] - x[i]) + amy * (y[j] - y[i])
        phi[h] = flux
    return phi


def _curl_from_A(hexagons, x, y, Ax, Ay) -> np.ndarray:
    curl = np.zeros(len(hexagons))
    for h, verts in enumerate(hexagons):
        verts_ccw = _order_ccw(verts, x, y)
        area  = _hexagon_area(verts_ccw, x, y)
        n_v   = len(verts_ccw)
        flux  = 0.0
        for e in range(n_v):
            i   = verts_ccw[e]
            j   = verts_ccw[(e + 1) % n_v]
            amx = 0.5 * (Ax[i] + Ax[j])
            amy = 0.5 * (Ay[i] + Ay[j])
            flux += amx * (x[j] - x[i]) + amy * (y[j] - y[i])
        curl[h] = flux / area
    return curl


def _B_avg_from_B(hexagons, x, y, B_z) -> np.ndarray:
    B_avg = np.zeros(len(hexagons))
    for h, verts in enumerate(hexagons):
        B_avg[h] = np.mean(B_z[verts])
    return B_avg


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(sim_dir: Path) -> str:
    """
    Load simulation outputs, compute gauge consistency metrics at peak signal,
    and return a space-separated string of 8 scalar values suitable for a sweep
    output table.

    Columns:
        corr_flux  alpha_flux  mean_rel_peak  max_rel_peak  mean_ratio_peak
        rms_flux_peak  dynamic_range  mean_rel_curl_peak
    """
    a_au    = _load_toml_lattice_const_au(sim_dir)
    cutoff  = a_au * 1.05
    x, y    = _load_lattice(sim_dir)
    n       = x.size
    bonds   = _build_nn_bonds(x, y, cutoff)
    neigh   = _neighbor_lists(n, bonds)
    hexagons = _find_hexagons(bonds, neigh)

    if len(hexagons) == 0:
        raise RuntimeError("No hexagons found — not a honeycomb lattice?")

    tB, B      = _load_B_ind(sim_dir)
    tA, Ax, Ay = _load_A_ind(sim_dir)

    if B.shape[1] != n or Ax.shape[1] != n:
        raise ValueError("Site count mismatch between lattice and field files.")
    if len(tA) != len(tB) or not np.allclose(tA, tB, atol=1e-10):
        raise ValueError("Time grids differ between A_ind and B_ind files.")

    n_t   = len(tA)
    n_hex = len(hexagons)

    phi_B  = np.zeros((n_t, n_hex))
    phi_A  = np.zeros((n_t, n_hex))
    curl_A = np.zeros((n_t, n_hex))
    B_avg  = np.zeros((n_t, n_hex))

    for k in range(n_t):
        phi_B[k]  = _flux_from_B(hexagons, x, y, B[k])
        phi_A[k]  = _flux_from_A(hexagons, x, y, Ax[k], Ay[k])
        curl_A[k] = _curl_from_A(hexagons, x, y, Ax[k], Ay[k])
        B_avg[k]  = _B_avg_from_B(hexagons, x, y, B[k])

    # --- Global flux metrics ---
    pB_flat   = phi_B.flatten()
    pA_flat   = phi_A.flatten()
    corr_flux = float(np.corrcoef(pB_flat, pA_flat)[0, 1])
    alpha_flux = float(np.dot(pB_flat, pA_flat) / (np.dot(pB_flat, pB_flat) + 1e-300))

    scale_flux   = np.max(np.abs(phi_B), axis=1)  # peak |Phi_B| per timestep
    global_scale = float(np.max(np.abs(phi_B)))

    # Pick peak-signal timestep
    k_peak = int(np.argmax(scale_flux))

    # Relative flux error at peak
    threshold  = 1e-3 * global_scale
    mask_peak  = np.abs(phi_B[k_peak]) > threshold
    if mask_peak.any():
        rel_err_peak = np.abs(phi_A[k_peak, mask_peak] - phi_B[k_peak, mask_peak]) \
                       / np.abs(phi_B[k_peak, mask_peak])
        mean_rel_peak = float(np.mean(rel_err_peak))
        max_rel_peak  = float(np.max(rel_err_peak))
        ratio_peak    = phi_A[k_peak, mask_peak] / phi_B[k_peak, mask_peak]
        mean_ratio_peak = float(np.mean(ratio_peak))
    else:
        mean_rel_peak = np.nan
        max_rel_peak  = np.nan
        mean_ratio_peak = np.nan

    rms_flux_peak = float(np.sqrt(np.mean((phi_B[k_peak] - phi_A[k_peak]) ** 2)))

    # Dynamic range: peak signal / late-time noise floor
    noise_floor   = float(np.median(scale_flux[n_t * 3 // 4:]))
    dynamic_range = float(global_scale / (noise_floor + 1e-300))

    # Curl relative error at peak
    global_scale_curl = float(np.max(np.abs(B_avg)))
    threshold_curl    = 1e-3 * global_scale_curl
    mask_curl         = np.abs(B_avg[k_peak]) > threshold_curl
    if mask_curl.any():
        rel_curl_peak      = np.abs(curl_A[k_peak, mask_curl] - B_avg[k_peak, mask_curl]) \
                             / np.abs(B_avg[k_peak, mask_curl])
        mean_rel_curl_peak = float(np.nanmean(rel_curl_peak))
    else:
        mean_rel_curl_peak = np.nan

    return (
        f"{corr_flux:.6e} {alpha_flux:.6e} {mean_rel_peak:.6e} {max_rel_peak:.6e} "
        f"{mean_ratio_peak:.6e} {rms_flux_peak:.6e} {dynamic_range:.6e} "
        f"{mean_rel_curl_peak:.6e}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <sim_dir_or_config.toml>", file=sys.stderr)
        print(NAN_LINE)
        sys.exit(0)

    try:
        sim_dir = _resolve_sim_dir(sys.argv[1])
        line    = compute_metrics(sim_dir)
        print(line)
    except Exception as e:
        print(f"gauge_batch error: {e}", file=sys.stderr)
        print(NAN_LINE)
        sys.exit(0)


if __name__ == "__main__":
    main()
