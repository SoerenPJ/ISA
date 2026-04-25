from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

AU_NM = 0.0529177   # Bohr radius in nm


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def fnv1a_64(data: bytes) -> int:
    h = 14695981039346656037
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def resolve_config_path(arg: str) -> Path:
    p = Path(arg)
    if p.exists() and p.is_file():
        return p
    p2 = Path(str(p) + ".toml")
    if p2.exists() and p2.is_file():
        return p2
    raise FileNotFoundError(f"Could not find config file: {arg}")


def simulation_dir_from_config(cfg_path: Path) -> Path:
    h = fnv1a_64(cfg_path.read_bytes())
    return Path("Simulations") / f"{cfg_path.stem}_{h:x}"


def load_toml_lattice_const_au(sim_dir: Path) -> float:
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
# Lattice topology
# ---------------------------------------------------------------------------

def build_nn_bonds(x: np.ndarray, y: np.ndarray, cutoff: float) -> List[Tuple[int, int]]:
    n = x.size
    c2 = cutoff * cutoff
    bonds: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx * dx + dy * dy <= c2:
                bonds.append((i, j))
    return bonds


def neighbor_lists(n: int, bonds: List[Tuple[int, int]]) -> List[List[int]]:
    neigh: List[List[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        neigh[i].append(j)
        neigh[j].append(i)
    return neigh


def find_hexagons(
    bonds: List[Tuple[int, int]],
    neigh: List[List[int]],
) -> List[List[int]]:
    """
    Find all elementary hexagonal plaquettes (6-cycles) in a honeycomb lattice.

    DFS from each site looking for closed paths of length exactly 6 that return
    to the start without revisiting any intermediate node.
    Each hexagon is stored once via canonical sorted-vertex key.
    """
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


def order_ccw(verts: List[int], x: np.ndarray, y: np.ndarray) -> List[int]:
    """Return vertex indices sorted counter-clockwise around their centroid."""
    cx = np.mean(x[verts])
    cy = np.mean(y[verts])
    angles = np.arctan2(y[verts] - cy, x[verts] - cx)
    return [verts[i] for i in np.argsort(angles)]


def hexagon_area(verts_ccw: List[int], x: np.ndarray, y: np.ndarray) -> float:
    """Signed area via shoelace formula (positive for CCW ordering)."""
    xs = x[verts_ccw]
    ys = y[verts_ccw]
    return 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))


# ---------------------------------------------------------------------------
# Flux computation
# ---------------------------------------------------------------------------

def flux_from_B(
    hexagons: List[List[int]],
    x: np.ndarray,
    y: np.ndarray,
    B_z: np.ndarray,
) -> np.ndarray:
    """
    Method A — flux:  Phi_hex = <B_ind,z>_corners * S_hex   (Eq. 19 + surface avg)

    Average B_ind,z (Biot-Savart) over the 6 corner sites and multiply by the
    hexagon area.  This approximates int int B·dS over the plaquette.
    """
    phi = np.zeros(len(hexagons))
    for h, verts in enumerate(hexagons):
        verts_ccw = order_ccw(verts, x, y)
        area = hexagon_area(verts_ccw, x, y)
        phi[h] = np.mean(B_z[verts]) * area
    return phi


def flux_from_A(
    hexagons: List[List[int]],
    x: np.ndarray,
    y: np.ndarray,
    Ax: np.ndarray,
    Ay: np.ndarray,
) -> np.ndarray:
    """
    Method B — flux:  Phi_hex = ∮ A · dl   (Stokes => flux of curl A, Eq. 27)

    Each bond contributes via the trapezoidal rule:
        A_mid · dl  =  0.5*(Ax[i]+Ax[j])*(x[j]-x[i])
                     + 0.5*(Ay[i]+Ay[j])*(y[j]-y[i])
    """
    phi = np.zeros(len(hexagons))
    for h, verts in enumerate(hexagons):
        verts_ccw = order_ccw(verts, x, y)
        n_v = len(verts_ccw)
        flux = 0.0
        for e in range(n_v):
            i = verts_ccw[e]
            j = verts_ccw[(e + 1) % n_v]
            amx = 0.5 * (Ax[i] + Ax[j])
            amy = 0.5 * (Ay[i] + Ay[j])
            dx  = x[j] - x[i]
            dy  = y[j] - y[i]
            flux += amx * dx + amy * dy
        phi[h] = flux
    return phi

def curl_from_A(
    hexagons: List[List[int]],
    x: np.ndarray,
    y: np.ndarray,
    Ax: np.ndarray,
    Ay: np.ndarray,
) -> np.ndarray:
    """
    Eq. 27: B^(curl A) at each hexagon centre, estimated as:
    (∮ A · dl) / S_hex  — i.e. flux divided by area = average curl
    """
    curl = np.zeros(len(hexagons))
    for h, verts in enumerate(hexagons):
        verts_ccw = order_ccw(verts, x, y)
        area = hexagon_area(verts_ccw, x, y)
        n_v = len(verts_ccw)
        flux = 0.0
        for e in range(n_v):
            i = verts_ccw[e]
            j = verts_ccw[(e + 1) % n_v]
            amx = 0.5 * (Ax[i] + Ax[j])
            amy = 0.5 * (Ay[i] + Ay[j])
            dx  = x[j] - x[i]
            dy  = y[j] - y[i]
            flux += amx * dx + amy * dy
        curl[h] = flux / area  # this is B^(curl A) at the hexagon
    return curl

def B_avg_from_B(
    hexagons: List[List[int]],
    x: np.ndarray,
    y: np.ndarray,
    B_z: np.ndarray,
) -> np.ndarray:
    """Average B_ind_z over hexagon corners — the direct Biot-Savart estimate."""
    B_avg = np.zeros(len(hexagons))
    for h, verts in enumerate(hexagons):
        B_avg[h] = np.mean(B_z[verts])
    return B_avg

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_B_ind(sim_dir: Path):
    path = sim_dir / "B_ind_z_time_evolution.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}\nRun with zeeman_induced=true, spin_on=true."
        )
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:]


def load_A_ind(sim_dir: Path):
    path = sim_dir / "A_ind_time_evolution.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}\nRun with self_consistent_phase=true."
        )
    data = np.loadtxt(path)
    t    = data[:, 0]
    rest = data[:, 1:]
    Ax   = rest[:, 0::2]                 # shape (n_t, n_sites)
    Ay   = rest[:, 1::2]
    return t, Ax, Ay


def load_lattice(sim_dir: Path):
    path = sim_dir / "lattice_points.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        return data, np.zeros_like(data)
    return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(sim_dir: Path) -> None:
    sim_dir = Path(sim_dir).resolve()
    out_dir = sim_dir / "gauge_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lattice
    a_au     = load_toml_lattice_const_au(sim_dir)
    cutoff   = a_au * 1.05
    x, y     = load_lattice(sim_dir)
    n        = x.size
    bonds    = build_nn_bonds(x, y, cutoff)
    neigh    = neighbor_lists(n, bonds)
    hexagons = find_hexagons(bonds, neigh)

    print(f"Lattice : {n} sites | {len(bonds)} bonds | {len(hexagons)} hexagons")
    if len(hexagons) == 0:
        raise RuntimeError(
            "No hexagons found — check lattice_points.txt and bond cutoff."
        )

    # Time series
    tB, B      = load_B_ind(sim_dir)
    tA, Ax, Ay = load_A_ind(sim_dir)

    if B.shape[1] != n or Ax.shape[1] != n:
        raise ValueError(
            f"Site count mismatch: lattice={n}, B cols={B.shape[1]}, A cols={Ax.shape[1]}"
        )
    if len(tA) != len(tB) or not np.allclose(tA, tB, atol=1e-10):
        raise ValueError("Time grids differ between A_ind and B_ind files.")
    t     = tA
    n_t   = len(t)
    n_hex = len(hexagons)

    # Compute flux stacks  shape (n_t, n_hex)
    phi_B = np.zeros((n_t, n_hex))
    phi_A = np.zeros((n_t, n_hex))
    for k in range(n_t):
        phi_B[k] = flux_from_B(hexagons, x, y, B[k])
        phi_A[k] = flux_from_A(hexagons, x, y, Ax[k], Ay[k])

    # Curl comparison — shape (n_t, n_hex)
    # These are B-field values (not fluxes), comparable directly
    curl_A = np.zeros((n_t, n_hex))
    B_avg  = np.zeros((n_t, n_hex))
    for k in range(n_t):
        curl_A[k] = curl_from_A(hexagons, x, y, Ax[k], Ay[k])
        B_avg[k]  = B_avg_from_B(hexagons, x, y, B[k])

    # Relative error for curl comparison
    diff_curl    = B_avg - curl_A
    global_scale_curl = np.max(np.abs(B_avg))

    rel_err_curl = np.full((n_t, n_hex), np.nan)
    for k in range(n_t):
        threshold = 1e-3 * global_scale_curl
        mask = np.abs(B_avg[k]) > threshold
        rel_err_curl[k, mask] = np.abs(curl_A[k, mask] - B_avg[k, mask]) / np.abs(B_avg[k, mask])

    mean_rel_curl = np.nanmean(rel_err_curl, axis=1)

    # ------------------------------------------------------------------
    # Global metrics — flux level
    # ------------------------------------------------------------------
    diff_flux   = phi_B - phi_A
    rms_flux    = np.sqrt(np.mean(diff_flux**2, axis=1))
    maxabs_flux = np.max(np.abs(diff_flux),     axis=1)
    scale_flux  = np.max(np.abs(phi_B), axis=1)
    global_scale = np.max(np.abs(phi_B))

    pB_flat = phi_B.flatten()
    pA_flat = phi_A.flatten()
    corr_flux  = np.corrcoef(pB_flat, pA_flat)[0, 1]
    alpha_flux = np.dot(pB_flat, pA_flat) / (np.dot(pB_flat, pB_flat) + 1e-300)

    ratio   = np.full((n_t, n_hex), np.nan)
    rel_err = np.full((n_t, n_hex), np.nan)
    ##GLOBAL SCALE
   
    for k in range(n_t):
        threshold = 1e-3 * global_scale
        mask = np.abs(phi_B[k]) > threshold
        ratio[k, mask]   = phi_A[k, mask] / phi_B[k, mask]
        rel_err[k, mask] = np.abs(phi_A[k, mask] - phi_B[k, mask]) / np.abs(phi_B[k, mask])
   

    mean_ratio = np.nanmean(ratio,   axis=1)
    max_rel    = np.nanmax(rel_err,  axis=1)
    mean_rel   = np.nanmean(rel_err, axis=1)

    print(f"\n--- Flux comparison (Phi_B vs Phi_A = ∮ A·dl) ---")
    print(f"  Global r              : {corr_flux:.6f}")
    print(f"  Best-fit slope α      : {alpha_flux:.6f}   (Phi_A = α·Phi_B)")
    print(f"  RMS error (last step) : {rms_flux[-1]:.4e}")
    print(f"  Max error (last step) : {maxabs_flux[-1]:.4e}")
    if scale_flux[-1] > 0:
        print(f"  Relative RMS          : {rms_flux[-1]/scale_flux[-1]:.4e}")
    print(f"\n--- Ratio Phi_A/Phi_B (last step) ---")
    print(f"  Mean ratio            : {mean_ratio[-1]:.4f}   (ideal = 1.0)")
    print(f"  Mean relative error   : {mean_rel[-1]:.4e}")
    print(f"  Max  relative error   : {max_rel[-1]:.4e}")
    from numpy.random import default_rng
    rng = default_rng(42)

    k_mid = n_t // 2      # reuse mid-time slice where signal is strong
    B_shuffled  = rng.permutation(B[k_mid])
    Ax_shuffled = rng.permutation(Ax[k_mid])
    Ay_shuffled = rng.permutation(Ay[k_mid])

    phi_B_shuf = flux_from_B(hexagons, x, y, B_shuffled)
    phi_A_shuf = flux_from_A(hexagons, x, y, Ax_shuffled, Ay_shuffled)
    r_shuf = np.corrcoef(phi_B_shuf, phi_A_shuf)[0, 1]

    signal_scale = np.max(scale_flux)
    noise_floor  = np.median(scale_flux[n_t * 3 // 4:])

    print(f"\n--- Sanity check (shuffled fields at t = {t[k_mid]:.4g} a.u.) ---")
    print(f"  r with real fields    : {np.corrcoef(phi_B[k_mid], phi_A[k_mid])[0,1]:.4f}")
    print(f"  r with shuffled fields: {r_shuf:.4f}   (should be ~0)")
    print(f"\n--- Signal vs noise ---")
    print(f"  Peak |Phi_B|          : {signal_scale:.4e}")
    print(f"  Late-time floor       : {noise_floor:.4e}")
    print(f"  Dynamic range         : {signal_scale / (noise_floor + 1e-300):.1f}x")


    # Save table
    with open(out_dir / "flux_comparison_timeseries.txt", "w") as f:
        f.write(f"# Flux:  r = {corr_flux:.6f}   alpha = {alpha_flux:.6f}\n")
       # In the save table section, add curl columns:
        f.write("# t[au]  rms_flux  max_flux  max_phi_B  mean_ratio mean_rel_err  max_rel_err  mean_rel_curl\n")
        for k in range(n_t):
            f.write(
                f"{t[k]:.6e}  {rms_flux[k]:.6e}  {maxabs_flux[k]:.6e}  {scale_flux[k]:.6e}"
                f"  {mean_ratio[k]:.6e}  {mean_rel[k]:.6e}  {max_rel[k]:.6e}  {mean_rel_curl[k]:.6e}\n"
            )

    # Hexagon centres for spatial plots
    hx = np.array([np.mean(x[h]) for h in hexagons])
    hy = np.array([np.mean(y[h]) for h in hexagons])

    # -----------------------------------------------------------------------
    # Plot 1: error vs time (left) and ratio vs time (right)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex=True)

    # Flux — RMS
    axes[0, 0].plot(t, rms_flux, color="C0", lw=1.4, label=f"r = {corr_flux:.4f}")
    axes[0, 0].set_ylabel(r"RMS$_{\rm hex}\,|\Phi_B - \Phi_A|$  [a.u.]")
    axes[0, 0].set_title(
        r"Flux: $\Phi_B = \langle B_z\rangle S$ vs $\Phi_A = \oint A\cdot dl$"
    )
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Flux — max absolute
    axes[1, 0].plot(t, maxabs_flux, color="C1", lw=1.4)
    axes[1, 0].set_ylabel(r"max$_{\rm hex}\,|\Phi_B - \Phi_A|$  [a.u.]")
    axes[1, 0].set_xlabel("Time [a.u.]")
    axes[1, 0].grid(True, alpha=0.3)

    # Ratio — mean over hexagons
    axes[0, 1].plot(t, mean_ratio, color="C2", lw=1.4, label=r"$\langle\Phi_A/\Phi_B\rangle$")
    axes[0, 1].axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5, label="ideal = 1")
    axes[0, 1].set_ylabel(r"$\langle \Phi_A / \Phi_B \rangle_{\rm hex}$")
    axes[0, 1].set_title("Mean flux ratio across hexagons")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Relative error — max over hexagons
    axes[1, 1].plot(t, max_rel,  color="C3", lw=1.4, label="max hex")
    axes[1, 1].plot(t, mean_rel, color="C4", lw=1.4, label="mean hex", ls="--")
    axes[1, 1].set_ylabel(r"$|\Phi_B - \Phi_A|/|\Phi_B|$")
    axes[1, 1].set_xlabel("Time [a.u.]")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        "Gauge check — absolute error (left) and relative ratio (right)", fontsize=11
    )
    plt.tight_layout()
    fig.savefig(out_dir / "flux_error_vs_time.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 2: scatter phi_B vs phi_A at three time slices
    # -----------------------------------------------------------------------
    idxs = [0, n_t // 2, n_t - 1]

    def _scatter_panel(ax, xa, ya, xlabel, ylabel, title):
        ax.scatter(xa, ya, s=50, alpha=0.75, color="C0")
        lim = max(np.max(np.abs(np.concatenate([xa, ya]))), 1e-30)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.4, label="y = x")
        if np.std(xa) > 1e-20 * lim:
            r_k = np.corrcoef(xa, ya)[0, 1]
            m, b = np.polyfit(xa, ya, 1)
            ax.plot([-lim, lim], [m*(-lim)+b, m*lim+b], "r-",
                    lw=1.0, alpha=0.7,
                    label=f"fit: y = {m:.3f}x + {b:.2e}  (r={r_k:.4f})")
        else:
            ax.text(0.05, 0.9, "no variance (t=0?)", transform=ax.transAxes,
                    fontsize=7, color="red")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for col, ki in enumerate(idxs):
        _scatter_panel(
            axs[col], phi_B[ki], phi_A[ki],
            r"$\Phi_B = \langle B_z\rangle S$  [a.u.]",
            r"$\Phi_A = \oint A\cdot dl$  [a.u.]",
            f"t = {t[ki]:.4g} a.u.",
        )
    plt.suptitle(
        r"Hexagon flux scatter: $\Phi_B$ vs $\Phi_A$ — each dot = one hexagon",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "flux_scatter.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 3: spatial maps at mid time — absolute discrepancy and ratio
    # -----------------------------------------------------------------------
    k_mid = n_t // 2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute flux discrepancy
    d    = diff_flux[k_mid]
    dmax = np.max(np.abs(d)) or 1.0
    sc = axes[0].scatter(hx, hy, c=d, cmap="coolwarm", s=150,
                         vmin=-dmax, vmax=dmax, edgecolors="none")
    plt.colorbar(sc, ax=axes[0], label=r"$\Phi_B - \Phi_A$  [a.u.]")
    axes[0].scatter(x, y, c="k", s=8, zorder=5, alpha=0.35, label="sites")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x [a.u.]")
    axes[0].set_ylabel("y [a.u.]")
    axes[0].set_title(f"Flux discrepancy at t = {t[k_mid]:.4g} a.u.")
    axes[0].legend(fontsize=8)

    # Right: per-hexagon ratio Phi_A / Phi_B
    r_mid  = ratio[k_mid]
    finite = np.isfinite(r_mid)
    if finite.any():
        r_center = np.nanmedian(r_mid)
        r_spread = max(np.nanmax(np.abs(r_mid[finite] - r_center)), 1e-30)
        sc2 = axes[1].scatter(
            hx[finite], hy[finite], c=r_mid[finite],
            cmap="RdBu_r", s=150,
            vmin=r_center - r_spread, vmax=r_center + r_spread,
            edgecolors="none",
        )
        plt.colorbar(sc2, ax=axes[1], label=r"$\Phi_A / \Phi_B$")
    axes[1].scatter(x, y, c="k", s=8, zorder=5, alpha=0.35, label="sites")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("x [a.u.]")
    axes[1].set_ylabel("y [a.u.]")
    axes[1].set_title(f"Flux ratio $\\Phi_A/\\Phi_B$ at t = {t[k_mid]:.4g} a.u.")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "flux_spatial_diff.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 4: time evolution — flux (left) and ratio (right) per hexagon
    # -----------------------------------------------------------------------
    n_show   = min(4, n_hex)
    sort_idx = np.argsort(hx)
    if n_show > 1:
        chosen = [sort_idx[int(i * (n_hex - 1) / (n_show - 1))] for i in range(n_show)]
    else:
        chosen = [0]

    fig, axes = plt.subplots(n_show, 2, figsize=(13, 2.8 * n_show), sharex=True)
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row, hi in enumerate(chosen):
        # Flux column
        axes[row, 0].plot(t, phi_B[:, hi], color="C0", lw=1.3,
                          label=r"$\Phi_B = \langle B_z\rangle S$")
        axes[row, 0].plot(t, phi_A[:, hi], color="C1", lw=1.3, ls="--",
                          label=r"$\Phi_A = \oint A\cdot dl$")
        axes[row, 0].set_ylabel(r"$\Phi$  [a.u.]")
        axes[row, 0].set_title(
            f"Flux — hex {hi}  (x={hx[hi]:.2f}, y={hy[hi]:.2f} a.u.)", fontsize=9
        )
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        # Ratio column
        axes[row, 1].plot(t, ratio[:, hi], color="C2", lw=1.3,
                          label=r"$\Phi_A / \Phi_B$")
        axes[row, 1].axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5, label="ideal = 1")
        axes[row, 1].set_ylabel(r"$\Phi_A / \Phi_B$")
        axes[row, 1].set_title(
            f"Ratio — hex {hi}  (x={hx[hi]:.2f}, y={hy[hi]:.2f} a.u.)", fontsize=9
        )
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time [a.u.]")
    axes[-1, 1].set_xlabel("Time [a.u.]")
    plt.suptitle(
        "Gauge time evolution — flux (left) and ratio (right) — selected hexagons",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "flux_time_evolution.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    # -----------------------------------------------------------------------
    # Plot 5: lattice structure with numbered hexagons (standalone reference)
    # -----------------------------------------------------------------------
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.patches as mpatches

    cmap_hex = plt.cm.tab20 if n_hex <= 20 else plt.cm.turbo
    colors   = [cmap_hex(i / max(n_hex - 1, 1)) for i in range(n_hex)]

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#f7f7f7")
    ax.set_facecolor("#f7f7f7")

    # Filled hexagon polygons
    patches = []
    for hi, verts in enumerate(hexagons):
        verts_ccw = order_ccw(verts, x, y)
        coords    = np.column_stack([x[verts_ccw], y[verts_ccw]])
        patches.append(Polygon(coords, closed=True))

    col = PatchCollection(patches, facecolors=colors, edgecolors="white",
                          linewidths=1.5, alpha=0.35, zorder=1)
    ax.add_collection(col)

    # Bonds
    for i, j in bonds:
        ax.plot([x[i], x[j]], [y[i], y[j]],
                color="#555555", lw=1.2, zorder=2, solid_capstyle="round")

    # Sites — colour by sublattice (coordination parity)
    coord = np.array([len(neigh[i]) for i in range(n)])
    for site in range(n):
        c = "#1a6faf" if coord[site] % 2 == 0 else "#c0392b"
        ax.plot(x[site], y[site], "o", color=c,
                markersize=7, zorder=4,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.text(x[site], y[site] + a_au * 0.38, str(site),
                ha="center", va="bottom", fontsize=5.5,
                color="#333333", zorder=5)

    # Hexagon centres with bold index
    for hi in range(n_hex):
        ax.plot(hx[hi], hy[hi], "D", color=colors[hi],
                markersize=9, zorder=6,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.text(hx[hi], hy[hi], str(hi),
                ha="center", va="center", fontsize=8,
                fontweight="bold", color="white", zorder=7)

    # Legend
    
    

    
    ax.set_xlabel("x [a.u.]", fontsize=11)
    ax.set_ylabel("y [a.u.]", fontsize=11)
    ax.set_title(
        f"Lattice structure — {n} sites | {len(bonds)} bonds | {n_hex} hexagons\n"
        f"Site labels = index,  ◆ = hexagon centre with hex index",
        fontsize=11, pad=12,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2, linestyle=":")
    plt.tight_layout()
    fig.savefig(out_dir / "lattice_structure.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    
    print(f"\nAll outputs written to: {out_dir}")
    


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gauge check: hexagon flux from B_ind,z vs loop integral of A_ind."
    )
    parser.add_argument(
        "config_or_dir",
        help="Path to .toml config or a Simulations/... directory",
    )
    args = parser.parse_args()
    arg = Path(args.config_or_dir)

    if arg.is_dir():
        sim_dir = arg
    else:
        cfg = resolve_config_path(str(arg))
        sim_dir = simulation_dir_from_config(cfg)
        if not sim_dir.is_dir():
            print(f"Simulation directory not found: {sim_dir}", file=sys.stderr)
            sys.exit(1)

    try:                          # ← ADD THIS
        run_analysis(sim_dir)     # ← AND THIS
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()