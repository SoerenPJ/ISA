from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple


def fnv1a_64(data: bytes) -> int:
    """
    64-bit FNV-1a hash, matching the C++ implementation in main.cpp:

        std::uint64_t h = 14695981039346656037ull;
        for (unsigned char c : s) {
            h ^= static_cast<std::uint64_t>(c);
            h *= 1099511628211ull;
        }
    """
    h = 14695981039346656037
    prime = 1099511628211
    for b in data:
        h ^= b
        h = (h * prime) & 0xFFFFFFFFFFFFFFFF  # keep as 64-bit
    return h


def load_lattice_points(path: Path) -> List[Tuple[float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"lattice_points.txt not found at {path}")

    points: List[Tuple[float, float]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            points.append((x, y))

    if not points:
        raise ValueError(f"No lattice points parsed from {path}")
    return points


def load_htb_matrix(path: Path) -> List[List[complex]]:
    if not path.is_file():
        raise FileNotFoundError(f"HTB.txt not found at {path}")

    rows: List[List[complex]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) % 2 != 0:
                raise ValueError(
                    f"HTB line has odd number of tokens ({len(parts)}): {line!r}"
                )
            row: List[complex] = []
            for i in range(0, len(parts), 2):
                re = float(parts[i])
                im = float(parts[i + 1])
                row.append(complex(re, im))
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows parsed from {path}")

    n = len(rows)
    for r in rows:
        if len(r) != n:
            raise ValueError(
                f"HTB matrix is not square: got {len(rows)} rows and "
                f"row length {len(r)}"
            )

    return rows


def compute_bond_diagnostics(
    points: List[Tuple[float, float]],
    H: List[List[complex]],
    hop_tol: float = 1e-12,
    geom_tolerance_rel: float = 0.05,
) -> None:
    n_sites = len(points)
    if len(H) != n_sites:
        print(
            f"WARNING: Number of lattice points ({n_sites}) does not match "
            f"H dimension ({len(H)}). Using min of both.",
            file=sys.stderr,
        )
        n_sites = min(n_sites, len(H))

    # Collect all bonds from H (non-zero hoppings).
    bond_distances: List[float] = []
    degrees = [0] * n_sites

    for i in range(n_sites):
        xi, yi = points[i]
        for j in range(i + 1, n_sites):
            hij = H[i][j]
            if abs(hij) > hop_tol:
                xj, yj = points[j]
                dx = xi - xj
                dy = yi - yj
                r = math.hypot(dx, dy)
                bond_distances.append(r)
                degrees[i] += 1
                degrees[j] += 1

    print(f"Number of sites (used): {n_sites}")
    print(f"Number of bonds (|H_ij| > {hop_tol:g}): {len(bond_distances)}")

    # Degree statistics
    if degrees:
        unique_degrees = sorted(set(degrees))
        print("\nNeighbour count per site (degree histogram):")
        for d in unique_degrees:
            count = sum(1 for x in degrees if x == d)
            print(f"  degree {d}: {count} sites")

    if not bond_distances:
        print("\nNo bonds detected from H (all hoppings below tolerance).")
        return

    # Bond length stats
    bond_distances_sorted = sorted(bond_distances)
    min_r = bond_distances_sorted[0]
    max_r = bond_distances_sorted[-1]
    mean_r = sum(bond_distances_sorted) / len(bond_distances_sorted)
    mid = len(bond_distances_sorted) // 2
    if len(bond_distances_sorted) % 2 == 1:
        median_r = bond_distances_sorted[mid]
    else:
        median_r = 0.5 * (
            bond_distances_sorted[mid - 1] + bond_distances_sorted[mid]
        )

    print("\nBond length statistics (from non-zero hoppings):")
    print(f"  min    = {min_r:.6f}")
    print(f"  max    = {max_r:.6f}")
    print(f"  mean   = {mean_r:.6f}")
    print(f"  median = {median_r:.6f}")

    # Define a "nearest neighbour" geometric cutoff from the median bond length.
    # Geometric tolerance is slightly larger than the TB construction tolerance
    # (0.01 in C++ vs default 0.05 here) to be forgiving of numerical noise.
    geom_cutoff = median_r * (1.0 + geom_tolerance_rel)
    print(f"\nGeometric NN cutoff (from median * (1+{geom_tolerance_rel})): "
          f"{geom_cutoff:.6f}")

    # Pre-compute pairwise distances for all i<j up to n_sites.
    dist: List[List[float]] = [[0.0] * n_sites for _ in range(n_sites)]
    for i in range(n_sites):
        xi, yi = points[i]
        for j in range(i + 1, n_sites):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            r = math.hypot(dx, dy)
            dist[i][j] = dist[j][i] = r

    missing_bonds: List[Tuple[int, int, float]] = []
    long_bonds: List[Tuple[int, int, float]] = []

    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            r = dist[i][j]
            hij = H[i][j]
            nonzero = abs(hij) > hop_tol
            if r <= geom_cutoff and not nonzero:
                missing_bonds.append((i, j, r))
            if nonzero and r > geom_cutoff:
                long_bonds.append((i, j, r))

    print("\nGeometric / TB consistency checks:")
    print(f"  Geometric NN pairs with NO hopping (potential missing bonds): "
          f"{len(missing_bonds)}")
    print(f"  Hopping pairs with LARGE distance (potential extra/long bonds): "
          f"{len(long_bonds)}")

    def print_some(label: str, items: List[Tuple[int, int, float]], limit: int = 10):
        if not items:
            return
        print(f"\n{label} (showing up to {limit} examples):")
        for (i, j, r) in items[:limit]:
            print(f"  (i={i}, j={j}), r={r:.6f}")

    print_some("Examples of missing bonds", missing_bonds)
    print_some("Examples of long bonds", long_bonds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check that tight-binding bonds in HTB.txt agree with geometry "
            "from lattice_points.txt for a given TOML config."
        )
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the TOML configuration file used for the simulation.",
    )
    parser.add_argument(
        "--sim-root",
        type=str,
        default="Simulations",
        help="Root directory where simulation folders are stored "
             "(default: Simulations).",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        return 1

    cfg_text = cfg_path.read_bytes()
    cfg_hash = fnv1a_64(cfg_text)
    hash_hex = format(cfg_hash, "x")  # matches std::hex default (lowercase, no 0x)

    stem = cfg_path.stem
    sim_root = Path(args.sim_root)
    out_dir = sim_root / f"{stem}_{hash_hex}"

    print(f"Config: {cfg_path}")
    print(f"FNV-1a hash: 0x{hash_hex}")
    print(f"Expected simulation folder: {out_dir}")

    if not out_dir.is_dir():
        print(
            f"ERROR: Simulation folder {out_dir} does not exist.\n"
            f"Run the C++ simulation with this config first.",
            file=sys.stderr,
        )
        return 1

    lattice_file = out_dir / "lattice_points.txt"
    htb_file = out_dir / "HTB.txt"

    try:
        points = load_lattice_points(lattice_file)
        H = load_htb_matrix(htb_file)
    except Exception as e:
        print(f"ERROR while loading data: {e}", file=sys.stderr)
        return 1

    compute_bond_diagnostics(points, H)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

