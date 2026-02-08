import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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
    if not p.suffix:
        p2 = Path(str(p) + ".toml")
        if p2.exists() and p2.is_file():
            return p2
    raise FileNotFoundError(f"Could not find config file: {arg} (tried '{p}' and '{p}.toml')")


def simulation_dir_from_config(cfg_path: Path) -> Path:
    h = fnv1a_64(cfg_path.read_bytes())
    return Path("Simulations") / f"{cfg_path.stem}_{h:016x}"

def count_unique_points(pts: np.ndarray, decimals: int = 6) -> int:
    if pts.size == 0:
        return 0
    rounded = np.round(pts[:, :2], decimals=decimals)
    unique = np.unique(rounded, axis=0)
    return int(unique.shape[0])


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 Ploting/plot_points_from_cpp.py <config.toml | configs/name | Simulations/folder>")
        return 1

    arg = Path(sys.argv[1])
    if arg.exists() and arg.is_dir():
        base_dir = arg
    else:
        cfg = resolve_config_path(sys.argv[1])
        base_dir = simulation_dir_from_config(cfg)

    pts_path = base_dir / "lattice_points.txt"
    if not pts_path.exists():
        raise FileNotFoundError(
            f"Missing {pts_path}\n"
            f"Run the simulator first, e.g.: ./sim_mkl {sys.argv[1]}"
        )

    pts = np.loadtxt(pts_path, comments="#")
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"Unexpected lattice_points.txt format: {pts.shape}")

    n_raw = int(len(pts))
    n_unique = count_unique_points(pts, decimals=6)
    if n_unique != n_raw:
        print(f"WARNING: lattice_points.txt contains duplicates: raw={n_raw}, unique(6dp)={n_unique}")

    x = pts[:, 0]
    y = pts[:, 1]

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=12, c="k")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.title(f"C++ lattice points: {base_dir.name} (raw={n_raw}, unique≈{n_unique})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

