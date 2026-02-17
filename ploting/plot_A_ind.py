import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path


def fnv1a_64(data: bytes) -> int:
    """Match the simulator's FNV-1a 64-bit hash for folder naming."""
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
    raise FileNotFoundError(f"Could not find config file: {arg}")


def simulation_dir_from_config(cfg_path: Path) -> Path:
    cfg_bytes = cfg_path.read_bytes()
    h = fnv1a_64(cfg_bytes)
    folder = f"{cfg_path.stem}_{h:x}"
    return Path("Simulations") / folder


def load_A_ind(sim_dir: Path):
    """Load A_ind_time_evolution.txt. Returns time, A_ind_x, A_ind_y (each shape (n_time, n_sites))."""
    path = Path(sim_dir) / "A_ind_time_evolution.txt"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path} (run with self_consistent_phase = true)")
    data = np.loadtxt(path)
    t = data[:, 0]
    # Columns: t, A_x_0, A_y_0, A_x_1, A_y_1, ...
    n_sites = (data.shape[1] - 1) // 2
    A_ind_x = data[:, 1 : 1 + 2 * n_sites : 2]   # (n_time, n_sites)
    A_ind_y = data[:, 2 : 1 + 2 * n_sites : 2]
    return t, A_ind_x, A_ind_y


def load_lattice(sim_dir: Path):
    """Load lattice_points.txt. Returns (x, y) each shape (n_sites,)."""
    path = Path(sim_dir) / "lattice_points.txt"
    if not path.exists():
        return None, None
    data = np.loadtxt(path)
    if data.ndim == 1:
        x = data
        y = np.zeros_like(x)
    else:
        x, y = data[:, 0], data[:, 1]
    return x, y


def plot_A_ind_vs_time(sim_dir: Path, site_indices=None, out_path=None):
    """Plot A_ind_x and A_ind_y vs time for selected sites."""
    t, A_ind_x, A_ind_y = load_A_ind(sim_dir)
    n_sites = A_ind_x.shape[1]
    if site_indices is None:
        site_indices = [0, n_sites // 2, n_sites - 1] if n_sites > 2 else list(range(n_sites))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    for i in site_indices:
        if i < n_sites:
            ax1.plot(t, A_ind_x[:, i], label=f"site {i}")
            ax2.plot(t, A_ind_y[:, i], label=f"site {i}")
    ax1.set_ylabel(r"$A_{\mathrm{ind},x}$")
    ax2.set_ylabel(r"$A_{\mathrm{ind},y}$")
    ax2.set_xlabel("Time [a.u.]")
    ax1.legend()
    ax2.legend()
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Induced vector potential vs time")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print("Saved", out_path)
    plt.show()





def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Example: python plot_A_ind.py configs/graphene_zigzag.toml")
        sys.exit(1)
    arg = Path(sys.argv[1])
    if arg.is_file():
        cfg = resolve_config_path(str(arg))
        sim_dir = simulation_dir_from_config(cfg)
    elif arg.is_dir():
        sim_dir = arg
    else:
        print("Not a file or directory:", arg)
        sys.exit(1)
    if not sim_dir.is_dir():
        print("Simulation directory not found:", sim_dir)
        print("Run ./sim_mkl configs/graphene_zigzag.toml first.")
        sys.exit(1)

    out_dir = sim_dir / "A_ind"
    out_dir.mkdir(parents=True, exist_ok=True)

    t, A_ind_x, A_ind_y = load_A_ind(sim_dir)
    print(f"Loaded A_ind: {len(t)} time steps, {A_ind_x.shape[1]} sites")
    # Sanity check: if C++ wrote time-varying A_ind, these ranges should be non-zero
    ax_span_x = A_ind_x.max() - A_ind_x.min()
    ax_span_y = A_ind_y.max() - A_ind_y.min()
    print(f"A_ind_x range over time: [{A_ind_x.min():.6e}, {A_ind_x.max():.6e}] (span {ax_span_x:.6e})")
    print(f"A_ind_y range over time: [{A_ind_y.min():.6e}, {A_ind_y.max():.6e}] (span {ax_span_y:.6e})")
    if ax_span_x < 1e-20 and ax_span_y < 1e-20:
        print("Warning: A_ind is constant in time. Rebuild C++ and re-run ./sim_mkl to regenerate A_ind_time_evolution.txt.")

    plot_A_ind_vs_time(sim_dir, out_path=out_dir / "A_ind_vs_time.png")


if __name__ == "__main__":
    main()
