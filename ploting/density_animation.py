import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path
import sys
import seaborn as sns


# ----------------- utilities (mirrored from base_plots.py) ----------------- #
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
    # allow calling with "configs/SSH" (no extension)
    if not p.suffix:
        p2 = Path(str(p) + ".toml")
        if p2.exists() and p2.is_file():
            return p2
    raise FileNotFoundError(f"Could not find config file: {arg} (tried '{p}' and '{p}.toml')")


def simulation_dir_from_config(cfg_path: Path) -> Path:
    cfg_bytes = cfg_path.read_bytes()
    h = fnv1a_64(cfg_bytes)
    folder = f"{cfg_path.stem}_{h:x}"
    return Path("Simulations") / folder


# ---------- constants (must match C++/other plotting scripts) ---------- #
au_eV = 27.2114
au_nm = 0.0529177
au_s = 2.41888e-17
au_fs = au_s * 1e15


def density_array(x_sites, y_sites, values, array_shape, sigma):
  
    nx, ny = array_shape

    padding = 3.0 * sigma  # 2–4 sigma is usually good

    x_min, x_max = x_sites.min() - padding, x_sites.max() + padding
    y_min, y_max = y_sites.min() - padding, y_sites.max() + padding

    x_lin = np.linspace(x_min, x_max, nx)
    y_lin = np.linspace(y_min, y_max, ny)
    XX, YY = np.meshgrid(x_lin, y_lin, indexing="ij")

    field = np.zeros_like(XX, dtype=float)

    # Gaussian smearing of each site onto the grid
    for xs, ys, v in zip(x_sites, y_sites, values):
        dx = XX - xs
        dy = YY - ys
        w = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        field += v * w

    return XX, YY, field


def animate_contourplot(rho_ind, t_fs, xx, yy, array_shape, cmap, tmax_fs=None, sigma=0.55, interval=100):
  
    # Limit to t <= tmax_fs if requested
    if tmax_fs is not None:
        mask = t_fs <= tmax_fs
        t_fs = t_fs[mask]
        rho_ind = rho_ind[:, mask]

    n_frames = rho_ind.shape[1]
    y_size = max(array_shape[1] / 20.0, 3.5)

    # Precompute smoothed fields for all frames to get global levels
    data = np.zeros((array_shape[0], array_shape[1], n_frames), dtype=float)
    for i in range(n_frames):
        XX, YY, data[:, :, i] = density_array(xx, yy, np.real(rho_ind[:, i]), array_shape, sigma)

    max_val = np.max(np.abs(data))
    if max_val <= 0.0:
        max_val = 1.0
    levels = np.linspace(-0.5 * max_val, 0.5 * max_val, 101)

    fig, ax = plt.subplots(figsize=(array_shape[0] / 20.0, y_size))

    # Initial empty contour plot
    contour = ax.contourf(XX, YY, np.zeros_like(data[:, :, 0]), cmap=cmap, levels=levels, extend="both")

    def animate(frame):
        ax.clear()
        contour = ax.contourf(XX, YY, data[:, :, frame], cmap=cmap, levels=levels, extend="both")
        ax.plot(xx, yy, "white", marker="o", ms=5, lw=0, zorder=10)
        ax.set_title(f"Time: {t_fs[frame]:.2f} fs")
        ax.axis("equal")
        ax.set_xlabel(r"$x_l$ (Å)", fontsize=14)
        ax.set_ylabel(r"$y_l$ (Å)", fontsize=14)
        #return contour.collections

    fig.colorbar(contour, ax=ax, label=r"$\rho_l^\mathrm{ind}$")

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=n_frames,
        interval=interval,
        blit=False,
    )
    plt.close(fig)
    return anim


def main():
    base_dir = Path(".")
    if len(sys.argv) >= 2:
        arg = Path(sys.argv[1])
        if arg.exists() and arg.is_dir():
            base_dir = arg
        else:
            cfg = resolve_config_path(sys.argv[1])
            base_dir = simulation_dir_from_config(cfg)
            if not base_dir.exists():
                raise FileNotFoundError(
                    f"Simulation output folder not found: {base_dir}\n"
                    f"Run the simulator first with: ./sim_mkl {cfg}"
                )

    # Load lattice positions (atomic units) and convert to Angstrom
    lattice_path = base_dir / "lattice_points.txt"
    lat = np.loadtxt(lattice_path, comments="#")
    xx_au = lat[:, 0]
    yy_au = lat[:, 1]
    xx = xx_au * au_nm * 10.0
    yy = yy_au * au_nm * 10.0

    # Load time-dependent diagonal densities: t (a.u.), rho_0 ... rho_{N-1}
    rho_diag_path = base_dir / "rho_diag_time_evolution.txt"
    rho_data = np.loadtxt(rho_diag_path, comments="#")
    t_au = rho_data[:, 0]
    rho_diag = rho_data[:, 1:]  # shape (N_time, N_sites)

    # Equilibrium density from rho0_l_space.txt
    rho0_l_space = np.loadtxt(base_dir / "rho0_l_space.txt")
    rho0_l = rho0_l_space[:, 0::2] + 1j * rho0_l_space[:, 1::2]
    rho0_diag = np.real(np.diag(rho0_l))  # shape (N_sites,)

    # Induced density: rho_ind(site, t) = rho(site, t) - rho0(site)
    rho_ind = (rho_diag - rho0_diag[None, :]).T  # (N_sites, N_time)

    # Time in femtoseconds
    t_fs = t_au * au_fs

    # Restrict to a short time window around the pulse maximum, e.g. 1500–1510 fs
    t_min_fs = 1500.0
    t_max_fs = 1510.0
    mask = (t_fs >= t_min_fs) & (t_fs <= t_max_fs)
    if not np.any(mask):
        raise RuntimeError(f"No time points found in range [{t_min_fs}, {t_max_fs}] fs")
    t_fs = t_fs[mask]
    rho_ind = rho_ind[:, mask]

    # Further limit the number of frames for the animation to keep memory reasonable
    max_frames = 300
    n_frames = rho_ind.shape[1]
    if n_frames > max_frames:
        idx = np.linspace(0, n_frames - 1, max_frames, dtype=int)
        rho_ind = rho_ind[:, idx]
        t_fs = t_fs[idx]

    # Aspect ratio and grid shape
    x_span = xx.max() - xx.min()
    y_span = yy.max() - yy.min()
    aspect = x_span / y_span if y_span != 0.0 else 1.0
    nx = 200
    ny = int(200 / aspect) if aspect != 0.0 else 200
    array_shape = (nx, max(ny, 10))

    
    cmap = sns.color_palette("icefire", as_cmap=True)
    anim = animate_contourplot(
        rho_ind=rho_ind,
        t_fs=t_fs,
        xx=xx,
        yy=yy,
        array_shape=array_shape,
        cmap=cmap,
        tmax_fs=None,
        sigma=0.55,
        interval=100,
    )

    out_path = base_dir / "density_animation.gif"
    anim.save(out_path, writer="pillow")
    print(f"Saved density animation to: {out_path}")


if __name__ == "__main__":
    main()

