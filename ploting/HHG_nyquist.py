#!/usr/bin/env python3
"""HHG.py — harmonic spectrum |p_ddot(omega)|^2 from a run's dipole_acc.txt.

Usage:
    python3 ploting/HHG.py Simulations/<run_dir>            [options]
    python3 ploting/HHG.py configs/graphene_zigzag_triangle.toml   (resolves the
                                                            run dir by config hash)

Options:
    --omega-0 <eV>    normalise by this instead of the config's drive frequency
    --harmonics <N>   plot out to this harmonic order (default: as far as the
                      data and the Nyquist limit allow, capped at 40)
    --no-tex          skip usetex (also skipped automatically if LaTeX is absent)

TWO THINGS THIS SCRIPT REFUSES TO DO, both of which used to make the figure
unreadable:

1. It does NOT take omega_0 from argmax of the spectrum. The simulator forms
   p_ddot(omega) = omega^2 * p(omega) (Observables/observables.cpp:120), so the
   omega^2 factor amplifies whatever sits at the top of the frequency grid by up
   to 1e8 relative to the drive. argmax then lands on that instead of on the
   fundamental, and omega/omega_0 squeezes the whole spectrum into [0, 1]. The
   fundamental is a KNOWN input — [field] omega in the run's input.toml — so it
   is read from there.

2. It does NOT plot past the Nyquist frequency of the solver's time grid. The
   adaptive solver's time step sets a hard ceiling on what the post-hoc DFT can
   resolve (typically ~30-40 eV); [analysis] omega_cut_off can and does ask for
   far more, and everything above the ceiling is aliasing multiplied by omega^2.
   The ceiling is computed from dipole_time_evolution.txt and reported.
"""

import sys
import re
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

au_eV = 27.2114

# Fallback ceiling when dipole_time_evolution.txt is not next to dipole_acc.txt
# and the Nyquist limit cannot be measured. Deliberately low: better a spectrum
# that stops early than one padded out with aliased noise.
FALLBACK_MAX_eV = 30.0
MAX_HARMONICS = 40


# ----------------- run-directory resolution ----------------- #
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
    # Match C++: std::hex with no setw/setfill -> unpadded hex
    folder = f"{cfg_path.stem}_{h:x}"
    return Path("Simulations") / folder


# ----------------- what the run itself says ----------------- #
def load_field_omega_eV(base_dir):
    """[field] omega in eV from the run's archived input.toml.

    This is the DRIVE frequency, i.e. the fundamental the harmonics are orders
    of — the only defensible omega_0. Parsed with a section-aware line regex
    rather than a TOML library, the same way ohno_U_sweep_plot.py:load_t_eV()
    reads t1, so the script keeps working with no extra dependency.
    """
    f = Path(base_dir) / "input.toml"
    if not f.is_file():
        return None
    section = None
    for line in f.read_text().splitlines():
        s = line.split("#")[0].strip()
        m = re.match(r"^\[([^\]]+)\]", s)
        if m:
            section = m.group(1).strip()
            continue
        if section == "field":
            m = re.match(r"^omega\s*=\s*(-?[\d.eE+-]+)", s)
            if m:
                return float(m.group(1))
    return None


def last_visible_harmonic(omega_eV, y, omega_0, n_max):
    """Highest harmonic order whose peak still clears the local noise floor.

    The plateau here is short and ends in a RISING floor, not a cutoff: past the
    last real harmonic the spectrum is the solver's noise multiplied by omega^2,
    which climbs with omega and can look like a plateau if you plot far enough.
    So the default x-range is set from where the comb actually dies rather than
    from the frequency grid, which would show mostly ramp.

    Per order: peak = max within +-0.3*omega_0 of n*omega_0, floor = median over
    +-0.5*omega_0. An order counts as visible at 10x contrast.

    The scan STOPS at the first order that fails rather than reporting the last
    one that passes. A comb is contiguous — harmonics fall off monotonically and
    do not reappear once they reach the floor — whereas the noise band above it
    is spiky enough that an isolated bin clears 10x the local median by chance.
    Taking the last success let one such bin at n = 17 stretch the axis over ten
    orders of pure floor.
    """
    last = 1
    for n in range(1, n_max + 1):
        c = n * omega_0
        peak_k = np.abs(omega_eV - c) <= 0.3 * omega_0
        band_k = np.abs(omega_eV - c) <= 0.5 * omega_0
        if not peak_k.any() or not band_k.any():
            break
        if y[peak_k].max() <= 10.0 * np.median(y[band_k]):
            break
        last = n
    return last


def nyquist_eV(base_dir):
    """Highest frequency the solver's time grid can actually carry, in eV.

    The propagator is adaptive, so the stamps in dipole_time_evolution.txt are
    non-uniform (and can repeat). The median step is what sets the usable
    ceiling: pi/dt. Frequencies above it are aliases, and since the acceleration
    carries an omega^2 factor they are aliases scaled up by orders of magnitude.
    """
    f = Path(base_dir) / "dipole_time_evolution.txt"
    if not f.is_file():
        return None
    t = np.loadtxt(f, comments="#", usecols=0)
    t = np.unique(t)                     # collapse duplicate stamps
    if t.size < 3:
        return None
    return float(np.pi / np.median(np.diff(t)) * au_eV)


# ----------------------------- main ----------------------------- #
def main():
    args = sys.argv[1:]

    def opt(name, cast, default=None):
        if name in args:
            return cast(args[args.index(name) + 1])
        return default

    omega_0 = opt("--omega-0", float)
    n_harm = opt("--harmonics", int)
    use_tex = "--no-tex" not in args and shutil.which("latex") is not None

    positional = [a for a in args if not a.startswith("--")]
    # drop values that belong to the options above
    for name in ("--omega-0", "--harmonics"):
        if name in args:
            v = args[args.index(name) + 1]
            if v in positional:
                positional.remove(v)

    base_dir = Path(".")
    if positional:
        arg = Path(positional[0])
        if arg.exists() and arg.is_dir():
            base_dir = arg
        else:
            cfg = resolve_config_path(positional[0])
            base_dir = simulation_dir_from_config(cfg)
            if not base_dir.exists():
                raise FileNotFoundError(
                    f"Simulation output folder not found: {base_dir}\n"
                    f"Run the simulator first with: ./sim_blas {cfg}"
                )

    out_dir = base_dir / "HHG_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = base_dir / "dipole_acc.txt"
    if not acc.is_file():
        # Distinguish "the run never computed it" from "wrong directory" —
        # run_dipole_acc defaults to false, so a missing file is usually the
        # config, not a mistake in the path.
        print(f"ERROR: {acc} not found.\n"
              f"       If {base_dir} is a finished run, set "
              f"[analysis] run_dipole_acc = true in its config and re-run; "
              f"the acceleration is not computed by default.")
        sys.exit(1)
    data = np.loadtxt(acc)
    omega_eV = data[:, 0] * au_eV                    # file is in a.u.
    y_axis = np.abs(data[:, 1] + 1j * data[:, 2]) ** 2
    print(f"  {len(omega_eV)} frequency bins, 0 .. {omega_eV[-1]:.4g} eV "
          f"(step {omega_eV[1] - omega_eV[0]:.4g} eV)")

    # ---- omega_0: the drive, never argmax (see module docstring) ------------
    if omega_0 is None:
        omega_0 = load_field_omega_eV(base_dir)
        if omega_0 is None:
            print("ERROR: could not read [field] omega from "
                  f"{base_dir/'input.toml'} — pass --omega-0 <eV> explicitly.")
            sys.exit(1)
        print(f"  omega_0 = {omega_0:g} eV   (drive, from input.toml)")
    else:
        print(f"  omega_0 = {omega_0:g} eV   (--omega-0 override)")

    # ---- ceiling: what the time grid can resolve ---------------------------
    nyq = nyquist_eV(base_dir)
    if nyq is None:
        nyq = FALLBACK_MAX_eV
        print(f"  [WARNING] dipole_time_evolution.txt not found — cannot measure "
              f"the Nyquist limit. Falling back to {nyq:g} eV.")
    else:
        print(f"  Nyquist limit of the time grid: {nyq:.4g} eV "
              f"= harmonic order {nyq/omega_0:.0f}")
        if omega_eV[-1] > nyq:
            print(f"  [WARNING] omega_cut_off reaches {omega_eV[-1]:.4g} eV, "
                  f"{omega_eV[-1]/nyq:.0f}x past that ceiling. Everything above "
                  f"{nyq:.4g} eV is aliasing amplified by the omega^2 factor and "
                  f"is NOT plotted. Lower [analysis] omega_cut_off — the extra "
                  f"bins cost DFT time and carry no information.")

    top_eV = min(nyq, omega_eV[-1])
    n_grid = min(MAX_HARMONICS, int(top_eV / omega_0))
    n_vis = last_visible_harmonic(omega_eV, y_axis, omega_0, n_grid)
    print(f"  harmonic comb resolved out to n = {n_vis}; above that the "
          f"spectrum is the solver noise floor scaled by omega^2 — a rising "
          f"tail, not a plateau.")
    if n_harm is None:
        # A little headroom past the last real order, so the noise floor is
        # visible for what it is rather than cropped out of sight.
        n_harm = int(min(n_grid, max(8, np.ceil(n_vis * 1.5))))
    top_eV = min(top_eV, n_harm * omega_0)

    k = omega_eV <= top_eV
    x_val = omega_eV[k] / omega_0
    j0 = int(np.argmin(np.abs(omega_eV - omega_0)))   # bin nearest the drive
    y_val = y_axis[k] / y_axis[j0]

    print(f"  plotting to omega/omega_0 = {n_harm} ({top_eV:.4g} eV); "
          f"|p_acc(omega_0)|^2 = {y_axis[j0]:.4g}")
    print("\n  === harmonic intensities (relative to the fundamental) ===")
    for n in range(1, min(n_harm, 12) + 1):
        j = int(np.argmin(np.abs(omega_eV - n * omega_0)))
        print(f"    n = {n:2d}   omega = {n*omega_0:6.2f} eV   "
              f"I/I_1 = {y_axis[j]/y_axis[j0]:.3e}")

    # ---- plot --------------------------------------------------------------
    # usetex is a hard failure without a LaTeX install, which looks like the
    # script itself is broken. Fall back to mathtext instead.
    if use_tex:
        plt.rc("text", usetex=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_val, y_val, linewidth=2)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\omega /\omega_0 $", fontsize=24)
    ax.set_ylabel(r"$|\ddot{p}(\omega)|^2 / |\ddot{p}(\omega_0)|^2$", fontsize=24)
    ax.tick_params(labelsize=12)

    ax.set_xlim(0, n_harm)
    step = 1 if n_harm <= 12 else (2 if n_harm <= 24 else 5)
    ax.xaxis.set_major_locator(FixedLocator(np.arange(0, n_harm + 1, step)))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.grid(which="major", linestyle="-", linewidth=0.8)

    plt.tight_layout()
    out = out_dir / "HHG.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nwrote {out}")
    plt.show()


if __name__ == "__main__":
    main()
