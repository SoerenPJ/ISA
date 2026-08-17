import os
import re
import glob
import sys
import numpy as np
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Extract the resonance frequency from sweep data.
#
# Sweep data layout (data_LLM):
#   sweep_data_mu_<structure>/
#       L0_mu_0.00/sigma_ext.txt
#       L0_mu_0.04/sigma_ext.txt
#       ...
#       L1_mu_<value>/sigma_ext.txt
#       L2_mu_<value>/sigma_ext.txt
#
# Each sigma_ext.txt has two columns:
#   col 0 -> omega   (atomic units, Hartree)
#   col 1 -> sigma   (extinction cross section, arb. units)
#
# For every chemical potential (mu) and every implementation level (L0/L1/L2)
# the resonance frequency is the omega of the dominant peak of sigma(omega).
# ---------------------------------------------------------------------------

# constants
HARTREE_TO_EV = 27.2114

# directory holding all the sweeps
DATA_ROOT = os.path.expanduser("~/University/masters/2.semester/ISA/scr/data_LLM")

# implementation levels present in the sweeps
LEVELS = ["L0", "L1", "L2"]

# regex: L<level>_mu_<value>
DIR_RE = re.compile(r"^(L\d+)_mu_(-?\d+(?:\.\d+)?)$")


def resonance_frequency(omega, sigma):
    """Return the omega of the dominant resonance peak of sigma(omega).

    Uses find_peaks with a prominence threshold and selects the most
    prominent peak; falls back to the global maximum if no peak is found.
    """
    if sigma.size == 0 or np.max(sigma) <= 0:
        return np.nan

    peaks, props = find_peaks(sigma, prominence=np.max(sigma) * 0.1)

    if peaks.size == 0:
        # no clear peak (e.g. monotonic / flat spectrum) -> global max
        return omega[int(np.argmax(sigma))]

    # pick the most prominent peak
    best = peaks[int(np.argmax(props["prominences"]))]
    return omega[best]


def extract_sweep(sweep_dir):
    """Build the resonance table for a single sweep directory.

    Returns (mu_unique, table) where table has one column of resonance
    frequencies (in eV) per level, in LEVELS order. Missing entries are NaN.
    """
    # collect resonance per (level, mu)
    res = {lvl: {} for lvl in LEVELS}

    for entry in sorted(os.listdir(sweep_dir)):
        m = DIR_RE.match(entry)
        if not m:
            continue
        level, mu_str = m.group(1), m.group(2)
        if level not in res:
            continue

        sigma_file = os.path.join(sweep_dir, entry, "sigma_ext.txt")
        if not os.path.isfile(sigma_file):
            continue

        data = np.loadtxt(sigma_file)
        if data.ndim != 2 or data.shape[0] < 2:
            continue

        omega = data[:, 0] * HARTREE_TO_EV
        sigma = data[:, 1]

        res[level][float(mu_str)] = resonance_frequency(omega, sigma)

    # union of all mu values across levels
    mu_unique = np.array(
        sorted({mu for lvl in LEVELS for mu in res[lvl]})
    )

    table = np.full((mu_unique.size, len(LEVELS)), np.nan)
    for j, lvl in enumerate(LEVELS):
        for i, mu in enumerate(mu_unique):
            if mu in res[lvl]:
                table[i, j] = res[lvl][mu]

    return mu_unique, table


def main():
    # sweep directories: from argv, else every sweep_data_* dir under DATA_ROOT
    if len(sys.argv) > 1:
        sweep_dirs = [os.path.abspath(p) for p in sys.argv[1:]]
    else:
        sweep_dirs = sorted(
            d for d in glob.glob(os.path.join(DATA_ROOT, "sweep_data_*"))
            if os.path.isdir(d)
        )

    if not sweep_dirs:
        print(f"No sweep directories found under {DATA_ROOT}")
        return

    for sweep_dir in sweep_dirs:
        name = os.path.basename(sweep_dir.rstrip("/"))
        mu_unique, table = extract_sweep(sweep_dir)

        if mu_unique.size == 0:
            print(f"[skip] {name}: no L*_mu_* spectra found")
            continue

        out = np.column_stack((mu_unique, table))
        output_file = os.path.join(DATA_ROOT, f"resonance_vs_mu_{name}.txt")

        header = "mu  " + "  ".join(f"omega_res_{lvl}_eV" for lvl in LEVELS)
        np.savetxt(output_file, out, header=header, fmt="%.6f")

        print(f"[ok] {name}: {mu_unique.size} mu values -> {output_file}")


if __name__ == "__main__":
    main()
