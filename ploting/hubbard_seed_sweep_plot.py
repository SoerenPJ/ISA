#!/usr/bin/env python3
"""
hubbard_seed_sweep_plot.py — the UHF ground state is independent of the initial
guess amplitude.

The initial guess is fully deterministic: a single number m_seed
([features] hubbard_seed) sets the staggered moment, n_up(i)=0.5+0.5*m_seed*s_i.
There is no random engine, so every run is exactly reproducible. This sweep sets
m_seed to many hand-chosen values and checks the loop reaches the SAME state for
EVERY value (not just a lucky one).

Reads the tables written by  hubbard_seed_sweep.sh:

    data/hubbard_seed_sweep_<tag>.txt
        # ... header incl. U_eV=.. mix=.. t1=<t> eV tag=..
        seed  S_total  sum_abs_m  gap_eV  converged  iters  final_error
    data/hubbard_seed_sweep_<tag>_traces/seed<v>.txt
        iter  error=max|dn_i|  N_total  S_z

Produces a 1x3 figure:

  (1) iterations vs seed              — every start converges (iters may vary)
  (2) N_total(iter) per seed          — population conserved throughout
  (3) residual vs iteration per seed  — every start finds the same loop

Usage:
    python3 ploting/hubbard_seed_sweep_plot.py data/hubbard_seed_sweep_<tag>.txt
"""

import sys
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 110,
})

C_ITER = "#2980b9"
C_NTOT = "#16a085"
C_BAD  = "#c0392b"

TOL = 1e-8


def parse_meta(path):
    meta = {"t": 2.8, "U": None, "mix": None, "tag": path.stem}
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            continue
        for key, dst in (("t1", "t"), ("U_eV", "U"), ("mix", "mix")):
            m = re.search(rf"{key}=([\-\d.eE+]+)", line)
            if m:
                meta[dst] = float(m.group(1))
        m = re.search(r"tag=(\S+)", line)
        if m:
            meta["tag"] = m.group(1)
    meta["t"] = abs(meta["t"])
    return meta


def load_by_seed(directory):
    """Return dict seed(float) -> ndarray for files matching seed<v>.txt."""
    out = {}
    if not directory.is_dir():
        return out
    for f in directory.glob("seed*.txt"):
        m = re.search(r"seed([\-\d.eE+]+)\.txt", f.name)
        if not m:
            continue
        arr = np.atleast_2d(np.loadtxt(f, comments="#"))
        if arr.size:
            out[float(m.group(1))] = arr
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    data_path = Path(sys.argv[1])
    meta = parse_meta(data_path)
    trace_dir = Path(str(data_path).replace(".txt", "_traces"))

    raw = np.atleast_2d(np.loadtxt(data_path, comments="#"))
    order = np.argsort(raw[:, 0])
    raw = raw[order]
    seed    = raw[:, 0]
    S_total = raw[:, 1]
    conv    = raw[:, 4].astype(int)
    iters   = raw[:, 5].astype(int)

    traces = load_by_seed(trace_dir)
    ok = conv == 1

    fig, (ax_it, ax_pop, ax_tr) = plt.subplots(1, 3, figsize=(17, 5))

    sub = []
    if meta["U"] is not None:
        sub.append(f"U = {meta['U']:g} eV")
    if meta["mix"] is not None:
        sub.append(f"mix = {meta['mix']:g}")
    subtitle = ("  (" + ", ".join(sub) + ")") if sub else ""
    fig.suptitle(f"Hubbard initial-guess sweep — outcome independent of the "
                 f"seed amplitude — {meta['tag']}{subtitle}", fontsize=13, y=0.99)

    XLAB = r"initial-guess amplitude  $m_\mathrm{seed}$"

    # ---- (1) iterations vs seed --------------------------------------------
    ax_it.plot(seed[ok], iters[ok], "-", color=C_ITER, lw=1.6, zorder=1)
    ax_it.scatter(seed[ok], iters[ok], color=C_ITER, s=45, zorder=3,
                  label="converged")
    if (~ok).any():
        ax_it.scatter(seed[~ok], iters[~ok], color=C_BAD, marker="x", s=70,
                      lw=2.2, zorder=4, label="NOT converged (hit max_iter)")
        ax_it.legend()
    ax_it.set_xlabel(XLAB)
    ax_it.set_ylabel("iterations to converge")
    ax_it.set_title("(1) initial guess vs iterations to converge")
    ax_it.grid(alpha=0.25)
    ax_it.set_ylim(bottom=0)

    # ---- (2) population conserved throughout -------------------------------
    if traces:
        for s in sorted(traces):
            arr = traces[s]
            ax_pop.plot(arr[:, 0], arr[:, 2], color=C_NTOT, lw=1.0, alpha=0.6)
        allN = np.concatenate([traces[s][:, 2] for s in traces])
        c = np.median(allN)
        ax_pop.set_ylim(c - 0.5, c + 0.5)
    ax_pop.set_xlabel("self-consistency iteration")
    ax_pop.set_ylabel(r"$N_\mathrm{total}$")
    ax_pop.set_title("(2) population conserved throughout")
    ax_pop.grid(alpha=0.25)

    # ---- (3) residual traces overlaid --------------------------------------
    if traces:
        svals = np.array(sorted(traces))
        norm = plt.Normalize(svals.min(), svals.max())
        cmap = cm.get_cmap("plasma")
        for s in svals:
            arr = traces[s]
            ax_tr.semilogy(arr[:, 0], np.maximum(arr[:, 1], 1e-16),
                           color=cmap(norm(s)), lw=1.3, alpha=0.9)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_tr, pad=0.02)
        cb.set_label(r"$m_\mathrm{seed}$")
    ax_tr.axhline(TOL, color="k", ls=":", lw=1.2, label=f"tol = {TOL:g}")
    ax_tr.set_xlabel("self-consistency iteration")
    ax_tr.set_ylabel(r"residual  $\max_i|\Delta n_i|$")
    ax_tr.set_title("(3) residual vs iteration per initial guess")
    ax_tr.grid(alpha=0.25, which="both")
    ax_tr.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = str(data_path).replace(".txt", ".png")
    out_pdf = str(data_path).replace(".txt", ".pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}\nwrote {out_pdf}")

    # ---- text summary -------------------------------------------------------
    print("\n=== Hubbard initial-guess sweep summary ===")
    print(f"  converged: {ok.sum()}/{len(seed)} seed amplitudes")
    if (~ok).any():
        print(f"  did NOT converge for seed = "
              f"{', '.join(f'{s:g}' for s in seed[~ok])}")
    if ok.any():
        print(f"  iters: min {iters[ok].min()}  max {iters[ok].max()}  "
              f"median {int(np.median(iters[ok]))}")
        print(f"  S_z spread over seeds: {np.ptp(S_total[ok]):.2e}  "
              f"(=> ground state is initial-guess-independent)")


if __name__ == "__main__":
    main()
