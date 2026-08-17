#!/usr/bin/env python3
"""
hubbard_mix_sweep_plot.py — how the linear-mixing factor affects UHF convergence.

Reads the table written by  hubbard_mix_sweep.sh:

    data/hubbard_mix_sweep_<tag>.txt
        # ... header incl.  U_eV=..  t1=<t> eV  tag=..
        mix  S_total  sum_abs_m  gap_eV  converged  iters  final_error
    data/hubbard_mix_sweep_<tag>_traces/mix<val>.txt   (per-mix residual traces:
        iter  error=max|dn_i|  N_total  S_z)

Produces a 1x3 figure:

  (1) iterations-to-converge vs mix    — the cost of the damping choice
  (2) final residual vs mix vs tol     — which mixes actually hit the tolerance
  (3) residual vs iteration per mix    — HOW mixing damps the Stoner loop

Usage:
    python3 ploting/hubbard_mix_sweep_plot.py data/hubbard_mix_sweep_<tag>.txt
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
C_ERR  = "#e67e22"
C_BAD  = "#c0392b"

TOL = 1e-8   # solver convergence tolerance (see solve_hubbard_mft)


def parse_meta(path):
    meta = {"t": 2.8, "U": None, "tag": path.stem}
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            continue
        for key, dst in (("t1", "t"), ("U_eV", "U")):
            m = re.search(rf"{key}=([\-\d.eE+]+)", line)
            if m:
                meta[dst] = abs(float(m.group(1)))
        m = re.search(r"tag=(\S+)", line)
        if m:
            meta["tag"] = m.group(1)
    return meta


def load_traces(trace_dir):
    """Return list of (mix, iters[], error[]) sorted by mix."""
    out = []
    if not trace_dir.is_dir():
        return out
    for f in trace_dir.glob("mix*.txt"):
        m = re.search(r"mix([\-\d.eE+]+)\.txt", f.name)
        if not m:
            continue
        arr = np.atleast_2d(np.loadtxt(f, comments="#"))
        if arr.size:
            out.append((float(m.group(1)), arr[:, 0], arr[:, 1]))
    out.sort(key=lambda z: z[0])
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
    mix     = raw[:, 0]
    S_total = raw[:, 1]
    conv    = raw[:, 4].astype(int)
    iters   = raw[:, 5].astype(int)
    ferr    = raw[:, 6]
    ok = conv == 1

    traces = load_traces(trace_dir)

    fig, (ax_it, ax_fe, ax_tr) = plt.subplots(1, 3, figsize=(17, 5))

    utitle = f"  (U = {meta['U']:g} eV)" if meta["U"] is not None else ""
    fig.suptitle(f"Hubbard mixing sweep — convergence vs linear-mixing factor "
                 f"— {meta['tag']}{utitle}", fontsize=13, y=0.99)

    # ---- (1) iterations vs mix ---------------------------------------------
    ax_it.plot(mix, iters, "-", color=C_ITER, lw=1.6, zorder=1)
    ax_it.scatter(mix[ok], iters[ok], color=C_ITER, s=45, zorder=3,
                  label="converged")
    if (~ok).any():
        ax_it.scatter(mix[~ok], iters[~ok], color=C_BAD, marker="x", s=70,
                      lw=2.2, zorder=4, label="NOT converged (hit max_iter)")
    ax_it.set_xlabel("mixing factor  mix")
    ax_it.set_ylabel("iterations to converge")
    ax_it.set_title("(1) mixing factor vs iterations to converge")
    ax_it.grid(alpha=0.25)
    ax_it.legend()

    # ---- (2) final residual vs mix -----------------------------------------
    ax_fe.semilogy(mix, np.maximum(ferr, 1e-16), "o-", color=C_ERR, lw=1.6, ms=5)
    if (~ok).any():
        ax_fe.scatter(mix[~ok], np.maximum(ferr[~ok], 1e-16), color=C_BAD,
                      marker="x", s=70, lw=2.2, zorder=4, label="NOT converged")
        ax_fe.legend()
    ax_fe.axhline(TOL, color="k", ls=":", lw=1.2, label=f"tol = {TOL:g}")
    ax_fe.set_xlabel("mixing factor  mix")
    ax_fe.set_ylabel("final residual")
    ax_fe.set_title("(2) mixing factor vs tolerance")
    ax_fe.grid(alpha=0.25, which="both")

    # ---- (3) residual traces overlaid --------------------------------------
    if traces:
        mvals = np.array([t[0] for t in traces])
        norm = plt.Normalize(mvals.min(), mvals.max())
        cmap = cm.get_cmap("viridis")
        for mv, it, err in traces:
            ax_tr.semilogy(it, np.maximum(err, 1e-16), color=cmap(norm(mv)),
                           lw=1.4, alpha=0.9)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_tr, pad=0.02)
        cb.set_label("mix")
    ax_tr.axhline(TOL, color="k", ls=":", lw=1.2, label=f"tol = {TOL:g}")
    ax_tr.set_xlabel("self-consistency iteration")
    ax_tr.set_ylabel(r"residual  $\max_i|\Delta n_i|$")
    ax_tr.set_title("(3) residual vs iteration per mixing factor")
    ax_tr.grid(alpha=0.25, which="both")
    ax_tr.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    out_png = str(data_path).replace(".txt", ".png")
    out_pdf = str(data_path).replace(".txt", ".pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}\nwrote {out_pdf}")

    # ---- text summary -------------------------------------------------------
    print("\n=== Hubbard mixing sweep summary ===")
    print(f"  converged: {ok.sum()}/{len(mix)} mixes")
    if (~ok).any():
        print(f"  did NOT converge in max_iter for mix = "
              f"{', '.join(f'{m:g}' for m in mix[~ok])}")
    if ok.any():
        print(f"  fastest: mix={mix[ok][np.argmin(iters[ok])]:g} "
              f"({iters[ok].min()} iters);  "
              f"slowest converged: mix={mix[ok][np.argmax(iters[ok])]:g} "
              f"({iters[ok].max()} iters)")
        print(f"  S_z spread over converged mixes: {np.ptp(S_total[ok]):.2e} "
              f"(=> the converged state is mix-independent)")


if __name__ == "__main__":
    main()
