#!/usr/bin/env python3
"""
hubbard_hartree_sweep_plot.py — Hartree ON vs OFF comparison of the UHF ground
state, across the same mixing grid.

Reads the table written by hubbard_hartree_sweep.sh:

    data/hubbard_hartree_sweep_<tag>.txt
        # ... header incl.  U_eV=..  t1=<t> eV  coulomb=..  tag=..
        mix  hartree  S_total  sum_abs_m  gap_eV  converged  iters  final_error  mismatch_warned
    data/hubbard_hartree_sweep_<tag>_traces/mix<val>_h<0|1>.txt   (residual traces:
        iter  error=max|dn_i|  N_total  S_z)

Produces a 2x2 figure:

  (1) iterations-to-converge vs mix, Hartree ON vs OFF  — does the extra
      nonlocal feedback make convergence harder?
  (2) converged S_total and sum|m| vs mix, ON vs OFF     — does Hartree change
      the magnetic ground state (it should, whenever there is charge to
      redistribute: doping, edges, broken sublattice symmetry)
  (3) converged gap vs mix, ON vs OFF
  (4) residual traces at the SLOWEST shared mix, ON vs OFF overlaid

Usage:
    python3 ploting/hubbard_hartree_sweep_plot.py data/hubbard_hartree_sweep_<tag>.txt
"""

import sys
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 110,
})

C_OFF = "#2980b9"   # hartree = 0
C_ON  = "#c0392b"   # hartree = 1
C_BAD = "#7f8c8d"

TOL = 1e-8   # solver convergence tolerance (see solve_hubbard_mft)


def parse_meta(path):
    meta = {"t": 2.8, "U": None, "coulomb": None, "tag": path.stem}
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            continue
        for key, dst in (("t1", "t"), ("U_eV", "U"), ("coulomb", "coulomb")):
            m = re.search(rf"{key}=(\S+)", line)
            if m:
                meta[dst] = m.group(1)
        m = re.search(r"tag=(\S+)", line)
        if m:
            meta["tag"] = m.group(1)
    return meta


def load_traces(trace_dir):
    """Return dict (hartree_flag) -> list of (mix, iters[], error[]) sorted by mix."""
    out = {0: [], 1: []}
    if not trace_dir.is_dir():
        return out
    for f in trace_dir.glob("mix*_h*.txt"):
        m = re.search(r"mix([\-\d.eE+]+)_h([01])\.txt", f.name)
        if not m:
            continue
        arr = np.atleast_2d(np.loadtxt(f, comments="#"))
        if arr.size:
            out[int(m.group(2))].append((float(m.group(1)), arr[:, 0], arr[:, 1]))
    for h in out:
        out[h].sort(key=lambda z: z[0])
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    data_path = Path(sys.argv[1])
    meta = parse_meta(data_path)
    trace_dir = Path(str(data_path).replace(".txt", "_traces"))

    raw = np.atleast_2d(np.loadtxt(data_path, comments="#"))
    order = np.lexsort((raw[:, 0], raw[:, 1]))
    raw = raw[order]
    mix        = raw[:, 0]
    hartree    = raw[:, 1].astype(int)
    S_total    = raw[:, 2]
    sum_abs_m  = raw[:, 3]
    gap        = raw[:, 4]
    conv       = raw[:, 5].astype(int)
    iters      = raw[:, 6].astype(int)
    ferr       = raw[:, 7]
    warned     = raw[:, 8].astype(int) if raw.shape[1] > 8 else np.zeros_like(conv)

    off = hartree == 0
    on  = hartree == 1
    ok  = conv == 1

    traces = load_traces(trace_dir)

    fig, ((ax_it, ax_s), (ax_gap, ax_tr)) = plt.subplots(2, 2, figsize=(13, 10))

    utitle = f"  (U = {meta['U']} eV, coulomb = {meta['coulomb']})" if meta["U"] else ""
    fig.suptitle(f"Hubbard Hartree ON/OFF comparison — {meta['tag']}{utitle}",
                 fontsize=13, y=0.995)

    def series(mask, color, label, ax, y):
        m, yv, okv = mix[mask], y[mask], ok[mask]
        order = np.argsort(m)
        m, yv, okv = m[order], yv[order], okv[order]
        ax.plot(m, yv, "-", color=color, lw=1.6, zorder=1)
        ax.scatter(m[okv], yv[okv], color=color, s=45, zorder=3, label=label)
        if (~okv).any():
            ax.scatter(m[~okv], yv[~okv], color=C_BAD, marker="x", s=70,
                       lw=2.2, zorder=4)

    # ---- (1) iterations vs mix, ON vs OFF ----------------------------------
    series(off, C_OFF, "hartree = off", ax_it, iters)
    series(on,  C_ON,  "hartree = on",  ax_it, iters)
    ax_it.set_xlabel("mixing factor  mix")
    ax_it.set_ylabel("iterations to converge")
    ax_it.set_title("(1) convergence cost: ON vs OFF")
    ax_it.grid(alpha=0.25)
    ax_it.legend()

    # ---- (2) S_total & sum|m| vs mix, ON vs OFF ----------------------------
    ax_s.plot(mix[off][np.argsort(mix[off])], sum_abs_m[off][np.argsort(mix[off])],
              "o-", color=C_OFF, lw=1.6, ms=5, label="sum|m|  off")
    ax_s.plot(mix[on][np.argsort(mix[on])], sum_abs_m[on][np.argsort(mix[on])],
              "o-", color=C_ON, lw=1.6, ms=5, label="sum|m|  on")
    ax_s2 = ax_s.twinx()
    ax_s2.plot(mix[off][np.argsort(mix[off])], S_total[off][np.argsort(mix[off])],
               "s--", color=C_OFF, lw=1.2, ms=4, alpha=0.6, label="S_total  off")
    ax_s2.plot(mix[on][np.argsort(mix[on])], S_total[on][np.argsort(mix[on])],
               "s--", color=C_ON, lw=1.2, ms=4, alpha=0.6, label="S_total  on")
    ax_s.set_xlabel("mixing factor  mix")
    ax_s.set_ylabel("sum|m_i|  (solid)")
    ax_s2.set_ylabel("S_total  (dashed)")
    ax_s.set_title("(2) converged magnetization: ON vs OFF")
    ax_s.grid(alpha=0.25)
    h1, l1 = ax_s.get_legend_handles_labels()
    h2, l2 = ax_s2.get_legend_handles_labels()
    ax_s.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")

    # ---- (3) gap vs mix, ON vs OFF ------------------------------------------
    series(off, C_OFF, "hartree = off", ax_gap, gap)
    series(on,  C_ON,  "hartree = on",  ax_gap, gap)
    ax_gap.set_xlabel("mixing factor  mix")
    ax_gap.set_ylabel("gap  [eV]")
    ax_gap.set_title("(3) converged gap: ON vs OFF")
    ax_gap.grid(alpha=0.25)
    ax_gap.legend()

    # ---- (4) residual traces at the slowest shared mix ----------------------
    shared = sorted(set(m for m, _, _ in traces[0]) & set(m for m, _, _ in traces[1]))
    if shared:
        mv = shared[-1]  # smallest mix present in both = slowest = most informative
        mv = min(shared)
        for h, color, label in ((0, C_OFF, "off"), (1, C_ON, "on")):
            for m, it, err in traces[h]:
                if m == mv:
                    ax_tr.semilogy(it, np.maximum(err, 1e-16), color=color, lw=1.6,
                                   label=f"hartree={label}  (mix={mv:g})")
        ax_tr.axhline(TOL, color="k", ls=":", lw=1.2, label=f"tol = {TOL:g}")
        ax_tr.set_xlabel("self-consistency iteration")
        ax_tr.set_ylabel(r"residual  $\max_i|\Delta n_i|$")
        ax_tr.set_title(f"(4) residual vs iteration at mix={mv:g}: ON vs OFF")
        ax_tr.grid(alpha=0.25, which="both")
        ax_tr.legend(fontsize=8)
    else:
        ax_tr.set_axis_off()
        ax_tr.text(0.5, 0.5, "no shared mix values with traces", ha="center", va="center")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = str(data_path).replace(".txt", ".png")
    out_pdf = str(data_path).replace(".txt", ".pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}\nwrote {out_pdf}")

    # ---- text summary --------------------------------------------------------
    print("\n=== Hubbard Hartree ON/OFF sweep summary ===")
    for h, label in ((0, "OFF"), (1, "ON")):
        mask = hartree == h
        n_ok = ok[mask].sum()
        print(f"  hartree={label}: converged {n_ok}/{mask.sum()} mixes", end="")
        if n_ok:
            print(f"   S_total in [{S_total[mask][ok[mask]].min():.4g}, "
                  f"{S_total[mask][ok[mask]].max():.4g}]"
                  f"   gap in [{gap[mask][ok[mask]].min():.4g}, "
                  f"{gap[mask][ok[mask]].max():.4g}] eV")
        else:
            print()
    if ok[off].any() and ok[on].any():
        dS = abs(np.mean(S_total[off][ok[off]]) - np.mean(S_total[on][ok[on]]))
        dgap = abs(np.mean(gap[off][ok[off]]) - np.mean(gap[on][ok[on]]))
        print(f"  |ΔS_total| (off vs on): {dS:.4g}    |Δgap| (off vs on): {dgap:.4g} eV")
        if dS < 1e-6 and dgap < 1e-6:
            print("  -> Hartree makes NO difference here (nothing to redistribute:"
                  " check doping/edges).")
        else:
            print("  -> Hartree changes the converged ground state, as expected"
                  " when there is charge to redistribute.")
    n_warn_off = warned[off].sum() if off.any() else 0
    n_warn_on  = warned[on].sum() if on.any() else 0
    print(f"  main.cpp coulomb/hartree mismatch warning fired: "
          f"{n_warn_off}/{off.sum()} (off runs), {n_warn_on}/{on.sum()} (on runs)"
          f"  [expected: all off, none on, whenever coulomb=true]")


if __name__ == "__main__":
    main()
