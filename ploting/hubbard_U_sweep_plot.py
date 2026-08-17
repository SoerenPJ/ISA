#!/usr/bin/env python3
"""
hubbard_U_sweep_plot.py — sensitivity analysis of the static UHF magnetism vs U.

Reads the table written by  hubbard_U_sweep.sh:

    data/hubbard_U_sweep_<tag>.txt
        # ... header, incl.  # lieb_S=..  N_A=..  N_B=..  and  t1=<t> eV
        U_eV  S_total  sum_abs_m  gap_eV  converged  iters  final_error
    data/hubbard_U_sweep_<tag>_traces/U<val>.txt   (per-U residual traces)

and produces a single-panel figure:

  spin gap vs U, with the mean-field validity anchors shaded:
        U < 2t         mean-field quantitatively validated vs QMC/ED  ("safe")
        2.23t          honeycomb MFT semimetal->AFM insulator transition (Uc)
        2t .. 3.63t    gray zone: MFT overestimates gaps; edge physics still OK
        3.63t          bare cRPA U (expect quantitative overestimation)

Usage:
    python3 ploting/hubbard_U_sweep_plot.py data/hubbard_U_sweep_<tag>.txt [--baseline 3.0]
"""

import sys
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": 110,
})

C_GAP   = "#c0392b"
C_NET   = "#2c3e50"
C_STAG  = "#8e44ad"
C_LIEB  = "#16a085"
C_ITER  = "#2980b9"
C_ERR   = "#e67e22"


def parse_meta(path):
    """Pull t1, lieb_S, N_A, N_B from the '#' header lines."""
    meta = {"t": 2.8, "lieb_S": None, "N_A": None, "N_B": None, "tag": path.stem}
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            continue
        m = re.search(r"t1=([\-\d.eE+]+)", line)
        if m:
            meta["t"] = abs(float(m.group(1)))
        for key in ("lieb_S", "N_A", "N_B"):
            m = re.search(rf"{key}=([\-\d.eE+]+)", line)
            if m:
                meta[key] = float(m.group(1))
        m = re.search(r"tag=(\S+)", line)
        if m:
            meta["tag"] = m.group(1)
    return meta


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    data_path = Path(sys.argv[1])
    baseline = None
    if "--baseline" in sys.argv:
        baseline = float(sys.argv[sys.argv.index("--baseline") + 1])

    meta = parse_meta(data_path)
    t = meta["t"]
    lieb_S = meta["lieb_S"]

    raw = np.loadtxt(data_path, comments="#")
    raw = np.atleast_2d(raw)
    order = np.argsort(raw[:, 0])
    raw = raw[order]
    U       = raw[:, 0]
    S_total = raw[:, 1]
    stag    = raw[:, 2]          # sum |m_i|
    gap     = raw[:, 3]
    conv    = raw[:, 4].astype(int)
    iters   = raw[:, 5].astype(int)
    ferr    = raw[:, 6]

    # literature anchors, in eV
    U_safe = 2.0 * t          # mean-field validated below this
    U_c    = 2.23 * t         # honeycomb MFT semimetal -> AFM insulator
    U_crpa = 3.63 * t         # bare cRPA value

    fig, ax1 = plt.subplots(figsize=(7.5, 5.5))

    def anchors(ax, ymax):
        ax.axvspan(0, U_safe, color="#2ecc71", alpha=0.08, zorder=0)
        ax.axvspan(U_safe, U_crpa, color="#f1c40f", alpha=0.08, zorder=0)
        ax.axvspan(U_crpa, max(U.max(), U_crpa) * 1.02,
                   color="#e74c3c", alpha=0.07, zorder=0)
        ax.axvline(U_crpa, color="#c0392b", ls=":", lw=1.2, zorder=1)
        if baseline is not None:
            ax.axvline(baseline, color="k", ls="-", lw=1.4, alpha=0.7, zorder=1)

    # ---- (1) spin gap vs U --------------------------------------------------
    anchors(ax1, gap.max())
    ax1.plot(U, gap, "o-", color=C_GAP, lw=2, ms=5)
    ax1.set_xlabel("Hubbard U  (eV)")
    ax1.set_ylabel("HOMO–LUMO spin gap  (eV)")
    ax1.set_title("Spin gap vs U — mean-field validity window")
    ax1.set_xlim(0, U.max() * 1.02)
    ax1.set_ylim(bottom=0)
    # secondary axis in units of t
    secx = ax1.secondary_xaxis("top", functions=(lambda x: x / t, lambda x: x * t))
    secx.set_xlabel("U / t")
    ax1.text(U_safe / 2, ax1.get_ylim()[1] * 0.92, "safe\n(U<2t)",
             ha="center", va="top", fontsize=9, color="#1e8449")
    ax1.text((U_safe + U_crpa) / 2, ax1.get_ylim()[1] * 0.92,
             "gray zone", ha="center", va="top", fontsize=9, color="#b7950b")
    
    ax1.annotate(f"cRPA $3.63t={U_crpa:.1f}$ eV", (U_crpa, ax1.get_ylim()[1] * 0.5),
                 rotation=90, va="center", ha="right", fontsize=8.5, color="#c0392b")

    fig.suptitle(f"Hubbard-U sensitivity analysis — {meta['tag']}  "
                 f"(t = {t:g} eV)", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    out_png = str(data_path).replace(".txt", ".png")
    out_pdf = str(data_path).replace(".txt", ".pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}\nwrote {out_pdf}")

    # ---- text summary / model-consistency cutoff ----------------------------
    print("\n=== Hubbard U sensitivity summary ===")
    print(f"  t = {t:g} eV   2t = {U_safe:.2f} eV   Uc = 2.23t = {U_c:.2f} eV"
          f"   cRPA 3.63t = {U_crpa:.2f} eV")
    if lieb_S is not None:
        near = np.abs(np.abs(S_total) - lieb_S) < 0.05
        if near.any():
            print(f"  net |S_z| reaches Lieb S={lieb_S:g} for U >= "
                  f"{U[near].min():.2f} eV")
    print(f"  spin gap at U=2t   ~ {np.interp(U_safe, U, gap):.2f} eV")
    print(f"  spin gap at U=Uc   ~ {np.interp(U_c,    U, gap):.2f} eV")
    print(f"  spin gap at cRPA U ~ {np.interp(U_crpa, U, gap):.2f} eV")
    print("  -> report a baseline U (justified) and show the gap leaving the")
    print("     model's validity window as U -> cRPA (known MFT overestimation).")


if __name__ == "__main__":
    main()
