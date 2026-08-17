#!/usr/bin/env python3
"""
hubbard_doping_maps.py — spatial charge & spin maps of the UHF ground state vs doping.

Reads the folder written by  hubbard_doping_sweep.sh:

    data/hubbard_doping_<tag>_maps/
        mu<val>_hartOFF.txt  magnetization.txt WITHOUT self-consistent Hartree
        mu<val>_hartON.txt   magnetization.txt WITH    self-consistent Hartree
        levels_mu<val>_*.txt eigenvalue spectrum per SCF iteration (ON/OFF/U0)
        bond_indices.txt     lattice skeleton (optional)

    Doping is set by the chemical potential mu (grand-canonical), so N floats and is
    read off each file (sum of n_up+n_dn); rows are labelled by mu, N and excess dN.

Each file has columns:  site x y sublattice n_up n_dn m_i.

Produces one figure, a grid with one row per doping Q and three columns:

    (1) probability density n_i = sum_occ |psi(i)|^2, NO self-consistency
    (2) probability density n_i, WITH self-consistency  -> Coulomb pushes weight to edges
    (3) spin m_i, WITH self-consistency                 -> how the magnet changes with doping

n_i (= n_up + n_dn) is the Fermi-weighted sum of the squared eigenvector amplitudes
over the occupied states, i.e. the electron probability density at site i.

Columns (1) vs (2) reproduce the paper's "w/o vs w/ self-consistency" comparison
(Yu/Cox/Garcia de Abajo, PRL 117, 123904, Fig. 2 / Fig. S1a).

Three further figures are written next to the maps:
    doping_residuals.png   SCF residual vs iteration + HOMO-LUMO gap vs doping
    doping_population.png  population drift in the loop + final N vs doping
    doping_level_flow.png  EIGENVALUES vs SCF iteration (one line per level, left
                           axis ticked by state index, right axis in eV), clean
                           Hartree (U=0) next to Hartree + Hubbard (U>0)

Usage:
    python3 ploting/hubbard_doping_maps.py data/hubbard_doping_<tag>_maps
    python3 ploting/hubbard_doping_maps.py data/hubbard_doping_<tag>_maps 0 1.5 3
        (extra args = only plot these mu values; default = every mu found)
    python3 ploting/hubbard_doping_maps.py data/hubbard_doping_<tag>_maps --zoom=10
        (level-flow figure keeps only 10 states either side of the Fermi level)
"""

import sys
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 110,
})

AU_NM = 0.0529177          # bohr -> nm
CMAP_DENS   = "viridis"    # probability density n_i >= 0 (sequential)
CMAP_SPIN   = "bwr_r"      # m_i: blue = up, red = down


def load(path):
    d = np.atleast_2d(np.loadtxt(path, comments="#"))
    return {"x": d[:, 1] * AU_NM, "y": d[:, 2] * AU_NM,
            "n": d[:, 4] + d[:, 5], "m": d[:, 6]}


def read_gap(path):
    """HOMO-LUMO gap (eV) parsed from a magnetization.txt header line
    (# ... gap_eV=<value> ...). Returns None if the file/field is missing."""
    try:
        with open(path) as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                m = re.search(r"gap_eV=([\-\d.eE+]+)", line)
                if m:
                    return float(m.group(1))
    except OSError:
        pass
    return None


def plot_residuals(folder, Qs, lab, tol=1e-8):
    """Second figure (two panels):
      LEFT  — SCF convergence residual max_i|dn_i| vs iteration for each doping value
              (needs conv_<pfx><val>_{ON,OFF}.txt, copied by the sweep).
      RIGHT — HOMO-LUMO gap vs doping (from the magnetization.txt headers). The gap
              collapse is the spectral reason the doped runs converge slowly:
              a near-degenerate frontier makes the fixed point ill-conditioned.
    Together they show cost (iterations) alongside its cause (the gap)."""
    OFFc, ONc = "#0072B2", "#D55E00"              # Hartree OFF / ON
    pfx, sym, unit, axlabel = lab["prefix"], lab["sym"], lab["unit"], lab["axlabel"]

    series = []                                   # (doping, tag, iters, residual)
    for Q in Qs:
        for tag in ("OFF", "ON"):
            hits = [folder / f"conv_{pfx}{Q:g}_{tag}.txt", folder / f"conv_{pfx}{Q}_{tag}.txt"]
            p = next((h for h in hits if h.exists()), None)
            if p is None:
                continue
            d = np.atleast_2d(np.loadtxt(p, comments="#"))
            if d.size and d.shape[1] >= 2:
                series.append((Q, tag, d[:, 0], d[:, 1]))
    if not series:
        print(f"no conv_{pfx}*_*.txt traces in folder — re-run the sweep to get the "
              "residuals figure (skipping).")
        return

    uQ = sorted({s[0] for s in series})
    cmap = plt.get_cmap("viridis")
    col = {q: cmap(0.12 + 0.76 * (i / max(1, len(uQ) - 1))) for i, q in enumerate(uQ)}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # -- left: residual vs iteration --
    for Q, tag, it, res in series:
        axL.semilogy(it, res, color=col[Q],
                     lw=2.2 if tag == "ON" else 1.4,
                     ls="-" if tag == "ON" else "--",
                     label=f"${sym}$={Q:g} {unit}  {tag}  ({len(it)} it)")
    axL.axhline(tol, color="0.5", lw=1.0, ls=":")
    axL.text(0.99, tol * 1.4, f"tol {tol:g}", transform=axL.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=8.5, color="0.4")
    axL.set_xlabel("self-consistency iteration")
    axL.set_ylabel(r"residual   $\max_i|\Delta n_i|$")
    axL.set_title("SCF convergence trace")
    axL.grid(True, which="major", color="0.85", lw=0.6)
    axL.legend(frameon=False, fontsize=8.5, ncol=2,
               title="solid = Hartree ON, dashed = OFF")

    # -- right: HOMO-LUMO gap vs mu --
    # A vanishing gap means a (near-)degenerate frontier; numerically it comes out
    # as ~1e-14 eV, which would stretch the log axis over ~14 dead decades. Floor
    # the display at 1 meV and flag those points as gapless so the meaningful
    # 0.03–2 eV variation stays readable.
    GAP_FLOOR = 1e-3
    gmax = GAP_FLOOR
    for tag, c in (("OFF", OFFc), ("ON", ONc)):
        pts = []
        for Q in Qs:
            for cand in (folder / f"{pfx}{Q:g}_hart{tag}.txt", folder / f"{pfx}{Q}_hart{tag}.txt"):
                if cand.exists():
                    g = read_gap(cand)
                    if g is not None:
                        pts.append((Q, g))
                    break
        if pts:
            xs, gs = zip(*sorted(pts))
            gd = [max(g, GAP_FLOOR) for g in gs]
            gmax = max(gmax, max(gs))
            axR.semilogy(xs, gd, "-o", color=c, lw=1.8, ms=6, label=f"Hartree {tag}")
            gapless = [(x, GAP_FLOOR) for x, g in zip(xs, gs) if g < GAP_FLOOR]
            if gapless:
                gx, gy = zip(*gapless)
                axR.scatter(gx, gy, marker="v", s=80, facecolors="none",
                            edgecolors=c, linewidths=1.4, zorder=5)
    axR.axhline(GAP_FLOOR, color="0.6", lw=0.8, ls=":")
    axR.text(0.015, GAP_FLOOR * 1.25, "gapless  (gap < 1 meV)",
             transform=axR.get_yaxis_transform(), fontsize=8.2, color="0.4", va="bottom")
    axR.set_ylim(GAP_FLOOR * 0.6, gmax * 3)
    axR.set_xlabel(axlabel)
    axR.set_ylabel("HOMO–LUMO gap  (eV)")
    axR.set_title("Frontier (HOMO–LUMO) gap vs doping")
    axR.grid(True, which="both", color="0.88", lw=0.5)
    axR.legend(frameon=False, fontsize=9)

    fig.suptitle(f"SCF convergence vs doping — {folder.name}", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"doping_residuals.{ext}", bbox_inches="tight")
    print(f"wrote {folder / 'doping_residuals.png'}\n"
          f"wrote {folder / 'doping_residuals.pdf'}")


def load_levels(path):
    """Read a levels_<pfx><val>_<tag>.txt trace written by the sweeps.

    Long format, one row per (sampled iteration, state):
        iter  state_index  energy_eV  occupation  spin(+1=up,-1=dn)
    Returns it reshaped to [n_sample, n_state] arrays. The spectrum is sorted
    ascending every iteration, so column k is the same state slot throughout."""
    d = np.atleast_2d(np.loadtxt(path, comments="#"))
    if d.size == 0 or d.shape[1] < 5:
        return None
    ns = len(np.unique(d[:, 1].astype(int)))
    if ns == 0 or d.shape[0] % ns:
        return None
    homo = None
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.search(r"homo_index=(\d+)", line)
            if m:
                homo = int(m.group(1))
    k = d[:, 1].reshape(-1, ns)[0].astype(int)
    # "homo" is returned as a COLUMN position, not a state index: a trace may hold
    # only a slice of the spectrum (older sweeps wrote a frontier window), in which
    # case the global HOMO index is not a valid column. k[] maps back to the labels.
    homo_col = int(np.clip(np.searchsorted(k, homo) if homo is not None else ns // 2 - 1,
                           0, ns - 1))
    return {"it": d[:, 0].reshape(-1, ns)[:, 0],
            "k":  k,
            "E":  d[:, 2].reshape(-1, ns),
            "f":  d[:, 3].reshape(-1, ns),
            "s":  d[:, 4].reshape(-1, ns).astype(int),
            "homo": homo_col}


def plot_level_flow(folder, Qs, lab, zoom=None):
    """Level-flow figure — how the EIGENVALUES move while the self-consistency loop
    runs. Every iteration rebuilds the mean-field H, so its spectrum drifts until it
    stops moving; this is that drift, one line per eigenvalue.

    Each panel:
        x  = self-consistency iteration
        y  = eigenvalue energy, one trajectory E_k(iteration) per state
        left  axis = ticked with the STATE INDEX k, each tick sitting at that state's
                     converged energy (eigenvalues are sorted, so k is a fixed slot)
        right axis = the same vertical coordinate read off in eV
    Blue = spin up, orange = spin down; solid = occupied, dashed = empty. The dashed
    grey line is the Fermi level (mid-gap of the converged spectrum).

    Two columns, the with/without-Hubbard comparison:
      LEFT  — clean Hartree, U = 0: only the charge Hartree phi_i relaxes; the
              spectrum shifts and settles, staying spin degenerate.
      RIGHT — Hartree + Hubbard, U > 0: the same relaxation PLUS the Stoner term,
              which pushes the up and down levels apart and opens the magnetic gap
              at the frontier.
    One row per doping value. Pass zoom=n to keep only the n states either side of
    the Fermi level (the frontier, where the magnetism happens)."""
    UPC, DNC = "#0072B2", "#D55E00"          # spin up / spin down levels
    pfx, sym, unit = lab["prefix"], lab["sym"], lab["unit"]

    def find(Q, tag):
        for cand in (folder / f"levels_{pfx}{Q:g}_{tag}.txt",
                     folder / f"levels_{pfx}{Q}_{tag}.txt"):
            if cand.exists():
                return cand
        return None

    # column pair: prefer the U=0 vs U>0 comparison the sweeps now write; fall back
    # to Hartree OFF vs ON if the folder predates the WITH_U0 runs.
    has_U0 = any(find(Q, "U0") for Q in Qs)
    cols = (("U0", "clean Hartree  ($U=0$, no Hubbard)"),
            ("ON", "Hartree + Hubbard  ($U>0$)")) if has_U0 else \
           (("OFF", "Hubbard only  (no self-consistent Hartree)"),
            ("ON",  "Hartree + Hubbard"))

    data = {(Q, tag): load_levels(p)
            for Q in Qs for tag, _ in cols
            if (p := find(Q, tag)) is not None}
    data = {k: v for k, v in data.items() if v is not None}
    if not data:
        print(f"no levels_{pfx}*_*.txt traces in folder — re-run the sweep to get the "
              "level-flow figure (skipping).")
        return

    rows = [Q for Q in Qs if any((Q, tag) in data for tag, _ in cols)]
    fig, axes = plt.subplots(len(rows), 2, figsize=(13.0, 4.2 * len(rows)),
                             squeeze=False)

    def index_ticks(L, k_lo, k_hi, span):
        """(energy, label) pairs for the left axis: an even spread of state indices
        over the plotted range, forced to include HOMO and LUMO, then thinned so
        labels sitting at nearly the same energy don't overlap. HOMO and LUMO get
        merged into one "45/46" tick when the gap itself is below the label spacing
        (the gapless case), which otherwise prints the two on top of each other."""
        homo = int(L["homo"])
        lbl  = lambda c: str(L["k"][c])        # column position -> printed state index
        must = [c for c in (homo, homo + 1) if k_lo <= c <= k_hi]
        out = []
        if len(must) == 2 and abs(L["E"][-1, must[1]] - L["E"][-1, must[0]]) < 0.03 * span:
            out.append((0.5 * (L["E"][-1, must[0]] + L["E"][-1, must[1]]),
                        f"{lbl(must[0])}/{lbl(must[1])}"))
        else:
            out += [(L["E"][-1, c], lbl(c)) for c in must]
        taken = [e for e, _ in out]
        for c in sorted(set(np.linspace(k_lo, k_hi, 9).round().astype(int)) - set(must)):
            e = L["E"][-1, c]
            if all(abs(e - t) > 0.045 * span for t in taken):
                out.append((e, lbl(c))); taken.append(e)
        return sorted(out)

    def draw(ax, L):
        homo = int(L["homo"])
        n_st = L["E"].shape[1]
        lumo = min(homo + 1, n_st - 1)
        k_lo, k_hi = (0, n_st - 1) if zoom is None else \
                     (max(0, homo - zoom + 1), min(n_st - 1, homo + zoom))

        # one line per eigenvalue. Two collections (occupied / empty) keep the
        # rendering cheap when a big flake contributes hundreds of levels.
        segs = {True: [], False: []}
        cols_ = {True: [], False: []}
        for k in range(k_lo, k_hi + 1):
            occ = L["f"][-1, k] > 0.5
            segs[occ].append(np.column_stack([L["it"], L["E"][:, k]]))
            cols_[occ].append(UPC if L["s"][-1, k] > 0 else DNC)
        ax.add_collection(LineCollection(segs[True], colors=cols_[True],
                                         linewidths=1.4, alpha=0.95, zorder=3))
        ax.add_collection(LineCollection(segs[False], colors=cols_[False],
                                         linewidths=0.9, alpha=0.6,
                                         linestyles="dashed", zorder=2))

        E = L["E"][:, k_lo:k_hi + 1]
        lo, hi = float(E.min()), float(E.max())
        span = max(hi - lo, 1e-6)
        ax.set_xlim(float(L["it"][0]), float(L["it"][-1]))
        ax.set_ylim(lo - 0.04 * span, hi + 0.04 * span)

        # Fermi level = mid-gap of the converged spectrum
        E_F = 0.5 * (L["E"][-1, homo] + L["E"][-1, lumo])
        ax.axhline(E_F, color="0.35", lw=1.1, ls="--", zorder=4)
        ax.text(0.012, E_F + 0.012 * span, "Fermi level", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=8.0, color="0.35")

        # LEFT axis: state index k, each tick placed at that state's converged energy.
        ticks = index_ticks(L, k_lo, k_hi, span)
        ax.set_yticks([e for e, _ in ticks])
        ax.set_yticklabels([s for _, s in ticks], fontsize=8.5)
        ax.tick_params(axis="y", length=3)
        # RIGHT axis (twin): the same vertical coordinate, read off in eV.
        ax2 = ax.twinx()
        ax2.set_ylim(*ax.get_ylim())
        ax2.set_ylabel("eigenvalue energy  (eV)", fontsize=9)
        ax2.tick_params(axis="y", labelsize=8.5)

        gap = L["E"][-1, lumo] - L["E"][-1, homo]
        ax.text(0.015, 0.02, f"{int(L['it'][-1])} iterations   final gap = {gap:.3f} eV",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
                color="0.2", bbox=dict(fc="white", ec="none", alpha=0.85, pad=2.0))
        ax.grid(True, axis="x", color="0.92", lw=0.5)

    for r, Q in enumerate(rows):
        for c, (tag, title) in enumerate(cols):
            ax = axes[r, c]
            L = data.get((Q, tag))
            if L is None:
                ax.text(0.5, 0.5, f"no {tag} run", transform=ax.transAxes,
                        ha="center", va="center", color="0.5")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            draw(ax, L)
            if r == 0:
                ax.set_title(title, fontsize=10.5)
            if r == len(rows) - 1:
                ax.set_xlabel("self-consistency iteration")
            if c == 0:
                ax.set_ylabel(f"${sym}$ = {Q:g} {unit}\nstate index  $k$")
    fig.suptitle(f"Eigenvalues during the self-consistency iteration — {folder.name}",
                 y=1.0)
    fig.tight_layout()
    # legend below the grid: the panels are full of lines, so anything inside a
    # panel would sit on top of data.
    fig.legend(handles=[
        Line2D([], [], color=UPC, lw=1.4, label=r"spin $\uparrow$, occupied"),
        Line2D([], [], color=DNC, lw=1.4, label=r"spin $\downarrow$, occupied"),
        Line2D([], [], color="0.4", lw=0.9, ls="--", label="empty"),
        Line2D([], [], color="0.35", lw=1.1, ls="--", label="Fermi level")],
        frameon=False, fontsize=9, ncol=4, loc="lower center",
        bbox_to_anchor=(0.5, -0.02))
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"doping_level_flow.{ext}", bbox_inches="tight")
    print(f"wrote {folder / 'doping_level_flow.png'}\n"
          f"wrote {folder / 'doping_level_flow.pdf'}")


def plot_population(folder, Qs, lab):
    """Population figure — is the electron count conserved?

      LEFT  — N_total vs SCF iteration, one line per doping value (Hartree ON solid /
              OFF dashed), read from conv_<pfx><val>_{ON,OFF}.txt (col 3 = N_total). A
              FLAT line means population is conserved through the self-consistency loop;
              any slope/jump is a leak. The 'does it stay constant' check the maps hide.
      RIGHT — final N_total vs doping (from population.txt, or the conv-trace last point).
              For the mu sweep this rises (grand-canonical). For the Q sweep it is the
              staircase N = N_sites + Q; it is the reference for reading the left panel.

    Both panels degrade gracefully: they use population.txt if the sweep wrote it, and
    fall back to the conv traces otherwise."""
    OFFc, ONc = "#0072B2", "#D55E00"
    pfx, sym, unit, axlabel = lab["prefix"], lab["sym"], lab["unit"], lab["axlabel"]

    # per-iteration N traces from the convergence files (col 3 = N_total)
    traces = []                                   # (doping, tag, iters, N_total)
    for Q in Qs:
        for tag in ("OFF", "ON"):
            p = next((h for h in (folder / f"conv_{pfx}{Q:g}_{tag}.txt",
                                  folder / f"conv_{pfx}{Q}_{tag}.txt") if h.exists()), None)
            if p is None:
                continue
            d = np.atleast_2d(np.loadtxt(p, comments="#"))
            if d.size and d.shape[1] >= 3:
                traces.append((Q, tag, d[:, 0], d[:, 2]))

    # final-N vs doping, preferring the explicit population.txt tracker
    finalN = {"OFF": [], "ON": []}                # tag -> list of (doping, N_final)
    expected = []                                 # (doping, N_expected) — Q sweep only (col 8)
    pop = folder / "population.txt"
    if pop.exists():
        for ln in pop.read_text().splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            c = ln.split()
            if len(c) >= 3 and c[1] in finalN:
                try:
                    finalN[c[1]].append((float(c[0]), float(c[2])))
                    if len(c) >= 8:               # canonical sweep also logs N_expected
                        expected.append((float(c[0]), float(c[7])))
                except ValueError:
                    pass
    if not finalN["OFF"] and not finalN["ON"]:     # fall back to last conv-trace point
        for Q, tag, it, N in traces:
            finalN[tag].append((Q, N[-1]))

    if not traces and not (finalN["OFF"] or finalN["ON"]):
        print(f"no population.txt / conv_{pfx}*.txt data in folder — re-run the sweep "
              "to get the population figure (skipping).")
        return

    uQ = sorted({t[0] for t in traces}) or Qs
    cmap = plt.get_cmap("viridis")
    col = {q: cmap(0.12 + 0.76 * (i / max(1, len(uQ) - 1))) for i, q in enumerate(uQ)}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # -- left: population DRIFT vs iteration (conservation check) --
    # Plot N(iter) - N(0), not raw N: the raw counts span ~62..100 across mu, which
    # would swamp any sub-electron leak on a shared axis. Referencing each trace to its
    # own start puts every run on a common zero baseline, so a drift of even 1e-4 e is
    # visible. A line pinned to 0 = population conserved through the loop.
    worst = 0.0
    for Q, tag, it, N in traces:
        dN = N - N[0]
        worst = max(worst, np.abs(dN).max())
        axL.plot(it, dN, color=col.get(Q, "0.5"),
                 lw=2.2 if tag == "ON" else 1.4,
                 ls="-" if tag == "ON" else "--",
                 label=f"${sym}$={Q:g} {unit}  {tag}  "
                       f"($\\Delta N$={dN[-1]:+.1e})")
    axL.axhline(0.0, color="0.5", lw=0.8, ls=":")
    axL.set_xlabel("self-consistency iteration")
    axL.set_ylabel(r"population drift  $N(\mathrm{it})-N(0)$   (electrons)")
    axL.set_title(f"Population drift vs SCF iteration\n"
                  f"(0 = conserved;  worst $|\\Delta N|$ = {worst:.1e} e)")
    axL.grid(True, color="0.88", lw=0.5)
    if traces:
        axL.legend(frameon=False, fontsize=8.0, ncol=2,
                   title="solid = Hartree ON, dashed = OFF")

    # -- right: final N vs doping (mu: grand-canonical rise; Q: N = N_sites + Q staircase) --
    # For the canonical (Q) sweep, overlay the target N_sites+Q as a reference: any gap
    # between it and the achieved N_final is the electron the fill missed (e.g. an unfilled
    # zero mode). This is the visual proof that the population does/doesn't hit its target.
    if not lab["grand"] and expected:
        xs, ne = zip(*sorted(set(expected)))
        axR.plot(xs, ne, "--s", color="0.45", lw=1.4, ms=6, mfc="none", zorder=1,
                 label=r"target  $N_{\rm sites}+Q$")
    for tag, c in (("OFF", OFFc), ("ON", ONc)):
        if finalN[tag]:
            xs, ns = zip(*sorted(finalN[tag]))
            axR.plot(xs, ns, "-o", color=c, lw=1.8, ms=6, label=f"Hartree {tag}")
    axR.set_xlabel(axlabel)
    axR.set_ylabel(r"final population  $N_{\rm total}$")
    axR.set_title("Population vs doping\n" + (
        "(grand-canonical: N floats with $\\mu$)" if lab["grand"]
        else "(canonical: N pinned to $N_{\\rm sites}+Q$)"))
    axR.grid(True, color="0.88", lw=0.5)
    if finalN["OFF"] or finalN["ON"]:
        axR.legend(frameon=False, fontsize=9)

    fig.suptitle(f"Population conservation & doping — {folder.name}", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"doping_population.{ext}", bbox_inches="tight")
    print(f"wrote {folder / 'doping_population.png'}\n"
          f"wrote {folder / 'doping_population.pdf'}")


def main():
    # --zoom=N restricts the level-flow figure to the N states either side of the
    # Fermi level; pull it out of argv so the rest stays "folder + doping values".
    argv = [a for a in sys.argv[1:] if not a.startswith("--zoom")]
    zoom = next((int(a.split("=", 1)[1]) for a in sys.argv[1:]
                 if a.startswith("--zoom=")), None)
    if not argv:
        print(__doc__); sys.exit(1)
    folder = Path(argv[0])
    if not folder.is_dir():
        sys.exit(f"not a folder: {folder}")

    # auto-detect the sweep type from the filenames so ONE plotter serves both scripts:
    #   hubbard_doping_sweep.sh   -> mu<val>_hart*.txt  (grand-canonical, N floats)
    #   hubbard_Qdoping_sweep.sh  -> Q<val>_hart*.txt   (canonical, N = N_sites + Q)
    if list(folder.glob("mu*_hart*.txt")):
        lab = {"prefix": "mu", "sym": r"\mu", "unit": "eV",
               "axlabel": r"chemical potential  $\mu$  (eV)", "grand": True}
    elif list(folder.glob("Q*_hart*.txt")):
        lab = {"prefix": "Q", "sym": "Q", "unit": "e",
               "axlabel": r"charge doping  $Q$  (e)", "grand": False}
    else:
        sys.exit(f"no mu*_hart*.txt or Q*_hart*.txt files in {folder}")
    pfx = lab["prefix"]

    # collect doping values that have at least the ON file
    Qs = sorted({float(m.group(1))
                 for f in folder.glob(f"{pfx}*_hart*.txt")
                 for m in [re.match(rf"{pfx}([\-\d.eE+]+)_hart", f.name)] if m})
    if not Qs:
        sys.exit(f"no {pfx}*_hart*.txt files in {folder}")

    # optional filter: any extra CLI args restrict the plot to those doping values
    # (so leftover files from a longer previous sweep don't sneak back into the grid)
    if len(argv) > 1:
        want = [float(a) for a in argv[1:]]
        kept = [q for q in Qs if any(abs(q - w) < 1e-6 for w in want)]
        missing = [w for w in want if not any(abs(q - w) < 1e-6 for q in Qs)]
        if missing:
            print(f"warning: requested {pfx} not found in {folder}: "
                  + ", ".join(f"{w:g}" for w in missing))
        if not kept:
            sys.exit(f"none of the requested {pfx} values are present")
        Qs = kept

    bonds = None
    bf = folder / "bond_indices.txt"
    if bf.exists():
        bonds = np.atleast_2d(np.loadtxt(bf, comments="#", dtype=int))

    def fpath(Q, tag):
        # match the sweep's formatting (it writes the value exactly as given on the grid)
        for cand in (f"{pfx}{Q:g}_hart{tag}.txt", f"{pfx}{Q}_hart{tag}.txt"):
            if (folder / cand).exists():
                return folder / cand
        hits = list(folder.glob(f"{pfx}{Q:g}_hart{tag}.txt"))
        return hits[0] if hits else None

    data = {}
    for Q in Qs:
        for tag in ("OFF", "ON"):
            p = fpath(Q, tag)
            if p:
                data[(Q, tag)] = load(p)

    # global colour scales so panels are comparable across doping:
    # density is >= 0 (sequential 0..nmax); spin is symmetric (+/- smax).
    nmax = max((d["n"].max() for d in data.values()), default=1.0) or 1.0
    smax = max((np.abs(d["m"]).max() for d in data.values()), default=1.0) or 1.0

    R = len(Qs)
    fig, axes = plt.subplots(R, 3, figsize=(12, 3.7 * R), squeeze=False)
    col_titles = ["prob. density  $n_i=\\sum_{occ}|\\psi_i|^2$\n(no self-consistency)",
                  "prob. density  $n_i$\n(self-consistent Hartree)",
                  "spin  $m_i$\n(self-consistent Hartree)"]

    def draw(ax, d, values, vmin, vmax, cmap):
        if bonds is not None:
            for i, j in bonds:
                ax.plot([d["x"][i], d["x"][j]], [d["y"][i], d["y"][j]],
                        color="0.75", lw=1.0, zorder=1)
        span = max(abs(vmin), abs(vmax)) or 1.0
        sizes = 25 + 260 * (np.abs(values) / span)
        sc = ax.scatter(d["x"], d["y"], c=values, s=sizes, cmap=cmap,
                        vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.3, zorder=2)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        return sc

    sc_charge = sc_spin = None
    for r, Q in enumerate(Qs):
        dON = data.get((Q, "ON"))
        dOFF = data.get((Q, "OFF"), dON)
        s1 = draw(axes[r, 0], dOFF, dOFF["n"], 0.0, nmax, CMAP_DENS)
        s2 = draw(axes[r, 1], dON,  dON["n"],  0.0, nmax, CMAP_DENS)
        s3 = draw(axes[r, 2], dON,  dON["m"],  -smax, smax, CMAP_SPIN)
        sc_charge, sc_spin = s2, s3
        Sz = dON["m"].sum()               # m_i = 0.5(n_up-n_dn), so S_z = sum_i m_i
        Ntot = dON["n"].sum()             # total electrons (floats with mu)
        dN   = Ntot - dON["n"].size       # excess vs neutral (N_sites electrons)
        axes[r, 0].set_ylabel(f"${lab['sym']}$ = {Q:g} {lab['unit']}\n"
                              f"N = {Ntot:.2f}  (+{dN:.2f} e)\n"
                              f"$S_z$ = {Sz:.2f}", fontsize=11)
        if r == 0:
            for c in range(3):
                axes[r, c].set_title(col_titles[c])

    # shared colourbars
    if sc_charge is not None:
        cb = fig.colorbar(sc_charge, ax=axes[:, :2].ravel().tolist(),
                          shrink=0.6, pad=0.02, location="right")
        cb.set_label(r"probability density  $n_i=\sum_{occ}|\psi_i|^2$  (electrons/site)")
    if sc_spin is not None:
        cb2 = fig.colorbar(sc_spin, ax=axes[:, 2].tolist(),
                           shrink=0.6, pad=0.02, location="right")
        cb2.set_label(r"spin  $m_i = \frac{1}{2}(n_{i\uparrow}-n_{i\downarrow})$")

    fig.suptitle(f"Ground-state charge & spin vs doping — {folder.name}", y=0.995)
    out_png = folder / "doping_maps.png"
    out_pdf = folder / "doping_maps.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}\nwrote {out_pdf}")

    # second figure: SCF convergence residuals vs iteration (one line per doping value)
    plot_residuals(folder, Qs, lab)

    # third figure: population tracker — N conservation in the loop + N vs doping
    plot_population(folder, Qs, lab)

    # fourth figure: level flow — eigenvalues vs SCF iteration, U=0 vs U>0
    plot_level_flow(folder, Qs, lab, zoom=zoom)

    # text summary
    print("\n=== doping summary (self-consistent Hartree) ===")
    for Q in Qs:
        d = data.get((Q, "ON"))
        if d is None:
            continue
        dq = d["n"] - 1.0
        print(f"  {pfx}={Q:g} {lab['unit']}:  N={d['n'].sum():.2f}  S_z={d['m'].sum():+.3f}"
              f"  sum|m|={np.abs(d['m']).sum():.3f}"
              f"  max|n_i-1|={np.abs(dq).max():.3f}"
              f"  charge std={d['n'].std():.3f}")

    plt.show()   # display both figures (maps + residuals)


if __name__ == "__main__":
    main()
