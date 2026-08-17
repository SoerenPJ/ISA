#!/usr/bin/env python3
"""
ohno_U_sweep_plot.py — how does the Ohno onsite U_0 reshape the Coulomb matrix
and move the extinction resonance?

Reads the sweep written by ohno_U_sweep.sh:

    data/ohnoUsweep_<tag>/
        lattice_points.txt
        summary.txt
        U_<val>/ V_ee.txt  sigma_ext.txt  eigenvalues.txt  magnetization.txt
                 dipole_time_evolution.txt  spin_diag_time_evolution.txt

The kernel is

    V(r_ij) = e^2 / sqrt( (e^2/U_0)^2 + r_ij^2 )

so U_0 sets the onsite value AND the softening length e^2/U_0, while the
long-range tail is e^2/r for EVERY U_0. Raising U_0 shrinks the softening
length; it does not rescale the potential.

Two figures:

  <dir>/ohno_U_matrix.png/pdf
      N x N V_ee heatmaps, one per U, on a shared colour scale, plus a radial
      panel overlaying V(r) for every U. The radial panel is the informative
      one: the curves separate at small r and collapse onto the same e^2/r tail.

  <dir>/ohno_U_moment.png/pdf
      Induced dipole p(t) and the per-site induced spin moment
      m_l = drho_ll,up - drho_ll,dn, plotted raw and signed on a shared time
      axis so the two can be correlated by eye. The TOTAL sum_l m_l is zero (S_z is
      conserved: nothing in the Hamiltonian flips spin) — the induced magnetism
      is a redistribution between sites. That sum is printed as a check.

  <dir>/ohno_U_sigma.png/pdf
      sigma_ext(omega) for every U_0 overlaid on one axis, coloured and labelled
      by U_0 (and U_0/t), with the parabola-refined peak marked on each curve,
      plus a panel of the peak dispersion omega_p vs U_0.

  <dir>/ohno_U_sigma_vs_moment.png/pdf
      sigma_ext(omega) and the Fourier transform of the induced spin moment
      sum_l |m_l(omega)|, ONE PANEL PER U_0 stacked on a shared omega axis:
      sigma on the left y-axis (solid), the moment spectrum on the right y-axis
      (dashed). sigma is RAW here (nm^2, NOT divided by the flake area) — the
      moment is not area-normalised either, and normalising only one of a pair
      that is meant to be compared puts a constant between the two curves. --per-panel N puts N values of U_0 on each panel instead (N =
      all of them reproduces the old single-axes overlay). The two y-axes always
      carry the SAME scale type — both log or both linear, set by --scale
      (default log) — so a feature at a given omega can be read off both curves
      without a hidden change of variable, and the y-LIMITS are shared across
      panels so the split does not rescale each U_0 against itself. The point of
      the figure is whether the spin response rings at the extinction resonance
      or somewhere else.

sigma_ext is normalised exactly as in hubbard_plasmon_plot.py: a.u.^2 -> nm^2,
divided by the graphene area of the flake, so the colour scale is the
dimensionless sigma_ext/A.

Usage:
    python3 ploting/ohno_U_sweep_plot.py data/ohnoUsweep_<tag> [--validity-max 6] [--omega-max ...] [--peak-min 0.3] [--peak-max ...] [--scale log|linear] [--per-panel 1]

Spectra are plotted up to VALIDITY_MAX (6 eV by default) — the range where the
nearest-neighbour p_z tight-binding model is trustworthy — NOT up to wherever
the data happens to stop. If a run's dominant feature lies above that ceiling
the script says so explicitly instead of quietly quoting the tallest surviving
bump as the resonance.
"""

import sys
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform, pdist

AU_EV = 27.2113834
AU_NM = 0.0529177

# Trust ceiling of the model, NOT of the data. A nearest-neighbour p_z
# tight-binding description of graphene stops being meaningful much above this,
# so spectra are plotted and peaks quoted only up to here. Override with
# --validity-max. See figure_sigma() for why this is kept separate from the
# sweep's omega_cut_off.
VALIDITY_MAX = 6.0

FONT_SIZE_GLOBAL = 16
FONT_SIZE_TITLE  = 18
FONT_SIZE_LABEL  = 17
FONT_SIZE_TICK   = 14
FONT_SIZE_LEGEND = 13

plt.rcParams.update({
    "font.size":       FONT_SIZE_GLOBAL,
    "axes.titlesize":  FONT_SIZE_TITLE,
    "axes.labelsize":  FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "figure.dpi": 110,
})


# ---------------------------------------------------------------- shared bits
# (get_acc_au / graphene_hex_area_nm2 / load_total_area_nm2 / peak_track are the
#  same recipes hubbard_plasmon_plot.py uses — kept identical on purpose so the
#  two sweeps' sigma_ext panels are directly comparable.)

def get_acc_au(lattice):
    tree = cKDTree(lattice)
    dists, _ = tree.query(lattice, k=2)
    positive = dists[:, 1][dists[:, 1] > 0.1]
    return positive.min()


def graphene_hex_area_nm2(N_atoms, a_cc_au):
    hex_area_au2 = (3.0 * np.sqrt(3.0) / 2.0) * a_cc_au**2
    total_area_au2 = (N_atoms / 2.0) * hex_area_au2
    return total_area_au2 * AU_NM**2


def load_lattice(sweep_dir):
    hits = [sweep_dir / "lattice_points.txt"] + \
           sorted(sweep_dir.glob("*/lattice_points.txt"))
    for path in hits:
        if path.is_file():
            return np.loadtxt(path, comments="#")
    return None


def peak_track(omega, sig, omega_min, omega_max):
    """Peak frequency per row of sig, refined below the FFT frequency spacing.

    The argmax alone snaps the peak onto the discrete omega grid, turning a
    smooth dispersion into a staircase. A parabola through the maximum bin and
    its two neighbours recovers the true vertex, so shifts smaller than one FFT
    bin stay visible. The upper bound excludes the FFT edge pile-up at the
    omega_cut_off of a finite-time signal.
    """
    mask = (omega >= omega_min) & (omega <= omega_max)
    om, s = omega[mask], sig[:, mask]
    idx = np.argmax(s, axis=1)

    pk = om[idx]
    dw = np.gradient(om)
    for i, j in enumerate(idx):
        if j == 0 or j == len(om) - 1:
            continue
        y0, y1, y2 = s[i, j - 1], s[i, j], s[i, j + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom == 0.0:
            continue
        delta = 0.5 * (y0 - y2) / denom          # vertex offset in bins
        if abs(delta) <= 1.0:
            pk[i] = om[j] + delta * dw[j]
    return pk


# ------------------------------------------------------------------- loading

def list_runs(sweep_dir):
    """[(U_eV, dir), ...] sorted by U."""
    runs = []
    for d in sweep_dir.glob("U_*"):
        m = re.search(r"U_(.+)$", d.name)
        if not m or not d.is_dir():
            continue
        try:
            runs.append((float(m.group(1)), d))
        except ValueError:
            continue
    runs.sort(key=lambda r: r[0])
    return runs


def load_t_eV(sweep_dir, runs):
    """Hopping |t1| in eV, from any run's archived input.toml."""
    for _, d in runs:
        f = d / "input.toml"
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            m = re.match(r"\s*t1\s*=\s*(-?[\d.eE+-]+)", line)
            if m:
                return abs(float(m.group(1)))
    return None


def load_sigma(runs, total_area_nm2):
    """(U[sorted], omega_eV, sigma/A [nU, nomega])."""
    rows = []
    for U, d in runs:
        f = d / "sigma_ext.txt"
        if not f.is_file():
            continue
        arr = np.loadtxt(f)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        sigma = arr[:, 1] * AU_NM**2
        if total_area_nm2 is not None:
            sigma /= total_area_nm2
        rows.append((U, arr[:, 0] * AU_EV, sigma))
    if not rows:
        return None, None, None
    U = np.array([r[0] for r in rows])
    omega = rows[0][1]
    sig = np.vstack([np.interp(omega, r[1], r[2]) for r in rows])
    return U, omega, sig


# -------------------------------------------------------------- figure A
def figure_matrix(sweep_dir, runs, lattice):
    """V_ee heatmap per U (shared scale) + radial V(r) overlay."""
    mats = []
    for U, d in runs:
        f = d / "V_ee.txt"
        if f.is_file():
            mats.append((U, np.loadtxt(f) * AU_EV))     # Hartree -> eV
    if not mats:
        print("ERROR: no V_ee.txt found")
        return

    n = len(mats)
    ncol = min(3, n)
    nrow_heat = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(5.0 * ncol, 4.3 * nrow_heat + 6.2))
    # Generous hspace: the shell panel carries a title AND a secondary top axis,
    # which together need about two text rows of clearance above it.
    gs = fig.add_gridspec(nrow_heat + 1, ncol,
                          height_ratios=[1] * nrow_heat + [1.35],
                          hspace=0.62, wspace=0.28)

    # Shared colour scale: comparing panels by eye is the point of the figure.
    vmax = max(M.max() for _, M in mats)
    print(f"  V_ee shared colour scale: 0 -> {vmax:.3f} eV")

    for k, (U, M) in enumerate(mats):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        im = ax.imshow(M, cmap="magma", vmin=0, vmax=vmax,
                       interpolation="nearest", origin="upper")
        ax.set_title(rf"$U_0 = {U:g}$ eV   ($V_{{00}}={M[0,0]:.2f}$)")
        ax.set_xlabel("site $j$"); ax.set_ylabel("site $i$")
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(r"$V_{ee}$  (eV)", fontsize=FONT_SIZE_TICK)
        cb.ax.tick_params(labelsize=FONT_SIZE_TICK - 2)

    # ---- shell panel: what a site feels from an electron n*a_cc away --------
    # Plotted on a ruler of multiples of the C-C bond a_cc rather than on the
    # raw pair distances. Raw distances run to 23 bohr on a linear axis, which
    # squeezes the only U-sensitive region (the first two shells) into a sliver
    # and makes the kernel look U-independent. Here n=1 IS the nearest
    # neighbour, so the 3x spread there is the first thing you read.
    axr = fig.add_subplot(gs[nrow_heat, :])
    if lattice is None:
        axr.text(0.5, 0.5, "lattice_points.txt not found — no radial plot",
                 ha="center", va="center", transform=axr.transAxes)
    else:
        R = squareform(pdist(lattice))               # bohr
        iu = np.triu_indices_from(R, 1)
        a_cc = get_acc_au(lattice)
        n_max = min(int(R.max() / a_cc), 10)
        n_grid = np.arange(0, n_max + 1)
        r_grid = n_grid * a_cc

        colors = plt.cm.viridis(np.linspace(0, 0.9, len(mats)))
        shells = []
        for (U, M), c in zip(mats, colors):
            # U0 from the matrix itself, not the directory label, so the ruler
            # curve cannot drift away from the data it is drawn beside.
            U0 = M[0, 0]
            V = AU_EV / np.sqrt((AU_EV / U0) ** 2 + r_grid ** 2)
            shells.append((U, V))
            axr.plot(n_grid, V, "-o", color=c, lw=1.8, ms=6,
                     label=rf"$U_0 = {U:g}$ eV")
            # The honeycomb only has sites at SOME multiples of a_cc, so the
            # ruler is an idealised sampling. The real pairs go underneath it as
            # faint dots to show the curve is interpolating, not inventing.
            axr.plot(R[iu] / a_cc, M[iu], ".", color=c, ms=2.5, alpha=0.25,
                     zorder=1)

        axr.plot(n_grid[1:], AU_EV / r_grid[1:], "k--o", lw=1.4, ms=4,
                 label=r"bare $e^2/r$")     # diverges at n=0, so start at n=1

        axr.set_xlabel(r"separation $n$   (units of $a_{cc}$)")
        axr.set_ylabel(r"$V(n\,a_{cc})$  (eV)")
        axr.set_xticks(n_grid)
        axr.set_xlim(-0.3, n_max + 0.3)
        axr.set_ylim(bottom=0)
        axr.grid(alpha=0.3)
        axr.legend(ncol=2)
        axr.tick_params(labelsize=FONT_SIZE_TICK)
        sec = axr.secondary_xaxis("top", functions=(lambda n: n * a_cc,
                                                    lambda r: r / a_cc))
        sec.set_xlabel(rf"$r$  (bohr)   ($a_{{cc}} = {a_cc:.3f}$ bohr)",
                       fontsize=FONT_SIZE_TICK)
        sec.tick_params(labelsize=FONT_SIZE_TICK)

        # Quote the spread rather than leaving it to be eyeballed.
        V_all = np.array([V for _, V in shells])
        spread = V_all.max(axis=0) / V_all.min(axis=0)
        n_far = min(5, n_max)
        axr.set_title(rf"$U_0$ splits $V$ by {spread[1]:.1f}$\times$ at the "
                      rf"nearest neighbour ($n=1$) and only "
                      rf"{spread[n_far]:.1f}$\times$ at $n={n_far}$")

        print(f"\n  === V(n * a_cc) in eV   (a_cc = {a_cc:.4f} bohr) ===")
        head = "".join(f"{U:>9g}" for U, _ in shells)
        print(f"    {'n':>2} {'r[bohr]':>9}{head}{'bare':>9}{'spread':>8}")
        for k in n_grid:
            row = "".join(f"{V[k]:9.2f}" for _, V in shells)
            bare = "      inf" if k == 0 else f"{AU_EV / r_grid[k]:9.2f}"
            print(f"    {k:>2} {r_grid[k]:9.3f}{row}{bare}{spread[k]:8.2f}")

    fig.suptitle(f"Ohno Coulomb matrix vs onsite $U_0$ — {sweep_dir.name}",
                 fontsize=FONT_SIZE_TITLE + 2)
    out = str(sweep_dir / "ohno_U_matrix")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=200)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"wrote {out}.png\nwrote {out}.pdf")


# -------------------------------------------------------------- figure C
def load_moment(run_dir):
    """(t, m_site[nt, N]) from spin_diag_time_evolution.txt.

    The file is ALREADY the induced quantity: Density.cpp:535-539 writes
    rho_ii(t) - rho0_ii for the up block then the down block, so

        m_l(t) = drho_ll,up(t) - drho_ll,dn(t)

    needs no further baseline subtraction. Columns are  t | up_0..up_{N-1} |
    dn_0..dn_{N-1}, i.e. 2N+1 wide.
    """
    f = run_dir / "spin_diag_time_evolution.txt"
    if not f.is_file():
        return None, None
    a = np.loadtxt(f, comments="#")
    if a.ndim != 2 or a.shape[1] < 3:
        return None, None
    t, d = a[:, 0], a[:, 1:]
    N = d.shape[1] // 2
    m = d[:, :N] - d[:, N:2 * N]

    # The adaptive solver can emit duplicate/non-monotonic stamps; collapse them
    # so the curve does not double back on itself.
    t, keep = np.unique(t, return_index=True)
    return t, m[keep]


def ft_like_sigma(t, y, omega):
    """P(omega) = integral y(t) exp(+i omega t) dt, exactly as sigma_ext is built.

    Mirrors compute_sigma_ext (Observables/observables.cpp:79-85), which forms
    y(t)*exp(+i*omega*t) and hands it to trapezoid() (:44-53). That means:

      * the kernel sign is POSITIVE, exp(+i omega t), not the usual exp(-i...);
      * integration is trapezoid on the RAW time stamps, so the solver's
        non-uniform steps are handled natively — no resampling and no FFT, and
        deliberately no window function, because the C++ applies none either.

    The trapezoid sum 0.5*sum_i dt_i*(f_i + f_{i+1}) is rewritten as a single
    weight vector so the whole transform is one matmul over all sites at once.

    t: (nt,)   y: (nt,) or (nt, nsite)   omega: (nw,) in the same units as 1/t.
    Returns (nw,) or (nw, nsite) complex.
    """
    dt = np.diff(t)
    w = np.zeros_like(t)
    w[:-1] += 0.5 * dt
    w[1:] += 0.5 * dt
    E = np.exp(1j * np.outer(omega, t))            # (nw, nt)
    return E @ (w[:, None] * y if y.ndim == 2 else w * y)


def load_omega_grid(run_dir):
    """The run's own omega grid (a.u.) from sigma_ext.txt column 0.

    Taken from the file rather than re-derived from omega_cut_off/fourier_dt_fs
    (main.cpp:863-867) so the moment transform lands on exactly the grid
    sigma_ext was evaluated on, with no chance of the two drifting apart.
    """
    f = run_dir / "sigma_ext.txt"
    if not f.is_file():
        return None
    a = np.loadtxt(f)
    return a[:, 0] if a.ndim == 2 and a.shape[0] > 1 else None


def moment_spectrum(t, m, omega_au):
    """(omega_eV, sum_l |m_l(omega)|) on the run's own sigma_ext omega grid.

    Transform per site FIRST, then sum magnitudes. sum_l m_l(t) == 0 identically
    (S_z is conserved), so summing the complex spectra would cancel to zero and
    leave nothing but round-off.
    """
    M = ft_like_sigma(t, m, omega_au)                  # (nw, nsite)
    return omega_au * AU_EV, np.abs(M).sum(axis=1)


def load_dipole(run_dir):
    """(t, p) from dipole_time_evolution.txt (# time  dipole_moment)."""
    f = run_dir / "dipole_time_evolution.txt"
    if not f.is_file():
        return None, None
    a = np.loadtxt(f, comments="#")
    if a.ndim != 2 or a.shape[1] < 2:
        return None, None
    t, keep = np.unique(a[:, 0], return_index=True)
    return t, a[keep, 1]


def figure_moment(sweep_dir, runs):
    """Induced spin moment m_l(t) and the dipole p(t), on a shared time axis.

    The moment panel plots the per-site m_l(t) as it comes out of the file — one
    signed trace per site, no sum and no absolute value. Rectifying or summing
    would destroy exactly what the panel is for: lining the ringing up against
    the dipole above it by eye.

    Note that sum_l m_l(t) is identically zero — nothing in the Hamiltonian flips
    spin (the Zeeman term couples sigma_z, the Peierls phase is spin-blind), so
    the up and down populations are separately conserved. The induced magnetism
    is a spatial redistribution between sites. That sum is printed per U as a
    conservation check.
    """
    series = []
    for U, d in runs:
        t_m, m = load_moment(d)
        t_p, p = load_dipole(d)
        if t_m is None and t_p is None:
            continue
        series.append((U, t_m, m, t_p, p, load_omega_grid(d)))

    if not series:
        print("\n  [moment figure] no dipole_time_evolution.txt / "
              "spin_diag_time_evolution.txt found under the U_* dirs.")
        print("  These are not kept by older sweeps. Re-run ohno_U_sweep.sh — it "
              "now sets save_spin_diag = true and copies both files.")
        return

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(series)))
    fig, (axp, axm, axf) = plt.subplots(
        3, 1, figsize=(13, 13.5), constrained_layout=True)

    rows, no_grid = [], []
    for (U, t_m, m, t_p, p, om_au), c in zip(series, colors):
        lbl = rf"$U_0 = {U:g}$ eV"
        if t_p is not None:
            axp.plot(t_p, p, "-", color=c, lw=1.5, label=lbl)
        if t_m is not None:
            # Time panel stays per SITE and signed — one thin trace per atom.
            # Reducing it would rectify away the oscillation, which is the whole
            # point of this panel.
            axm.plot(t_m, m, "-", color=c, lw=0.5, alpha=0.5)
            axm.plot([], [], "-", color=c, lw=1.5, label=lbl)   # legend proxy
            rows.append((U, float(np.abs(m.sum(axis=1)).max()),
                         float(np.abs(m).max())))

            if om_au is None:
                no_grid.append(f"{U:g}")
            else:
                # The FT panel IS reduced to one curve per U_0: 46 overlaid
                # spectra hid the line structure.
                om_eV, spec = moment_spectrum(t_m, m, om_au)
                axf.plot(om_eV, spec, "-", color=c, lw=1.5, label=lbl)

    axp.set_ylabel(r"dipole $p(t)$  (a.u.)")
    axp.set_title("Induced dipole")
    axp.grid(alpha=0.3); axp.legend(ncol=3, fontsize=FONT_SIZE_LEGEND - 1)
    axp.tick_params(labelsize=FONT_SIZE_TICK)

    axm.set_xlabel("time  (a.u.)")
    axm.set_ylabel(r"$m_l(t) = \Delta\rho_{ll\uparrow} - \Delta\rho_{ll\downarrow}$")
    axm.set_title("Induced spin moment")
    axm.grid(alpha=0.3); axm.legend(ncol=3, fontsize=FONT_SIZE_LEGEND - 1)
    axm.tick_params(labelsize=FONT_SIZE_TICK)

    axf.set_xlabel(r"$\omega$  (eV)")
    axf.set_ylabel(r"$\sum_l |m_l(\omega)|$")
    axf.set_title(r"Fourier transform of the induced spin moment  ")
    axf.set_yscale("log")
    axf.grid(alpha=0.3, which="both")
    axf.legend(ncol=3, fontsize=FONT_SIZE_LEGEND - 1)
    axf.tick_params(labelsize=FONT_SIZE_TICK)
    if no_grid:
        print(f"  [moment FT] no sigma_ext.txt for U_0 = {', '.join(no_grid)} eV "
              f"— no omega grid, transform skipped for those runs.")

    fig.suptitle(f"Induced moment vs dipole — {sweep_dir.name}",
                 fontsize=FONT_SIZE_TITLE + 2)
    out = str(sweep_dir / "ohno_U_moment")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=200)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"wrote {out}.png\nwrote {out}.pdf")

    if rows:
        print("\n  === induced moment vs U_0 ===")
        print(f"    {'U_eV':>7} {'|sum m_l| (=0?)':>16} {'max_l |m_l|':>14}")
        for U, tot, mx in rows:
            print(f"    {U:7g} {tot:16.3e} {mx:14.4e}")
        print("    (column 2 must be ~0 — S_z is conserved; a nonzero value "
              "means a spin-flip term entered the Hamiltonian)")


# -------------------------------------------------------------- figure D
def figure_sigma_vs_moment(sweep_dir, runs, t_eV, omega_max, peak_min, scale,
                           per_panel):
    """sigma_ext and the induced-spin-moment FT, stacked panels on a shared omega.

    One panel per U_0 by default (--per-panel groups more than one onto a panel).
    Within a panel: sigma_ext on the left y-axis (solid), sum_l |m_l(omega)| on
    the right y-axis (dashed), same colour per U_0 as every other figure here.

    sigma here is NOT divided by the flake area, unlike figure_sigma. The moment
    is not area-normalised either, and normalising one curve of a pair that is
    meant to be compared puts a constant between them for no physical reason.
    The unit is therefore nm^2, not the dimensionless sigma_ext/A.

    All panels share the omega axis, so a feature lines up vertically across the
    whole column, and they also share y-LIMITS, so a curve that is taller in one
    panel really is taller — the split is only a decluttering of x-overlap, it
    must not silently rescale each U_0 against itself.

    The two quantities carry different units and differ by orders of magnitude,
    so they cannot share one y-axis without normalising — and normalising would
    make the amplitudes uncomparable between U_0. A twin axis keeps both raw. The
    one rule enforced here is that BOTH y-axes get the SAME scale type: mixing a
    log axis against a linear one puts a change of variable between the two
    curves, and a coincidence of peaks would then be an artefact of the drawing.

    Both curves are evaluated on the run's own sigma_ext omega grid (the moment
    transform is handed that grid directly), so the shared x-axis is exact — no
    interpolation between the two.
    """
    series, missing = [], []
    for U, d in runs:
        om_au = load_omega_grid(d)
        if om_au is None:
            missing.append(f"{U:g}")
            continue
        # sigma is RAW here (a.u.^2 -> nm^2, no division by the flake area),
        # unlike figure_sigma. The moment carries no area normalisation either,
        # so dividing only one of the two pair would put a factor between the
        # curves that has nothing to do with the physics being compared.
        arr = np.loadtxt(d / "sigma_ext.txt")
        s = arr[:, 1] * AU_NM**2
        om_eV = om_au * AU_EV

        t_m, m = load_moment(d)
        spec = None
        if t_m is not None:
            _, spec = moment_spectrum(t_m, m, om_au)
        series.append((U, om_eV, s, spec))

    if not series:
        print("\n  [sigma-vs-moment] no sigma_ext.txt under the U_* dirs.")
        return
    if missing:
        print(f"  [sigma-vs-moment] no sigma_ext.txt for U_0 = "
              f"{', '.join(missing)} eV — skipped.")

    data_max = max(om.max() for _, om, _, _ in series)
    if omega_max is None:
        omega_max = VALIDITY_MAX
    omega_max = min(omega_max, data_max)

    def show(y):
        """Non-positive samples become gaps on a log axis, not spikes.

        sigma_ext dips slightly negative between lines (finite-time trapezoid
        transform of a decaying signal). Left as-is, matplotlib clips those
        points and the line plunges to the axis floor and back, which reads as
        structure. NaN breaks the line there instead.
        """
        return np.where(y > 0, y, np.nan) if scale == "log" else y

    # Global extrema FIRST: the panels share y-limits, so they have to be known
    # before anything is drawn (see the docstring — per-panel autoscale would
    # make a weak U_0 look as strong as a strong one).
    smax = mmax = 0.0
    for U, om, s, spec in series:
        k = (om >= 0) & (om <= omega_max)
        smax = max(smax, float(np.nanmax(s[k])))
        if spec is not None:
            mmax = max(mmax, float(np.nanmax(spec[k])))

    per_panel = max(1, per_panel)
    groups = [series[i:i + per_panel] for i in range(0, len(series), per_panel)]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(series)))

    fig, axes = plt.subplots(len(groups), 1, sharex=True,
                             figsize=(13, 3.3 * len(groups) + 1.4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    dropped = 0
    ci = 0
    for gi, (ax, group) in enumerate(zip(axes, groups)):
        axm = ax.twinx()                 # shares x with ax by construction
        labels = []
        for U, om, s, spec in group:
            c = colors[ci]; ci += 1
            lbl = rf"$U_0 = {U:g}$ eV"
            if t_eV:
                lbl += rf"  $({U/t_eV:g}\,t)$"
            labels.append((lbl, c))
            k = (om >= 0) & (om <= omega_max)
            ax.plot(om, show(s), "-", color=c, lw=1.9)
            if spec is not None:
                axm.plot(om, show(spec), "--", color=c, lw=1.6)
            if scale == "log":
                dropped += int(np.count_nonzero(s[k] <= 0))

        # The whole point of the figure: identical scale type on both axes.
        ax.set_yscale(scale)
        axm.set_yscale(scale)
        if scale == "log":
            # Same number of decades on both axes, anchored on each one's own
            # global peak, so the vertical extent of a feature means the same
            # thing left and right AND from panel to panel. Without this the
            # autoscale gives sigma ~7 decades (it reaches far lower between
            # lines) against ~5 for the moment.
            DECADES = 5.0
            ax.set_ylim(smax / 10**DECADES, smax * 2.0)
            if mmax > 0:
                axm.set_ylim(mmax / 10**DECADES, mmax * 2.0)
        else:
            ax.set_ylim(0, smax * 1.06)
            if mmax > 0:
                axm.set_ylim(0, mmax * 1.06)

        ax.set_ylabel(r"$\sigma^\mathrm{ext}$  (nm$^2$)",
                      fontsize=FONT_SIZE_TICK + 1)
        axm.set_ylabel(r"$\sum_l |m_l(\omega)|$", fontsize=FONT_SIZE_TICK + 1)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        axm.tick_params(labelsize=FONT_SIZE_TICK)

        # Which U_0 a panel holds goes in its TITLE, not a legend box: on a log
        # axis the curves fill the upper half of every panel, so any in-axes
        # legend lands on data. A title cannot overlap anything.
        # The solid/dashed key is stated once, on the top panel.
        key = (rf"solid: $\sigma^\mathrm{{ext}}$ (left)   ·   "
               rf"dashed: $\sum_l |m_l(\omega)|$ (right)   ·   "
               rf"both axes {scale}") if gi == 0 else ""
        if key:
            # Separate Text object from the left title, so the U_0 label can be
            # tinted with the curve colour without tinting the key with it.
            ax.set_title(key, loc="right", fontsize=FONT_SIZE_LEGEND - 1,
                         color="0.25")
        if len(group) == 1:
            ax.set_title(labels[0][0], loc="left",
                         fontsize=FONT_SIZE_TICK + 1, color=labels[0][1])
        else:
            # Grouped panels need colour to tell the U_0 apart, which a title
            # cannot carry — fall back to a legend here.
            handles = [plt.Line2D([], [], color=c, lw=2.0, label=l)
                       for l, c in labels]
            ax.legend(handles=handles, ncol=len(labels), loc="upper right",
                      fontsize=FONT_SIZE_LEGEND - 1)

    axes[-1].set_xlim(0, omega_max)
    axes[-1].set_xlabel(r"$\omega$  (eV)")
    fig.suptitle(f"Extinction vs induced-spin-moment spectrum — "
                 f"{sweep_dir.name}", fontsize=FONT_SIZE_TITLE + 2)
    if dropped:
        print(f"  [sigma-vs-moment] log axis: {dropped} non-positive sigma "
              f"samples in window shown as gaps (sigma dips below 0 between "
              f"lines).")

    out = str(sweep_dir / "ohno_U_sigma_vs_moment")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=200)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"wrote {out}.png\nwrote {out}.pdf")

    # Do the comparison the eye is meant to make, numerically as well.
    print("\n=== sigma peak vs moment peak (same window) ===")
    print(f"    {'U_eV':>7} {'omega_sigma':>12} {'omega_moment':>13} "
          f"{'delta':>8}")
    for U, om, s, spec in series:
        k = (om >= peak_min) & (om <= omega_max)
        if not k.any():
            continue
        ps = peak_track(om[k], s[None, k], peak_min, omega_max)[0]
        if spec is None:
            print(f"    {U:7g} {ps:12.3f} {'—':>13} {'—':>8}")
            continue
        pm = peak_track(om[k], spec[None, k], peak_min, omega_max)[0]
        print(f"    {U:7g} {ps:12.3f} {pm:13.3f} {pm - ps:+8.3f}")


# -------------------------------------------------------------- figure B
def figure_sigma(sweep_dir, runs, t_eV, omega_max, peak_min, peak_max):
    lattice = load_lattice(sweep_dir)
    area = None
    if lattice is not None:
        a_cc = get_acc_au(lattice)
        area = graphene_hex_area_nm2(len(lattice), a_cc)
        print(f"  area: a_cc={a_cc:.4f} a.u., N={len(lattice)}, A={area:.3f} nm^2")
    else:
        print("  [WARNING] lattice_points.txt not found — sigma left unnormalised")

    U, om_full, sig_full = load_sigma(runs, area)
    if U is None:
        print("ERROR: no sigma_ext.txt found")
        return

    # Two different ceilings, deliberately kept apart:
    #   omega_max     — how far the DATA goes (the sweep's omega_cut_off)
    #   VALIDITY_MAX  — how far the MODEL is trustworthy. A nearest-neighbour p_z
    #                   tight-binding graphene model is not meaningful much above
    #                   ~6 eV, so that is the range we plot and quote peaks in.
    # They are not the same thing, and collapsing them is a trap: truncating the
    # DATA at 6 eV destroys the very evidence that tells you the dominant feature
    # has moved outside the trustworthy range. omega_cut_off only sets how many
    # bins the post-hoc DFT evaluates (main.cpp:864) — the time evolution is
    # already done — so running the sweep with a WIDE omega_cut_off costs
    # essentially nothing and lets this function report honestly.
    data_max = om_full.max()
    if omega_max is None:
        omega_max = VALIDITY_MAX
    omega_max = min(omega_max, data_max)
    if peak_max is None:
        peak_max = 0.95 * omega_max

    k = om_full <= omega_max
    om, sig = om_full[k], sig_full[:, k]
    print(f"  data reaches {data_max:.2f} eV;  plotting/peak-searching "
          f"0 .. {omega_max:.2f} eV (TB validity)")

    pk = peak_track(om, sig, peak_min, peak_max)

    # --- is the quoted peak the real one, or just the tallest survivor? -------
    # If the data extends past the validity ceiling this is exact: compare the
    # global maximum against the window directly.
    # If the data STOPS at the ceiling it can only be guessed, from weight still
    # piled up in the last few bins. That test catches a spectrum still rising at
    # the boundary, but is blind to a resonance sitting far above with a lull at
    # the boundary. Measured on this sweep against a wide-window reference
    # (last-3%-band / in-window-max):
    #     U_0 = 1t 0.11, 2t 0.36  -> genuinely inside   (correctly cleared)
    #          3t 0.77, 6t 0.67  -> outside            (correctly flagged)
    #          4t 0.02, 5t 0.01  -> outside at 9.1 and 12.9 eV, and INVISIBLE
    # That blind spot is not a tuning problem: truncated data does not contain
    # the evidence. A wide run is the only reliable answer, and it is free —
    # omega_cut_off only sets how many bins the post-hoc DFT evaluates
    # (main.cpp:864), the time evolution is already done.
    exact = data_max > omega_max + 1e-9
    suspect = []
    edge_band = om >= 0.97 * omega_max
    for i, u in enumerate(U):
        inwin = sig[i][(om >= peak_min) & (om <= peak_max)]
        if inwin.size == 0 or inwin.max() <= 0:
            continue
        note = None
        if exact:
            sel = om_full > peak_min
            om_out = om_full[sel][int(np.argmax(sig_full[i][sel]))]
            if om_out > omega_max:
                note = f"true maximum at {om_out:.2f} eV"
        elif sig[i][edge_band].max() / inwin.max() > 0.5:
            note = (f"still {sig[i][edge_band].max()/inwin.max():.2f}x the "
                    f"in-window max at the ceiling — spectrum continues")
        if note:
            suspect.append((u, note))

    if suspect:
        print(f"\n  [WARNING] dominant feature lies ABOVE the {omega_max:.1f} eV "
              f"validity window for:")
        for u, note in suspect:
            print(f"      U_0 = {u:g} eV: {note}")
        print("      Their quoted omega_p is the tallest surviving in-window "
              "feature, NOT the resonance.")
    if not exact:
        print(f"      NOTE: data stops at {data_max:.2f} eV, so this check is a "
              f"hint only — a resonance far above the ceiling with a lull at the\n"
              f"      boundary is undetectable here. Re-run with a wide "
              f"OMEGA_CUT (e.g. 25) to know for certain; it costs no sim time.")

    # Overlaid spectra, one curve per U_0 — with only a handful of U values this
    # beats a heatmap: lineshape, width and relative amplitude are all visible,
    # and the peaks span 3-15 eV so the curves barely overlap. Amplitudes cover
    # only ~5x, so a linear y-axis needs no normalisation.
    # The peak dispersion gets its own short panel rather than an inset: an inset
    # large enough to read collides with a 6-entry legend on this data.
    fig, (ax, axp) = plt.subplots(
        2, 1, figsize=(12, 9.5), height_ratios=[3.0, 1.15],
        constrained_layout=True)
    # same viridis mapping as the radial panel of figure A, so a given U_0 keeps
    # its colour across both figures
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(U)))
    for u, s, p, c in zip(U, sig, pk, colors):
        lbl = rf"$U_0 = {u:g}$ eV"
        if t_eV:
            lbl += rf"  $({u/t_eV:g}\,t)$"
        ax.plot(om, s, "-", color=c, lw=1.9, label=lbl)
        j = int(np.argmin(np.abs(om - p)))
        ax.plot([p], [s[j]], "o", color=c, ms=7, mec="k", mew=0.7, zorder=5)

    ax.set_xlabel(r"$\omega$  (eV)")
    ax.set_ylabel(r"$\sigma^\mathrm{ext}/A$")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, omega_max)      # the TB-validity ceiling, see above
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.legend(title=r"markers = peak $\omega_p$", ncol=2, loc="upper right")

    ax.set_title(f"Extinction vs Ohno onsite $U_0$ — {sweep_dir.name}")

    # peak dispersion: what the overlay only shows qualitatively
    axp.plot(U, pk, "-", color="#555555", lw=1.8, zorder=1)
    for u, p, c in zip(U, pk, colors):
        axp.plot([u], [p], "o", color=c, ms=9, mec="k", mew=0.8, zorder=5)
    axp.set_xlabel(r"Ohno onsite $U_0$  (eV)")
    axp.set_ylabel(r"peak $\omega_p$  (eV)")
    axp.set_xticks(U)
    axp.grid(alpha=0.3)
    axp.tick_params(labelsize=FONT_SIZE_TICK)
    if t_eV:
        secp = axp.secondary_xaxis("top", functions=(lambda u: u / t_eV,
                                                     lambda x: x * t_eV))
        secp.set_xlabel(rf"$U_0 / t$   ($t = {t_eV:g}$ eV)",
                        fontsize=FONT_SIZE_TICK)
        secp.tick_params(labelsize=FONT_SIZE_TICK)

    out = str(sweep_dir / "ohno_U_sigma")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=200)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"wrote {out}.png\nwrote {out}.pdf")

    # An edge-pinned peak means the resonance left the window — say so rather
    # than reporting the FFT pile-up as a plasmon.
    dw = float(np.median(np.diff(om)))
    edge = [f"{u:g}" for u, p in zip(U, pk) if p >= peak_max - 2.0 * dw]
    if edge:
        print(f"  [WARNING] peak pinned to the {peak_max:.2f} eV search edge at "
              f"U_0 = {', '.join(edge)} eV — the resonance is likely ABOVE the "
              f"window. Re-run the sweep with a larger OMEGA_CUT.")

    print("\n=== peak frequency vs U_0 ===")
    for u, p in zip(U, pk):
        ut = f"{u/t_eV:.2f}" if t_eV else "?"
        print(f"  U_0 = {u:6.3g} eV ({ut} t):  omega_p = {p:7.3f} eV")
    print(f"  total shift over the grid: {pk[-1] - pk[0]:+.3f} eV")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sweep_dir = Path(sys.argv[1])
    omega_max = None
    peak_min = 0.3
    peak_max = None
    scale = "log"
    per_panel = 1
    if "--per-panel" in sys.argv:
        per_panel = int(sys.argv[sys.argv.index("--per-panel") + 1])
    if "--scale" in sys.argv:
        scale = sys.argv[sys.argv.index("--scale") + 1]
        if scale not in ("log", "linear"):
            print("ERROR: --scale takes 'log' or 'linear'")
            sys.exit(1)
    if "--omega-max" in sys.argv:
        omega_max = float(sys.argv[sys.argv.index("--omega-max") + 1])
    if "--peak-min" in sys.argv:
        peak_min = float(sys.argv[sys.argv.index("--peak-min") + 1])
    if "--peak-max" in sys.argv:
        peak_max = float(sys.argv[sys.argv.index("--peak-max") + 1])
    if "--validity-max" in sys.argv:
        global VALIDITY_MAX
        VALIDITY_MAX = float(sys.argv[sys.argv.index("--validity-max") + 1])

    runs = list_runs(sweep_dir)
    if not runs:
        print(f"ERROR: no U_* run directories under {sweep_dir}")
        sys.exit(1)
    t_eV = load_t_eV(sweep_dir, runs)
    print(f"  {len(runs)} runs: U_0 = {', '.join(f'{u:g}' for u, _ in runs)} eV"
          + (f"   (t = {t_eV:g} eV)" if t_eV else ""))

    figure_matrix(sweep_dir, runs, load_lattice(sweep_dir))
    figure_sigma(sweep_dir, runs, t_eV, omega_max, peak_min, peak_max)
    figure_moment(sweep_dir, runs)
    figure_sigma_vs_moment(sweep_dir, runs, t_eV, omega_max, peak_min, scale,
                           per_panel)
    plt.show()


if __name__ == "__main__":
    main()
