#!/usr/bin/env python3
"""
hubbard_plasmon_plot.py — does the new model shift the plasmon frequency?

Reads the two-model Q-doping sweep written by hubbard_plasmon_sweep.sh:

    data_LLM/plasmon_Q_<tag>/
        L2_Q_<Q>/sigma_ext.txt        old model (hubbard = hubbard_hartree = false)
        L2HH_Q_<Q>/sigma_ext.txt      new model (hubbard = hubbard_hartree = true)

Both arms hold spin_on / self_consistent_phase / zeeman_induced fixed, so the
only difference is the Hubbard/Hartree block.

Each sigma_ext.txt is two columns:  omega [a.u.]   sigma_ext.
sigma_ext is normalised exactly as in linear_response_vol2_rot0.py: converted
from a.u.^2 to nm^2 and divided by the graphene area of the flake (from
lattice_points.txt), so the colour scale shows the dimensionless sigma_ext/A.
Builds sigma_ext(omega, Q) heatmaps for both models, a signed (new - old)
difference map, and the extracted plasmon-peak frequency vs Q with both models
overlaid, so any shift is read off directly.

Each heatmap carries its OWN colour scale. The two arms often differ in
amplitude by enough that one shared scale renders the weaker one as a black
panel with no visible structure. The cost is that panel brightness is no longer
comparable between arms — so each panel's max is printed in its title, the ratio
goes to stdout, and the difference map (still one symmetric scale) stays the
quantitative comparison. Pass --shared-scale for the old single normalisation.

The plotted omega window defaults to whatever the sweep actually wrote (its
omega_cut_off); --omega-max narrows it. Do NOT narrow it below the new model's
U-split gap: with the Hubbard on, the main resonance follows that gap (13.57 eV
against a 13.41 eV gap on the 5x5 zigzag triangle at U = 15.72 eV), so a 6 eV
window shows an almost empty panel and the peak tracker locks onto the FFT edge
pile-up instead.

Usage:
    python3 ploting/hubbard_plasmon_plot.py data_LLM/plasmon_Q_<tag> [--omega-max 25] [--peak-min 0.3] [--peak-max 23] [--shared-scale]
"""

import sys
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

AU_EV = 27.2113834
AU_NM = 0.0529177

FONT_SIZE_GLOBAL = 18
FONT_SIZE_TITLE  = 20
FONT_SIZE_LABEL  = 19
FONT_SIZE_TICK   = 16
FONT_SIZE_LEGEND = 15

plt.rcParams.update({
    "font.size":       FONT_SIZE_GLOBAL,
    "axes.titlesize":  FONT_SIZE_TITLE,
    "axes.labelsize":  FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "figure.dpi": 110,
})

Q_LABEL = r"charge doping $Q$  (extra electrons)"


def get_acc_au(lattice):
    tree = cKDTree(lattice)
    dists, _ = tree.query(lattice, k=2)
    positive = dists[:, 1][dists[:, 1] > 0.1]
    return positive.min()


def graphene_hex_area_nm2(N_atoms, a_cc_au):
    hex_area_au2 = (3.0 * np.sqrt(3.0) / 2.0) * a_cc_au**2
    total_area_au2 = (N_atoms / 2.0) * hex_area_au2
    return total_area_au2 * AU_NM**2


def load_total_area_nm2(sweep_dir):
    """Flake area in nm^2 from lattice_points.txt (same recipe as vol2)."""
    hits = [sweep_dir / "lattice_points.txt"] + \
           sorted(sweep_dir.glob("*/lattice_points.txt"))
    for path in hits:
        if path.is_file():
            lattice = np.loadtxt(path, comments="#")
            a_cc_au = get_acc_au(lattice)
            area = graphene_hex_area_nm2(len(lattice), a_cc_au)
            print(f"  area: a_cc={a_cc_au:.4f} a.u., N={len(lattice)}, "
                  f"A={area:.3f} nm^2")
            return area
    print("  [WARNING] lattice_points.txt not found — sigma left unnormalised")
    return None


def load_model(sweep_dir, model, total_area_nm2):
    """Return (Q[sorted], omega_eV, sigma/A [nQ, nomega]) for one model."""
    rows = []
    for d in sweep_dir.glob(f"{model}_Q_*"):
        m = re.search(rf"{model}_Q_(.+)$", d.name)
        f = d / "sigma_ext.txt"
        if not m or not f.is_file():
            continue
        try:
            Q = float(m.group(1))
        except ValueError:
            continue
        arr = np.loadtxt(f)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        sigma = arr[:, 1] * AU_NM**2
        if total_area_nm2 is not None:
            sigma = sigma / total_area_nm2
        rows.append((Q, arr[:, 0] * AU_EV, sigma))
    if not rows:
        return None, None, None
    rows.sort(key=lambda r: r[0])
    Q = np.array([r[0] for r in rows])
    omega = rows[0][1]                       # common frequency grid
    sig = np.vstack([np.interp(omega, r[1], r[2]) for r in rows])
    return Q, omega, sig


def peak_track(Q, omega, sig, omega_min, omega_max):
    """Plasmon-peak frequency vs Q, refined below the FFT frequency spacing.

    The argmax alone snaps the peak onto the discrete omega grid, which turns a
    smooth dispersion into a staircase. A parabola through the maximum bin and
    its two neighbours recovers the true vertex, so shifts smaller than one FFT
    bin stay visible.

    The upper bound excludes the FFT edge pile-up that accumulates right at the
    omega_cut_off of a finite-time signal, which would otherwise be mistaken for
    the plasmon. Widen it with --peak-max if your resonance sits higher.
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


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sweep_dir = Path(sys.argv[1])
    omega_max = None
    peak_min = 0.3
    peak_max = None
    if "--omega-max" in sys.argv:
        omega_max = float(sys.argv[sys.argv.index("--omega-max") + 1])
    if "--peak-min" in sys.argv:
        peak_min = float(sys.argv[sys.argv.index("--peak-min") + 1])
    if "--peak-max" in sys.argv:
        peak_max = float(sys.argv[sys.argv.index("--peak-max") + 1])
    shared_scale = "--shared-scale" in sys.argv

    models = [
        ("L2",   "L2  (old: no Hubbard/Hartree)",   "#2980b9"),
        ("L2HH", "L2 + Hubbard + Hartree  (new)",   "#c0392b"),
    ]

    total_area_nm2 = load_total_area_nm2(sweep_dir)
    raw = {}
    for key, _, _ in models:
        Q, om, s = load_model(sweep_dir, key, total_area_nm2)
        if Q is None:
            print(f"ERROR: could not load {key} sigma_ext under {sweep_dir}")
            sys.exit(1)
        raw[key] = (Q, om, s)

    # Default to the window the sweep actually wrote (its omega_cut_off). A
    # hardcoded default would silently crop the new model's gap-following
    # resonance out of the figure.
    if omega_max is None:
        omega_max = min(om.max() for _, om, _ in raw.values())
    if peak_max is None:
        peak_max = 0.95 * omega_max      # skip the FFT edge pile-up by default
    print(f"  omega window: 0 .. {omega_max:.2f} eV   "
          f"peak search: {peak_min:.2f} .. {peak_max:.2f} eV")

    data = {}
    for key, _, _ in models:
        Q, om, s = raw[key]
        k = om <= omega_max
        data[key] = (Q, om[k], s[:, k])

    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5), constrained_layout=True)
    axmaps = [axes[0, 0], axes[0, 1]]
    axdiff = axes[1, 0]
    axpk = axes[1, 1]

    # Each arm gets its OWN colour scale by default. A shared scale is only
    # honest when the two arms have comparable amplitude; here they routinely do
    # not, and the weaker arm is then rendered as an almost uniformly black
    # panel with no visible structure at all. Independent scales cost the
    # cross-panel amplitude comparison, so that information is preserved
    # explicitly instead: each panel's own max is printed in its title, the
    # ratio goes to stdout, and the signed difference map below is still on one
    # symmetric scale and remains the quantitative comparison.
    # --shared-scale restores the old single-normalisation behaviour.
    vmaxes = {key: float(data[key][2].max()) for key, _, _ in models}
    if shared_scale:
        v = max(vmaxes.values())
        vmaxes = {key: v for key in vmaxes}
        print(f"  shared colour scale: 0 -> {v:.4f}")
    else:
        for key, _, _ in models:
            print(f"  colour scale {key}: 0 -> {vmaxes[key]:.4f}")
        lo, hi = min(vmaxes.values()), max(vmaxes.values())
        if lo > 0:
            print(f"  amplitude ratio between arms: {hi / lo:.2f}x "
                  f"— panels are independently normalised, so do NOT compare "
                  f"their brightness; use the difference map.")

    def heat(ax, Q, om, s, title, vmax):
        pcm = ax.pcolormesh(Q, om, s.T, shading="auto", cmap="magma",
                            vmin=0, vmax=vmax, rasterized=True)
        ax.set_xlabel(Q_LABEL)
        ax.set_ylabel(r"$\omega$  (eV)")
        ax.set_title(title)
        ax.set_ylim(0, omega_max)
        ax.set_xticks(Q)
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        cb = fig.colorbar(pcm, ax=ax, label=r"$\sigma^\mathrm{ext}/A$", pad=0.02)
        cb.ax.tick_params(labelsize=FONT_SIZE_TICK)

    peaks = {}
    for ax, (key, title, _) in zip(axmaps, models):
        Q, om, s = data[key]
        # The max is in the title because independent colorbars make two panels
        # of very different amplitude look equally bright.
        panel_title = title if shared_scale else \
            f"{title}\n" + rf"max $\sigma^\mathrm{{ext}}/A$ = {vmaxes[key]:.3g}"
        heat(ax, Q, om, s, panel_title, vmaxes[key])
        pk = peak_track(Q, om, s, peak_min, peak_max)
        peaks[key] = (Q, pk)
        # A peak sitting on the search boundary is the signature of a truncated
        # spectrum: the real resonance is outside the window and the tracker has
        # latched onto the FFT edge pile-up. Say so rather than plotting it
        # as if it were a plasmon.
        dw = float(np.median(np.diff(om)))
        edge = [f"{q:g}" for q, p in zip(Q, pk) if p >= peak_max - 2.0 * dw]
        if edge:
            print(f"  [WARNING] {key}: peak pinned to the {peak_max:.2f} eV search "
                  f"edge at Q = {', '.join(edge)} — the resonance is likely ABOVE "
                  f"the window. Re-run the sweep with a larger OMEGA_CUT.")

    # --- difference map: new - old, on the Q values both arms actually have ---
    Q_old, om_ref, s_old = data["L2"]
    Q_new, _,      s_new = data["L2HH"]
    Q_common = np.intersect1d(Q_old, Q_new)
    if Q_common.size:
        d = s_new[np.searchsorted(Q_new, Q_common)] - \
            s_old[np.searchsorted(Q_old, Q_common)]
        vlim = np.nanmax(np.abs(d))
        pcm = axdiff.pcolormesh(Q_common, om_ref, d.T, shading="auto",
                                cmap="RdBu_r", vmin=-vlim, vmax=+vlim,
                                rasterized=True)
        axdiff.set_xticks(Q_common)
        cb = fig.colorbar(pcm, ax=axdiff, label=r"$\Delta\sigma^\mathrm{ext}/A$",
                          pad=0.02)
        cb.ax.tick_params(labelsize=FONT_SIZE_TICK)
        print(f"  difference map: +/- {vlim:.4f}  over Q = {list(Q_common)}")
    else:
        axdiff.text(0.5, 0.5, "no common Q values", ha="center", va="center",
                    transform=axdiff.transAxes)
    axdiff.set_xlabel(Q_LABEL)
    axdiff.set_ylabel(r"$\omega$  (eV)")
    axdiff.set_title("difference:  new $-$ old")
    axdiff.set_ylim(0, omega_max)
    axdiff.tick_params(labelsize=FONT_SIZE_TICK)

    # comparison panel: both peak tracks overlaid. L2 is dashed because the two
    # tracks can lie on top of each other over part of the Q range.
    for key, title, col in models:
        Q, pk = peaks[key]
        ls = "--" if key == "L2" else "-"
        axpk.plot(Q, pk, ls, color=col, lw=2.2, marker="o", ms=5, label=title)
    axpk.set_xlabel(Q_LABEL)
    axpk.set_ylabel(r"plasmon peak $\omega_p$  (eV)")
    axpk.set_title(r"Peak frequency vs $Q$")
    axpk.set_xticks(data["L2"][0])

    # zoom onto the peak tracks: the full 0-omega_max window compresses the
    # shift into an unreadable sliver
    pk_all = np.concatenate([pk for _, pk in peaks.values()])
    lo, hi = np.nanmin(pk_all), np.nanmax(pk_all)
    pad = max(0.08 * (hi - lo), 0.05)
    axpk.set_ylim(max(lo - pad, 0.0), min(hi + pad, omega_max))
    axpk.grid(alpha=0.3)
    axpk.tick_params(labelsize=FONT_SIZE_TICK)
    axpk.legend()

    fig.suptitle(f"Plasmon frequency vs doping — old L2 vs new (Hubbard + Hartree) — "
                 f"{sweep_dir.name}", fontsize=FONT_SIZE_TITLE + 2)

    out = str(sweep_dir / "plasmon_Q_compare")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"wrote {out}.png\nwrote {out}.pdf")
    plt.show()

    # quick numeric summary: shift of the new model vs the old L2 baseline
    Q_ref, pk_ref = peaks["L2"]
    Q_new, pk_new = peaks["L2HH"]
    common = np.intersect1d(Q_ref, Q_new)
    if common.size:
        dpk = pk_new[np.searchsorted(Q_new, common)] - \
              pk_ref[np.searchsorted(Q_ref, common)]
        print("\n=== plasmon peak shift, new - old L2 ===")
        for Q, dv in zip(common, dpk):
            print(f"  Q = {Q:+.2f}: {dv:+.4f} eV")
        j = np.nanargmax(np.abs(dpk))
        print(f"  mean {np.nanmean(dpk):+.3f} eV   "
              f"max |shift| {np.abs(dpk[j]):.3f} eV (Q={common[j]:.2f})")


if __name__ == "__main__":
    main()
