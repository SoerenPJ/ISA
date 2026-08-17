# =============================================================================
# Figure 2 - Light-driven spin current in real space
#
# WHAT TO RUN: ONE simulation with a strong pulse, using a build that writes the
# per-bond spin current:
#   [field] mode = "time_impulse", spin_on = true, hubbard = true,
#   hubbard_dynamic = true, save_spin_diag = true
# (re-run it after the per-bond-spin-current change so J_spin_bond_time_evolution.txt exists)
# Point `path` at that output folder.
#
# Top : three snapshots (rising / peak / falling of the response envelope).
#       Node COLOUR = induced local moment m_i(t)-m_i^eq (where spin sits);
#       ARROWS on bonds = spin current J^spin_{ll'} (where spin flows).
# Bot : staggered moment m_stag(t) and |J_spin|(t); total S_z is conserved.
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt

path = "Simulations/graphene_zigzag_triangle_fe5e6d629795c05"
AU_FS = 0.0241888

# --- geometry ---
xy = np.loadtxt(path + "/lattice_points.txt", comments="#")
x, y = xy[:, 0], xy[:, 1]
sub = np.loadtxt(path + "/magnetization.txt", comments="#")[:, 3]     # sublattice +/-1

# --- induced local moment m_i(t) - m_i^eq ---
sd = np.loadtxt(path + "/spin_diag_time_evolution.txt", comments="#")
t = sd[:, 0]; ind = sd[:, 1:]
N = ind.shape[1] // 2
dm = ind[:, :N] - ind[:, N:]
m_stag = dm @ sub
Sz_ind = 0.5 * dm.sum(axis=1)

# --- net spin current magnitude (for the bottom trace) ---
sc = np.loadtxt(path + "/spin_current_time_evolution.txt", comments="#")
tj, Jmag = sc[:, 0], np.hypot(sc[:, 5], sc[:, 6])

# --- per-bond spin current (for the arrows) ---
bonds = np.loadtxt(path + "/bond_indices.txt", comments="#").astype(int)
jsb = np.loadtxt(path + "/J_spin_bond_time_evolution.txt", comments="#")
tjb, Jbonds = jsb[:, 0], jsb[:, 1:]                        # (time, N_bonds)
bx0, by0, bx1, by1 = x[bonds[:, 0]], y[bonds[:, 0]], x[bonds[:, 1]], y[bonds[:, 1]]
midx, midy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
ex, ey = bx1 - bx0, by1 - by0
L = np.hypot(ex, ey); ex, ey = ex / L, ey / L
a_bond = np.median(L)

# --- pick 3 vivid snapshot times: rising / peak / falling of the envelope ---
amp = np.sqrt((dm ** 2).sum(axis=1))
w = max(3, len(t) // 120); box = np.ones(w) / w
env = np.convolve(amp, box, mode="same")
pk = int(np.argmax(env)); half = 0.5 * env[pk]
rise = int(np.argmax(env > half))
fall = len(env) - 1 - int(np.argmax(env[::-1] > half))
W = max(2, len(t) // 60)
picks = [lo + int(np.argmax(amp[lo:hi]))
         for tgt in (rise, pk, fall)
         for lo, hi in [(max(0, tgt - W), min(len(amp), tgt + W))]]
vmax = np.abs(dm[picks]).max() or 1e-30

# per-bond spin current at the snapshot times (interpolated onto each bond's series)
Js_pk = np.array([[np.interp(t[i], tjb, Jbonds[:, b]) for b in range(bonds.shape[0])]
                  for i in picks])
Jbmax = np.abs(Js_pk).max() or 1e-30

# --- figure ---
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.25, wspace=0.05)
snap_ax = [fig.add_subplot(gs[0, k]) for k in range(3)]

for k, (ax, i, lab) in enumerate(zip(snap_ax, picks, ["rising", "peak", "falling"])):
    scat = ax.scatter(x, y, c=dm[i], cmap="seismic", vmin=-vmax, vmax=vmax,
                      s=110, edgecolors="0.3", linewidths=0.4, zorder=2)
    # spin-current arrows, centred on each bond, length ∝ |J^spin|
    ux = 0.55 * a_bond * (Js_pk[k] / Jbmax) * ex
    uy = 0.55 * a_bond * (Js_pk[k] / Jbmax) * ey
    ax.quiver(midx - 0.5 * ux, midy - 0.5 * uy, ux, uy,
              angles="xy", scale_units="xy", scale=1.0,
              color="k", width=0.006, headwidth=4, headlength=5, zorder=3)
    ax.set_title(f"{lab}\nt = {t[i] * AU_FS:.1f} fs", fontsize=10)
    ax.set_aspect("equal"); ax.axis("off")
cb = fig.colorbar(scat, ax=snap_ax, location="right", shrink=0.75, pad=0.01)
cb.set_label(r"induced moment  $m_i(t)-m_i^{\rm eq}$   (arrows: spin current)")

# --- time traces ---
axb = fig.add_subplot(gs[1, :])
axb.plot(t * AU_FS, m_stag, color="tab:orange", lw=0.4, alpha=0.3)
axb.plot(t * AU_FS, np.convolve(m_stag, box, mode="same"),
         color="tab:orange", lw=1.8, label=r"$m_{\rm stag}(t)$  (staggered moment)")
axb.set_xlabel("time [fs]")
axb.set_ylabel(r"$m_{\rm stag}(t)$", color="tab:orange")
axb.tick_params(axis="y", labelcolor="tab:orange")
for i in picks:
    axb.axvline(t[i] * AU_FS, color="0.6", ls=":", lw=0.8)

axr = axb.twinx()
axr.plot(tj * AU_FS, Jmag, color="tab:blue", lw=1.2, label=r"$|J_{\rm spin}|(t)$")
axr.set_ylabel(r"$|J_{\rm spin}|(t)$", color="tab:blue")
axr.tick_params(axis="y", labelcolor="tab:blue")

axb.text(0.99, 0.06,
         rf"total $S_z$ conserved:  max$|\Delta S_z|={np.abs(Sz_ind).max():.0e}$",
         transform=axb.transAxes, ha="right", fontsize=9, color="0.3")
h1, l1 = axb.get_legend_handles_labels()
h2, l2 = axr.get_legend_handles_labels()
axb.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False)

fig.suptitle("Light-driven spin current (colour = spin density, arrows = spin flow)", fontsize=13)
plt.savefig("spin_current_realspace.png", dpi=150, bbox_inches="tight")
print("saved: spin_current_realspace.png")
plt.show()
