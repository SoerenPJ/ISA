# Hubbard mean-field magnetism — feature documentation

This document explains the self-consistent Hubbard (unrestricted Hartree–Fock,
"UHF") magnetism feature added to the tight-binding code: the physics, the
equations before and after, the algorithm, every file that changed, the config
knobs, the outputs, and the validation.

---

## 1. Why this was added

Before this feature the model had **no equilibrium magnetism**:

* The equilibrium density matrix was a plain Fermi–Dirac filling of the *bare*
  tight-binding eigenstates (`Rho_0` in `main.cpp`). The spin-up and spin-down
  blocks of the Hamiltonian were **identical**, so every eigenstate was
  spin-degenerate and the net moment was exactly zero.
* The only Coulomb term acting during the dynamics is a **charge** Hartree
  term, `V_ee · (ρ − ρ₀)` (`add_Hartree_to_H`). It is spin-summed and vanishes
  at equilibrium, so it can neither split spins nor create a magnet.
* There is no spin–orbit coupling, so `S_z` commutes with `H` and is exactly
  conserved.

Consequence: the only spin signal in the whole simulation was the tiny
(~10⁻⁹) dynamical splitting from the self-consistent induced magnetic field.
Every spin observable just followed the driving pulse and then relaxed with it,
because there was no intrinsic spin physics in the Hamiltonian.

The zigzag graphene triangle, however, *should* be magnetic: it has a
**sublattice imbalance** `N_A − N_B = 4` (25 A-sites, 21 B-sites), which shows
up as **4 zero-energy modes per spin** in the non-interacting spectrum. Lieb's
theorem then predicts a magnetic ground state with total spin

```
S = |N_A − N_B| / 2 = 2.
```

The Hubbard feature puts that physics into the Hamiltonian.

---

## 2. The physics and how the equations change

### 2.1 Starting point — non-interacting tight binding

The spinful tight-binding Hamiltonian is block-diagonal in spin (site basis,
ordering `[up_0 … up_{N-1}, dn_0 … dn_{N-1}]`):

```
H_TB = Σ_{⟨ij⟩, σ}  t_ij  c†_{iσ} c_{jσ}
```

Because the up block and the down block are the *same* matrix, the spectrum is
spin-degenerate and `⟨n_{i↑}⟩ = ⟨n_{i↓}⟩` at every site → **no magnetization**.

### 2.2 Add the Hubbard interaction

The Hubbard model adds an on-site Coulomb repulsion — an energy cost `U` for
putting two (opposite-spin) electrons on the same site:

```
H = H_TB + U Σ_i  n_{i↑} n_{i↓}
```

This term is **quartic** (a product of two number operators), so `H` can no
longer be diagonalized as a single-particle problem. We treat it at the
**mean-field (unrestricted Hartree–Fock) level**.

### 2.3 Mean-field decoupling

Replace the product of operators by fluctuations around their averages and drop
the (second-order) fluctuation–fluctuation term:

```
n_{i↑} n_{i↓}  ≈  ⟨n_{i↑}⟩ n_{i↓} + n_{i↑} ⟨n_{i↓}⟩ − ⟨n_{i↑}⟩⟨n_{i↓}⟩
```

Substituting back, the interaction becomes a **single-particle, spin-dependent
on-site potential**: an electron of spin σ on site i feels the average density
of the *opposite* spin on the same site:

```
H_MF = H_TB + Σ_{i,σ}  ε_{iσ} n_{iσ} − U Σ_i ⟨n_{i↑}⟩⟨n_{i↓}⟩ ,

     with   ε_{iσ} = U ⟨n_{i,−σ}⟩ .
```

The last (constant) term only shifts the total energy and is dropped from `H`.

**Particle–hole-symmetric form.** We implement the standard shifted version

```
V_{iσ} = U ( ⟨n_{i,−σ}⟩ − 1/2 ) ,
```

i.e. the on-site shift is measured relative to half-filling `⟨n⟩ = ½`. This
keeps the spectrum symmetric about `E = 0` at charge neutrality, so the
chemical potential stays at `μ = 0` (matching how `Rho_0` is used elsewhere).

### 2.4 What actually changed in the Hamiltonian

The up and down blocks are **no longer identical**. On the diagonal:

```
H_MF(i, i)         =  H_TB(i, i)         + U ( n_{i↓} − ½ )     ← spin up
H_MF(N+i, N+i)     =  H_TB(N+i, N+i)     + U ( n_{i↑} − ½ )     ← spin down
```

If a site has more down-electrons than up (`n_{i↓} > n_{i↑}`), the up level is
pushed *up* in energy and the down level *down*, which makes it even more
favourable for down to occupy that site. This positive feedback is the
**Stoner mechanism**: above (here, any) critical `U` the symmetric solution
`n_{i↑} = n_{i↓}` becomes unstable and the system spontaneously develops a
site-resolved **magnetization**

```
m_i = ½ ( n_{i↑} − n_{i↓} ) ,      S_z = Σ_i m_i .
```

For the bipartite graphene flake the moments arrange antiferromagnetically
(A-sublattice up, B-sublattice down), and the **net** moment is carried by the
sublattice imbalance → `S_z = |N_A − N_B|/2 = 2`, exactly Lieb's theorem.

---

## 3. The self-consistency algorithm

`V_{iσ}` depends on the occupations `{n_{iσ}}`, which in turn depend on
`V_{iσ}` through the eigenstates. This is solved by iteration
(`solve_hubbard_mft` in `Hamiltonians/hubbard.cpp`):

1. **Seed (symmetry breaking).** 2-colour the lattice from the bond graph
   (BFS → sublattice label `s_i = ±1`) and start from a staggered guess
   ```
   n_{i↑} = ½ + ½ · m_seed · s_i ,   n_{i↓} = ½ − ½ · m_seed · s_i .
   ```
   Without a seed the symmetric solution `n↑ = n↓` is a fixed point and no
   magnetism would appear.
2. **Build** `H_MF` by adding `U(n_{i,−σ} − ½)` to the diagonal of `H_TB`.
3. **Diagonalize** `H_MF` (`compute_eigenpairs`).
4. **Fill** to get the new density: `ρ₀ = Rho_0(ε, μ, T)` (or `Rho_0_charge`
   when `use_charge_doping`), transform to the site basis
   (`ρ_l = V ρ₀ V†`), and read off `n_{i↑} = ρ_l(i,i)`, `n_{i↓} = ρ_l(N+i,N+i)`.
5. **Mix** (linear): `n ← (1 − mix)·n_old + mix·n_new`. Mixing damps the
   Stoner feedback so the loop converges instead of oscillating.
6. **Check convergence:** `max_i |Δn_{iσ}| < tol` (default `1e-8`). Otherwise
   go to 2. (default `max_iter = 400`).
7. **Finalize:** one more build/diagonalize to get the converged occupations,
   the on-site potentials `V_{iσ}`, the magnetization `m_i`, the net `S_z`,
   `Σ|m_i|`, and the HOMO–LUMO **spin gap**.

The converged **`V_{iσ}` is applied in two places** so the magnetic state is the
equilibrium seen by both the static eigenproblem and the dynamics:

* **Eigenproblem / `rho0`:** folded into `Hc_eig` in `main.cpp` (a copy), so
  the eigenvalues, `HTB.txt` and the initial density matrix are magnetic.
* **Time evolution:** kept **out of the tight-binding base `Hc` / `H0`** and
  stored in the solver (`hub_V_up`, `hub_V_dn`); it is **re-added on the
  diagonal every step inside `build_H_for_time`**, in *every* branch.

Why the second point matters (this was a bug during development): when
`self_consistent_phase = true`, `build_H_for_time` **rebuilds `H_out` from the
hopping** each step (`TB_hamiltonian_from_points_with_phases`). If the exchange
field lived only in `H0`, that rebuild would drop it, the magnetic `rho0` would
**not be stationary**, and it would churn at the ~eV exchange scale — which both
gave wrong dynamics and forced the adaptive ODE solver into tiny time steps
(a ~4× slowdown). Re-adding `V_{iσ}` in every branch keeps `rho0` stationary
(net induced `S_z` stays ~1e-9) and removes the slowdown (the UHF solve itself
is ~0.1 s).

---

## 4. Files changed / added

| File | Change |
|------|--------|
| `Hamiltonians/hubbard.hpp` | **New.** `HubbardResult` struct + `solve_hubbard_mft(...)` declaration. |
| `Hamiltonians/hubbard.cpp` | **New.** UHF self-consistency loop, bipartite seed (`bipartite_colour`), occupation helper (`occupations_from_H`). |
| `main.cpp` | Include `hubbard.hpp`; after the spinful `Hc` is built and gated on `p.spin_on && p.hubbard`, run the solver, store `V_{iσ}` in `params`, fold it into `Hc_eig` (eigenproblem only), print a summary, write `magnetization.txt`. |
| `DensityMatrix/Density.hpp` / `Density.cpp` | Solver carries `hubbard_on`, `hub_V_up`, `hub_V_dn`; `build_H_for_time` re-adds the exchange field on the diagonal in every branch. `Rho_0` clamps negligible Fermi tails to exact 0/1. |
| `params/params.hpp` / `params.cpp` | New config members: `hubbard`, `hubbard_U_eV`, `hubbard_seed`, `hubbard_mix` (from `[features]`); derived `hub_active`, `hub_V_up`, `hub_V_dn`. |
| `configs/graphene_zigzag_triangle.toml` | Documented the new toggles; flagged that `coulomb_onsite_eV` is unused. |
| `build_BLAS.sh`, `build_fast.sh`, `build_mkl.sh` | Added `Hamiltonians/hubbard.cpp` to the compile list. |
| `ploting/magnetization.py` | **New.** Plots and quantifies the magnetization (map + Lieb check + edge localization). |

No existing behaviour changes when `hubbard = false` (default): the solver is
never called and `Hc` is untouched.

---

## 5. The value of U (important detail)

The Hubbard `U` is, by default, **`vvR(0.0)`** — the on-site value of the
graphene-specific fitted Coulomb kernel `vvR(R)` (`Hamiltonians/potential.cpp`),
`≈ 0.578 a.u. ≈ 15.72 eV`. This is the same value the rest of the code uses for
the on-site Coulomb: `build_coulomb_matrix` sets `V_ee(i,i) = vvR(0)`.

> Note: the TOML key `coulomb_onsite_eV` is **not used anywhere** in the code —
> in `build_coulomb_matrix` the line that would use it is commented out in
> favour of `vvR(0)`. So it does not, and now visibly should not, control `U`.

You can **override** `U` from the config with `hubbard_U_eV` (see below), e.g.
to scan `U` or to use a more screened effective value. If unset (negative),
`vvR(0)` is used.

---

## 6. Config knobs (`[features]`)

```toml
[features]
hubbard          = true   # turn the self-consistent UHF magnetism on/off (default false)
# hubbard_U_eV = 3.0      # optional U override in eV; if unset/negative -> vvR(0) ~ 15.7 eV
hubbard_seed     = 0.5    # initial staggered moment for symmetry breaking (0..1)
hubbard_mix      = 0.3    # linear mixing factor for the self-consistency loop (0..1)
hubbard_fd_fill  = true   # Fermi-Dirac smeared canonical fill in the SCF loop (default true)
# hubbard_smear_T = 300   # smearing temperature [K] for that fill; unset/negative -> [thermo] T
```

`hubbard` requires `spin_on = true`. Filling follows the existing `[thermo]`
controls (`T`, `mu`, `use_charge_doping`, `Q_doping`).

### `hubbard_fd_fill` — why the SCF loop smears

The fill is always **canonical**: the total is pinned to `N + Q` exactly, so the doping
is never affected. `hubbard_fd_fill` only decides whether levels within ~`kT` of the
Fermi level are occupied fractionally (`true`, default) or by a hard 1/0 step (`false`,
the pre-fix behaviour).

The hard step breaks down whenever the doped electron lands in a near-degenerate shell.
The mean field moves the diagonal by `U·|Δn|` each iteration; with `U = 15.7 eV` that is
~1.4 eV at a residual of 0.09, so any two levels closer than that swap order every
iteration and a full electron jumps between them. The residual then cannot fall below
that jump and the loop **limit-cycles forever** — it is a discontinuous refill, not an
overshoot, so no value of `hubbard_mix` helps.

Measured on the 5×5 **armchair** triangle (`data/Qsweep_armchair_triangle_5x5_rot0`),
which is sublattice-balanced and therefore has no protected zero-mode multiplet — the
neutral flake is a closed shell with a 13.3 eV U-split gap, and the first doped electrons
drop into the dense near-continuum just above it:

| Q | hard fill | smeared fill |
|---|---|---|
| 0 | converged, 1144 iters, gap 13.2815 eV | **identical**, bit-for-bit |
| 1 | **never converges**, 50000 iters, residual stuck in [0.081, 0.108] | converged, 5706 iters, residual 1.0e-8 |
| 2 | **never converges**, 50000 iters | converged, 8285 iters, residual 1.0e-8 |

The smearing costs nothing where it is not needed: a gap large against `kT` gives an FD
correction of `e^{−gap/kT}`, which at `T = 300 K` and a 13.4 eV gap is `~e^{−514}`. The
5×5 zigzag `Q = 0` Lieb `S = 2` ground state comes back identical line-for-line.

**Read the converged small-gap results carefully.** For armchair `Q = 1` the true
self-consistent state has levels 90/91 *exactly* degenerate at 8.59937 eV with occupation
0.4259 each, plus level 92 at 0.1481 — one electron shared over a three-fold shell, gap
`1.9e-9 eV`. The old "0.0075 eV gap" was itself an artifact of the limit cycle: the hard
fill was forced to put the doped electron *entirely* into one member of a degenerate pair,
and that arbitrarily symmetry-broken occupation is what the loop kept flip-flopping over.
`S_total` correspondingly stops being a clean half-integer (armchair `Q = 1`: −0.499991);
that is the honest finite-`T` answer for a degenerate open shell, not a bug.

### This also fixes the dynamics

The hard-fill state was not merely unconverged, it was a **saddle** of the mean field: a UHF
solution with a negative eigenvalue of the RPA/TDHF stability matrix has a linear-response
root at imaginary ω, so the induced dipole grows as `e^{|Im ω|t}` instead of oscillating.
That is exactly what the old runs did — armchair 5×5 `Q = 1` grew from 7e-3 to 2.8 with the
driving pulse long over (rate ≈ 0.22 eV, against only `γ/2 = 0.025 eV` of damping), saturated
when the occupations hit the [0,1] bound, and produced a meaningless `sigma_ext` with large
*negative* (gain) excursions.

Occupying the degenerate shell *equally and fractionally* restores its symmetry, so the
unstable mode is gone. Full 180 fs runs, unchanged configs:

| case | max\|d\| before → after | `sigma_ext` range before → after |
|---|---|---|
| armchair 5×5 `Q = 1` | 2.81 → **1.84e-2** | [−1654, 1948] → **[0, 7.76]** |
| zigzag 5×5 `Q = 1`   | 0.783 → **9.26e-3** | [−1064, 1111] → **[−0.006, 30.2]** |

Both dipole envelopes now decay monotonically over three decades instead of growing.
Note zigzag `Q = 1` had reported `converged = 1` before — it had simply landed on a
different, spurious branch with a 0.0336 eV gap; the smeared fill finds a properly gapped
state (0.397 eV, `S_total = 1.5` exactly). **A `converged = 1` flag was never sufficient
evidence that the state was stable.**

Sweep `hubbard_smear_T` to tell a genuinely converged result from a smeared one — the same
discipline as the `hubbard_seed` and `hubbard_mix` sweeps. Armchair `Q = 1` at 100 / 300 /
600 K gives `S_total` = −0.5 / −0.499991 / −0.496228 and `sum_abs_m` = 37.2197 / 37.2200 /
37.2206, i.e. the degenerate-shell state is real and not a smearing artifact. Note also that
UHF is nonlinear with multiple stationary points, so changing the fill can change *which*
solution the loop lands on: armchair `Q = 4` converges in both modes but to different
branches (gap 0.682 eV vs 0.304 eV). The onsite mean field is a
single unified spin-resolved term `V_{iσ} = U(n_{i,−σ} − ½)` that is live in the
dynamics by construction — there is no separate static/dynamic toggle or Stoner
`U` (see Section 10).

---

## 7. Outputs

* **`magnetization.txt`** (written to the run's `Simulations/<...>/` folder):
  header with `U_eV`, `S_total`, `sum_abs_m`, `gap_eV`, `converged`, `iters`,
  then one row per site: `site  x  y  sublattice  n_up  n_dn  m_i`.
* **Console:** `[Hubbard] UHF mean-field U = … eV (…) (converged in N iters)`
  and `net spin S_z = …  sum|m_i| = …  spin gap = … eV`.
* **`ploting/magnetization.py <SIM_DIR>`** → `magnetization.png/.pdf`:
  * spin-density map on the lattice (blue = up, red = down, size ∝ |m|),
  * net moment and sublattice-resolved moments,
  * Lieb check `S = |N_A − N_B|/2`,
  * `|m_i|` vs distance from the flake centre (edge localization).

---

## 8. Validation

For the 5×5 zigzag triangle (46 sites, `N_A = 25`, `N_B = 21`):

* **Net moment `S_z = 2.000`**, exactly `|N_A − N_B|/2` (Lieb's theorem) and
  consistent with the 4 non-interacting zero modes per spin.
* Staggered antiferromagnetic texture: A-sublattice `m ≈ +0.46`,
  B-sublattice `m ≈ −0.44` at `U = vvR(0)`; edge sites carry slightly larger
  moments than the interior.
* **`U`-scan** (net moment is Lieb-protected, the magnitude/gap scale with `U`):

  | U (eV) | net S_z | Σ\|mᵢ\| | spin gap |
  |--------|---------|---------|----------|
  | 15.72 (vvR(0)) | 2 | 19.44 | 13.41 eV |
  | 3.0  | 2 | 3.00 | 0.62 eV |
  | 0.5  | 2 | 2.11 | 0.085 eV |

  The net `S = 2` survives down to tiny `U` because the four zero modes are
  exactly degenerate at `E = 0`; any repulsion spin-polarizes them
  (flat-band / Stoner ferromagnetism). The AF background `Σ|m_i|` and the gap
  grow with `U`; at `vvR(0) ≈ 15.7 eV` the flake is in strong coupling
  (near-saturated moments, Mott-like gap `~U`).

---

## 9. Scope and next steps

* The onsite mean field is **fully time-dependent (TDUHF) by construction**: the
  ground state is solved once (Section 3), then the *same* unified onsite field
  `V_{iσ}(t) = U(n_{i,−σ}(t) − ½)` is re-evaluated against the live `ρ(t)` at every
  step — see **Section 10**. There is no static/dynamic toggle.
* No double counting at equilibrium: the frozen equilibrium part lives in `Hc`,
  while the live deviations (onsite Hubbard and nonlocal Hartree) both use
  `(ρ(t) − ρ₀)`, which is zero at `t = 0`, so `ρ₀` stays stationary.
* A second independent toggle for **spin–orbit coupling** (Kane–Mele + Rashba)
  would add spin-Hall physics and genuine spin relaxation on top of the magnet.

---

## 10. Unified time-dependent onsite Hubbard field (TDUHF)

Following the "New model" of `SSH__Spin_current-1.pdf` (Eq. 25/26), the onsite mean
field is a **single unified spin-resolved term** with one `U`,

```
V_{iσ}(t) = U ( n_{i,−σ}(t) − ½ ),      n_{iσ}(t) = Re ρ_{iσ,iσ}(t)
```

evaluated against the live `ρ(t)` at every step, so the magnet responds to the
drive (spin ringing, driven demagnetization, exchange spin current). There is **no
charge/spin channel split and no separate Stoner `U`** — that split was a relic of
the old model, valid only while the onsite Hartree was retained. Now the onsite
Coulomb `v(0)` is carried entirely by `U` (the Hartree diagonal is zeroed), so the
whole onsite response is this one term.

### 10.1 What is added each step (frozen anchor + live deviation)

Inside `build_H_for_time` (spin block) the equilibrium part sits in the frozen
field `hub_V_{up,dn} = φ_final + U(n_{i,−σ}^eq − ½)`; we add only the **live
opposite-spin deviation** with the same single `U`:

```
H(i,i)     += hub_V_up(i) + U ( n_{i↓}(t) − n_{i↓}^eq )      (up block)
H(N+i,N+i) += hub_V_dn(i) + U ( n_{i↑}(t) − n_{i↑}^eq )      (dn block)
```

* It is a **functional of the instantaneous ρ(t)** (like the Hartree term), so the
  dopri5 stages carry the self-consistency — **no inner SCF loop, ~O(N) extra work
  per step**.
* Anchoring on the frozen field (deviation `n(t) − n^eq` rather than bare `n(t)`)
  makes `ρ₀` **stationary for any filling** — the deviation vanishes at `t = 0`,
  recovering exactly the equilibrium field. This matters because graphene **edges
  are not half-filled**, so a bare form would leave an onsite mismatch at `t = 0`.
* Equivalently `U(n_{i,−σ}(t) − n_{i,−σ}^eq) = (U/2)Δn_i ∓ (U/2)Δm_i`: the same
  single `U` supplies **both** the charge restoring force (`U/2`, halved because
  the Fock term cancels the onsite self-interaction) and the spin exchange. Keeping
  the charge restoring force is what stabilizes the dynamics (see §12).

### 10.2 Stability

Time-dependent UHF around a broken-symmetry state can be **dynamically unstable**
(a Thouless/RPA instability) when that state is a saddle rather than a true
minimum — most easily at **small `U` / small HOMO–LUMO gap**. The charge restoring
force embedded in the unified field (the `U/2` onsite stiffness) suppresses this.
An earlier scheme that instead *deleted* the onsite term to make the full field
dynamic removed that restoring force and blew up from noise under zero drive; the
unified field keeps it.

### 10.3 Files changed

| File | Change |
|------|--------|
| `params/params.{hpp,cpp}` | Removed `hubbard_dynamic`, `hubbard_U_spin_eV`, `hub_U_spin`, `hub_m_eq`; derived `hub_U`, `hub_n_up_eq`, `hub_n_dn_eq`. |
| `main.cpp` | Store `hub_U` and the equilibrium occupations `hub_n_up_eq = n_up`, `hub_n_dn_eq = n_dn`; updated console label. |
| `DensityMatrix/Density.{hpp,cpp}` | Solver carries `hub_U`, `hub_n_up_eq`, `hub_n_dn_eq`; `build_H_for_time` adds the live unified deviation `U(n_{-σ}(t) − n_{-σ}^eq)` (classical Hartree left unchanged). |

### 10.4 Validation

* **Zero drive** (`intensity = 0`): at `t = 0` the deviation vanishes, so the onsite
  field equals the frozen field and `ρ₀` stays stationary (no dipole / spin drift).
* A former `hubbard_dynamic = true` run with `U_spin = U` is algebraically identical
  to the new default and reproduces it to numerical precision.

---

## 11. Self-consistent nonlocal charge Hartree in the ground state (`hubbard_hartree`)

By default the ground state is blind to added electrons: `Rho_0_charge` just fills
higher bare eigenstates, so a doped flake never feels the mutual Coulomb repulsion
of the extra charge. With `[features] hubbard_hartree = true` the **nonlocal charge
Hartree** is solved self-consistently *in the same UHF loop*, so doping `Q` makes the
density redistribute (edge accumulation), and that rearrangement is baked into `ρ₀`.

### 11.1 What is added — charge/spin decomposition

With Hartree on, the onsite Coulomb energy is split into its **charge** and **spin**
channels, which are orthogonal (so nothing is double-counted):

> **Superseded by §12.** The equations below describe the version in which the
> onsite `v(0)` stayed inside the charge Hartree. As of §12 the onsite element is
> excluded from `φ` and carried by `U` alone. Kept here because it documents why
> the onsite term was put in the charge channel in the first place.

```
V_{iσ} = φ_i  ∓  (U/2) M_i ,        M_i = n_{i↑} − n_{i↓}   (− for ↑, + for ↓)
φ_i    = Σ_j  V_ee(i,j) ( n_j − 1 ),    n_j = n_{j↑}+n_{j↓}   (INCLUDING onsite j=i)
         └──── full charge Hartree (spin-blind) ────┘   └── Stoner spin ──┘
```

* **Charge** is driven by the *full* Coulomb `V_ee` — onsite `v(0)=vvR(0)` **plus**
  nonlocal. The strong onsite (`≈15.7 eV`) pins `n_i ≈ 1`, so doped electrons
  redistribute *smoothly to the edges* (paper Fig. 2 / S1a) instead of forming a
  spurious sublattice CDW. (An earlier version excluded the onsite and let the small
  overridden `U` set the charge stiffness — `U/2≈1.8 eV` ≪ nearest-neighbour
  `≈8.6 eV` — which produced a violent checkerboard; including `v(0)` fixes it.)
* **Spin** is driven by the onsite `U`, which (as of §12) is the same single `U`
  that sets the onsite charge stiffness — there is no separate Stoner coupling.

`V_up/V_dn` are folded into `Hc_eig` and re-added every step in `build_H_for_time`;
the dynamical `V_ee·(ρ−ρ₀)` stays referenced to this `ρ₀`, so static + dynamical
Hartree never double-count. **Hartree off** keeps the plain UHF onsite term
`U(n_{i,−σ} − ½)` unchanged (validation in §8 is untouched).

### 11.2 Correspondence to the paper

Matches Yu, Cox & García de Abajo, *PRL* **117**, 123904 (2016), SI Eqs. (S1)–(S2):
`V_Hartree(l) = Σ_{l''} v_{ll''} n_{l''}` with the `+1`/site ionic background
(`n⁰ = 1`, **not** a neutral-SC-density reference), `H = H_TB + V_Hartree` (no analyte).
The paper mixes the Hamiltonian with `β = 0.01` over "a few hundred iterations"; we
mix the density, so use a **small `hubbard_mix` (~0.05–0.1)** for doped runs.

### 11.3 Config

```toml
[thermo]
use_charge_doping = true
Q_doping          = 4        # extra electrons vs neutral
[features]
hubbard          = true      # required (combined UHF)
hubbard_hartree  = true      # self-consistent nonlocal charge Hartree
hubbard_mix      = 0.1       # small mixing; charge Hartree is stiffer than the spin loop
```

**Doping by chemical potential instead of fixed count.** By default the SCF loop
fills *canonically* (fixed `N + Q`). Set `[features] hubbard_mu_filling = true` to fill
*grand-canonically* by Fermi–Dirac at `([hamiltonian] mu, [thermo] T)` — the electron
number then floats and is set by `mu` (gate/chemical-potential doping). `mu` is in **eV**.

```toml
[hamiltonian]
mu = 1.0                     # chemical potential (eV); electron number floats
[thermo]
use_charge_doping = false    # required: keeps main's final rho0 grand-canonical too
[features]
hubbard          = true
hubbard_hartree  = true      # optional; charging energy then shifts the neutrality point off mu=0
hubbard_mu_filling = true
```

Verified (5×5 zigzag triangle, no Hartree): `N_total` rises with `mu` — 46 (μ=0) →
50 (μ=0.5–2 eV plateau) → 56 (μ=3 eV), the plateaus being the quantum-dot shell gaps.
Note: for the *magnetic* zero-mode flake, canonical filling is the safe default at
neutrality; μ-mode is meant for gated / away-from-neutral / non-magnetic studies (the
exchange gap does keep `S_z = 2` at μ=0 here, but that is not guaranteed in general).

### 11.4 Spatial maps & validation (5×5 zigzag triangle, U = 3.64 eV)

Test/visualise with the doping sweep + map plotter (reproduces the paper's w/o-vs-w/
self-consistency comparison, Fig. 2 / S1a, plus the spin texture):

```
./hubbard_doping_sweep.sh configs/graphene_zigzag_triangle.toml   # Q_LIST, U_EV, MIX
python3 ploting/hubbard_doping_maps.py data/hubbard_doping_<tag>_maps
```

* **Neutral, `Q = 0`:** Hartree on **exactly preserves** the magnet
  (`Σ|m_i| = 3.37`, `S_z = 2`, charge `std = 0.000`) — the strong onsite `v(0)` pins
  `n_i = 1`, so the magnetism knob (`U`) is untouched at neutrality.
* **Doped:** charge `std ≈ 0.065 (Q=2) / 0.073 (Q=4) / 0.100 (Q=6)` with **smooth
  edge/corner accumulation** (no CDW); the self-consistent column shows stronger edge
  localization than the non-self-consistent one. The magnet collapses with doping,
  `S_z = 2 → 1 → 0 → 1`, as the four zero modes fill.
* **Convergence:** open-shell doped *magnetic* configurations (e.g. `Q = 2`, `S = 1`)
  resist linear mixing even at `mix = 0.03` (near-degenerate partial shells slosh) —
  these need Pulay/DIIS. Closed-shell / larger-gap dopings converge (`Q = 6` at
  `mix = 0.05`, ~586 iters; `Q = 0, 4` at `mix = 0.1`). Use the `hubbard_mix` sweep to
  pick `mix` per doping.

---

## 12. `v(0)` as the Hubbard energy — onsite Coulomb excluded from the Hartree

**Prescription (this section supersedes the onsite bookkeeping of §10.1 and §11.1).**
The `i = j` element of the density–density Coulomb *is* the Hubbard term. So:

* `U = v(0) = vvR(0) ≈ 15.72 eV` is the Hubbard energy (now the default whenever
  `hubbard_U_eV` is unset — the key defaults to `-1.0`, not `0.0` as before);
* whenever `[features] hubbard = true`, the **onsite element of `V_ll'` is removed
  from every density–density evaluation**, static and dynamic, so it is counted
  once, through `U`, and acts **spin-resolved** instead of spin-blind.

### 12.1 Equations

Static SCF (`Hamiltonians/hubbard.cpp`), one expression for both Hartree on/off:

```
V_{iσ} = φ_i + U ( n_{i,−σ} − ½ ) ,     φ_i = Σ_{j≠i} V_ee(i,j) ( n_j − 1 )
         └ nonlocal only ┘   └── onsite Coulomb, spin-resolved ──┘
```

Split into channels, the single onsite `U` supplies **both**:

```
U ( n_{i,−σ} − ½ )  =  (U/2)( n_i − 1 )   ∓  (U/2) m_i
                       └ charge stiffness ┘  └ spin exchange ┘
```

The charge stiffness is `U/2`, **not** `U`: the Fock term exactly cancels the
same-spin onsite Hartree, i.e. an electron does not repel itself. That halving is
the whole physical content of the change.

Dynamics (`DensityMatrix/Density.cpp`, `build_H_for_time`): the Hartree runs on
`V_ee_hartree` (= `V_ee` with a **zeroed diagonal** when the Hubbard is on) and the
onsite response is added in the Hubbard block as deviations from equilibrium:

```
dV_{i,↑}(t) = U ( n_{i↓}(t) − n_{i↓}^eq )    (= (U/2)Δn_i − (U/2)Δm_i)
dV_{i,↓}(t) = U ( n_{i↑}(t) − n_{i↑}^eq )    (= (U/2)Δn_i + (U/2)Δm_i)
```

a single unified `U` supplying both channels at once. Both vanish at `t = 0`, so
`ρ₀` stays stationary. `V_ee` itself (with its diagonal)
is untouched for the induced-vector-potential / current kernel — the Hubbard
replaces the density–density onsite Coulomb, not the current–current one.

### 12.2 Consequence: the charge kernel becomes indefinite (CDW runaway)

The spin-blind charge kernel the dynamics sees is `K_ij = V_ee(i,j)` for `i≠j` and
`K_ii = U/2`. Measured on the 5×5 zigzag triangle (46 sites, `Q = 1`):

| onsite diagonal | value | min eig(K) | dynamics |
|---|---|---|---|
| `v(0)` (old behaviour) | 15.72 eV | **+3.69 eV** | stable |
| `U/2`, `U = v(0)` | 7.86 eV | **−4.17 eV** | runs away |
| `U/2`, `U = 3.64 eV` | 1.82 eV | **−10.21 eV** | runs away |
| zeroed, no replacement | 0 | −12.03 eV | runs away |

A negative eigenvalue is an exponentially growing charge-density-wave mode. Verified
directly: with `intensity = 0`, `gamma = 0` and **no** driving field, site occupations
drift from `1.00–1.05` to `0.21–1.80` over 40 fs; the same run with `hubbard = false`
stays exactly on `1.00–1.06`. It is the charge channel (the `U/2` onsite stiffness),
independent of the spin part. §10.2 and §11.1 record the same failure from earlier
attempts to drop the onsite term.

The cause is a model inconsistency, not a bug: this `vvR` kernel has
`v(0)/v(NN) = 15.72/8.61 = 1.83`, so `v(0)/2 < v(NN)` — the flake sits on the
CDW-unstable side of the extended-Hubbard phase boundary. (Ohno/PPP graphene,
`U = 11.13`, `V_NN = 5.4 eV`, gives `U/2 = 5.57 > V_NN` and stays marginally stable.)

Ways out, in order of increasing effort:

1. **Screen the nonlocal tail.** Scale off-diagonal `v(R)` by `≤ 0.653` (`ε_r ≥ 1.53`)
   for `U = v(0)`; by `≤ 0.151` (`ε_r ≥ 6.6`) for `U = 3.64 eV`. Restores positive
   definiteness and is defensible — the tail *is* screened by the substrate/σ-bands.
2. **Add the intersite Fock (exchange) term.** The proper fix: it halves the
   same-spin nonlocal repulsion the same way the onsite Fock term does, which is
   exactly the piece whose absence unbalances the kernel. Not implemented.
3. **Keep the onsite in the Hartree** (`hubbard = false`, or revert §12) — stable,
   but double-counts `v(0)` when `U = v(0)`.

`main.cpp` computes `min eig(K)` at startup, prints it, and emits a loud warning
when it is negative, so the runaway is announced rather than silent.

---

## 13. Hubbard-free self-consistent Hartree ground state (`hartree_scf`)

The mode §12.2 point 3 points at, made explicit: **no Hubbard at all**, but the ground
state is still found self-consistently — purely from the electrons' own electrostatics,
using the **full** `V_ee` kernel with the **onsite diagonal included**.

```
[features]
coulomb     = true
hubbard     = false     # no U, no exchange, no magnetism
hartree_scf = true      # SCF ground state from the full V_rr', diagonal INCLUDED
hubbard_mix = 0.02      # NOTE: the Hubbard default 0.05 sloshes here — see below
```

`hartree_scf` is ignored (with a note on stderr) when `hubbard = true`: exactly one
channel may own the onsite Coulomb, never both.

### 13.1 Equations

The same SCF loop as §11, run at `U = 0` and with the `j = i` term kept:

```
    phi_i    = sum_j  V_ee(i,j) ( n_j - 1 )        <- ALL j, diagonal included
    V_{i,up} = V_{i,dn} = phi_i                    <- spin-blind: no exchange term
```

versus the Hubbard branch's `phi_i = sum_{j != i} V_ee(i,j)(n_j - 1)` plus
`U(n_{i,-σ} - 1/2)`. The reference is the same neutral ionic `n_j^0 = 1` (SI Eq. S1).

Because both spin blocks see an identical field, the converged state carries no moment
(`S_z` and `max|m_i|` come out at ~1e-15). `hubbard_seed` is forced to 0 in this mode —
a staggered seed cannot survive a spin-blind field, so starting polarised only wastes
iterations.

**This is a bare classical Hartree.** With no Fock partner the onsite self-interaction
is *not* removed — that removal is exactly what turns `v(0)` into the `U/2` stiffness of
the Hubbard branch (§12.1). Every electron here feels its own onsite charge. That is the
deliberate content of the mode, and it is what makes the charge kernel `K = V_ee`
unchanged and positive definite (`min eig = +3.69 eV`, the stable top row of §12.2's
table) instead of indefinite.

### 13.2 Static/dynamic consistency

`Density.cpp` normally zeroes the `V_ee_hartree` diagonal whenever an SCF ran, so `v(0)`
is not counted twice (once as `U`, once as the Hartree diagonal). Under `hartree_scf`
there is no `U`, so the diagonal is **kept**, and the static and dynamic Hartree are the
same model:

```
    static  :  phi_i^eq = sum_j V_ee(i,j) ( n_j^eq - 1 )          (folded into hub_V_up/dn)
    dynamic :  + sum_j V_ee(i,j) ( n_j(t) - n_j^eq )              (add_Hartree_to_H)
    total   :  sum_j V_ee(i,j) ( n_j(t) - 1 )                     <- all j, no double count
```

The live `U(n_{-σ}(t) - n_{-σ}^eq)` term of §10 is identically zero (`hub_U = 0`), so the
dynamics is spin-blind too.

### 13.3 Filling: FD smearing is mandatory here

With `U = 0` nothing splits the flake's degenerate zero-mode shell, so the hard canonical
fill has to pick arbitrarily among degenerate states; the pick flips between iterations
and the loop limit-cycles. Measured on the 5×5 zigzag triangle at `Q = 1`:

| fill | mixing | result |
|---|---|---|
| hard canonical | 0.05 / 0.01 / 0.002 | **period-3 limit cycle**, residual pinned at ~0.15 |
| canonical + FD smearing | 0.05 | sloshes, residual pinned at ~0.099 |
| canonical + FD smearing | 0.02 | **converged, 729 iters** |
| canonical + FD smearing | 0.01 | converged, 1464 iters |
| canonical + FD smearing | 0.005 | converged, 2934 iters |

So `hartree_scf` forces `fd_canonical` (the `Rho_0_canonical_fd` bisection fill) whatever
the doping knob, exactly as the `hubbard_mu_filling` path already did. The electron count
stays pinned to `N + Q`, so the doping is unchanged; only the occupation of the
Fermi-level shell becomes fractional and continuous. The shrinking mixing is needed
because the retained `v(0) = 15.72 eV` diagonal makes the charge channel stiff.

### 13.4 Validation (5×5 zigzag triangle, 46 sites, `Q_doping = 1`, `mix = 0.02`)

| check | result |
|---|---|
| convergence | 729 iters, `tol = 1e-8` |
| electron count | `N_tot = 47.000` (46 + 1) |
| magnetism | `max|m_i| = 1.9e-15`, `S_z = -8.9e-15` — spin-blind, as designed |
| charge redistribution | `n_i` spans 1.0041 (interior) → 1.0497 (outer corners): the doped electron goes to the edges, the §11.4 / paper Fig. 2 behaviour, now driven by the Hartree alone |
| charge kernel | `min eig(V_ee) = +3.69 eV` — positive definite, no CDW mode |
| stationarity | `max\|n(0) - n_eq\| = 0`, `max\|[H_eq, rho(0)]\| = 1.3e-16` a.u. field-free |

At **neutrality** (`Q = 0`) on a bipartite flake, particle-hole symmetry pins `n_i = 1`
exactly, so `phi ≡ 0` and the loop converges in 1 iteration to the bare tight-binding
ground state. That is correct, not a failure: this Hartree does something only when the
density is pushed off `n = 1` (doping, or inequivalent sites).

### 13.5 Files changed

| file | change |
|---|---|
| `params/params.hpp/.cpp` | `hartree_scf` flag; forced off (with a note) when `hubbard = true` |
| `Hamiltonians/hubbard.hpp/.cpp` | `hartree_onsite` argument: keeps `j = i` in `phi`; also forces `fd_canonical` |
| `main.cpp` | SCF gate opened to `hubbard \|\| hartree_scf`; passes `U = 0`, `hartree_on = true`, `seed = 0`; separate `[Hartree]` logging and kernel check |
| `DensityMatrix/Density.cpp` | keeps the `V_ee_hartree` diagonal when `hartree_scf` |
| `configs/zigzag_triangle_hartree_scf.toml` | ready-to-run example |
