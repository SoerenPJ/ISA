#include "hubbard.hpp"
#include "../DensityMatrix/Density.hpp"

#include <queue>
#include <cmath>
#include <algorithm>
#include <iostream>

// 2-colour the lattice from the bond graph (BFS). Graphene is bipartite, so
// this returns a consistent +1 / -1 sublattice label used only to seed the
// symmetry-breaking initial guess; frustrated bonds (if any) are ignored.
static std::vector<int> bipartite_colour(int N_sites, const std::vector<Bond>& bonds)
{
    std::vector<std::vector<int>> adj(N_sites);
    for (const auto& b : bonds) {
        if (b.first >= 0 && b.first < N_sites && b.second >= 0 && b.second < N_sites) {
            adj[b.first].push_back(b.second);
            adj[b.second].push_back(b.first);
        }
    }
    std::vector<int> colour(N_sites, 0);
    for (int s = 0; s < N_sites; ++s) {
        if (colour[s] != 0) continue;
        colour[s] = 1;
        std::queue<int> q;
        q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : adj[u]) {
                if (colour[v] == 0) {
                    colour[v] = -colour[u];
                    q.push(v);
                }
            }
        }
    }
    return colour;
}

// Canonical fill with Fermi-Dirac smearing: pick an INTERNAL chemical potential
// mu* (by bisection) so the total occupation equals N_target exactly, and fill
// with the FD factor f((E-mu*)/kT). This pins the particle number (like a hard
// canonical fill: no charge runaway, and two runs with the same N_target get the
// same population) while keeping the fill SMOOTH across near-degenerate Fermi-level
// states (like grand-canonical FD: no hard-T=0 degenerate-swap limit cycle). mu*
// is re-solved every call because the mean-field spectrum moves between iterations.
// capacity: electrons each level can hold (1 for spinful spin-orbitals, 2 for a
// spinless/implicit-spin-degenerate orbital). Only the bisection target scales
// with it; the RETURNED Rho(k,k) stays the bare FD fraction (0..1) of that
// capacity, exactly the convention Rho_0_charge's degenerate-block fill uses, so
// callers on either side of the spin_on divide can transform it the same way.
static MatrixC Rho_0_canonical_fd(const Eigen::VectorXcd& evals, double N_target, double T,
                                   double capacity = 1.0)
{
    const int    M  = static_cast<int>(evals.size());
    const double kb = 8.617e-5 / 27.2113834;       // Boltzmann const [a.u./K]
    const double kT = kb * T;
    auto Nof = [&](double mu) {
        double s = 0.0;
        for (int k = 0; k < M; ++k)
            s += capacity / (std::exp((evals(k).real() - mu) / kT) + 1.0);
        return s;
    };
    // bracket mu* between the lowest and highest eigenvalue (+/- a margin), then bisect
    double lo = evals(0).real(), hi = evals(0).real();
    for (int k = 1; k < M; ++k) { lo = std::min(lo, evals(k).real()); hi = std::max(hi, evals(k).real()); }
    lo -= 10.0 * kT + 1.0;  hi += 10.0 * kT + 1.0;
    for (int it = 0; it < 200; ++it) {             // ~60 bits; converges well before 200
        double mid = 0.5 * (lo + hi);
        (Nof(mid) < N_target) ? lo = mid : hi = mid;
    }
    const double mu_star = 0.5 * (lo + hi);
    MatrixC Rho = MatrixC::Zero(M, M);
    for (int k = 0; k < M; ++k)
        Rho(k, k) = std::complex<double>(1.0 / (std::exp((evals(k).real() - mu_star) / kT) + 1.0), 0.0);
    return Rho;
}

// Occupations n_up[i], n_dn[i] from a spinful Hamiltonian: diagonalize, fill,
// transform the eigenbasis density back to the site basis.
// Optional outputs evals_out / occ_out / spin_out expose the spectrum of THIS
// iteration (energies, Fermi factors, and which spin block each eigenstate lives
// in) so the caller can trace the level flow across the self-consistency loop.
static void occupations_from_H(
    const MatrixC& H, int N_sites, double N_target, double T,
    bool use_charge_doping, double Q, bool fd_canonical,
    Eigen::VectorXd& n_up, Eigen::VectorXd& n_dn,
    Eigen::VectorXd* evals_out = nullptr,
    Eigen::VectorXd* occ_out   = nullptr,
    Eigen::VectorXi* spin_out  = nullptr)
{
    auto [evals, evecs] = compute_eigenpairs(H);
    // Two filling rules for the self-consistency loop:
    //  - fd_canonical = true: CANONICAL fill to N_target with Fermi-Dirac smearing
    //    (chemical-potential / gate doping, mu already mapped to N_target on the bare
    //    band). The total is pinned to N_target so Hartree-ON and Hartree-OFF get the
    //    same population and the strong onsite Coulomb cannot drive a charge runaway;
    //    the FD smear keeps near-degenerate Fermi states fractionally shared (smooth).
    //  - fd_canonical = false: CANONICAL hard fill to a FIXED electron number N + Q
    //    every iteration (default magnet path), conserving the population through the
    //    loop and keeping the zero modes integer-filled. Neutral (Q = 0) unless charge
    //    doping is on.
    MatrixC rho0 = fd_canonical
        ? Rho_0_canonical_fd(evals, N_target, T)
        : Rho_0_charge(evals, N_sites, use_charge_doping ? Q : 0.0, /*spin_on=*/true);
    MatrixC rho_l = rho_l_space(evecs, rho0);          // site basis
    for (int i = 0; i < N_sites; ++i) {
        n_up(i) = std::real(rho_l(i, i));
        n_dn(i) = std::real(rho_l(N_sites + i, N_sites + i));
    }
    if (evals_out) {
        evals_out->resize(evals.size());
        for (int k = 0; k < evals.size(); ++k) (*evals_out)(k) = evals(k).real();
    }
    if (occ_out) {
        occ_out->resize(rho0.rows());
        for (int k = 0; k < rho0.rows(); ++k) (*occ_out)(k) = std::real(rho0(k, k));
    }
    if (spin_out) {
        // up/dn blocks never mix, so each eigenstate sits entirely in one of them:
        // label it by which block carries its weight.
        const int M = static_cast<int>(evals.size());
        spin_out->resize(M);
        for (int k = 0; k < M; ++k) {
            double w_up = 0.0, w_dn = 0.0;
            for (int i = 0; i < N_sites; ++i) {
                w_up += std::norm(evecs(i, k));
                w_dn += std::norm(evecs(N_sites + i, k));
            }
            (*spin_out)(k) = (w_up >= w_dn) ? +1 : -1;
        }
    }
}

HubbardResult solve_hubbard_mft(
    const MatrixC& Hc_spinful,
    double U,
    int N_sites,
    double mu,
    double T,
    bool use_charge_doping,
    double Q,
    const std::vector<Bond>& bonds,
    double m_seed,
    int    max_iter,
    double tol,
    double mix,
    const Eigen::MatrixXd& V_ee,
    bool   hartree_on,
    bool   hartree_onsite,
    bool   use_mu,
    bool   fd_fill,
    double T_smear)
{
    HubbardResult R;
    R.sublattice = bipartite_colour(N_sites, bonds);

    if (hartree_on && (V_ee.rows() != N_sites || V_ee.cols() != N_sites)) {
        std::cerr << "[Hubbard] hartree_on requested but V_ee is "
                  << V_ee.rows() << "x" << V_ee.cols() << " (need "
                  << N_sites << "x" << N_sites << "); disabling Hartree.\n";
        hartree_on = false;
    }

    // --- chemical-potential doping mapped to a FIXED electron number ---------
    // When the fill is driven by a chemical potential mu ("gate doping"), do NOT
    // fill grand-canonically on the mean-field spectrum every iteration: that has
    // no particle-number pinning, so with the strong onsite Coulomb v(0) the
    // charge susceptibility is enormous and the total electron count runs away
    // (it oscillates empty<->full and never converges once mu != 0), AND the
    // converged N then depends on the model, so the Hartree-ON and Hartree-OFF
    // runs disagree at the same mu.
    //
    // Instead convert mu -> a target electron number ONCE, using the BARE (non-
    // interacting) tight-binding spectrum: N_target(mu) = sum_k f((E_k^bare-mu)/kT).
    // This is the standard "gate charge": the doping level is set by mu on the
    // fixed band structure, independent of U or the Hartree. Then fill CANONICALLY
    // to that same N_target every iteration for both cases. Result: (1) the total
    // is pinned so the loop converges, (2) Hartree-ON and Hartree-OFF fill to the
    // IDENTICAL population by construction, and the Hartree only redistributes the
    // charge spatially (to the edges), reproducing the paper's Fig. 2 comparison.
    double N_target = static_cast<double>(N_sites)
                    + (use_charge_doping ? Q : 0.0);   // only used when fd_canonical
    if (use_mu) {
        auto [bare_evals, bare_evecs] = compute_eigenpairs(Hc_spinful);
        const double kb = 8.617e-5 / 27.2113834;   // Boltzmann const [a.u./K]
        N_target = 0.0;
        for (int k = 0; k < bare_evals.size(); ++k) {
            double x = (bare_evals(k).real() - mu) / (kb * T);
            N_target += 1.0 / (std::exp(x) + 1.0);
        }
    }
    // The mu (gate) path fills canonically to N_target with FD smearing; the default
    // path keeps its plain canonical hard fill to N_sites + Q. fd_canonical selects it.
    //
    // The U = 0 Hartree mode (hartree_onsite) also needs the smeared fill, whatever
    // the doping knob: with no U there is nothing to split the flake's degenerate
    // zero-mode shell, so a hard fill has to pick arbitrarily among degenerate states.
    // The pick then flips from iteration to iteration and the loop settles into a
    // limit cycle instead of converging — measured on the 5x5 zigzag triangle at
    // Q = 1: period-3, residual pinned at ~0.15, and shrinking the mixing all the way
    // to 0.002 does NOT help, because the cycle is a discontinuous refill, not an
    // overshoot. FD smearing occupies the degenerate shell fractionally and
    // continuously, which restores ordinary linear-mixing convergence (same case:
    // 729 iters at mix = 0.02, 1464 at 0.01). The total electron count stays pinned
    // to N_target, so the doping is unchanged. Note the onsite v(0) makes the charge
    // channel stiff, so the mixing must be smaller than the Hubbard default: mix =
    // 0.05 still sloshes, mix <= 0.02 converges.
    //
    // fd_fill extends the SAME smeared fill to the ordinary canonical path (neutral
    // or Q-doped, U != 0), because the identical failure appears there whenever the
    // doped level lands in a near-degenerate shell, which is the generic case for a
    // sublattice-BALANCED flake. Armchair triangles have no protected zero-mode
    // multiplet, so the neutral state is a closed shell with a large U-split gap and
    // the first doped electron drops into a dense near-continuum just above it.
    // Measured on the 5x5 armchair triangle at Q = 1: HOMO/LUMO are levels 90/91 at
    // 8.60797 / 8.61550 eV, BOTH in the spin-down block, i.e. 0.0075 eV = 2.8e-4 a.u.
    // apart. Rho_0_charge's degenerate-block tolerance is 1e-8 a.u., four orders of
    // magnitude tighter, so that pair is filled as a hard 1/0 step. The mean field
    // moves the diagonal by U*|dn| each iteration, and at the observed residual
    // (|dn| ~ 0.09) with U = 15.72 eV that is ~1.4 eV — some 200x the gap. The two
    // levels therefore swap order essentially every iteration, a full electron jumps
    // between them, and the residual can never drop below that jump: all 50000
    // iterations sit in a quasi-periodic band [0.081, 0.108] with no decay at all.
    // Mixing damps OVERSHOOT (a smooth Jacobian with |lambda| > 1); this is a step
    // function, so no mix helps — same conclusion as the hartree_onsite case above.
    //
    // The smearing costs nothing where it is not needed: Rho_0_canonical_fd pins the
    // total to N_target exactly (so Q is untouched to machine precision), and a gap
    // large against kT gives an FD correction of e^{-gap/kT}. At T = 300 K
    // (kT = 0.0259 eV) the neutral flakes' 13.3 eV gap makes that ~e^{-514}, i.e.
    // bit-for-bit the hard fill. Only the near-degenerate shells change, and there
    // the fractional occupation is the honest finite-T mean-field answer: such a
    // state is genuinely not a pure spin eigenstate, so S_total stops being a clean
    // half-integer. Sweep T_smear to tell a converged result from a smeared one.
    const bool   fd_canonical = use_mu || hartree_onsite || fd_fill;
    const double T_fill       = (T_smear > 0.0) ? T_smear : T;

    // Spin-blind charge-Hartree onsite potential from a given occupation pair:
    //   phi_i = sum_{j != i} V_ee(i,j) ( n_up_j + n_dn_j - 1 )     (NONLOCAL ONLY)
    // with the neutral ionic reference n_j^0 = 1 per site (paper SI Eq. S1).
    //
    // The onsite term j == i is DELIBERATELY EXCLUDED: the i == j element of the
    // density-density Coulomb *is* the Hubbard term, and it is carried by U below
    // (U = v(0) by default). Including it here as well would double-count the
    // onsite Coulomb. Returns a zero vector when Hartree is off (callers stay
    // branch-free).
    //
    // hartree_onsite flips exactly that choice: the sum then runs over ALL j, so
    // the onsite Coulomb is carried by the Hartree diagonal. That is the U = 0
    // ([features] hartree_scf) mode — a Hubbard-free, spin-blind, purely
    // electrostatic self-consistency over the full V_ee. Exactly one of the two
    // channels may own the onsite element, never both.
    auto hartree_phi = [&](const Eigen::VectorXd& nu,
                           const Eigen::VectorXd& nd) -> Eigen::VectorXd {
        Eigen::VectorXd phi = Eigen::VectorXd::Zero(N_sites);
        if (!hartree_on) return phi;
        Eigen::VectorXd dq(N_sites);
        for (int j = 0; j < N_sites; ++j) dq(j) = nu(j) + nd(j) - 1.0;
        for (int i = 0; i < N_sites; ++i) {
            phi(i) = V_ee.row(i).dot(dq);
            if (!hartree_onsite) phi(i) -= V_ee(i, i) * dq(i);   // drop onsite j == i
        }
        return phi;
    };

    // Build the mean-field Hamiltonian for a given occupation.
    //
    // ONE onsite term for both branches — the UHF Hubbard field
    //     V_{iσ} = U ( n_{i,-σ} - 1/2 ),
    // i.e. an electron of spin σ feels only the OPPOSITE spin on its own site
    // (the Fock term cancels the same-spin onsite Hartree exactly, so there is no
    // onsite self-interaction). Split into charge and spin channels this reads
    //     V_{iσ} = (U/2)( n_i - 1 )  ∓  (U/2) M_i ,   M_i = n_{i↑} - n_{i↓},
    // so the SAME U supplies both the onsite charge stiffness (U/2, half of the
    // classical v(0) — that halving is the self-interaction removal) and the spin
    // exchange that makes the system spin-resolved.
    //
    // Hartree ON adds, on top, the NONLOCAL (j != i) spin-blind charge Hartree
    // φ_i, which lets doped charge redistribute self-consistently to the edges.
    // Hartree OFF is the pure onsite UHF, unchanged.
    auto build_meanfield_H = [&](const Eigen::VectorXd& nu,
                                 const Eigen::VectorXd& nd) -> MatrixC {
        MatrixC H = Hc_spinful;
        Eigen::VectorXd phi = hartree_phi(nu, nd);   // 0 if Hartree off; nonlocal only if on
        for (int i = 0; i < N_sites; ++i) {
            H(i, i)                     += phi(i) + U * (nd(i) - 0.5);   // up feels dn
            H(N_sites + i, N_sites + i) += phi(i) + U * (nu(i) - 0.5);   // dn feels up
        }
        return H;
    };

    // --- symmetry-breaking seed: staggered moment on the two sublattices ---
    // The initial guess is fully deterministic and set by the single number
    // m_seed (config [features] hubbard_seed): the moment on site i points along
    // its sublattice s_i = +/-1 with amplitude m_seed. Sweeping m_seed then
    // checks the loop converges to the same state for every chosen value (not a
    // lucky one), while every run stays exactly reproducible.
    Eigen::VectorXd n_up(N_sites), n_dn(N_sites);
    for (int i = 0; i < N_sites; ++i) {
        double s = static_cast<double>(R.sublattice[i]);       // +1 / -1
        n_up(i) = 0.5 + 0.5 * m_seed * s;
        n_dn(i) = 0.5 - 0.5 * m_seed * s;
    }

    // --- level-flow sampling ------------------------------------------------
    // Store the FULL spectrum (all 2*N_sites eigenvalues) at a subset of iterations.
    // The number of iterations is not known in advance (it can be 10 or 50000), so
    // sample every lvl_stride-th one and, whenever the buffer exceeds MAX_SAMPLES,
    // throw away every second sample and double the stride. The kept samples stay
    // uniformly spaced and always span the whole loop, at bounded cost: the total
    // number of recorded (sample x state) entries never exceeds ~150k.
    const int M_states    = 2 * N_sites;
    const int MAX_SAMPLES = std::clamp(150000 / std::max(1, M_states), 100, 1500);
    R.lvl_homo = std::clamp(static_cast<int>(std::llround(N_target)) - 1, 0, M_states - 2);
    R.lvl_stride = 1;

    // keep samples 0, 2, 4, ... and drop the odd ones (halves the time resolution)
    auto decimate = [](auto& v) {
        size_t w = 0;
        for (size_t r = 0; r < v.size(); r += 2) v[w++] = std::move(v[r]);
        v.resize(w);
    };
    auto record_levels = [&](int iter, const Eigen::VectorXd& E,
                             const Eigen::VectorXd& f, const Eigen::VectorXi& s) {
        R.hist_lvl_iter.push_back(iter);
        R.hist_lvl_E.emplace_back(E.data(), E.data() + M_states);
        R.hist_lvl_f.emplace_back(f.data(), f.data() + M_states);
        R.hist_lvl_s.emplace_back(s.data(), s.data() + M_states);
    };

    // --- self-consistency loop ---
    Eigen::VectorXd n_up_new(N_sites), n_dn_new(N_sites);
    Eigen::VectorXd it_evals, it_occ;
    Eigen::VectorXi it_spin;
    double diff = 0.0;
    int it = 0;
    for (; it < max_iter; ++it) {
        // build mean-field Hamiltonian: onsite Hubbard U(n_{-sigma} - 1/2) plus,
        // if enabled, the nonlocal charge Hartree phi_i on the diagonal.
        MatrixC H = build_meanfield_H(n_up, n_dn);

        // T_fill, not T: the loop fill and the final-pass fill below MUST use the
        // identical rule AND temperature, or H_final and rho0 stop commuting.
        occupations_from_H(H, N_sites, N_target, T_fill, use_charge_doping, Q, fd_canonical,
                           n_up_new, n_dn_new, &it_evals, &it_occ, &it_spin);

        // level flow: THIS iteration's spectrum (subsampled, see MAX_SAMPLES above)
        if (it % R.lvl_stride == 0) {
            record_levels(it, it_evals, it_occ, it_spin);
            if (static_cast<int>(R.hist_lvl_iter.size()) > MAX_SAMPLES) {
                decimate(R.hist_lvl_iter);
                decimate(R.hist_lvl_E);
                decimate(R.hist_lvl_f);
                decimate(R.hist_lvl_s);
                R.lvl_stride *= 2;
            }
        }

        diff = 0.0;
        for (int i = 0; i < N_sites; ++i) {
            diff = std::max(diff, std::abs(n_up_new(i) - n_up(i)));
            diff = std::max(diff, std::abs(n_dn_new(i) - n_dn(i)));
        }

        // record the trace for this iteration (before mixing): the residual, the
        // total electron count (population conservation) and the net spin.
        R.hist_error.push_back(diff);
        R.hist_Ntot.push_back(n_up_new.sum() + n_dn_new.sum());
        R.hist_Sz.push_back(0.5 * (n_up_new.sum() - n_dn_new.sum()));

        // linear mixing
        n_up = (1.0 - mix) * n_up + mix * n_up_new;
        n_dn = (1.0 - mix) * n_dn + mix * n_dn_new;

        if (diff < tol) { ++it; R.converged = true; break; }
    }
    R.iterations = it;

    // --- final pass: build H ONCE from the converged input occupations, then fill it.
    //
    // Which occupations feed which output matters, and the two roles are NOT
    // interchangeable at the 1e-8 level:
    //
    //   * the reported FIELDS (phi_final, V_up, V_dn) must come from n_in — the
    //     occupations H_final was actually built from — so that
    //     Hc + diag(V_up, V_dn) reproduces H_final EXACTLY;
    //   * the reported OCCUPATIONS (n_up, n_dn) must be the ones that come back out
    //     of filling H_final, so they are exactly the diagonal of rho0_site.
    //
    // The dynamics then evaluates H(0) = Hc + diag(V) + U(diag(rho0) - n_eq), and
    // both corrections vanish identically: the U-term because diag(rho0) == n_eq by
    // construction, so H(0) == H_final and [H(0), rho0] == 0 to machine precision,
    // independent of the SCF tolerance. Using n_out for BOTH roles (the obvious but
    // wrong choice) leaves V off by U*(n_out - n_in) ~ U*tol, which makes the
    // magnetic ground state drift under zero driving field.
    const Eigen::VectorXd n_up_in = n_up;   // what H_final is built from -> fields
    const Eigen::VectorXd n_dn_in = n_dn;
    MatrixC H = build_meanfield_H(n_up_in, n_dn_in);

    auto [evals_c, evecs] = compute_eigenpairs(H);
    MatrixC rho0_eig = fd_canonical
        ? Rho_0_canonical_fd(evals_c, N_target, T_fill)    // smeared canonical fill to N_target
        : Rho_0_charge(evals_c, N_sites, use_charge_doping ? Q : 0.0, /*spin_on=*/true);
    MatrixC rho0_site = rho_l_space(evecs, rho0_eig);       // site basis

    // Occupations that come OUT of H_final: exactly diag(rho0_site), so the dynamics'
    // equilibrium reference and the propagated rho(0) agree to machine precision.
    for (int i = 0; i < N_sites; ++i) {
        n_up(i) = std::real(rho0_site(i, i));
        n_dn(i) = std::real(rho0_site(N_sites + i, N_sites + i));
    }

    const int M_final = static_cast<int>(evals_c.size());
    Eigen::VectorXd evals(M_final);
    for (int k = 0; k < M_final; ++k) evals(k) = evals_c(k).real();
    it_occ.resize(M_final);
    it_spin.resize(M_final);
    for (int k = 0; k < M_final; ++k) it_occ(k) = std::real(rho0_eig(k, k));

    // append the final (post-mixing) pass to the level flow, so the last sample of
    // the trace is exactly the converged spectrum written to hubbard_spectrum.txt
    // (spin labels filled in with the spectrum block below).
    // --- converged spin-resolved spectrum (for the energy-eigenstate picture) ---
    // Since the up/down blocks never mix, each eigenstate lives entirely in one
    // block; label it by which block holds its weight, and read its occupation off
    // the equilibrium density matrix (diagonal in the eigenbasis).
    R.spec_energy = Eigen::VectorXd(M_final);
    R.spec_spin   = Eigen::VectorXi(M_final);
    R.spec_occ    = Eigen::VectorXd(M_final);
    for (int k = 0; k < M_final; ++k) {
        R.spec_energy(k) = evals(k);
        R.spec_occ(k)    = it_occ(k);
        double w_up = 0.0, w_dn = 0.0;
        for (int i = 0; i < N_sites; ++i) {
            w_up += std::norm(evecs(i, k));
            w_dn += std::norm(evecs(N_sites + i, k));
        }
        R.spec_spin(k) = (w_up >= w_dn) ? +1 : -1;
        it_spin(k)     = R.spec_spin(k);
    }
    record_levels(it, evals, it_occ, it_spin);

    // Converged onsite fields, INCLUDING the nonlocal Hartree, so folding V_up/V_dn
    // into the eigenproblem and re-adding them each dynamics step carries the full
    // static mean field (spin + charge). The dynamical Hartree V_ee*(rho-rho0) then
    // stays referenced to this rho0 AND runs with a zeroed onsite element (the
    // onsite channel is the Hubbard U, added separately), so the two never
    // double-count. Built from n_*_in — see the note above.
    Eigen::VectorXd phi_final = hartree_phi(n_up_in, n_dn_in);  // 0 if Hartree off; nonlocal only if on
    R.n_up = n_up;
    R.n_dn = n_dn;
    R.V_up = Eigen::VectorXd(N_sites);
    R.V_dn = Eigen::VectorXd(N_sites);
    R.m    = Eigen::VectorXd(N_sites);
    for (int i = 0; i < N_sites; ++i) {
        // nonlocal charge Hartree (0 if off) + onsite UHF Hubbard field
        R.V_up(i) = phi_final(i) + U * (n_dn_in(i) - 0.5);
        R.V_dn(i) = phi_final(i) + U * (n_up_in(i) - 0.5);
        R.m(i) = 0.5 * (n_up(i) - n_dn(i));
    }
    R.S_total = 0.5 * (n_up.sum() - n_dn.sum());
    R.m_abs   = R.m.cwiseAbs().sum();

    // The converged equilibrium state itself, so the caller propagates THIS rho
    // instead of re-deriving one with an independent filling rule.
    R.rho0_site = rho0_site;
    R.rho0_eig  = rho0_eig;
    R.evecs     = evecs;
    R.evals     = evals_c;

    // HOMO-LUMO gap around the filled electron count
    int N_e = static_cast<int>(std::llround(n_up.sum() + n_dn.sum()));
    N_e = std::clamp(N_e, 1, static_cast<int>(evals.size()) - 1);
    R.gap = evals(N_e) - evals(N_e - 1);

    return R;
}

HubbardResult solve_hartree_scf_spinless(
    const MatrixC& Hc,
    int N_sites,
    bool use_charge_doping,
    double Q,
    double T,
    const std::vector<Bond>& bonds,
    int    max_iter,
    double tol,
    double mix,
    const Eigen::MatrixXd& V_ee)
{
    HubbardResult R;
    R.sublattice = bipartite_colour(N_sites, bonds);

    if (V_ee.rows() != N_sites || V_ee.cols() != N_sites) {
        std::cerr << "[Hartree] spinless hartree_scf requires V_ee " << N_sites << "x"
                  << N_sites << "; got " << V_ee.rows() << "x" << V_ee.cols()
                  << ". SCF not run.\n";
        return R;
    }

    constexpr double capacity = 2.0;   // one spatial orbital, two implicit spins
    const double N_target = static_cast<double>(N_sites) + (use_charge_doping ? Q : 0.0);

    // phi_i = sum_j V_ee(i,j) (n_phys_j - 1), full kernel, onsite included
    // (this IS the hartree_scf mode: U = 0, the onsite Coulomb is carried by phi).
    auto hartree_phi = [&](const Eigen::VectorXd& n_phys) -> Eigen::VectorXd {
        return V_ee * (n_phys.array() - 1.0).matrix();
    };
    auto build_H = [&](const Eigen::VectorXd& n_phys) -> MatrixC {
        MatrixC H = Hc;
        Eigen::VectorXd phi = hartree_phi(n_phys);
        for (int i = 0; i < N_sites; ++i) H(i, i) += phi(i);
        return H;
    };

    // Spin-blind: no seed can survive an identical field on both implicit spins,
    // so start neutral (n_phys_i = 1 electron/site) same as the spinful hartree_only path.
    Eigen::VectorXd n_phys = Eigen::VectorXd::Constant(N_sites, 1.0);

    const int M_states    = N_sites;
    const int MAX_SAMPLES = std::clamp(150000 / std::max(1, M_states), 100, 1500);
    R.lvl_homo   = std::clamp(static_cast<int>(std::llround(N_target / capacity)) - 1, 0, M_states - 2);
    R.lvl_stride = 1;

    auto decimate = [](auto& v) {
        size_t w = 0;
        for (size_t r = 0; r < v.size(); r += 2) v[w++] = std::move(v[r]);
        v.resize(w);
    };
    auto record_levels = [&](int iter, const Eigen::VectorXd& E, const Eigen::VectorXd& f) {
        R.hist_lvl_iter.push_back(iter);
        R.hist_lvl_E.emplace_back(E.data(), E.data() + M_states);
        R.hist_lvl_f.emplace_back(f.data(), f.data() + M_states);
        R.hist_lvl_s.emplace_back(M_states, 0);   // 0: spin-blind, no up/dn split
    };

    Eigen::VectorXd n_phys_new(N_sites);
    double diff = 0.0;
    int it = 0;
    for (; it < max_iter; ++it) {
        MatrixC H = build_H(n_phys);
        auto [evals_c, evecs] = compute_eigenpairs(H);
        MatrixC rho0_eig = Rho_0_canonical_fd(evals_c, N_target, T, capacity);
        MatrixC rho_l    = rho_l_space(evecs, rho0_eig);
        for (int i = 0; i < N_sites; ++i)
            n_phys_new(i) = capacity * std::real(rho_l(i, i));

        if (it % R.lvl_stride == 0) {
            Eigen::VectorXd E(M_states), f(M_states);
            for (int k = 0; k < M_states; ++k) {
                E(k) = evals_c(k).real();
                f(k) = std::real(rho0_eig(k, k));
            }
            record_levels(it, E, f);
            if (static_cast<int>(R.hist_lvl_iter.size()) > MAX_SAMPLES) {
                decimate(R.hist_lvl_iter);
                decimate(R.hist_lvl_E);
                decimate(R.hist_lvl_f);
                decimate(R.hist_lvl_s);
                R.lvl_stride *= 2;
            }
        }

        diff = (n_phys_new - n_phys).cwiseAbs().maxCoeff();
        R.hist_error.push_back(diff);
        R.hist_Ntot.push_back(n_phys_new.sum());
        R.hist_Sz.push_back(0.0);

        n_phys = (1.0 - mix) * n_phys + mix * n_phys_new;
        if (diff < tol) { ++it; R.converged = true; break; }
    }
    R.iterations = it;

    // Final pass: same n_in/n_out split as solve_hubbard_mft (see its comment) — the
    // reported field phi must come from n_in (what H_final was built from), the
    // reported occupation n_out from what comes back out of filling H_final, so
    // H(0) == H_final and [H(0), rho0] == 0 to machine precision at t = 0.
    const Eigen::VectorXd n_in = n_phys;
    MatrixC H_final = build_H(n_in);
    auto [evals_c, evecs] = compute_eigenpairs(H_final);
    MatrixC rho0_eig  = Rho_0_canonical_fd(evals_c, N_target, T, capacity);
    MatrixC rho0_site = rho_l_space(evecs, rho0_eig);   // capacity-fraction convention

    Eigen::VectorXd n_out(N_sites);
    for (int i = 0; i < N_sites; ++i) n_out(i) = capacity * std::real(rho0_site(i, i));

    Eigen::VectorXd E(M_states), occ(M_states);
    for (int k = 0; k < M_states; ++k) {
        E(k)   = evals_c(k).real();
        occ(k) = std::real(rho0_eig(k, k));
    }
    record_levels(it, E, occ);

    Eigen::VectorXd phi_final = hartree_phi(n_in);

    R.n_up = 0.5 * n_out;
    R.n_dn = 0.5 * n_out;
    R.V_up = phi_final;
    R.V_dn = phi_final;
    R.m    = Eigen::VectorXd::Zero(N_sites);
    R.S_total = 0.0;
    R.m_abs   = 0.0;

    R.rho0_site = rho0_site;
    R.rho0_eig  = rho0_eig;
    R.evecs     = evecs;
    R.evals     = evals_c;

    R.spec_energy = E;
    R.spec_spin   = Eigen::VectorXi::Zero(M_states);
    R.spec_occ    = occ;

    const int N_levels_filled =
        std::clamp(static_cast<int>(std::llround(N_target / capacity)), 1, M_states - 1);
    R.gap = E(N_levels_filled) - E(N_levels_filled - 1);

    return R;
}
