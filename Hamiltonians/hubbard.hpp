#pragma once

#include <Eigen/Dense>
#include <vector>
#include <utility>

#include "../DensityMatrix/Density.hpp"   // MatrixC
#include "hamiltonian.hpp"                 // Bond

// ---------------------------------------------------------------------------
// Self-consistent Hubbard mean-field (unrestricted Hartree-Fock) ground state.
//
//   H_Hubbard = H_TB + U sum_i n_{i,up} n_{i,dn}
//
// decoupled at mean-field level (per spin sigma, per site i):
//
//   V_{i,sigma} = U ( <n_{i,-sigma}> - 1/2 )
//
// The -1/2 keeps particle-hole symmetry so half-filling stays at mu = 0.
// For a bipartite flake with a sublattice imbalance (e.g. a zigzag graphene
// triangle) this produces a magnetic ground state whose net moment follows
// Lieb's theorem, S = |N_A - N_B| / 2.
//
// This is a *static* (frozen) mean field: the converged onsite potential is
// returned so the caller can add it to the base Hamiltonian, making the
// magnetic state the equilibrium the dynamics is perturbed around.
// ---------------------------------------------------------------------------

struct HubbardResult {
    Eigen::VectorXd V_up;        // onsite potential for spin up  (N_sites) [a.u.]
    Eigen::VectorXd V_dn;        // onsite potential for spin dn  (N_sites) [a.u.]
    Eigen::VectorXd n_up;        // converged up occupations       (N_sites)
    Eigen::VectorXd n_dn;        // converged dn occupations       (N_sites)
    Eigen::VectorXd m;           // magnetization per site 0.5*(n_up - n_dn)
    std::vector<int> sublattice; // bipartite label +1 / -1 (0 if uncolored)
    double S_total   = 0.0;      // net spin 0.5*sum(n_up - n_dn)
    double m_abs     = 0.0;      // sum_i |m_i|  (staggered magnitude)
    double gap       = 0.0;      // HOMO-LUMO gap of the converged spectrum [a.u.]
    int    iterations = 0;
    bool   converged  = false;

    // --- self-consistency trace (one entry per iteration) ---
    std::vector<double> hist_error;   // max_i |Delta n_i| (residual, before mixing)
    std::vector<double> hist_Ntot;    // total electrons sum(n_up)+sum(n_dn) (should stay constant)
    std::vector<double> hist_Sz;      // net spin 0.5*(sum n_up - sum n_dn)

    // --- LEVEL FLOW: the whole eigenvalue spectrum sampled along the loop --------
    // The mean field changes every iteration, so the single-particle spectrum does
    // too: the eigenvalues drift, and (with U) the up/dn levels split apart until
    // the magnetic gap opens. Every sample holds ALL 2*N_sites eigenvalues, sorted
    // ascending, so state index k is the same slot in every sample.
    // Sampling is decimated on the fly (keep every lvl_stride-th iteration, doubling
    // the stride whenever the buffer fills) so a 50k-iteration run costs a bounded
    // amount of memory and still covers the loop uniformly from start to end.
    int                              lvl_homo   = -1;  // index of the highest occupied state
    int                              lvl_stride = 1;   // iterations between samples (final value)
    std::vector<int>                 hist_lvl_iter; // iteration each sample was taken at
    std::vector<std::vector<double>> hist_lvl_E;    // [sample][k] eigenvalue [a.u.]
    std::vector<std::vector<double>> hist_lvl_f;    // [sample][k] occupation (Fermi factor)
    std::vector<std::vector<int>>    hist_lvl_s;    // [sample][k] +1 = up state, -1 = dn state

    // --- converged spin-resolved spectrum (2*N_sites eigenstates) ---
    // The up/down blocks never mix, so every eigenstate is purely up or down.
    Eigen::VectorXd  spec_energy;     // eigenvalue of each state [a.u.]
    Eigen::VectorXi  spec_spin;       // +1 = up state, -1 = down state
    Eigen::VectorXd  spec_occ;        // occupation (Fermi factor) of each state

    // --- converged equilibrium density matrix (site basis, 2N x 2N) ------------
    // THE ground state of the converged mean-field H, filled with the SAME rule the
    // self-consistency loop used. The caller MUST propagate this rho, not re-derive
    // one with an independent filling rule: any other fill gives occupations that
    // differ from n_up/n_dn, so the live Hubbard term U(n(t) - n_eq) is nonzero at
    // t = 0, H(0) != H_converged, and [H(0), rho(0)] != 0 — the "ground state" then
    // evolves under zero driving field, and the gamma-relaxation reference is not a
    // self-consistent state.
    MatrixC rho0_site;                // rho^sigma_{ll'} at equilibrium, site basis
    MatrixC rho0_eig;                 // same, diagonal in the converged eigenbasis
    MatrixC evecs;                    // eigenvectors of the converged mean-field H
    Eigen::VectorXcd evals;           // eigenvalues of the converged mean-field H [a.u.]
};

// Solve the self-consistent UHF problem.
//   Hc_spinful : 2N x 2N base tight-binding Hamiltonian, block diagonal in spin
//                ([0..N-1] = up block, [N..2N-1] = dn block).
//   U          : Hubbard U in atomic units.
//   mu, T      : filling controls (Fermi-Dirac); mu = 0 is charge neutrality.
//   use_charge_doping, Q : if true, fill to N_sites + Q electrons instead.
//   bonds      : nearest-neighbour list, used to 2-colour the lattice for the
//                symmetry-breaking seed.
//   V_ee       : full N x N Coulomb kernel v(R_ij) (site basis). Only used when
//                hartree_on = true, and then only its OFF-DIAGONAL part.
//   hartree_on : if true, also solve the self-consistent NONLOCAL charge Hartree in
//                the same loop:
//                    V_{iσ} = φ_i + U ( n_{i,-σ} - 1/2 ),
//                    φ_i    = sum_{j != i} V_ee(i,j) ( n_j - 1 )   (onsite EXCLUDED).
//                The onsite element v(0) is deliberately dropped from φ: the i == j
//                piece of the density-density Coulomb IS the Hubbard term, carried
//                by U (= v(0) by default). Keeping both would double-count it.
//                The single onsite U term supplies the charge stiffness (U/2 per
//                unit of δn — half of the classical v(0), which is exactly the
//                Hartree-Fock removal of the onsite self-interaction) AND the spin
//                exchange (∓U/2 per unit of local moment), so the onsite Coulomb is
//                no longer spin-blind. Doped charge still redistributes smoothly to
//                the edges (paper Fig. 2 / S1) via φ. Hartree OFF keeps the plain
//                UHF onsite term U(n_{-σ}-1/2), unchanged.
//   hartree_onsite : keep the j == i element in φ, i.e. use the FULL kernel
//                    φ_i = sum_j V_ee(i,j) ( n_j - 1 )   (diagonal INCLUDED).
//                Only meaningful with U = 0 (config [features] hartree_scf): the
//                onsite Coulomb is then carried by the Hartree diagonal instead of
//                by the Hubbard U, giving a Hubbard-free, spin-blind, purely
//                electrostatic self-consistent ground state. Note this is a bare
//                classical Hartree: with no Fock partner the onsite self-interaction
//                is NOT removed (that removal is what turns v(0) into U/2 in the
//                Hubbard branch), and v(0) is large, so the charge channel is stiff —
//                expect to need a small hubbard_mix. Never combine with U != 0: that
//                double-counts the onsite Coulomb.
HubbardResult solve_hubbard_mft(
    const MatrixC& Hc_spinful,
    double U,
    int N_sites,
    double mu,
    double T,
    bool use_charge_doping,
    double Q,
    const std::vector<Bond>& bonds,
    double m_seed  = 0.5,
    int    max_iter = 50000,
    double tol     = 1e-8,
    double mix     = 0.3,
    const Eigen::MatrixXd& V_ee = Eigen::MatrixXd(),
    bool   hartree_on     = false,
    bool   hartree_onsite = false,
    bool   use_mu         = false,    // true: grand-canonical Fermi-Dirac fill at (mu,T)
                                  // each iteration (chemical-potential doping); false:
                                  // canonical fill to N+Q (default, magnet-preserving).
    bool   fd_fill        = true,     // true: fill CANONICALLY to N+Q but with Fermi-Dirac
                                  // SMEARING instead of the hard T = 0 step. The total is
                                  // still pinned exactly to N+Q, so the doping is
                                  // unchanged; only the occupation of levels within ~kT of
                                  // the Fermi level becomes fractional. Required whenever
                                  // the HOMO-LUMO gap is comparable to the per-iteration
                                  // mean-field shift U*|dn|: the hard fill then swaps which
                                  // level is occupied every iteration and the loop
                                  // limit-cycles at a finite residual forever, and no value
                                  // of mix helps (see the discussion at the top of the
                                  // fd_canonical block in hubbard.cpp).
    double T_smear        = -1.0);    // smearing temperature [K] for that fill; < 0 => use T.

// ---------------------------------------------------------------------------
// Spinless (implicit spin-degeneracy) counterpart of the hartree_onsite branch
// above: a Hubbard-FREE (U = 0), spin-blind self-consistent Hartree ground state,
// run directly on the PLAIN N x N Hamiltonian instead of a spin-doubled 2N x 2N
// one. Physically identical to solve_hubbard_mft(..., hartree_onsite=true) on the
// spin-doubled Hc (both spin channels see an identical field there too, so the
// converged state is spin-blind either way) — this variant just never builds the
// 2N x 2N matrix, so it works when [hamiltonian] spin_on = false.
//
//   phi_i = sum_j V_ee(i,j) ( n_j - 1 ),   n_j = physical site occupation (ref. 1)
//
// Every eigenstate of Hc has capacity 2 (one spatial orbital, two implicit spins),
// same convention as Rho_0_charge(..., spin_on=false). Filling is CANONICAL with
// FD smearing (mandatory: U = 0 leaves the degenerate zero-mode shell unsplit, so
// a hard fill limit-cycles, same reasoning as hartree_onsite above).
//
// R.rho0_site / R.rho0_eig come back in the same CAPACITY-FRACTION convention
// used everywhere else in the spinless pipeline (Rho_0_charge, rho_l_space,
// rho_diag_from_rho) so the caller can propagate them verbatim. R.V_up == R.V_dn
// == phi (there is no spin channel to split), already in physical (energy) units,
// ready to fold into the site Hamiltonian diagonal unconditionally.
HubbardResult solve_hartree_scf_spinless(
    const MatrixC& Hc,
    int N_sites,
    bool use_charge_doping,
    double Q,
    double T,
    const std::vector<Bond>& bonds,
    int    max_iter = 50000,
    double tol      = 1e-8,
    double mix      = 0.3,
    const Eigen::MatrixXd& V_ee = Eigen::MatrixXd());
