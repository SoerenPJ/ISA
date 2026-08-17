#pragma once

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <string>

struct Params
{
    // ======================
    // ---- CONSTANTS ----
    // ======================
    double au_eV, au_nm, au_s, au_fs, au_c;
    double au_kg, au_kB, au_m, au_J, au_w, au_I, au_me;
    double alpha;
    int au_hbar, e;
    

    // ======================
    // ---- SYSTEM (TOML) ----
    // ======================
    int N;
    bool B_ext = false;
    bool two_dim = false;
    bool spin_on = false;
    std::string lattice = "chain";
    std::string formation = "zigzag"; // "zigzag" | "armchair"
    std::string formation_shape;  //rectangle | triangle
    int size_x = 0;                             
    int size_y = 0;
    // In-plane rotation angle for 2D lattices (degrees, about z-axis through origin).
    double rotation_angle_deg = 0.0;
    // ======================
    // ---- HAMILTONIAN (TOML) ----
    // ======================
    double t1, t2;
    double mu;
    double gamma;

    // ======================
    // ---- TIME (TOML) ----
    // ======================
    double dt;
    double max_internal_dt;
    double t_end;

    // ======================
    // ---- SOLVER ----
    // ======================
    bool use_strict_solver = false;
    double t0   = 0.0;
    double a_tol;
    double r_tol;

    // ======================
    // ---- FIELD (TOML) ----
    // ======================
    double Intensity;
    std::string field_mode; // "time_impulse" | "sinus" | "ddf"
    double field_phase = 0.0;                // phase offset [rad]
    double au_omega;
    double au_omega_ddf = 0.1 / 27.2113834;  // ddf omega (a.u.), default 0.1 eV
    double t_shift;
    double sigma_gaus;
    double sigma_ddf;
    double omega_cut_off;
    /** Time step (fs) for uniform mesh used in Fourier / dipole-acceleration analysis.
     *  Match bachelor reference (e.g. 0.005 or 0.0025 fs) so adaptive trajectory
     *  is resampled onto a fixed grid before ∫ p(t) e^(iωt) dt. */
    double fourier_dt_fs;
    /** If false, main simulation skips sigma_ext / alpha_ext (set true in TOML when needed). */
    bool run_sigma_ext = false;
    /** If false, main simulation skips dipole acceleration spectrum (set true in TOML when needed). */
    bool run_dipole_acc = false;
    /** If true, stream the full induced density matrix rho(t)-rho0 (all elements) on the
     *  output stride to rho_full_induced_time_evolution.txt. Off by default: file is O(N^2) per step.
     *  Only needed for off-diagonal coherences (e.g. ploting/Spin_matrix.py). */
    bool save_rho_full = false;
    /** If true, stream only the spin-resolved induced diagonal rho_ii(t)-rho0_ii on the output
     *  stride to spin_diag_time_evolution.txt (1 + N_mat reals/row: [up_0..up_{N-1}, dn_0..dn_{N-1}]).
     *  This is all the spin-density plots need, and is ~O(N) instead of O(N^2) per step. */
    bool save_spin_diag = false;

    // ======================
    // ---- THERMO (TOML) ----
    // ======================
    int T;
    bool   use_charge_doping = false;
    double Q_doping          = 0.0;   // extra electrons (can be non‑integer)

    // ======================
    // ---- FEATURES (TOML) ----
    // ======================
    bool coulomb_on;
    // Which Coulomb kernel v(R) builds V_ee: "vvR" (the ab-initio rational fit,
    // default) or "ohno" (analytic Ohno interpolation). Selects the kernel for the
    // WHOLE pipeline, the Hubbard onsite U = v(0) included.
    std::string coulomb_kernel = "vvR";
    double ohno_U   = 0.5777610; // Ohno U_0 in V(r) = e^2/sqrt((e^2/U_0)^2 + r^2), i.e. the
                                 // onsite value V(0). Stored in HARTREE; the TOML key is
                                 // [features] ohno_U_eV and is given in eV. Default matches
                                 // vvR(0) = 0.5777610 Ha = 15.7216761 eV exactly, so switching
                                 // kernels does not move the onsite scale. This is the
                                 // kernel's ONLY parameter.
    bool hubbard = false;        // self-consistent Hubbard mean-field (UHF) magnetic ground state
    double hubbard_U_eV = -1.0;  // optional Hubbard U override (eV); unset/< 0 => use v(0)
                                 // of the ACTIVE kernel (coulomb_kernel),
                                 // i.e. the onsite element of the Coulomb kernel V_ee (the
                                 // default: the onsite v(0) IS the Hubbard energy).
                                 // vvR KERNEL ONLY: under coulomb_kernel = "ohno" this is
                                 // IGNORED (params.cpp resets it to -1 with a NOTE), because
                                 // ohno_U_eV already IS the onsite V(0). Honouring both would
                                 // leave V_ee a hybrid — Ohno tail for one U_0, diagonal
                                 // pinned to another — which is not any single kernel.
    double hubbard_seed = 0.5;   // initial staggered moment for symmetry breaking
    double hubbard_mix  = 0.3;   // linear mixing factor for the self-consistency loop
    int    hubbard_max_iter = 50000; // iteration cap for the self-consistency loop. Small
                                    // hubbard_mix needs a bigger cap: the paper's beta =
                                    // 0.01 takes a few hundred to a few thousand iters.
    double hubbard_tol = 1e-8;   // convergence criterion: max over sites AND both spins
                                 // of |Delta n| between consecutive iterations.
    bool   hubbard_hartree = false; // also solve the self-consistent NONLOCAL charge
                                    // Hartree in the ground state (combined UHF), so
                                    // added electrons (Q_doping) redistribute self-
                                    // consistently. Onsite Coulomb stays the Hubbard U.
    bool   hubbard_mu_filling = false; // fill the SCF loop grand-canonically at (mu,T)
                                    // instead of canonically to N+Q: doping is then set
                                    // by the chemical potential mu (see [thermo] mu, T).
    bool   hubbard_fd_fill = true;  // fill the SCF loop with Fermi-Dirac SMEARING (still
                                    // canonical, total pinned to N+Q) instead of the hard
                                    // T = 0 step of Rho_0_charge. Needed whenever the
                                    // doped level lands in a near-degenerate shell: the
                                    // hard fill then flips which of two levels a few meV
                                    // apart is occupied every iteration and the loop
                                    // limit-cycles forever (measured: armchair 5x5
                                    // triangle at Q = 1, gap 0.0075 eV, residual pinned
                                    // at ~0.093 for all 50000 iterations). No mixing can
                                    // fix that — it is a discontinuous refill, not an
                                    // overshoot. Set false only to reproduce old runs.
    double hubbard_smear_T = -1.0;  // smearing temperature [K] for that fill. < 0 (default)
                                    // => use [thermo] T. Separate knob so the SCF smearing
                                    // can be swept independently of the dynamics
                                    // temperature (same robustness check as hubbard_seed /
                                    // hubbard_mix): a result that moves with it is a
                                    // genuinely smeared open shell, not a converged state.
    bool   hartree_scf = false;  // HUBBARD-FREE self-consistent ground state: run the SCF
                                 // loop with U = 0 and the FULL Coulomb kernel V_ee in the
                                 // Hartree, onsite diagonal INCLUDED:
                                 //     phi_i = sum_j V_ee(i,j) ( n_j - 1 )   (all j)
                                 // There is no Hubbard U and no exchange, so the ground
                                 // state is spin-blind (no magnetism); the onsite Coulomb
                                 // is carried by the Hartree diagonal instead of by U, in
                                 // the static SCF and in the dynamics alike. Ignored when
                                 // hubbard = true (the Hubbard owns the onsite channel).
    bool self_consistent_phase;  // current -> A_ind -> phi_ind -> update hopping (induced phase)
    bool peierls_induced = false;  // apply induced Peierls phases (L2); false = Zeeman-only (L1)
    bool zeeman_external;   // include external B in Zeeman diagonal μ_B σ·B
    bool zeeman_induced;    // include induced B (from A_ind) in Zeeman diagonal

    // ======================
    // ---- DERIVED ----
    // ======================
    double a;          // lattice spacing (a.u.)
    double E0;         // field amplitude
    double au_mu_0;    // vacuum permeability in a.u. (4*pi/c^2)
    double area_2d;    // effective area for 2D A_ind (graphene unit-cell related)
    Eigen::MatrixXd V_ee;

    // Converged Hubbard mean-field onsite potential (set in main when hubbard on).
    // Kept separate from the tight-binding Hc so it can be re-added every step.
    bool hub_active = false;
    double hub_U = 0.0;          // converged Hubbard U (a.u.), single unified onsite U
    Eigen::VectorXd hub_V_up;
    Eigen::VectorXd hub_V_dn;
    Eigen::VectorXd hub_n_up_eq; // equilibrium spin-up occupation per site
    Eigen::VectorXd hub_n_dn_eq; // equilibrium spin-dn occupation per site

    // ======================
    // ---- GEOMETRY ----
    // ======================
    std::vector<double> xl_1D;
    std::vector<std::array<double,2>> xl_2D;

    // ======================
    // ---- API ----
    // ======================
    Params();
    void load_from_toml(const std::string& filename);
    void finalize();

private:
    void build_lattice();
};
