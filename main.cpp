    #include "params/params.hpp"
    #include "DensityMatrix/Density.hpp"
    #include "Hamiltonians/hamiltonian.hpp"
    #include "Hamiltonians/hubbard.hpp"
    #include "Hamiltonians/potential.hpp"
    #include "Observables/observables.hpp"

    #include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
    #include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
    #include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
    #include <boost/numeric/odeint/algebra/range_algebra.hpp>

    #include <Eigen/Dense>
    #include <algorithm>
    #include <cmath>
    #include <iostream>
    #include <fstream>
    #include <complex>
    #include <vector>
    #include <iomanip>
    #include <filesystem>
    #include <sstream>
    #include <cstdint>
    #include <iterator>
    #include <complex>
    using namespace std::complex_literals;


    using namespace std;
    using namespace Eigen;
    using namespace boost::numeric::odeint;

    static std::string read_file_to_string(const std::filesystem::path& p)
    {
        std::ifstream in(p, std::ios::binary);
        if (!in)
            return {};
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    
    // Stable 64-bit FNV-1a hash (good enough for folder naming)
    static std::uint64_t fnv1a_64(std::string_view s)
    {
        std::uint64_t h = 14695981039346656037ull;
        for (unsigned char c : s) {
            h ^= static_cast<std::uint64_t>(c);
            h *= 1099511628211ull;
        }
        return h;
    }

    int main(int argc, char** argv)
    {
        // ===========================
        // Argument check
        // ===========================
        if (argc < 2) {
            cerr << "Usage: ./sim_mkl <config.toml>\n";
            return 1;
        }

        // ===========================
        // Load parameters from TOML
        // ===========================
        const std::string config_path = argv[1];

        Params p;
        p.load_from_toml(config_path);
        p.finalize();

#ifdef USE_EXTERNAL_INPUTS
        if (p.use_external_inputs && !p.external_positions_file.empty()) {
            if (!load_external_positions(p.external_positions_file, p)) {
                std::cerr << "Error: failed to load external positions. Aborting.\n";
                return 1;
            }
        }
#endif

        // ===========================
        // Field mode (from TOML)
        // ===========================
        string mode = p.field_mode;

        // ===========================
        // Output folder: Simulations/<timestamp>_<config-stem>/
        // ===========================
        namespace fs = std::filesystem;

        const fs::path cfg_p = fs::path(config_path);
        const std::string cfg_stem = cfg_p.stem().string();

        // B) overwrite only if identical config:
        // Use a deterministic folder name based on the input TOML file *contents*.
        // If the TOML text is unchanged, it maps to the same folder and outputs are overwritten.
        const std::string cfg_text = read_file_to_string(cfg_p);
        const std::uint64_t cfg_hash = fnv1a_64(cfg_text);
        std::ostringstream hhex;
        hhex << std::hex << cfg_hash;

        const fs::path out_dir = fs::path("Simulations") / (cfg_stem + "_" + hhex.str());
        fs::create_directories(out_dir);

        // copy input toml into the folder (so the simulation is reproducible)
        try {
            fs::copy_file(cfg_p, out_dir / "input.toml", fs::copy_options::overwrite_existing);
        } catch (...) {
        }

        // ===========================
        // Save lattice points generated in C++
        // ===========================
        {
            ofstream fout(out_dir / "lattice_points.txt");
            fout << "# x y\n";
            if (p.two_dim) {
                for (const auto& r : p.xl_2D)
                    fout << r[0] << " " << r[1] << "\n";
            } else {
                for (double x : p.xl_1D)
                    fout << x << " 0\n";
            }
        }

        // ===========================
        // Build potential and Coulomb
        // ===========================
        Potential pot(p);

#ifdef USE_EXTERNAL_INPUTS
        if (p.use_external_inputs && !p.external_coulomb_file.empty()) {
            int N_coulomb = p.N;
            if (load_external_coulomb(p.external_coulomb_file, N_coulomb, p.V_ee)) {
                p.N = N_coulomb;
            } else {
                std::cerr << "Warning: failed to load external Coulomb matrix, falling back to internal builder.\n";
                p.V_ee = pot.build_coulomb_matrix();
            }
        } else {
            p.V_ee = pot.build_coulomb_matrix();
        }
#else
        p.V_ee = pot.build_coulomb_matrix();
#endif



        // Onsite Hubbard U = the onsite element of the Coulomb kernel, U = v_ll =
        // v(0) ~ 15.7 eV, taken from whichever kernel [features] coulomb_kernel
        // selects (vvR or ohno) via pot.v_kernel() — the SAME call build_coulomb_matrix()
        // uses for the diagonal, so U and V_ee(0,0) cannot drift apart when the
        // kernel is swapped. That identity is what makes the double-counting
        // bookkeeping exact: whenever the Hubbard is active the onsite element is
        // dropped from every Hartree evaluation (static SCF and dynamics) and put
        // back, spin-resolved, through U.
        //
        // [features] hubbard_U_eV overrides U (to scan it, or to use a more screened
        // value). Carry that override into the V_ee diagonal too, BEFORE V_ee is
        // saved or used anywhere: otherwise the code removes v(0) from the Hartree
        // and adds back a different U, injecting (U - vvR(0)) of onsite interaction
        // from nowhere — with the configured 31.44 eV override that is a spurious
        // +15.7 eV. (TOML coulomb_onsite_eV is NOT used here, for the same reason it
        // is unused in build_coulomb_matrix().)
        const bool   U_override = (p.hubbard_U_eV >= 0.0);
        const double U_au       = U_override ? (p.hubbard_U_eV / p.au_eV) : pot.v_kernel(0.0);

        if (p.spin_on && p.hubbard && U_override &&
            p.V_ee.rows() == p.N && p.V_ee.cols() == p.N) {
            const double v0_old = p.V_ee(0, 0);
            if (std::abs(U_au - v0_old) > 1e-12) {
                // retune the V_ee diagonal so U = v_ll stays exact
                p.V_ee.diagonal().setConstant(U_au);
            }
        }

        // Save V_ee (Coulomb / VLL) matrix for plotting
        {
            ofstream fout(out_dir / "V_ee.txt");
            for (int i = 0; i < p.V_ee.rows(); ++i) {
                for (int j = 0; j < p.V_ee.cols(); ++j)
                    fout << p.V_ee(i, j) << " ";
                fout << "\n";
            }
        }

        if (p.spin_on) {
            const int N_sites = p.N;
            const int N_spin  = 2 * N_sites;
            Eigen::MatrixXd V_spin = Eigen::MatrixXd::Zero(N_spin, N_spin);
            V_spin.block(0,           0,           N_sites, N_sites) = p.V_ee;
            V_spin.block(N_sites, N_sites, N_sites, N_sites) = p.V_ee;

            ofstream fout_spin(out_dir / "V_ee_spin.txt");
            for (int i = 0; i < N_spin; ++i) {
                for (int j = 0; j < N_spin; ++j)
                    fout_spin << V_spin(i, j) << " ";
                fout_spin << "\n";
            }
        }

    
        if (p.lattice == "graphene" && p.two_dim && p.spin_on && p.B_ext) {
            pot.export_peierls_phases(out_dir / "peierls_phases.txt");
        }

        // ===========================
        // Build base tight-binding Hamiltonian (no Zeeman yet).
        // External magnetic field enters here only via Peierls phases;
        // Zeeman coupling is added later in the time-dependent builder so
        // it is not double-counted.
        // ===========================
        MatrixC Hc;

#ifdef USE_EXTERNAL_INPUTS
        bool used_external_H = false;
        if (p.use_external_inputs && !p.external_hamiltonian_file.empty()) {
            int N_H = p.N;
            if (load_external_hamiltonian(p.external_hamiltonian_file, N_H, Hc)) {
                used_external_H = true;
                p.N = N_H;
            } else {
                std::cerr << "Warning: failed to load external Hamiltonian, falling back to internal builder.\n";
            }
        }
        if (!used_external_H) {
#endif
            if (p.lattice == "graphene" || p.lattice == "pentalene") {
                // Graphene / Pentalene: build from points (remove "|| p.lattice == \"pentalene\"" if pentalene preset removed).
                Hc = TB_hamiltonian_from_points(p.xl_2D, p.a, p.t1, 1e-5);
                if (p.lattice == "pentalene") {
                    int n_bonds = 0;
                    for (int i = 0; i < Hc.rows(); ++i)
                        for (int j = i + 1; j < Hc.cols(); ++j)
                            if (std::abs(Hc(i, j)) > 1e-12) n_bonds++;
                    cout << "Pentalene: a=" << p.a << " a.u., bonds=" << n_bonds << " (expect 15)\n";
                    if (n_bonds == 0)
                        cerr << "WARNING: Pentalene has 0 bonds; dipole will be zero. Check bond length a.\n";
                }
            } else if (p.lattice == "ssh" || p.lattice == "chain") {
                // SSH: same freedom as graphene — with or without external phase on bonds.
                if (p.B_ext) {
                    auto bonds = pot.get_bonds();
                    auto phi_ext = pot.build_ssh_external_phases(pot.compute_Bz());
                    if (phi_ext.size() == bonds.size())
                        Hc = TB_hamiltonian_SSH_with_phases(p.N, p.t1, p.t2, bonds, phi_ext);
                    else
                        Hc = TB_hamiltonian(p.N, p.t1, p.t2);
                } else {
                    Hc = TB_hamiltonian(p.N, p.t1, p.t2);
                }
        } else {
            Hc = TB_hamiltonian(p.N, p.t1, p.t2);
        }

        if (p.spin_on) {
            Hc = spin_tonian(Hc);
            if (p.B_ext) {
                // Apply external Peierls phases only to the spinful hopping;
                // Zeeman from B_ext will be added consistently during time
                // evolution via build_H_for_time.
                pot.apply_peierls_to_spinful_hamiltonian(Hc);
            }
        }

        if (p.spin_on)
            cout << "Number of sites, spin resolved = " << 2 * p.xl_1D.size() << endl;
        else
            cout << "Number of sites, spin degenerate = " << p.xl_1D.size() << endl;

        // ===========================
        // Self-consistent Hubbard mean-field (UHF) magnetic ground state.
        // Feature-gated: [features] hubbard = true/false. When on, the converged
        // spin-dependent onsite potential U(<n_-sigma> - 1/2) is folded into the
        // base Hc, so the magnetic state is the equilibrium seen by both the
        // eigenproblem and the time evolution.
        // ===========================
        // Converged Hubbard equilibrium state, kept alive past the block below so the
        // time evolution propagates the SCF's OWN ground state instead of re-deriving
        // one with an independent filling rule (see the note at the rho0 build).
        MatrixC hub_rho0_l, hub_rho0_eig, hub_evecs;
        Eigen::VectorXcd hub_evals;

        // [features] hartree_scf runs the very same SCF loop with U = 0 and the FULL
        // Coulomb kernel (onsite diagonal INCLUDED) in the Hartree: a Hubbard-free,
        // spin-blind, purely electrostatic self-consistent ground state. The onsite
        // Coulomb is then owned by the Hartree diagonal instead of by U — in the
        // static SCF and, via V_ee_hartree in Density.cpp, in the dynamics too.
        const bool   hartree_only    = p.hartree_scf && !p.hubbard;
        const double U_eff           = hartree_only ? 0.0 : U_au;
        const bool   hartree_on_eff  = hartree_only ? true : p.hubbard_hartree;

        // NOTE: [features] hubbard = true additionally requires [hamiltonian]
        // spin_on = true (magnetism needs explicit spin channels); it is silently
        // ignored otherwise, and the printed model line reports what actually ran.
        //
        // hartree_only never needs spin_on: it is spin-blind by construction (both
        // implicit spins see the identical field), so it runs on the plain N x N Hc
        // via solve_hartree_scf_spinless when spin_on = false, instead of silently
        // doing nothing. True Hubbard magnetism (p.hubbard) still requires spin_on.
        if ((p.spin_on && p.hubbard) || hartree_only) {
            // U_au / U_override were resolved above, before V_ee was saved, so that
            // an overridden U could be carried into the V_ee diagonal (U = v_ll).

            // The ground state and the dynamics must use the same model. With
            // hubbard_hartree = false the SCF has NO nonlocal Coulomb, but the
            // propagator still adds V_ee*(rho - rho0) whenever coulomb = true — a
            // nonlocal Hartree referenced to rho0 instead of the ionic n = 1. That is
            // stationary at t = 0, so nothing blows up AT t = 0, but it is not the
            // reference model: the static part sum_{j!=i} v_ij (n_j^eq - 1) is simply
            // missing, so rho0 is a FALSE equilibrium — the SCF placed the charge
            // using only the onsite U, i.e. without letting the nonlocal Coulomb say
            // where it should sit.
            //
            // Being stationary is NOT the same as being safe under driving. Once a
            // pulse displaces rho, the nonlocal term switches on around an equilibrium
            // that was never relaxed against it. Whether that matters is decided by the
            // ground-state GAP, because the gap is the only thing resisting the
            // rearrangement. Measured on the 5x5 armchair triangle (time_impulse,
            // 1e13 W/cm^2, omega = 0.5 eV), dipole after the pulse has fully passed:
            //
            //   Q  gap_eV     coulomb=true            coulomb=false (consistent)
            //   0  13.28      fine                    fine
            //   1  6e-15      mean +2.7e-2, rms 2.5e-2  mean -1.1e-10, rms 7.7e-8
            //   2  3.30       mean +1.4e-11             fine
            //
            // Q = 1 is the odd-electron case: with no nonlocal Hartree to break the
            // symmetry the SCF settles on a spin-UNPOLARISED S = 0 state whose Fermi
            // level is a 4-fold EXACTLY degenerate shell (two up, two down, 0.25 each).
            // Rearranging charge inside that shell costs zero energy and commutes with
            // H, so nothing restores it: the flake parks in a different configuration
            // with a permanent dipole plus an undamped ~0.054 eV mode that outlives the
            // gamma damping time (1088 a.u.) by an order of magnitude. Note the ground
            // state is IDENTICAL in both columns — only the propagated model differs —
            // so this is the model mismatch, not the degeneracy on its own.
            //
            // Safe combinations: hubbard_hartree = true with coulomb = true (nonlocal
            // in both), or coulomb = false (nonlocal in neither). Splitting them is
            // only defensible when the converged gap_eV is comfortably nonzero.
            HubbardResult hub = p.spin_on
                ? solve_hubbard_mft(
                      Hc, U_eff, p.N, p.mu, static_cast<double>(p.T),
                      p.use_charge_doping, p.Q_doping, pot.get_bonds(),
                      // With U = 0 the symmetry-breaking seed cannot survive (both spin
                      // blocks see the identical spin-blind field), so start unpolarised.
                      hartree_only ? 0.0 : p.hubbard_seed,
                      p.hubbard_max_iter, p.hubbard_tol, p.hubbard_mix,
                      p.V_ee, /*hartree_on=*/hartree_on_eff,
                      /*hartree_onsite=*/hartree_only,
                      /*use_mu=*/p.hubbard_mu_filling,
                      /*fd_fill=*/p.hubbard_fd_fill,
                      /*T_smear=*/p.hubbard_smear_T)
                : solve_hartree_scf_spinless(
                      Hc, p.N, p.use_charge_doping, p.Q_doping,
                      static_cast<double>(p.T), pot.get_bonds(),
                      p.hubbard_max_iter, p.hubbard_tol, p.hubbard_mix, p.V_ee);

            // Keep Hc (the tight-binding base) PURE. Store the exchange field so
            // it is re-added every step in build_H_for_time (including branches
            // that rebuild H from hopping). It is folded into Hc_eig below for
            // the static eigenproblem / rho0 only.
            p.hub_active   = true;
            p.hub_U        = U_eff;     // converged Hubbard U (single unified onsite U); 0 in hartree_scf mode
            p.hub_n_up_eq  = hub.n_up;  // equilibrium spin-resolved occupations per site
            p.hub_n_dn_eq  = hub.n_dn;
            p.hub_V_up     = hub.V_up;
            p.hub_V_dn     = hub.V_dn;

            // The converged equilibrium density matrix and the decomposition it came
            // from. Reused verbatim below for rho0 / the eigenproblem so no second,
            // divergent filling rule can creep in.
            hub_rho0_l   = hub.rho0_site;
            hub_rho0_eig = hub.rho0_eig;
            hub_evecs    = hub.evecs;
            hub_evals    = hub.evals;

            // --- SCF report: which model ran, did it converge, what did it fill ---
            // Deliberately terse: the active model, the convergence verdict with its
            // iteration count, and the converged spin-resolved electron count. Every
            // other quantity (U, S_z, gap, moments) is written to magnetization.txt.
            {
                const char* model = hartree_only
                    ? "Hartree only (U = 0, spin-blind)"
                    : (p.hubbard_hartree ? "Hubbard + nonlocal Hartree"
                                         : "Hubbard only (no nonlocal Hartree)");
                const double N_up = hub.n_up.sum();
                const double N_dn = hub.n_dn.sum();

                cout << "\nModel: " << model << "\n";
                if (hub.converged)
                    cout << "SCF converged in " << hub.iterations << " iterations\n";
                else
                    cout << "SCF NOT CONVERGED after " << hub.iterations << " iterations\n";
                cout << "Electrons: N_up = " << N_up << "   N_dn = " << N_dn
                     << "   (total " << N_up + N_dn << ")\n";
            }

            // save the magnetization texture for plotting
            ofstream fmag(out_dir / "magnetization.txt");
            fmag << "# self-consistent UHF magnetization\n"
                 << "# U_eV=" << U_eff * p.au_eV << " S_total=" << hub.S_total
                 << " sum_abs_m=" << hub.m_abs << " gap_eV=" << hub.gap * p.au_eV
                 << " converged=" << hub.converged << " iters=" << hub.iterations << "\n"
                 << "# site  x  y  sublattice  n_up  n_dn  m_i=0.5*(n_up-n_dn)\n";
            for (int i = 0; i < p.N; ++i) {
                double xi = p.two_dim ? p.xl_2D[i][0] : p.xl_1D[i];
                double yi = p.two_dim ? p.xl_2D[i][1] : 0.0;
                fmag << i << " " << xi << " " << yi << " " << hub.sublattice[i] << " "
                     << hub.n_up(i) << " " << hub.n_dn(i) << " " << hub.m(i) << "\n";
            }

            // save the self-consistency trace: error and population per iteration.
            // A smooth decreasing error means healthy convergence; sudden jumps
            // hint at a bad local minimum, and N_tot should stay flat (population
            // conservation).
            ofstream fconv(out_dir / "hubbard_convergence.txt");
            fconv << "# self-consistency convergence trace\n"
                  << "# iter  error=max|dn_i|  N_total  S_z\n";
            fconv << std::setprecision(12);   // resolve sub-1e-4 population drift
            for (size_t it = 0; it < hub.hist_error.size(); ++it) {
                fconv << it << " " << hub.hist_error[it] << " "
                      << hub.hist_Ntot[it] << " " << hub.hist_Sz[it] << "\n";
            }

            // save the LEVEL FLOW: the whole eigenvalue spectrum sampled along the
            // self-consistency loop, i.e. how the levels move while the mean field
            // converges (with U the up/dn levels split and the magnetic gap opens;
            // with U = 0 it is the pure charge-Hartree relaxation). Long format,
            // one row per (sampled iteration) x (state index), so the plotter can
            // slice it any way it likes. The solver already decimated the sampling
            // (see hub.lvl_stride) to keep long runs bounded; the last sample is
            // always the converged spectrum.
            {
                ofstream flev(out_dir / "hubbard_levels.txt");
                flev << "# SCF level flow: eigenvalue spectrum vs self-consistency iteration\n"
                     << "# U_eV=" << U_eff * p.au_eV
                     << " hartree=" << (hartree_on_eff ? 1 : 0)
                     << " hartree_onsite=" << (hartree_only ? 1 : 0)
                     << " homo_index=" << hub.lvl_homo
                     << " n_states=" << (p.spin_on ? 2 : 1) * p.N
                     << " n_samples=" << hub.hist_lvl_iter.size()
                     << " stride=" << hub.lvl_stride
                     << " iters=" << hub.iterations << "\n"
                     << "# iter  state_index  energy_eV  occupation  spin(+1=up,-1=dn)\n";
                for (size_t s = 0; s < hub.hist_lvl_iter.size(); ++s) {
                    for (size_t k = 0; k < hub.hist_lvl_E[s].size(); ++k) {
                        flev << hub.hist_lvl_iter[s] << " " << k << " "
                             << hub.hist_lvl_E[s][k] * p.au_eV << " "
                             << hub.hist_lvl_f[s][k] << " "
                             << hub.hist_lvl_s[s][k] << "\n";
                    }
                }
            }

            // save the converged spin-resolved spectrum: the energy eigenstates,
            // which spin they carry, and how they are filled.
            ofstream fspec(out_dir / "hubbard_spectrum.txt");
            fspec << "# converged UHF spin-resolved spectrum\n"
                  << "# index  energy_eV  spin(+1=up,-1=dn)  occupation\n";
            for (int k = 0; k < hub.spec_energy.size(); ++k) {
                fspec << k << " " << hub.spec_energy(k) * p.au_eV << " "
                      << hub.spec_spin(k) << " " << hub.spec_occ(k) << "\n";
            }
        }

        // Copy for eigenproblem / diagnostics: include Zeeman here so that
        // eigenvalues, eigenvectors and saved HTB reflect the full static
        // Hamiltonian with external B, while the base Hc (without Zeeman)
        // is passed into the time-evolution where Zeeman is added once.
        MatrixC Hc_eig = Hc;
        // Fold the frozen Hubbard exchange field into the static Hamiltonian used
        // for eigenvalues / rho0 (Hc itself stays pure for the time evolution).
        if (p.hub_active) {
            for (int i = 0; i < p.N; ++i) {
                Hc_eig(i, i) += p.hub_V_up(i);
                if (p.spin_on)
                    Hc_eig(p.N + i, p.N + i) += p.hub_V_dn(i);
            }
        }
        if (p.spin_on && p.B_ext && p.zeeman_external) {
            add_Zeeman_diagonal(Hc_eig, pot.compute_Bz(), p.N, true, 0.5);
        }

        // Save Hamiltonian (including Zeeman, if present) for plotting
        {
            ofstream fout(out_dir / "HTB.txt");
            for (int i = 0; i < Hc_eig.rows(); ++i) {
                for (int j = 0; j < Hc_eig.cols(); ++j)
                    fout << Hc_eig(i, j).real() << " " << Hc_eig(i, j).imag() << " ";
                fout << "\n";
            }
        }

        Eigen::VectorXd xl_eig;
        if (p.two_dim) {
            xl_eig.resize(static_cast<int>(p.xl_2D.size()));
            for (int i = 0; i < xl_eig.size(); ++i)
                xl_eig[i] = p.xl_2D[i][0];
        } else {
            xl_eig = Eigen::Map<Eigen::VectorXd>(p.xl_1D.data(), p.xl_1D.size());
        }
        setprecision(12);
        // ===========================
        // Eigenproblem  (use Hamiltonian including Zeeman, if any)
        // ===========================
        auto [eigenvalues, eigenvectors] = compute_eigenpairs(Hc_eig);

        // With the Hubbard on, Hc_eig IS the converged mean-field H (hub_V_up/dn are
        // exactly the diagonal the SCF added), so reuse the decomposition the solver
        // already produced rather than a fresh one: in a degenerate subspace the two
        // diagonalizations can return different basis vectors, and rho0 below must be
        // expressed in the SAME basis the eigenvectors file records.
        if (p.hub_active && hub_evals.size() == eigenvalues.size()) {
            eigenvalues  = hub_evals;
            eigenvectors = hub_evecs;
        }

        // save eigenvalues
        {
            ofstream fout(out_dir / "eigenvalues.txt");
            for (int i = 0; i < eigenvalues.size(); ++i)
                fout << eigenvalues(i).real() << " "
                    << eigenvalues(i).imag() << "\n";
        }//save eigenvectors

        {
            ofstream fout(out_dir / "eigenvectors.txt");
            for (int i = 0; i < eigenvectors.size(); ++i)
                fout << eigenvectors(i).real() << " "
                    << eigenvectors(i).imag() << "\n";
        }

        // ===========================
        // Initial density matrix
        // ===========================
        // With the Hubbard on, rho0 MUST be the state the SCF converged to, filled with
        // the SAME rule the loop used. Re-deriving it here with the plain Rho_0 /
        // Rho_0_charge rules silently disagrees whenever those rules differ from the
        // solver's: the mu-driven path (hubbard_mu_filling) fills canonically to a
        // target derived from the BARE band, while Rho_0(mu,T) fills grand-canonically
        // on the U-shifted mean-field spectrum — with U ~ 30 eV those are simply
        // different electron counts. The occupations would then not match hub_n_*_eq,
        // so the live term U(n(0) - n_eq) is nonzero, H(0) != Hc_eig, [H(0), rho0] != 0,
        // and the magnetic ground state evolves with no driving field at all.
        MatrixC rho0;
        if (p.hub_active && hub_rho0_eig.rows() == Hc_eig.rows()) {
            rho0 = hub_rho0_eig;
        } else if (p.use_charge_doping) {
            rho0 = Rho_0_charge(eigenvalues, p.N, p.Q_doping, p.spin_on);
        } else {
            rho0 = Rho_0(eigenvalues, p.mu, p.T);
        }

        // save rho0 in j-space
        {
            ofstream fout(out_dir / "rho0_j_space.txt");
            for (int i = 0; i < rho0.rows(); ++i) {
                for (int j = 0; j < rho0.cols(); ++j)
                    fout << rho0(i,j).real() << " "
                        << rho0(i,j).imag() << " ";
                fout << "\n";
            }
        }

        MatrixC rho_l = (p.hub_active && hub_rho0_l.rows() == Hc_eig.rows())
                      ? hub_rho0_l                              // SCF ground state, verbatim
                      : rho_l_space(eigenvectors, rho0);

        // save rho0 in l-space 
        {
            ofstream fout(out_dir / "rho0_l_space.txt");
            for (int i = 0; i < rho_l.rows(); ++i) {
                for (int j = 0; j < rho_l.cols(); ++j)
                    fout << rho_l(i,j).real() << " "
                        << rho_l(i,j).imag() << " ";
                fout << "\n";
            }
        }

        MatrixC rho_l_site(p.N, p.N);
        rho_l_site.setZero();
        if (!p.spin_on) {
            for (int i = 0; i < p.N; ++i)
                for (int j = 0; j < p.N; ++j)
                    rho_l_site(i,j) = rho_l(i,j);
        } else {
            const int N_sites = p.N;
            for (int i = 0; i < N_sites; ++i)
                for (int j = 0; j < N_sites; ++j)
                    rho_l_site(i,j) =
                        rho_l(i, j) +
                        rho_l(i + N_sites, j + N_sites);
        }

        cout << "\nStart simulation...\n";

        // ===========================
        // Build equilibrium diag and xl for online dipole computation
        // ===========================
        VectorXd rho0_diag_online(p.N);
        for (int i = 0; i < p.N; ++i)
            rho0_diag_online(i) = std::real(rho_l_site(i, i));

        VectorXd xl_x_online;
        if (p.two_dim) {
            xl_x_online.resize(static_cast<int>(p.xl_2D.size()));
            for (int i = 0; i < xl_x_online.size(); ++i)
                xl_x_online[i] = p.xl_2D[i][0];
        } else {
            xl_x_online = Eigen::Map<VectorXd>(p.xl_1D.data(), p.xl_1D.size());
        }

        // ===========================
        // Open streaming output files for large per-step arrays
        // ===========================
        const auto bonds_list = pot.get_bonds();
        const int N_bonds_sys = static_cast<int>(bonds_list.size());

        {
            ofstream f_bi(out_dir / "bond_indices.txt");
            f_bi << "# i j  (0-based site indices for each bond)\n";
            for (const auto& b : bonds_list)
                f_bi << b.first << " " << b.second << "\n";
        }

        const bool bs_zeeman_active = p.zeeman_induced && p.spin_on && p.two_dim && !p.xl_2D.empty();
        const bool sc_active = p.self_consistent_phase && p.two_dim && !p.xl_2D.empty();

        ofstream f_J_bond_zeeman_s, f_B_ind_z_zeeman_s;
        ofstream f_J_bond_sc_s, f_B_ind_z_sc_s, f_B_ind_z_curl_s, f_A_ind_s;
        ofstream f_rho_diag_s;
        ofstream f_rho_full_s;
        ofstream f_spin_diag_s;
        ofstream f_J_spin_bond_s;

        const bool spin_bond_active = p.spin_on && p.two_dim && N_bonds_sys > 0;
        if (spin_bond_active) {
            f_J_spin_bond_s.open(out_dir / "J_spin_bond_time_evolution.txt");
            f_J_spin_bond_s << "# t";
            for (int b = 0; b < N_bonds_sys; ++b) f_J_spin_bond_s << "  Js_bond_" << b;
            f_J_spin_bond_s << '\n';
        }

        if (bs_zeeman_active && N_bonds_sys > 0) {
            f_J_bond_zeeman_s.open(out_dir / "J_bond_time_evolution.txt");
            f_J_bond_zeeman_s << "# t";
            for (int b = 0; b < N_bonds_sys; ++b) f_J_bond_zeeman_s << "  J_bond_" << b;
            f_J_bond_zeeman_s << '\n';

            f_B_ind_z_zeeman_s.open(out_dir / "B_ind_z_time_evolution.txt");
            f_B_ind_z_zeeman_s << "# t";
            for (int i = 0; i < p.N; ++i) f_B_ind_z_zeeman_s << "  B_z_" << i;
            f_B_ind_z_zeeman_s << '\n';
        }
        if (sc_active && N_bonds_sys > 0) {
            f_J_bond_sc_s.open(out_dir / "J_bond_sc_time_evolution.txt");
            f_J_bond_sc_s << "# t";
            for (int b = 0; b < N_bonds_sys; ++b) f_J_bond_sc_s << "  J_bond_sc_" << b;
            f_J_bond_sc_s << '\n';

            f_B_ind_z_sc_s.open(out_dir / "B_ind_z_sc_time_evolution.txt");
            f_B_ind_z_sc_s << "# t";
            for (int i = 0; i < p.N; ++i) f_B_ind_z_sc_s << "  B_z_sc_" << i;
            f_B_ind_z_sc_s << '\n';

            f_B_ind_z_curl_s.open(out_dir / "B_ind_z_curl_time_evolution.txt");
            f_B_ind_z_curl_s << "# t";
            for (int i = 0; i < p.N; ++i) f_B_ind_z_curl_s << "  B_z_curl_" << i;
            f_B_ind_z_curl_s << '\n';

            f_A_ind_s.open(out_dir / "A_ind_time_evolution.txt");
            f_A_ind_s << "# t  A_ind_x_0 A_ind_y_0 ... (N_sites = " << p.N << ")\n";
        }

        f_rho_diag_s.open(out_dir / "rho_diag_time_evolution.txt");
        f_rho_diag_s << "# t";
        for (int i = 0; i < p.N; ++i) f_rho_diag_s << " rho_" << i;
        f_rho_diag_s << '\n';

        // Full induced density matrix rho(t)-rho0 (site basis), streamed on the output stride.
        // Format: each line is  t  then Re Im pairs for all N_mat*N_mat elements (row-major),
        // where N_mat = N (spinless) or 2*N (spin_on). Off unless [analysis] save_rho_full = true.
        if (p.save_rho_full) {
            const int N_mat = static_cast<int>(Hc.rows());
            f_rho_full_s.open(out_dir / "rho_full_induced_time_evolution.txt");
            f_rho_full_s << "# induced rho(t)-rho0, N_mat=" << N_mat
                         << " ; line: t  [Re Im]*(N_mat*N_mat) row-major\n";
        }

        // Lean spin-resolved induced diagonal (all the spin-density plots need this).
        // Format: each line is  t  then N_mat reals = rho_ii(t)-rho0_ii,
        // ordered [up_0..up_{N-1}, dn_0..dn_{N-1}] when spin_on. ~O(N) per step.
        if (p.save_spin_diag) {
            const int N_mat = static_cast<int>(Hc.rows());
            f_spin_diag_s.open(out_dir / "spin_diag_time_evolution.txt");
            f_spin_diag_s << "# induced diagonal rho_ii(t)-rho0_ii, N_mat=" << N_mat
                          << " ; line: t  rho_ind_0 .. rho_ind_" << (N_mat - 1)
                          << "  (up block then down block)\n";
        }

        // ===========================
        // Time evolution
        // ===========================
        RhoHistory history;

        MatrixC rho_final = evolve_rho_over_time(
            rho_l, Hc, pot, mode, p, history,
            bs_zeeman_active && N_bonds_sys > 0 ? &f_J_bond_zeeman_s  : nullptr,
            bs_zeeman_active && N_bonds_sys > 0 ? &f_B_ind_z_zeeman_s : nullptr,
            sc_active        && N_bonds_sys > 0 ? &f_J_bond_sc_s      : nullptr,
            sc_active        && N_bonds_sys > 0 ? &f_B_ind_z_sc_s     : nullptr,
            sc_active        && N_bonds_sys > 0 ? &f_B_ind_z_curl_s   : nullptr,
            sc_active        && N_bonds_sys > 0 ? &f_A_ind_s          : nullptr,
            &f_rho_diag_s,
            p.save_rho_full ? &f_rho_full_s : nullptr,
            p.save_spin_diag ? &f_spin_diag_s : nullptr,
            spin_bond_active ? &f_J_spin_bond_s : nullptr,
            &rho0_diag_online, &xl_x_online, p.e, p.spin_on);

        cout << "\n simulation is done\n";
        // save final rho in j-space (site basis) as pairs: Re Im
        {
            ofstream fout(out_dir / "rho_j_space.txt");
            for (int i = 0; i < rho_final.rows(); ++i) {
                for (int j = 0; j < rho_final.cols(); ++j)
                    fout << rho_final(i,j).real() << " "
                         << rho_final(i,j).imag() << " ";
                fout << "\n";
            }
        }
       

        // Close streaming files — they are now fully written
        f_J_bond_zeeman_s.close();
        f_B_ind_z_zeeman_s.close();
        f_J_bond_sc_s.close();
        f_B_ind_z_sc_s.close();
        f_B_ind_z_curl_s.close();
        f_A_ind_s.close();
        f_J_spin_bond_s.close();
        f_rho_diag_s.close();
        f_rho_full_s.close();
        f_spin_diag_s.close();

        // ===========================
        // Save dipole evolution
        // ===========================
        const size_t N_t = history.time.size();
        VectorXd time_vec(N_t);
        VectorXd dipole_t(N_t);

        {
            ofstream fout(out_dir / "dipole_time_evolution.txt");
            fout << "# time   dipole_moment\n";

            if (!history.dipole_t.empty()) {
                // Online path: dipole already computed per-step during evolution
                for (size_t k = 0; k < N_t; ++k) {
                    time_vec[k] = history.time[k];
                    dipole_t[k] = history.dipole_t[k];
                    fout << history.time[k] << " " << history.dipole_t[k] << "\n";
                }
            } else {
                // Fallback: compute from hist.diag (used when rho_diag was not streamed)
                for (size_t k = 0; k < N_t; ++k) {
                    VectorXd rho_diag = Eigen::Map<const VectorXd>(history.diag[k].data(), p.N);
                    double dip = compute_dipole_moment_from_diag(rho_diag, rho0_diag_online,
                                                                  xl_eig, p.e, p.spin_on);
                    time_vec[k] = history.time[k];
                    dipole_t[k] = dip;
                    fout << history.time[k] << " " << dip << "\n";
                }
            }
        }

        //  compute current Jx, Jy
        {
            ofstream fout(out_dir / "current_time_evolution.txt");
            fout << "# t  Jx  Jy\n";

            for (size_t k = 0; k < N_t; ++k) {
                fout << history.time[k] << " "
                    << history.J_x[k] << " "
                    << history.J_y[k] << "\n";
            }
        }

        if (!history.J_up_x.empty()) {
            ofstream fout(out_dir / "spin_current_time_evolution.txt");
            fout << "# t  J_up_x  J_up_y  J_dn_x  J_dn_y  J_spin_x  J_spin_y\n";
            for (size_t k = 0; k < N_t; ++k) {
                const double Js_x = history.J_up_x[k] - history.J_dn_x[k];
                const double Js_y = history.J_up_y[k] - history.J_dn_y[k];
                fout << history.time[k]    << " "
                     << history.J_up_x[k] << " " << history.J_up_y[k] << " "
                     << history.J_dn_x[k] << " " << history.J_dn_y[k] << " "
                     << Js_x              << " " << Js_y               << "\n";
            }
        }

        // L1: per-bond scalar currents — only write from history if not already streamed to disk
        if (!history.J_bond.empty()) {
            ofstream fout(out_dir / "J_bond_time_evolution.txt");
            const int N_b = static_cast<int>(history.J_bond[0].size());
            fout << "# t";
            for (int b = 0; b < N_b; ++b) fout << "  J_bond_" << b;
            fout << "\n";
            for (size_t k = 0; k < history.time.size(); ++k) {
                fout << history.time[k];
                for (int b = 0; b < N_b; ++b) fout << " " << history.J_bond[k][b];
                fout << "\n";
            }
        }

        // L1: site-resolved B_ind_z — only write from history if not already streamed
        if (!history.B_ind_z.empty()) {
            ofstream fout(out_dir / "B_ind_z_time_evolution.txt");
            const int N_s = static_cast<int>(history.B_ind_z[0].size());
            fout << "# t";
            for (int i = 0; i < N_s; ++i) fout << "  B_z_" << i;
            fout << "\n";
            for (size_t k = 0; k < history.time.size(); ++k) {
                fout << history.time[k];
                for (int i = 0; i < N_s; ++i) fout << " " << history.B_ind_z[k][i];
                fout << "\n";
            }
        }

        // rho_diag — only write from history if not already streamed to disk
        if (!history.diag.empty()) {
            ofstream fout(out_dir / "rho_diag_time_evolution.txt");
            fout << "# t";
            for (int i = 0; i < p.N; ++i) fout << " rho_" << i;
            fout << "\n";
            for (size_t k = 0; k < N_t; ++k) {
                fout << history.time[k];
                const auto &diag_k = history.diag[k];
                for (int i = 0; i < p.N; ++i)
                    fout << " " << diag_k[i];
                fout << "\n";
            }
        }

        // A_ind — only write from history if not already streamed to disk
        if (!history.A_ind_x.empty() && history.A_ind_x.size() == N_t) {
            ofstream fout(out_dir / "A_ind_time_evolution.txt");
            const int N_s = static_cast<int>(history.A_ind_x[0].size());
            fout << "# t  A_ind_x_0 A_ind_y_0  A_ind_x_1 A_ind_y_1  ... (N_sites = " << N_s << ")\n";
            for (size_t k = 0; k < N_t; ++k) {
                fout << history.time[k];
                for (int i = 0; i < N_s; ++i)
                    fout << " " << history.A_ind_x[k][i] << " " << history.A_ind_y[k][i];
                fout << "\n";
            }
        }

        if (time_vec.size() < 2) {
            cerr << "Error: Not enough time points for Fourier analysis (need at least 2)\n";
            return 1;
        }

        // Controlled by [analysis] run_sigma_ext and run_dipole_acc in TOML (default false = skip for fast runs)
        if (p.run_sigma_ext || p.run_dipole_acc) {
            double freq_step_eV_au = p.fourier_dt_fs / p.au_eV;
            int N_omega = static_cast<int>((p.omega_cut_off) / freq_step_eV_au);
            VectorXd omega_fourier(N_omega);
            for (int i = 0; i < N_omega; ++i)
                omega_fourier(i) = i * freq_step_eV_au;

            if (p.run_sigma_ext) {
                cout << "\n Calculating Sigma_ext\n";
                VectorXd sigma_ext;
                VectorXcd alpha;
                compute_sigma_ext(
                    dipole_t, time_vec, omega_fourier,
                    p.a, p.au_fs, p.E0, p.N, p.au_c, p.sigma_ddf,
                    sigma_ext, alpha, p.spin_on);
                {
                    ofstream fout(out_dir / "alpha_ext.txt");
                    for (int i = 0; i < alpha.size(); ++i)
                        fout << alpha(i).real() << " " << alpha(i).imag() << "\n";
                }
                {
                    ofstream fout(out_dir / "sigma_ext.txt");
                    for (int i = 0; i < sigma_ext.size(); ++i)
                        fout << omega_fourier(i) << " " << sigma_ext(i) << "\n";
                }
            }

            if (p.run_dipole_acc) {
                 cout << "\n Calculating dipole acceleration\n";
                Eigen::VectorXcd dipole_acc;
                compute_dipole_acceleration(dipole_t, time_vec, omega_fourier, dipole_acc);
                {
                    ofstream fout(out_dir / "dipole_acc.txt");
                    for (int i = 0; i < dipole_acc.size(); ++i)
                        fout << omega_fourier(i)<< " " << dipole_acc(i).real() << " " << dipole_acc(i).imag() << "\n";
                }
            }
        }

        cout << "All outputs saved under: " << out_dir << endl;

        return 0;
    }
