    #include "params/params.hpp"
    #include "DensityMatrix/Density.hpp"
    #include "Hamiltonians/hamiltonian.hpp"
    #include "Hamiltonians/potential.hpp"
    #include "Observables/observables.hpp"

    #include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
    #include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
    #include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
    #include <boost/numeric/odeint/algebra/range_algebra.hpp>

    #include <Eigen/Dense>
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
        p.V_ee = pot.build_coulomb_matrix();



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
        // Build Hamiltonian (initial: no induced phase; external phase only if B_ext)
        // ===========================
        MatrixC Hc;
        if (p.lattice == "graphene" || p.lattice == "pentalene") {
            // Graphene / Pentalene: build from points (remove "|| p.lattice == \"pentalene\"" if pentalene preset removed).
            Hc = TB_hamiltonian_from_points(p.xl_2D, p.a, p.t1, 0.001);
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
                pot.apply_peierls_to_spinful_hamiltonian(Hc);
            }
        }

        if (p.spin_on && p.B_ext && p.zeeman_external) {
            add_Zeeman_diagonal(Hc, pot.compute_Bz(), p.N, true, 0.5);
        }


        // Save Hamiltonian 
        {
            ofstream fout(out_dir / "HTB.txt");
            for (int i = 0; i < Hc.rows(); ++i) {
                for (int j = 0; j < Hc.cols(); ++j)
                    fout << Hc(i, j).real() << " " << Hc(i, j).imag() << " ";
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

        cout << "\nxl_1D size = " << p.xl_1D.size() << endl;

        // ===========================
        // Eigenproblem 
        // ===========================
        auto [eigenvalues, eigenvectors] = compute_eigenpairs(Hc);


        // save eigenvalues
        {
            ofstream fout(out_dir / "eigenvalues.txt");
            for (int i = 0; i < eigenvalues.size(); ++i)
                fout << eigenvalues(i).real() << " "
                    << eigenvalues(i).imag() << "\n";
        }

        // ===========================
        // Initial density matrix
        // ===========================
        MatrixC rho0 = Rho_0(eigenvalues, p.mu, p.T);

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

        MatrixC rho_l = rho_l_space(eigenvectors, rho0);

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
        // Time evolution
        // ===========================
        RhoHistory history;

        MatrixC rho_final =
            evolve_rho_over_time(rho_l, Hc, pot, mode, p, history);

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
       

        // ===========================
        // Save dipole evolution (use diagonal-only)
        // ===========================
        VectorXd time_vec(history.time.size());
        VectorXd dipole_t(history.time.size());
        VectorXd rho0_diag(p.N);
        for (int i = 0; i < p.N; ++i)
            rho0_diag(i) = std::real(rho_l_site(i, i));

        {
            ofstream fout(out_dir / "dipole_time_evolution.txt");
            fout << "# time   dipole_moment\n";

            for (size_t k = 0; k < history.time.size(); ++k) {
                VectorXd rho_diag = Eigen::Map<const VectorXd>(history.diag[k].data(), p.N);
                double dip = compute_dipole_moment_from_diag(rho_diag, rho0_diag, xl_eig, p.e, p.spin_on);
                time_vec[k] = history.time[k];
                dipole_t[k] = dip;
                fout << history.time[k] << " " << dip << "\n";
            }

        }

        //  compute current Jx, Jy
        {
            ofstream fout(out_dir / "current_time_evolution.txt");
            fout << "# t  Jx  Jy\n";
        
            for (size_t k = 0; k < history.time.size(); ++k) {
                fout << history.time[k] << " "
                    << history.J_x[k] << " "
                    << history.J_y[k] << "\n";
            }
        }

        // Save site-resolved density diagonal as a function of time
        {
            ofstream fout(out_dir / "rho_diag_time_evolution.txt");
            fout << "# t";
            for (int i = 0; i < p.N; ++i)
                fout << " rho_" << i;
            fout << "\n";

            for (size_t k = 0; k < history.time.size(); ++k) {
                fout << history.time[k];
                const auto &diag_k = history.diag[k];
                for (int i = 0; i < p.N; ++i)
                    fout << " " << diag_k[i];
                fout << "\n";
            }
        }
        
        // Induced vector potential A_ind_x, A_ind_y per site (when self_consistent_phase was on)
        if (!history.A_ind_x.empty() && history.A_ind_x.size() == history.time.size()) {
            ofstream fout(out_dir / "A_ind_time_evolution.txt");
            const int N_s = static_cast<int>(history.A_ind_x[0].size());
            fout << "# t  A_ind_x_0 A_ind_y_0  A_ind_x_1 A_ind_y_1  ... (N_sites = " << N_s << ")\n";
            for (size_t k = 0; k < history.time.size(); ++k) {
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
