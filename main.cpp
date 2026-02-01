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
    #include <iostream>
    #include <fstream>
    #include <complex>
    #include <vector>
    #include <iomanip>

    using namespace std;
    using namespace Eigen;
    using namespace boost::numeric::odeint;

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
        Params p;
        p.load_from_toml(argv[1]);
        p.finalize();

        // ===========================
        // Simulation mode
        // ===========================
        string mode = "time_impulse";

        // ===========================
        // Build potential and Coulomb
        // ===========================
        Potential pot(p);
        p.coulomb_on = false;   // keep off unless you want it on
        p.V_ee = pot.build_coulomb_matrix();

        // ===========================
        // Build Hamiltonian
        // ===========================
        MatrixC Hc = TB_hamiltonian(p.N, p.t1, p.t2);

        Eigen::VectorXd xl_eig =
            Eigen::Map<Eigen::VectorXd>(p.xl_1D.data(), p.xl_1D.size());

        cout << "\nxl_1D size = " << p.xl_1D.size() << endl;

        // ===========================
        // Eigenproblem
        // ===========================
        VectorXcd eigenvalues  = compute_eigenvalues(Hc);
        MatrixXcd eigenvectors = compute_eigenvectors(Hc);

        // save eigenvalues
        {
            ofstream fout("eigenvalues.txt");
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
            ofstream fout("rho0_j_space.txt");
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
            ofstream fout("rho0_l_space.txt");
            for (int i = 0; i < rho_l.rows(); ++i) {
                for (int j = 0; j < rho_l.cols(); ++j)
                    fout << rho_l(i,j).real() << " "
                        << rho_l(i,j).imag() << " ";
                fout << "\n";
            }
        }

        cout << "\nStart simulation...\n";

        // ===========================
        // Time evolution
        // ===========================
        RhoHistory history;

        MatrixC rho_final =
            evolve_rho_over_time(rho_l, Hc, pot, mode, p, history);

        // ===========================
        // Save dipole evolution
        // ===========================
        {
            ofstream fout("dipole_time_evolution.txt");
            fout << "# time   dipole_moment\n";

            for (size_t k = 0; k < history.time.size(); ++k) {

                MatrixC rho_t(p.N, p.N);
                rho_t.setZero();

                for (int i = 0; i < p.N; ++i)
                    rho_t(i, i) = history.diag[k][i];

                double dip =
                    compute_dipole_moment(rho_t, rho_l, xl_eig, p.e);

                fout << history.time[k] << " " << dip << "\n";
            }

            cout << "\nDipole moment saved to dipole_time_evolution.txt\n";
        }

        return 0;
    }
