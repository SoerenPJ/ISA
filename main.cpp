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

int main() {

    Params p;
    p.spin_on = false;
    p.build_lattice();

    Potential pot(p);

    MatrixC Hc = TB_hamiltonian(p.N, p.t1, p.t2);
    Eigen::VectorXd xl_eig = Eigen::Map<Eigen::VectorXd>(p.xl_1D.data(),
                                                         p.xl_1D.size());
    cout << "\nxl_1D size = " << p.xl_1D.size() << endl;


    VectorXcd eigenvalues   = compute_eigenvalues(Hc);
    MatrixXcd eigenvectors  = compute_eigenvectors(Hc);

    // save eigenvalues
    {
        ofstream fout("eigenvalues.txt");
        for (int i = 0; i < eigenvalues.size(); ++i)
            fout << eigenvalues(i).real() << " " << eigenvalues(i).imag() << "\n";
    }

    MatrixC rho0 = Rho_0(eigenvalues, p.mu, p.T);

    // save rho0 in j-space
    {
        ofstream fout("rho0_j_space.txt");
        for (int i = 0; i < rho0.rows(); ++i) {
            for (int j = 0; j < rho0.cols(); ++j)
                fout << rho0(i,j).real() << " " << rho0(i,j).imag() << " ";
            fout << "\n";
        }
    }

    MatrixC rho_l = rho_l_space(eigenvectors, rho0);

    // save rho0 in l-space
    {
        ofstream fout("rho0_l_space.txt");
        for (int i = 0; i < rho_l.rows(); ++i) {
            for (int j = 0; j < rho_l.cols(); ++j)
                fout << rho_l(i,j).real() << " " << rho_l(i,j).imag() << " ";
            fout << "\n";
        }
    }

    cout << "\nStart simulation...\n";

    RhoHistory history;

    p.use_strict_solver = false;
    string mode = "sinus";

    MatrixC rho_final =
        evolve_rho_over_time(rho_l, Hc, pot, mode, p, history);

    Eigen::VectorXd rho0_diag(p.N);
    for (int i = 0; i < p.N; ++i)
        rho0_diag(i) = real(rho_l(i, i));

    

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

            double dip = compute_dipole_moment(rho_t, rho_l, xl_eig, p.e);
            fout << history.time[k] << " " << dip << "\n";
        }

        cout << "\nDipole moment saved to dipole_time_evolution.txt\n";
    }

   

    return 0;
}
