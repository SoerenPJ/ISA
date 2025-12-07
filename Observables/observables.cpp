#include "observables.hpp"

using namespace std;
using namespace Eigen;

double compute_dipole_moment(
    const MatrixXcd &rho_t,
    const MatrixXcd &rho0,
    const VectorXd  &xl,
    int e)
{
    int N = xl.size();
    VectorXd rho_ind_diag(N);

    for(int i = 0; i < N; ++i)
    {
        double val = std::real(rho_t(i,i) - rho0(i,i));
        rho_ind_diag(i) = -2.0 * e * val;
    }

    return rho_ind_diag.dot(xl);
}

