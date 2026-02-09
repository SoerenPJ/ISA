#include "observables.hpp"

using namespace std;
using namespace Eigen;
//======================DIPOLE MOMENT=========================
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

//======================SIMPSON INTEGRATION=========================
std::complex<double> trapezoid(const Eigen::VectorXd &t, const Eigen::VectorXcd &f) {
    assert(t.size() == f.size());
    std::complex<double> integral = 0.0;

    for (int i = 0; i < t.size()-1; i++) {
        double dt = t[i+1] - t[i];
        integral += 0.5 * dt * (f[i] + f[i+1]);
    }
    return integral;
}

//======================SIGMA EXT=========================
void compute_sigma_ext( const VectorXcd &dipole_t, const VectorXd  &t, const VectorXd  &omega_fourier,
    double au_fs, double E0, double au_c, double sigma_ddf, VectorXd &sigma_ext)
{
    const int Nt = t.size(); 
    const int Nw = omega_fourier.size();
    
    sigma_ext.resize(Nw);

    VectorXcd dipole_c = dipole_t.cast<complex<double>>();

    for (int w = 0; w < Nw; ++w)
    {
        VectorXcd expo(Nt);
        for (int i = 0; i < Nt; ++i)
            expo(i) = exp(complex<double>(0.0, omega_fourier(w) * t(i)));

        VectorXcd integrand = dipole_c.array() * expo.array();
        
        complex<double> P_w = trapezoid(t, integrand);

        complex<double> alpha = 2.0 * P_w / (au_fs * sigma_ddf * E0); // remove 2 when spin is added

        sigma_ext(w) =
            4.0 * M_PI * (omega_fourier(w) / au_c) * imag(alpha);
    } 

}





