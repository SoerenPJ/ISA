#include "observables.hpp"
#include <iostream> 


using namespace std;
using namespace Eigen;
//======================DIPOLE MOMENT=========================
double compute_dipole_moment(
    const MatrixXcd &rho_t,
    const MatrixXcd &rho0,
    const VectorXd  &xl,
    int e,
    bool spin_on)
{
    int N = xl.size();
    VectorXd rho_ind_diag(N);

    // If spin is not explicitly resolved, multiply by 2 for spin degeneracy.
    const double prefactor = (spin_on ? -1.0 : -2.0) * static_cast<double>(e);

    for(int i = 0; i < N; ++i)
    {
        double val = std::real(rho_t(i,i) - rho0(i,i));
        rho_ind_diag(i) = prefactor * val;
    }

    return rho_ind_diag.dot(xl);
}

//======================trapezoid INTEGRATION=========================
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
void compute_sigma_ext(
    const VectorXd &dipole_t,
    const VectorXd  &t,
    const VectorXd  &omega_fourier,
    double a,
    double au_fs,
    double E0,
    int    N,
    double au_c,
    double sigma_ddf,
    VectorXd &sigma_ext, VectorXcd &alpha)
{
    const int Nt = t.size();
    const int Nw = omega_fourier.size();
    alpha.resize(Nw);   
    
    double hex_area = (3.0 * std::sqrt(3.0) / 2.0) * a * a;
    double total_area = (static_cast<double>(N) / 2.0) * hex_area;

    sigma_ext.resize(Nw);
   
    for (int w = 0; w < Nw; ++w) {
        VectorXcd expo(Nt);
        for (int i = 0; i < Nt; ++i)
            expo(i) = std::exp(std::complex<double>(0.0, omega_fourier(w) * t(i)));

        auto integrand = dipole_t.array() * expo.array();
        auto P_w = trapezoid(t, integrand);

        std::complex<double> alpha_w =  P_w / (  2.0* (N*a*a) *sigma_ddf * E0);

        alpha(w) = alpha_w;

        sigma_ext(w) =
            4.0 * M_PI * (omega_fourier(w) / (au_c )) * std::imag(alpha_w);
        
        
    }
}


void compute_dipole_acceleration(
    const Eigen::VectorXd &dipole_t, 
    const Eigen::VectorXd &t, 
    const Eigen::VectorXd &omega_fourier, Eigen::VectorXcd &dipole_acc
)
{
    const int Nt = t.size();
    const int Nw = omega_fourier.size();

    dipole_acc.resize(Nw);

    for (int w = 0; w < Nw; ++w)
    {
        VectorXcd expo(Nt);

        for (int i = 0; i < Nt; ++i)
            {
                expo(i) = std::exp(std::complex<double>(0.0, omega_fourier(w) * t(i)));
            }

        VectorXcd integrand = dipole_t.cast<std::complex<double>>().array() * expo.array();

        std::complex<double> integral = trapezoid(t, integrand);

        dipole_acc(w) = (omega_fourier(w)*omega_fourier(w)) * integral; 
    }

}




