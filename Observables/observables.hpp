#pragma once
#include <Eigen/Dense>

double compute_dipole_moment(
    const Eigen::MatrixXcd &rho_t,
    const Eigen::MatrixXcd &rho0,
    const Eigen::VectorXd  &xl, int e
);


std::complex<double> trapezoid(const Eigen::VectorXd& t, const Eigen::VectorXcd& f);


void compute_sigma_ext(
    const Eigen::VectorXd &dipole_t,
    const Eigen::VectorXd &t,
    const Eigen::VectorXd &omega_fourier,
    double a,
    double au_fs,
    double E0,
    int N,
    double au_c,
    double sigma_ddf,
    Eigen::VectorXd &sigma_ext,
    Eigen::VectorXcd &alpha);