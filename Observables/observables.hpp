#pragma once
#include <Eigen/Dense>

double compute_dipole_moment(
    const Eigen::MatrixXcd &rho_t,
    const Eigen::MatrixXcd &rho0,
    const Eigen::VectorXd  &xl,
    int e,
    bool spin_on);

/** Faster: dipole from diagonal occupations only (avoids building NxN rho per step). */
double compute_dipole_moment_from_diag(
    const Eigen::VectorXd &rho_diag,
    const Eigen::VectorXd &rho0_diag,
    const Eigen::VectorXd &xl,
    int e,
    bool spin_on);


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
    Eigen::VectorXcd &alpha, bool spin_on);

void compute_dipole_acceleration(
    const Eigen::VectorXd &diople_t, 
    const Eigen::VectorXd &t, 
    const Eigen::VectorXd &omega_fourier, 
    Eigen::VectorXcd &dipole_acc
);