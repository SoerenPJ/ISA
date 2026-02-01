#pragma once

#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <functional>
#include <iostream>   // <-- added for std::cout

using MatrixC = Eigen::Matrix<
    std::complex<double>,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor
>;

// Forward declarations
class Potential;
class Params;
struct TimeTonianSolver;   // <<< ADD THIS


// =====================================================
// Store time evolution of diagonal density elements
// =====================================================
struct RhoHistory {
    std::vector<double> time;
    std::vector<std::vector<double>> diag;   // diag[t][i] = rho_ii(t)
};

// =====================================================
// Observer called by Boost.Odeint at each time step
// =====================================================
struct RhoObserver {
    int N;
    RhoHistory &hist;
    TimeTonianSolver *solver;

    int counter = 0;
    int stride  = 200;

    RhoObserver(int N_, RhoHistory &h, TimeTonianSolver *s, int stride_ = 200);

    void operator()(const std::vector<std::complex<double>> &rho_vec, double t);
};


// =====================================================
// TimeTonian = RHS of Liouville-von Neumann equation
// =====================================================
struct TimeTonianSolver {

    int N;
    MatrixC H0;
    std::function<Eigen::VectorXd(double)> get_potential;
    bool coulomb_on;
    Eigen::MatrixXd V_ee;
    Eigen::VectorXd rho0_diag;    
    double e_charge;

    double gamma;                  // relaxation rate
    MatrixC rho0;                  // equilibrium density matrix

    MatrixC rho_tmp;
    MatrixC H_tmp;
    MatrixC V_tmp;
    MatrixC comm_tmp;
    MatrixC drho_dt_tmp;
    Eigen::VectorXd Vext_tmp;

    //EEI_related
    Eigen::VectorXd rho_diag;       // size N
    Eigen::VectorXd rho_l_ind;      // size N
    Eigen::VectorXd V_ind_vector;   // size N
    Eigen::VectorXd V_hartree_cached;     // size N


    void operator()(const std::vector<std::complex<double>> &rho_vec,
                    std::vector<std::complex<double>> &drho_dt_vec,
                    const double t);
};

// =====================================================
// Function declarations
// =====================================================

Eigen::VectorXcd compute_eigenvalues(const Eigen::MatrixXcd& H);
Eigen::MatrixXcd compute_eigenvectors(const Eigen::MatrixXcd& H);

MatrixC Rho_0(const Eigen::VectorXcd &epsilon, double mu, double T);

MatrixC rho_l_space(const Eigen::MatrixXcd &A_jl, const MatrixC &rho_j);

// Evolve density matrix over time (records diagonals into history)
MatrixC evolve_rho_over_time(
    const MatrixC &rho_initial_l,
    const MatrixC &Hc,
    Potential &pot,
    const std::string &mode,
    const Params &p,
    RhoHistory &history);
