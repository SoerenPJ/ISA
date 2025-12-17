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
    int counter = 0;     // counts observer calls
    int stride  = 200;   // print every 200 steps (adjust as needed)

    RhoObserver(int N_, RhoHistory &h, int stride_ = 200)
        : N(N_), hist(h), stride(stride_) {}

    void operator()(const std::vector<std::complex<double>> &rho_vec, double t) {
        counter++;

        // store time
        hist.time.push_back(t);

        // store diagonal rho_ii(t)
        hist.diag.emplace_back(N);
        for(int i = 0; i < N; ++i)
            hist.diag.back()[i] = std::real(rho_vec[i*N + i]);

        // ----- PRINT ONLY EVERY "stride" STEPS -----
        
        if (counter % stride == 0) {
            double t_fs = t * 2.418884326505e-17 * 1e15; // au → fs
            std::cout << "t = " << t_fs << " fs" << std::endl;
        }
        
    }
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
    Eigen::VectorXd rho_diag;       // size N
    Eigen::VectorXd rho_l_ind;      // size N
    Eigen::VectorXd V_ind_vector;   // size N


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
