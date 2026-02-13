#pragma once

#include <Eigen/Dense>
#include <utility>
#include <vector>
#include <array>
#include <complex>
#include <functional>
#include <iostream>

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
    std::vector<MatrixC> rho_full;
    std::vector<double> J_x;
    std::vector<double> J_y;
    // Induced vector potential per site when self_consistent_phase is on: A_ind_[xy][t][i]
    std::vector<std::vector<double>> A_ind_x;
    std::vector<std::vector<double>> A_ind_y;
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

    // Dimension of Hilbert space (rows/cols of H0)
    int N_mat;
    // Number of spatial sites/orbitals (from Params::N, without spin duplication)
    int N_sites;
    // Whether spin-doubling was applied (H0 is 2*N_sites × 2*N_sites)
    bool spin_on;

    MatrixC H0;
    std::function<Eigen::VectorXd(double)> get_potential;
    bool coulomb_on;
    // Coulomb matrix in site space (N_sites × N_sites)
    Eigen::MatrixXd V_ee;
    // Equilibrium site occupations (size N_sites)
    Eigen::VectorXd rho0_diag;
    double e_charge;
    double gamma;                  // relaxation rate
    MatrixC rho0;                  // equilibrium density matrix

    MatrixC rho_tmp;
    MatrixC H_tmp;
    MatrixC V_tmp;
    MatrixC comm_tmp;
    MatrixC drho_dt_tmp;
    // external potential in site space (size N_sites)
    Eigen::VectorXd Vext_tmp;

    // EEI-related, all in site space (size N_sites)
    Eigen::VectorXd rho_diag;          // instantaneous site occupations
    Eigen::VectorXd rho_l_ind;         // induced density (rho - rho0)
    Eigen::VectorXd V_ind_vector;      // not currently used, keep for compatibility
    Eigen::VectorXd V_hartree_cached;  // Hartree potential per site


    // geometry
    Eigen::VectorXd x_pos;
    Eigen::VectorXd y_pos;

    // dipole / current
    MatrixC P_x;
    MatrixC P_y;
    double hbar;

    // ---------- Self-consistent induced phase (current -> A_ind -> phi_ind -> hopping) ----------
    bool self_consistent_on = false;
    MatrixC H_prev;   // Hamiltonian from previous RHS call (for current and next H build)
    std::vector<std::pair<int,int>> bonds;
    std::vector<double> phi_ext;   // external Peierls phase per bond (from B_ext)
    double au_mu_0 = 0.0;
    double area_2d = 1.0;
    Eigen::VectorXd J_l_x;
    Eigen::VectorXd J_l_y;
    Eigen::VectorXd J_l_x_eq;   // equilibrium site current (so A_ind from J_l - J_l_eq starts at 0)
    Eigen::VectorXd J_l_y_eq;
    bool J_l_eq_computed = false;
    Eigen::VectorXd A_ind_x;
    Eigen::VectorXd A_ind_y;
    std::vector<double> phi_ind;
    std::vector<double> combined_phase;  // phi_ext + phi_ind per bond
    std::vector<std::array<double,2>> points_2d;  // copy of xl_2D for rebuilding H with phases
    double t_hop = 0.0;   // hopping t (e.g. p.t1)
    double a_bond = 0.0;  // bond length for TB_hamiltonian_from_points_with_phases



    void operator()(const std::vector<std::complex<double>> &rho_vec,
                    std::vector<std::complex<double>> &drho_dt_vec,
                    const double t);
};

// =====================================================
// Function declarations
// =====================================================

std::pair<Eigen::VectorXcd, Eigen::MatrixXcd> compute_eigenpairs(const Eigen::MatrixXcd& H);

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

// Post-process stored rho(t) to compute current Jx, Jy and write current_time_evolution.txt
void compute_current_from_history(
    const Params &p,
    const MatrixC &Hc,
    Potential &pot,
    const RhoHistory &history,
    const std::string &mode,
    const std::string &out_dir);
