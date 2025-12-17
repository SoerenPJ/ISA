#pragma once

#include <Eigen/Dense>   // REQUIRED for Eigen::MatrixXd
#include <vector>
#include <array>
#include <string>
using namespace std;

struct Params {
    // Atomic units
    double au_eV; 
    double au_nm;
    double au_s;
    double au_fs; 
    double au_kg;
    double au_kB; 
    double au_m;
    double au_J;
    double au_w;
    double au_I;
    double au_me;
    double alpha;
    double au_mu_0;
    double a;
    double Intensity;
    double E0; 
    double mu;
    int au_hbar;
    double au_c;
    int e;

    // Physical parameters
    double t0;
    double t1; 
    double t2; 
    double gamma;

    // Simulation parameters
    int N; 
    int T; 
    double t_end;
    double t_shift;
    double sigma_gaus; 
    double sigma_ddf; 
    double dt;
    double a_tol; 
    double r_tol; 

    bool coulomb_on = false;    // turn on/off Coulomb Hartree term
    Eigen::MatrixXd V_ee;       // Coulomb matrix (NxN)

    // External field
    string potential_mode = "time_impulse";
    double au_omega;
    double au_omega_fourier;

    bool spin_on; 
    bool two_dim; 

    // Solver
    bool use_strict_solver = false;

    // Lattice positions
    vector<double> xl_1D;
    vector<array<double, 2>> xl_2D;

    Params();
    void build_lattice();
};
