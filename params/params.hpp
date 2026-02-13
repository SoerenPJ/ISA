#pragma once

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <string>

struct Params
{
    // ======================
    // ---- CONSTANTS ----
    // ======================
    double au_eV, au_nm, au_s, au_fs, au_c;
    double au_kg, au_kB, au_m, au_J, au_w, au_I, au_me;
    double alpha;
    int au_hbar, e;
    

    // ======================
    // ---- SYSTEM (TOML) ----
    // ======================
    int N;
    bool B_ext = false;
    bool two_dim = false;
    bool spin_on = false;
    std::string lattice = "chain";
    std::string formation = "zigzag"; // "zigzag" | "armchair"
    std::string formation_shape;  //rectangle | triangle
    int size_x = 0;                             
    int size_y = 0;
    // ======================
    // ---- HAMILTONIAN (TOML) ----
    // ======================
    double t1, t2;
    double mu;
    double gamma;

    // ======================
    // ---- TIME (TOML) ----
    // ======================
    double dt;
    double t_end;

    // ======================
    // ---- SOLVER ----
    // ======================
    bool use_strict_solver = false;
    double t0   = 0.0;
    double a_tol = 1e-6;
    double r_tol = 1e-8;

    // ======================
    // ---- FIELD (TOML) ----
    // ======================
    double Intensity;
    std::string field_mode = "time_impulse"; // "time_impulse" | "sinus" | "ddf"
    double field_phase = 0.0;                // phase offset [rad]
    double au_omega;
    double au_omega_ddf = 0.1 / 27.2113834;  // ddf omega (a.u.), default 0.1 eV
    double t_shift;
    double sigma_gaus;
    double sigma_ddf;
    double omega_cut_off;

    // ======================
    // ---- THERMO (TOML) ----
    // ======================
    int T;

    // ======================
    // ---- FEATURES (TOML) ----
    // ======================
    bool coulomb_on = false;
    double coulomb_onsite_eV = 10.0;  // Hubbard U / onsite repulsion (eV), e.g. 5-10 for graphene
    bool self_consistent_phase = false;  // current -> A_ind -> phi_ind -> update hopping (induced phase)

    // ======================
    // ---- DERIVED ----
    // ======================
    double a;          // lattice spacing (a.u.)
    double E0;         // field amplitude
    double au_mu_0;    // vacuum permeability in a.u. (4*pi/c^2)
    double area_2d;    // effective area for 2D A_ind (graphene unit-cell related)
    Eigen::MatrixXd V_ee;

    // ======================
    // ---- GEOMETRY ----
    // ======================
    std::vector<double> xl_1D;
    std::vector<std::array<double,2>> xl_2D;

    // ======================
    // ---- API ----
    // ======================
    Params();
    void load_from_toml(const std::string& filename);
    void finalize();

private:
    void build_lattice();
};
