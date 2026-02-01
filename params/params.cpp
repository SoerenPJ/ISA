#include "params.hpp"
#include <toml++/toml.h>
#include <cmath>

// ======================
// Constructor: constants only
// ======================
Params::Params()
{
    au_eV = 27.2114;
    au_nm = 0.0529177;
    au_s  = 2.41888e-17;
    au_fs = au_s * 1e15;
    au_c  = 137.036;

    au_kg = 9.10938e-31;
    au_kB = 3.16681e-6;
    au_m  = 1.0;
    au_J  = 4.35974e-18;
    au_w  = 4.13414e16;
    au_I  = 3.50945e16;
    au_me = 9.10938e-31;

    alpha = 1.0 / au_c;
    au_hbar = 1;
    e = 1;
}

// ======================
// Load TOML
// ======================
void Params::load_from_toml(const std::string& filename)
{
    auto tbl = toml::parse_file(filename);

    // ---- system ----
    N = tbl["system"]["N"].value_or(50);
    a = tbl["system"]["lattice_const"].value_or(0.142) / au_nm;
    two_dim = tbl["system"]["two_dim"].value_or(false);

    // ---- hamiltonian ----
    t1 = tbl["hamiltonian"]["t1"].value_or(-2.8) / au_eV;
    t2 = tbl["hamiltonian"]["t2"].value_or(-2.8) / au_eV;
    mu = tbl["hamiltonian"]["mu"].value_or(0.0) / au_eV;
    gamma = tbl["hamiltonian"]["gamma"].value_or(0.01) / au_eV;

    // ---- simulation ----
    // NOTE: TOML time inputs are assumed to be in femtoseconds (fs).
    // Convert to atomic units consistently (same convention as t_max, t_shift, sigma_*).
    dt    = tbl["simulation"]["dt"].value_or(0.2) / au_fs;
    t_end = tbl["simulation"]["t_max"].value_or(500.0) / au_fs;
    t0    = tbl["simulation"]["t0"].value_or(0.0) / au_fs;
    a_tol = tbl["simulation"]["a_tol"].value_or(1e-4);
    r_tol = tbl["simulation"]["r_tol"].value_or(1e-6);
    // ---- solver ----
    // Allow configuring strict vs adaptive via TOML (support both [solver] and [simulation]).
    use_strict_solver =
        tbl["solver"]["use_strict_solver"].value_or(
            tbl["simulation"]["use_strict_solver"].value_or(false)
        );


    // ---- field ----
    Intensity   = tbl["field"]["intensity"].value_or(1e13);
    au_omega    = tbl["field"]["omega"].value_or(0.2) / au_eV;
    t_shift     = tbl["field"]["t_shift"].value_or(200.0) / au_fs;
    sigma_gaus  = tbl["field"]["sigma_gaus"].value_or(60.0) / au_fs;
    sigma_ddf   = tbl["field"]["sigma_ddf"].value_or(0.01) / au_fs;

    // ---- thermo ----
    T = tbl["thermo"]["T"].value_or(300);

    // ---- features ----
    coulomb_on = tbl["features"]["coulomb"].value_or(false);
}

// ======================
// Derived quantities
// ======================
void Params::finalize()
{
    E0 = std::sqrt((2.0 * M_PI * Intensity) / au_c);
    build_lattice();
}

// ======================
// Geometry
// ======================
void Params::build_lattice()
{
    xl_1D.clear();
    xl_1D.reserve(N);

    for (int i = 0; i < N; ++i)
        xl_1D.push_back(a * i);
}
