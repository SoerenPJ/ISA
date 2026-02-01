#include "params.hpp"
#include <toml++/toml.h>
#include <cmath>

// ======================
// Constructor: constants only
// ======================
Params::Params()
{
    // Keep these consistent with the project's reference Python constants.
    au_eV = 27.2113834;
    au_nm = 0.05291772083;
    au_s  = 2.418884326502e-17;

    // NOTE: In this codebase, `au_fs` means "femtoseconds per atomic unit of time".
    // (If you prefer "atomic units per femtosecond", that's 1/au_fs.)
    au_fs = au_s * 1e15; // ~0.0241888 fs

    au_c  = 137.03599971;

    au_kg = 9.10938291e-31;
    au_J  = 4.3597447222060e-18;      // 1 Hartree in Joule
    au_m  = 5.29177210544e-11;        // Bohr radius in meter

    // kB in atomic units (Hartree/K). Same as (8.617e-5 eV/K) / au_eV.
    au_kB = 8.617e-5 / au_eV;

    // Keep as a reference constant (not used for the legacy intensity convention)
    au_I  = 3.50944506e16;

    // Derived / convenience
    alpha = 1.0 / au_c;
    au_hbar = 1;
    e = 1;

    // Keep for compatibility if used elsewhere
    au_w  = 4.1341373336493e16; // Eh / ħ in s^-1 (approx)
    au_me = 9.10938291e-31;
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
    
    dt    = tbl["simulation"]["dt"].value_or(0.2); // do not change, this is the way 
    t_end = tbl["simulation"]["t_max"].value_or(500.0) / au_fs;
    t0    = tbl["simulation"]["t0"].value_or(0.0) / au_fs;
    a_tol = tbl["simulation"]["a_tol"].value_or(1e-4);
    r_tol = tbl["simulation"]["r_tol"].value_or(1e-6);
    // ---- solver ----
    use_strict_solver = tbl["simulation"]["use_strict_solver"].value_or(false);

    
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
    
    const double intensity_au = (Intensity / au_kg) * (au_s * au_s * au_s);
    E0 = std::sqrt((2.0 * M_PI * intensity_au) / au_c);
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
