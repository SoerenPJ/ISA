    #include "params.hpp"
    #include <cmath>

    using namespace std;
    Params::Params() {
        // Atomic units
        au_eV = 27.2114;
        au_nm = 0.0529177;
        au_s = 2.41888e-17;
        au_fs = au_s * 1e15;
        au_c = 137.036;
        au_kg = 9.10938e-31;
        au_kB = 3.16681e-6;
        au_hbar = 1.0;
        au_m = 1.0;
        au_J = 4.35974e-18;
        au_w = 4.13414e16;
        au_I = 3.50945e16;
        au_me = 9.10938e-31;
        alpha = 1.0 / au_c;
        e = 1;
        au_mu_0 = 4 * M_PI * 1e-7 * (au_eV * au_s) / (au_J * au_m);

        
        a = 0.142 / au_nm;           // Lattice constant 
        
        mu = 0.0/au_eV;                     // Chemical potential
        // Physical parameters
        N = 100;
        T = 100;
        t1 = -2.8/au_eV;
        t2 = t1;// + 0.5;
        spin_on = true;
        two_dim = false;
        
        
        // external field parameters
        Intensity = (1e13/au_kg) * (au_s*au_s*au_s);  // time pules = 1e13 ddf = 1e15
        E0 = sqrt((2 *M_PI* Intensity) / (au_c)); // Electric field amplitude
        
        t_shift = 0.5 / au_fs;
        sigma_gaus = 60.07/au_fs; 
        sigma_ddf = 0.01/au_fs;

        gamma = 0.01/au_eV;
        au_omega = 0.2/au_eV;
        au_omega_fourier = 0.0;// for now


        // simulation parameters    
        dt = 0.2;
        t0 = 0.0;
        t_end = 500 / au_fs;
        r_tol = 1e-6; // 
        a_tol = 1e-5;
        

        

        // Build lattice positions
        build_lattice();
    }

    void Params::build_lattice() {
        xl_1D.clear();
        xl_1D.reserve(N * (spin_on ? 2 : 1));

        // physical positions
        for (int i = 0; i < N; ++i) {
            xl_1D.push_back(a * i);
        }

    }