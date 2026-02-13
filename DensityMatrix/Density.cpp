#include <Eigen/Eigenvalues>
#include "Density.hpp"
#include "params/params.hpp"
#include "Hamiltonians/potential.hpp"

#include <numeric>
#include <algorithm>
#include <complex>
#include <fstream>

#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/algebra/range_algebra.hpp>
#include <boost/numeric/odeint/algebra/default_operations.hpp>

using namespace std;
using namespace boost::numeric::odeint;



// -------------------------------------------------------------
RhoObserver::RhoObserver(
    int N_,
    RhoHistory &h,
    TimeTonianSolver *s,
    int stride_
)
    : N(N_), hist(h), solver(s), stride(stride_)
{}
// -------------------------------------------------------------
// RhoObserver operator() IMPLEMENTATION
// -------------------------------------------------------------
// =============================================================
void RhoObserver::operator()(
    const std::vector<std::complex<double>> &rho_vec,
    double t
) {
    // store every observer call (i.e., every integrator step)
    counter++;

    // Use solver's sizes so this works with and without explicit spin
    const int N_sites = solver->N_sites;
    const int N_mat   = solver->N_mat;
    const bool spin   = solver->spin_on;

    hist.time.push_back(t);
    hist.diag.emplace_back(N_sites);

    // rho_vec is flattened row-major N_mat × N_mat
    // Map to site occupations; if spin is on, sum the diagonal of both spin blocks.
    for (int i = 0; i < N_sites; ++i) {
        const int idx_up = i * N_mat + i;
        double occ = std::real(rho_vec[idx_up]);
        if (spin) {
            const int j = i + N_sites;
            const int idx_down = j * N_mat + j;
            occ += std::real(rho_vec[idx_down]);
        }
        hist.diag.back()[i] = occ;
    }

    // Store full rho(t) for post-processing current calculation
    MatrixC rho_k(N_mat, N_mat);
    for (int i = 0; i < N_mat; ++i)
        for (int j = 0; j < N_mat; ++j)
            rho_k(i, j) = rho_vec[i * N_mat + j];
    hist.rho_full.push_back(rho_k);

    // ==============================
    // Compute current 
    // ==============================
    const std::complex<double> im(0.0, 1.0);
    double e_over_hbar = solver->e_charge / solver->hbar;

    // Hamiltonian used in this step
   MatrixC H_t = solver->H0;

    // external potential
    Eigen::VectorXd Vext = solver->get_potential(t);

    if (solver->coulomb_on) {

    Eigen::VectorXd rho_diag(solver->N_sites);

    if (!solver->spin_on) {
        for (int i = 0; i < solver->N_sites; ++i)
            rho_diag(i) = std::real(rho_k(i, i));
    } else {
        for (int i = 0; i < solver->N_sites; ++i)
            rho_diag(i) =
                std::real(rho_k(i, i)) +
                std::real(rho_k(i + solver->N_sites,
                                i + solver->N_sites));
    }

    const double hartree_factor = solver->spin_on ? 1.0 : 2.0;

    Eigen::VectorXd rho_l_ind =
        hartree_factor * solver->e_charge *
        (rho_diag - solver->rho0_diag);

    Eigen::VectorXd V_hartree =
        solver->V_ee * rho_l_ind;

    for (int i = 0; i < solver->N_sites; ++i) {
        std::complex<double> v(
            solver->e_charge * V_hartree(i), 0.0);

        H_t(i,i) += v;

        if (solver->spin_on)
            H_t(i + solver->N_sites,
                i + solver->N_sites) += v;
    }
}

    for (int i = 0; i < solver->N_sites; ++i) {
        std::complex<double> v(Vext(i), 0.0);
        H_t(i, i) += v;

        if (solver->spin_on)
            H_t(i + solver->N_sites, i + solver->N_sites) += v;
    }


    // Current operators
    MatrixC J_x(N_mat, N_mat);
    MatrixC J_y(N_mat, N_mat);

    J_x.noalias() = -im * e_over_hbar * (H_t * solver->P_x - solver->P_x * H_t);
    J_y.noalias() = -im * e_over_hbar * (H_t * solver->P_y - solver->P_y * H_t);

    // Expectation values
    double Jx_exp = std::real((rho_k * J_x).trace());
    double Jy_exp = std::real((rho_k * J_y).trace());

    // Store for plotting
    hist.J_x.push_back(Jx_exp);
    hist.J_y.push_back(Jy_exp);

    // simple progress print
    if (counter % 5000 == 0) {
        double t_fs = t * 2.418884326505e-17 * 1e15;
        std::cout << "\r t = " << t_fs << " fs" << std::flush;
    }
}





// -------------------------------------------------------------
// 1. Eigenpairs (single solver — eigenvalues and eigenvectors paired)
// For real symmetric H (SSH, graphene from points) use SelfAdjointEigenSolver
// so eigenvectors are real and orthonormal; avoids phase ambiguity from ComplexEigenSolver.
// -------------------------------------------------------------
std::pair<Eigen::VectorXcd, Eigen::MatrixXcd> compute_eigenpairs(const Eigen::MatrixXcd& H) {
    const int N = H.rows();
    const double imag_norm = H.imag().cwiseAbs().maxCoeff();
    const double herm_err = (H - H.adjoint()).cwiseAbs().maxCoeff();

    Eigen::VectorXcd eigvals_sorted(N);
    Eigen::MatrixXcd eigvecs_sorted(N, N);

    if (imag_norm < 1e-14 && herm_err < 1e-14) {
        // Real symmetric: use SelfAdjointEigenSolver for stable real eigenvectors
        Eigen::MatrixXd Hreal = H.real();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Hreal);
        if (solver.info() != Eigen::Success) {
            // fallback to complex solver if something goes wrong
            Eigen::ComplexEigenSolver<Eigen::MatrixXcd> csolver(H);
            eigvals_sorted = csolver.eigenvalues();
            eigvecs_sorted = csolver.eigenvectors();
        } else {
            Eigen::VectorXd evals = solver.eigenvalues();
            Eigen::MatrixXd evecs = solver.eigenvectors();
            for (int i = 0; i < N; ++i)
                eigvals_sorted(i) = std::complex<double>(evals(i), 0.0);
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    eigvecs_sorted(i, j) = std::complex<double>(evecs(i, j), 0.0);
        }
    } else {
        // General complex Hermitian: use ComplexEigenSolver
        Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(H);
        eigvals_sorted = solver.eigenvalues();
        eigvecs_sorted = solver.eigenvectors();
    }

    // Sort by real part of eigenvalue (ascending)
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
        [&](int a, int b) {
            return eigvals_sorted(a).real() < eigvals_sorted(b).real();
        });

    Eigen::VectorXcd eigvals_out(N);
    Eigen::MatrixXcd eigvecs_out(N, N);
    for (int k = 0; k < N; ++k) {
        eigvals_out(k) = eigvals_sorted(idx[k]);
        eigvecs_out.col(k) = eigvecs_sorted.col(idx[k]);
    }

    return {eigvals_out, eigvecs_out};
}

// -------------------------------------------------------------
// 3. Rho_0 — now RETURNS MatrixC (row-major)
// -------------------------------------------------------------
MatrixC Rho_0(const Eigen::VectorXcd& epsilon, double mu, double T) {
    const double kb = 8.617e-5 / 27.2113834;
    int N = epsilon.size();

    MatrixC Rho = MatrixC::Zero(N, N);

    for (int i = 0; i < N; ++i) {
        double Ei = epsilon(i).real();
        double x  = (Ei - mu) / (kb * T);
        double f  = 1.0 / (exp(x) + 1.0);
        Rho(i, i) = complex<double>(f, 0.0);
    }

    return Rho;
}

// -------------------------------------------------------------
// 4. Transform density to l-space — now MatrixC everywhere
// -------------------------------------------------------------
MatrixC rho_l_space(const Eigen::MatrixXcd& U,
                    const MatrixC& rho_eig)
{
    MatrixC Udag = U.adjoint();
    return U * (rho_eig * Udag);
}


// -------------------------------------------------------------
// 5. RHS of dρ/dt — all matrices are MatrixC
// -------------------------------------------------------------

void TimeTonianSolver::operator()(
    const std::vector<std::complex<double>> &rho_vec,
    std::vector<std::complex<double>> &drho_dt_vec,
    const double t)
{
    const int N_sites_local = N_sites;
    const int N_mat_local   = N_mat;

    // --- reshape rho_vec → rho_tmp ---
    for (int i = 0, k = 0; i < N_mat_local; ++i)
        for (int j = 0; j < N_mat_local; ++j, ++k)
            rho_tmp(i,j) = rho_vec[k];

    // --- get diagonal potential Vext(t) ---
    const Eigen::VectorXd Vext = get_potential(t);

    // --- H_tmp = H0 (copy) ---
    H_tmp = H0;


    /* RHS call counter for debugging
    static size_t rhs_calls = 0;
    rhs_calls++;
    if (rhs_calls % 10000 == 0)
        std::cout << "RHS calls = " << rhs_calls << std::endl;
    */

    // --- add diagonal external potential ---
    // Vext is defined per site; if spin is on, apply to both spin blocks.
    if (!spin_on) {
        for (int i = 0; i < N_sites_local; ++i)
            H_tmp(i,i) += std::complex<double>(Vext[i], 0.0);
    } else {
        for (int i = 0; i < N_sites_local; ++i) {
            const std::complex<double> v(Vext[i], 0.0);
            H_tmp(i,i) += v;                                    // spin-up
            H_tmp(i + N_sites_local, i + N_sites_local) += v;   // spin-down
        }
    }

    //==========================================================
    //                 ELECTRON–ELECTRON (Hartree) TERM
    //==========================================================
    if (coulomb_on) {
        // Current site occupations from rho_tmp (sum over spin if present)
        if (!spin_on) {
            for (int i = 0; i < N_sites_local; ++i)
                rho_diag(i) = std::real(rho_tmp(i, i));
        } else {
            for (int i = 0; i < N_sites_local; ++i) {
                rho_diag(i) =
                    std::real(rho_tmp(i, i)) +
                    std::real(rho_tmp(i + N_sites_local, i + N_sites_local));
            }
        }

        // rho_l_ind = (2e or 1e) * (rho - rho0) in site space.
        // Without explicit spin we multiply by 2 for spin degeneracy.
        // With explicit spin (spin_on == true) rho_diag already includes both spins,
        // so we drop the extra factor of 2.
        const double hartree_factor = spin_on ? 1.0 : 2.0;
        rho_l_ind.noalias() = hartree_factor * e_charge * (rho_diag - rho0_diag);

        // V_hartree_cached = V_ee * rho_l_ind in site space
        V_hartree_cached.noalias() = V_ee * rho_l_ind;

        // Add Hartree potential back to H_tmp diagonal (duplicate for spin)
        if (!spin_on) {
            for (int i = 0; i < N_sites_local; ++i)
                H_tmp(i,i) += std::complex<double>(e_charge * V_hartree_cached(i), 0.0);
        } else {
            for (int i = 0; i < N_sites_local; ++i) {
                const std::complex<double> v(e_charge * V_hartree_cached(i), 0.0);
                H_tmp(i,i) += v;
                H_tmp(i + N_sites_local, i + N_sites_local) += v;
            }
        }
    }

    

    //==========================================================

    // --- commutator [H,ρ] ---
    comm_tmp.noalias() = H_tmp * rho_tmp;
    comm_tmp.noalias() -= rho_tmp * H_tmp;

    const std::complex<double> minus_i(0.0, -1.0);

    // --- drho/dt ---
    for(int i = 0, k = 0; i < N_mat_local; ++i)
        for(int j = 0; j < N_mat_local; ++j, ++k)
        {
            std::complex<double> comm_val = comm_tmp(i,j);
            std::complex<double> relax = rho_tmp(i,j) - rho0(i,j);

            drho_dt_tmp(i,j) =
                minus_i * comm_val - (gamma * 0.5) * relax;

            drho_dt_vec[k] = drho_dt_tmp(i,j);
        }
}


// -------------------------------------------------------------
// 6. Time evolution — now using MatrixC everywhere
// -------------------------------------------------------------
MatrixC evolve_rho_over_time( const MatrixC &rho_initial_l, const MatrixC &Hc, Potential &pot, 
    const std::string &mode, 
    const Params &p,
    RhoHistory &history)

{
    const int N_sites = p.N;         // number of spatial sites
    const int N_mat   = Hc.rows();   // full Hilbert-space dimension (with or without spin)
    using state_type = vector<complex<double>>;
    state_type rho_vec(N_mat * N_mat);

    for(int i = 0; i < N_mat; ++i)
        for(int j = 0; j < N_mat; ++j)
            rho_vec[i*N_mat + j] = rho_initial_l(i,j);

    TimeTonianSolver solver;
    solver.N_mat   = N_mat;
    solver.N_sites = N_sites;
    solver.spin_on = p.spin_on;
    solver.H0      = Hc;
    solver.coulomb_on = p.coulomb_on;
    solver.V_ee       = p.V_ee;          // site-space Coulomb
    solver.e_charge   = p.e;
            
    solver.gamma = p.gamma;
    solver.rho0  = rho_initial_l;

    //===========calc current====================
    solver.hbar = p.au_hbar;
    solver.x_pos.resize(N_sites);
    solver.y_pos.resize(N_sites);

    // make the xl-cords
    for (int i = 0; i<N_sites; i++){
        if (p.two_dim){
            solver.x_pos(i) = p.xl_2D[i][0];
            solver.y_pos(i) = p.xl_2D[i][1];
        } else{
            solver.x_pos(i) = p.xl_1D[i];
            solver.y_pos(i) = 0.0;
        }
    }

    //calc the dipole operator
    solver.P_x = MatrixC::Zero(N_mat, N_mat);
    solver.P_y = MatrixC::Zero(N_mat, N_mat);

    for (int i = 0; i < N_sites; ++i) {
        std::complex<double> px(p.e * solver.x_pos(i), 0.0);
        std::complex<double> py(p.e * solver.y_pos(i), 0.0);

        solver.P_x(i, i) = px;
        solver.P_y(i, i) = py;

        if (p.spin_on) {
            solver.P_x(i + N_sites, i + N_sites) = px;
            solver.P_y(i + N_sites, i + N_sites) = py;
        }
    }

    // i am speed boosted now
    solver.rho_tmp.resize(N_mat, N_mat);
    solver.V_tmp.resize(N_mat, N_mat);
    solver.H_tmp.resize(N_mat, N_mat);
    solver.comm_tmp.resize(N_mat, N_mat);
    solver.drho_dt_tmp.resize(N_mat, N_mat);
    solver.Vext_tmp.resize(N_sites);
    
    solver.rho_diag.resize(N_sites);
    solver.rho_l_ind.resize(N_sites);
    solver.V_ind_vector.resize(N_sites);
    solver.V_hartree_cached.resize(N_sites);
    solver.V_hartree_cached.setZero();

    if (solver.coulomb_on) {
        // Equilibrium site occupations from initial rho (sum over spin if needed)
        solver.rho0_diag.resize(N_sites);
        if (!solver.spin_on) {
            for (int i = 0; i < N_sites; ++i)
                solver.rho0_diag(i) = std::real(rho_initial_l(i, i));
        } else {
            for (int i = 0; i < N_sites; ++i) {
                solver.rho0_diag(i) =
                    std::real(rho_initial_l(i, i)) +
                    std::real(rho_initial_l(i + N_sites, i + N_sites));
            }
        }

        // Start with zero induced density / Hartree potential
        solver.rho_diag.setZero();
        solver.rho_l_ind.setZero();
        solver.V_hartree_cached.setZero();
    }


    solver.rho_tmp.setZero();
    solver.V_tmp.setZero();
    solver.H_tmp.setZero();
    solver.comm_tmp.setZero();
    solver.drho_dt_tmp.setZero();

    solver.get_potential = [&](double t){
        return pot.get_potential(t, mode);
    };

   RhoObserver observer(N_sites, history, &solver);

    if(p.use_strict_solver) {
        typedef runge_kutta_dopri5<state_type> strict_stepper;
        strict_stepper stepper;

        boost::numeric::odeint::integrate_const(
            stepper,
            solver,
            rho_vec,
            p.t0,
            p.t_end,
            p.dt,
            observer);
    }
    else {
        typedef runge_kutta_dopri5<state_type> error_stepper_type;
        typedef controlled_runge_kutta<error_stepper_type> controlled_stepper_type;

        controlled_stepper_type controlled_stepper(
            default_error_checker<
                double, range_algebra, default_operations
            >(p.a_tol, p.r_tol, 1, 1)
        );

        boost::numeric::odeint::integrate_adaptive(
            controlled_stepper,
            solver,
            rho_vec,
            p.t0,
            p.t_end,
            p.dt,
            observer
        );
    }

    // Return final density in site space (N_sites × N_sites)
    MatrixC rho_final(N_sites, N_sites);
    for(int i = 0; i < N_sites; ++i)
        for(int j = 0; j < N_sites; ++j) {
            const int idx_up = i * N_mat + j;
            std::complex<double> val = rho_vec[idx_up];
            if (p.spin_on) {
                const int ii = i + N_sites;
                const int jj = j + N_sites;
                const int idx_down = ii * N_mat + jj;
                val += rho_vec[idx_down];
            }
            rho_final(i,j) = val;
        }

    return rho_final;
}

// -------------------------------------------------------------
// Post-process rho(t) history to compute current Jx, Jy; write .txt
// -------------------------------------------------------------
void compute_current_from_history(
    const Params &p,
    const MatrixC &Hc,
    Potential &pot,
    const RhoHistory &history,
    const std::string &mode,
    const std::string &out_dir)
{
    const int N_sites = p.N;
    const int N_mat   = Hc.rows();
    if (history.rho_full.empty() || history.time.size() != history.rho_full.size()) {
        std::cerr << "compute_current_from_history: empty or mismatched history, skipping.\n";
        return;
    }

    const double e_over_hbar = static_cast<double>(p.e) / static_cast<double>(p.au_hbar);
    const std::complex<double> im(0.0, 1.0);

    // Equilibrium site occupations (for Hartree) from first stored rho
    Eigen::VectorXd rho0_diag(N_sites);
    const MatrixC &rho0_full = history.rho_full[0];
    if (!p.spin_on) {
        for (int i = 0; i < N_sites; ++i)
            rho0_diag(i) = std::real(rho0_full(i, i));
    } else {
        for (int i = 0; i < N_sites; ++i)
            rho0_diag(i) = std::real(rho0_full(i, i)) + std::real(rho0_full(i + N_sites, i + N_sites));
    }

    // Dipole (position) matrices P_x, P_y: diagonal in site space, duplicated for spin if needed
    MatrixC P_x = MatrixC::Zero(N_mat, N_mat);
    MatrixC P_y = MatrixC::Zero(N_mat, N_mat);
    
    for (int i = 0; i < N_sites; ++i) {
        double xi = p.two_dim ? p.xl_2D[i][0] : p.xl_1D[i];
        double yi = p.two_dim ? p.xl_2D[i][1] : 0.0;
        std::complex<double> vx(static_cast<double>(p.e) * xi, 0.0);
        std::complex<double> vy(static_cast<double>(p.e) * yi, 0.0);
        P_x(i, i) = vx;
        P_y(i, i) = vy;
        if (p.spin_on) {
            P_x(i + N_sites, i + N_sites) = vx;
            P_y(i + N_sites, i + N_sites) = vy;
        }
    }

    MatrixC H_t(N_mat, N_mat);
    MatrixC J_x(N_mat, N_mat), J_y(N_mat, N_mat);
    Eigen::VectorXd rho_diag(N_sites);
    Eigen::VectorXd rho_l_ind(N_sites);
    Eigen::VectorXd V_hartree(N_sites);
    const double hartree_factor = p.spin_on ? 1.0 : 2.0;

    std::ofstream fout(out_dir + "/current_time_evolution.txt");
    if (!fout) {
        std::cerr << "compute_current_from_history: could not open current_time_evolution.txt\n";
        return;
    }
    fout << "# t  Jx  Jy\n";

    for (size_t k = 0; k < history.time.size(); ++k) {
        const double t = history.time[k];
        const MatrixC &rho_k = history.rho_full[k];








        H_t = Hc;
        Eigen::VectorXd Vext = pot.get_potential(t, mode);
        for (int i = 0; i < N_sites; ++i) {
            std::complex<double> v(Vext(i), 0.0);
            H_t(i, i) += v;
            if (p.spin_on)
                H_t(i + N_sites, i + N_sites) += v;
        }

        if (p.coulomb_on) {
            if (!p.spin_on) {
                for (int i = 0; i < N_sites; ++i)
                    rho_diag(i) = std::real(rho_k(i, i));
            } else {
                for (int i = 0; i < N_sites; ++i)
                    rho_diag(i) = std::real(rho_k(i, i)) + std::real(rho_k(i + N_sites, i + N_sites));
            }
            rho_l_ind.noalias() = hartree_factor * static_cast<double>(p.e) * (rho_diag - rho0_diag);
            V_hartree.noalias() = p.V_ee * rho_l_ind;
            for (int i = 0; i < N_sites; ++i) {
                std::complex<double> v(static_cast<double>(p.e) * V_hartree(i), 0.0);
                H_t(i, i) += v;
                if (p.spin_on)
                    H_t(i + N_sites, i + N_sites) += v;
            }
        }

        // J_x = (-i e/hbar) [H_t, P_x], J_y = (-i e/hbar) [H_t, P_y]
        J_x.noalias() = -im * e_over_hbar * (H_t * P_x - P_x * H_t);
        J_y.noalias() = -im * e_over_hbar * (H_t * P_y - P_y * H_t);

        double Jx_exp = std::real((rho_k * J_x).trace());
        double Jy_exp = std::real((rho_k * J_y).trace());
        fout << t << " " << Jx_exp << " " << Jy_exp << "\n";
    }
}
