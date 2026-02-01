#include <Eigen/Eigenvalues>
#include "Density.hpp"
#include "params/params.hpp"
#include "Hamiltonians/potential.hpp"

#include <numeric>
#include <algorithm>
#include <complex>

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
    static size_t counter = 0;
    counter++;

    // store data
    hist.time.push_back(t);
    hist.diag.emplace_back(N);
    for (int i = 0; i < N; ++i)
        hist.diag.back()[i] = std::real(rho_vec[i*N + i]);

    // simple progress print
    if (counter % 5000 == 0) {
        double t_fs = t * 2.418884326505e-17 * 1e15;
        std::cout << "\r t = " << t_fs << " fs" << std::flush;
    }
}





// -------------------------------------------------------------
// 1. Eigenvalues  
// -------------------------------------------------------------
Eigen::VectorXcd compute_eigenvalues(const Eigen::MatrixXcd& H) {
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(H); //selfadjont eigen

    Eigen::VectorXcd eigvals = solver.eigenvalues();
    int N = eigvals.size();

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(),
        [&](int a, int b) {
            return eigvals(a).real() < eigvals(b).real();
        });

    Eigen::VectorXcd eigvals_sorted(N);
    for (int k = 0; k < N; ++k)
        eigvals_sorted(k) = eigvals(idx[k]);

    return eigvals_sorted;
}

// -------------------------------------------------------------
// 2. Eigenvectors 
// -------------------------------------------------------------
Eigen::MatrixXcd compute_eigenvectors(const Eigen::MatrixXcd& H) {
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(H);

    Eigen::MatrixXcd eigvecs = solver.eigenvectors();
    Eigen::VectorXcd eigvals = solver.eigenvalues();
    int N = eigvals.size();

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(),
        [&](int a, int b) {
            return eigvals(a).real() < eigvals(b).real();
        });

    Eigen::MatrixXcd eigvecs_sorted(N, N);
    for (int k = 0; k < N; ++k)
        eigvecs_sorted.col(k) = eigvecs.col(idx[k]);

    return eigvecs_sorted;
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
    // --- reshape rho_vec → rho_tmp ---
    for (int i = 0, k = 0; i < N; ++i)
        for (int j = 0; j < N; ++j, ++k)
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
    for(int i = 0; i < N; ++i)
        H_tmp(i,i) += std::complex<double>(Vext[i], 0.0);

    //==========================================================
    //                 ELECTRON–ELECTRON (Hartree) TERM
    //==========================================================
   
        if (coulomb_on) {
            for (int i = 0; i < N; ++i)
                H_tmp(i,i) += std::complex<double>(
                e_charge * V_hartree_cached(i), 0.0
        );
    }


    //==========================================================

    // --- commutator [H,ρ] ---
    comm_tmp.noalias() = H_tmp * rho_tmp;
    comm_tmp.noalias() -= rho_tmp * H_tmp;

    const std::complex<double> minus_i(0.0, -1.0);

    // --- drho/dt ---
    for(int i = 0, k = 0; i < N; ++i)
        for(int j = 0; j < N; ++j, ++k)
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
    int N = p.N;

    using state_type = vector<complex<double>>;
    state_type rho_vec(N * N);

    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            rho_vec[i*N+j] = rho_initial_l(i,j);

    TimeTonianSolver solver;
    solver.N     = N;
    solver.H0    = Hc;   
    solver.coulomb_on = p.coulomb_on;                      // NEW
    solver.V_ee       = p.V_ee;                            // NEW
    solver.rho0_diag  = rho_initial_l.diagonal().real();   // NEW
    solver.e_charge   = p.e;                               // NEW
            
    solver.gamma = p.gamma;
    solver.rho0  = rho_initial_l;

    // i am speed boosted now
    solver.rho_tmp.resize(N, N);
    solver.V_tmp.resize(N, N);
    solver.H_tmp.resize(N, N);
    solver.comm_tmp.resize(N, N);
    solver.drho_dt_tmp.resize(N, N);
    solver.Vext_tmp.resize(N);
    
    solver.rho_diag.resize(N);
    solver.rho_l_ind.resize(N);
    solver.V_ind_vector.resize(N);
    solver.V_hartree_cached.resize(N);
    solver.V_hartree_cached.setZero();

    if (solver.coulomb_on) {

        // extract diagonal of initial rho
        for (int i = 0; i < N; ++i)
            solver.rho_diag(i) = std::real(rho_initial_l(i, i));

        // rho_l_ind = 2 e (rho - rho0)
        solver.rho_l_ind.noalias() =
            2.0 * solver.e_charge *
            (solver.rho_diag - solver.rho0_diag);

        // V_hartree_cached = V_ee * rho_l_ind
        solver.V_hartree_cached.noalias() =
            solver.V_ee * solver.rho_l_ind;
    }


    solver.rho_tmp.setZero();
    solver.V_tmp.setZero();
    solver.H_tmp.setZero();
    solver.comm_tmp.setZero();
    solver.drho_dt_tmp.setZero();

    solver.get_potential = [&](double t){
        return pot.get_potential(t, mode);
    };

   RhoObserver observer(N, history, &solver);

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

    MatrixC rho_final(N, N);
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            rho_final(i,j) = rho_vec[i*N+j];

    return rho_final;
}
