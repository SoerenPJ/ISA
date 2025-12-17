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


// -------------------------------------------------------------
// 1. Eigenvalues  
// -------------------------------------------------------------
Eigen::VectorXcd compute_eigenvalues(const Eigen::MatrixXcd& H) {
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(H);

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

    // --- add diagonal external potential ---
    for(int i = 0; i < N; ++i)
        H_tmp(i,i) += std::complex<double>(Vext[i], 0.0);

    //==========================================================
    //                 ELECTRON–ELECTRON (Hartree) TERM
    //==========================================================
    if (coulomb_on)
    {
        // Step 1: extract real diag(rho) (NO new allocation)
        for (int i = 0; i < N; ++i)
            rho_diag(i) = rho_tmp(i,i).real();

        // Step 2: rho_l_ind = 2 * e * (rho_diag - rho0_diag)
        rho_l_ind.noalias() = 2.0 * e_charge * (rho_diag - rho0_diag);

        // Step 3: V_ind_vector = V_ee * rho_l_ind  (MKL GEMV)
        V_ind_vector.noalias() = V_ee * rho_l_ind;

        // Step 4: add induced potential to diagonal of Hamiltonian
        for (int i = 0; i < N; ++i)
            H_tmp(i,i) += std::complex<double>(
                e_charge * V_ind_vector(i), 0.0
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


    solver.rho_tmp.setZero();
    solver.V_tmp.setZero();
    solver.H_tmp.setZero();
    solver.comm_tmp.setZero();
    solver.drho_dt_tmp.setZero();

    solver.get_potential = [&](double t){
        return pot.get_potential(t, mode);
    };

    RhoObserver observer(N, history);

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
