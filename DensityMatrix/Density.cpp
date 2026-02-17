#include <Eigen/Eigenvalues>
#include "Density.hpp"
#include "params/params.hpp"
#include "Hamiltonians/potential.hpp"
#include "Hamiltonians/hamiltonian.hpp"

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

// ---------------------------------------------------------------------------
// Helpers (internal use only) — reduce duplication between observer, RHS, and post-process
// ---------------------------------------------------------------------------
namespace {

constexpr double AU_TIME_TO_FS = 2.418884326505e-17 * 1e15;

// Site occupations from density matrix (sum over spin if spin_on).
Eigen::VectorXd rho_diag_from_rho(const MatrixC& rho, int N_sites, bool spin_on) {
    Eigen::VectorXd rho_diag(N_sites);
    if (!spin_on) {
        for (int i = 0; i < N_sites; ++i)
            rho_diag(i) = std::real(rho(i, i));
    } else {
        for (int i = 0; i < N_sites; ++i)
            rho_diag(i) = std::real(rho(i, i)) + std::real(rho(i + N_sites, i + N_sites));
    }
    return rho_diag;
}

// Bond currents J_l_x, J_l_y from rho and current operators J_x, J_y.
void compute_bond_currents(
    const MatrixC& rho,
    const MatrixC& J_x, const MatrixC& J_y,
    const std::vector<std::pair<int,int>>& bonds,
    bool spin_on, int N_sites,
    Eigen::VectorXd& J_l_x, Eigen::VectorXd& J_l_y)
{
    MatrixC rho_J_x = rho * J_x;
    MatrixC rho_J_y = rho * J_y;
    J_l_x.setZero();
    J_l_y.setZero();
    for (size_t b = 0; b < bonds.size(); ++b) {
        const int i = bonds[b].first;
        const int j = bonds[b].second;
        double cx = std::imag(rho_J_x(i, j));
        double cy = std::imag(rho_J_y(i, j));
        if (spin_on) {
            cx += std::imag(rho_J_x(i + N_sites, j + N_sites));
            cy += std::imag(rho_J_y(i + N_sites, j + N_sites));
        }
        J_l_x(i) += cx;
        J_l_x(j) -= cx;
        J_l_y(i) += cy;
        J_l_y(j) -= cy;
    }
}

// Current operators J_x, J_y = (-i e/hbar) [H, P_x/y].
void compute_current_operators(
    const MatrixC& H, const MatrixC& P_x, const MatrixC& P_y,
    double e_over_hbar,
    MatrixC& J_x, MatrixC& J_y)
{
    const std::complex<double> im(0.0, 1.0);
    J_x.noalias() = -im * e_over_hbar * (H * P_x - P_x * H);
    J_y.noalias() = -im * e_over_hbar * (H * P_y - P_y * H);
}

// Add external potential to H diagonal (per site; duplicate for spin if spin_on).
void add_Vext_to_H(MatrixC& H, const Eigen::VectorXd& Vext, bool spin_on, int N_sites) {
    for (int i = 0; i < N_sites; ++i) {
        std::complex<double> v(Vext(i), 0.0);
        H(i, i) += v;
        if (spin_on)
            H(i + N_sites, i + N_sites) += v;
    }
}

// Add Hartree potential to H diagonal from rho_diag and rho0_diag.
void add_Hartree_to_H(
    MatrixC& H,
    const Eigen::VectorXd& rho_diag, const Eigen::VectorXd& rho0_diag,
    const Eigen::MatrixXd& V_ee, double e_charge, double hartree_factor,
    bool spin_on, int N_sites)
{
    Eigen::VectorXd rho_l_ind = hartree_factor * e_charge * (rho_diag - rho0_diag);
    Eigen::VectorXd V_hartree = V_ee * rho_l_ind;
    for (int i = 0; i < N_sites; ++i) {
        std::complex<double> v(e_charge * V_hartree(i), 0.0);
        H(i, i) += v;
        if (spin_on)
            H(i + N_sites, i + N_sites) += v;
    }
}

// B_ind_z from discrete curl of A_ind on 2D lattice: B_z = (∂A_y/∂x - ∂A_x/∂y).
// Uses bonds and points_2d; returns per-site B_ind_z (a.u.).
Eigen::VectorXd compute_B_ind_z_2d(TimeTonianSolver* s) {
    const int N_sites = s->N_sites;
    Eigen::VectorXd B_ind_z = Eigen::VectorXd::Zero(N_sites);
    if (s->points_2d.size() != static_cast<size_t>(N_sites) || s->bonds.empty() || s->a_bond <= 0.0)
        return B_ind_z;

    Eigen::VectorXd sum_curl(N_sites);
    Eigen::VectorXd neighbor_count(N_sites);
    sum_curl.setZero();
    neighbor_count.setZero();

    for (size_t b = 0; b < s->bonds.size(); ++b) {
        const int i = s->bonds[b].first;
        const int j = s->bonds[b].second;
        if (i < 0 || j < 0 || i >= N_sites || j >= N_sites) continue;
        double xi = s->points_2d[i][0], yi = s->points_2d[i][1];
        double xj = s->points_2d[j][0], yj = s->points_2d[j][1];
        double dx = xj - xi, dy = yj - yi;
        // (∂A_y/∂x - ∂A_x/∂y) approximated from bond (i,j)
        double circ = (s->A_ind_y(j) - s->A_ind_y(i)) * dx - (s->A_ind_x(j) - s->A_ind_x(i)) * dy;
        sum_curl(i) += circ;
        neighbor_count(i) += 1.0;
    }

    const double a2 = s->a_bond * s->a_bond;
    for (int i = 0; i < N_sites; ++i) {
        if (neighbor_count(i) > 0.0 && a2 > 0.0)
            B_ind_z(i) = sum_curl(i) / (neighbor_count(i) * a2);
    }
    return B_ind_z;
}

// Add Zeeman term μ_B σ·B to diagonal: spin-up +μ_B B_z, spin-down −μ_B B_z (σ_z convention).
void add_Zeeman_to_H(MatrixC& H, const Eigen::VectorXd& B_total_z, bool spin_on, int N_sites, double au_mu_B) {
    for (int i = 0; i < N_sites; ++i) {
        double zeeman = au_mu_B * B_total_z(i);
        H(i, i) += std::complex<double>(zeeman, 0.0);
        if (spin_on)
            H(i + N_sites, i + N_sites) += std::complex<double>(-zeeman, 0.0);
    }
}

// Ensure equilibrium bond currents J_l_eq are computed once (side effect on solver).
void ensure_J_l_eq_computed(TimeTonianSolver* s) {
    if (s->J_l_eq_computed)
        return;
    const int N = s->N_sites;
    const int N_mat = s->N_mat;
    const std::complex<double> im(0.0, 1.0);
    const double eoh = s->e_charge / s->hbar;

    MatrixC H_eq = s->H0;
    Eigen::VectorXd Vext0 = s->get_potential(0.0);
    add_Vext_to_H(H_eq, Vext0, s->spin_on, N);

    MatrixC J_x_eq(N_mat, N_mat), J_y_eq(N_mat, N_mat);
    compute_current_operators(H_eq, s->P_x, s->P_y, eoh, J_x_eq, J_y_eq);

    s->J_l_x_eq.resize(N);
    s->J_l_y_eq.resize(N);
    compute_bond_currents(s->rho0, J_x_eq, J_y_eq, s->bonds, s->spin_on, N, s->J_l_x_eq, s->J_l_y_eq);
    s->J_l_eq_computed = true;
}

// Build full H for time t and density rho (TB with phases when self-consistent, then + Vext + Hartree).
// Does not update H_prev (caller does that in RHS if needed).
void build_H_for_time(TimeTonianSolver* s, const MatrixC& rho, double t, MatrixC& H_out) {
    const int N_sites = s->N_sites;
    const int N_mat   = s->N_mat;
    const bool spin   = s->spin_on;
    const double eoh  = s->e_charge / s->hbar;
    Eigen::VectorXd Vext = s->get_potential(t);

    const bool can_use_phases_2d = s->self_consistent_on && !s->use_ssh_phases && !s->bonds.empty() &&
        static_cast<int>(s->points_2d.size()) == N_sites &&
        static_cast<int>(s->phi_ext.size()) == static_cast<int>(s->bonds.size()) &&
        s->H_prev.rows() == N_mat;

    const bool can_use_phases_1d = s->self_consistent_on && s->use_ssh_phases && !s->bonds.empty() &&
        static_cast<int>(s->phi_ext.size()) == static_cast<int>(s->bonds.size()) &&
        s->H_prev.rows() == N_mat;

    if (can_use_phases_2d) {
        MatrixC J_x(N_mat, N_mat), J_y(N_mat, N_mat);
        compute_current_operators(s->H_prev, s->P_x, s->P_y, eoh, J_x, J_y);

        Eigen::VectorXd J_l_x(N_sites), J_l_y(N_sites);
        compute_bond_currents(rho, J_x, J_y, s->bonds, spin, N_sites, J_l_x, J_l_y);

        ensure_J_l_eq_computed(s);

        Eigen::VectorXd J_l_x_ind = J_l_x - s->J_l_x_eq;
        Eigen::VectorXd J_l_y_ind = J_l_y - s->J_l_y_eq;
        s->A_ind_x.noalias() = (s->au_mu_0 / s->area_2d) * (s->V_ee * J_l_x_ind);
        s->A_ind_y.noalias() = (s->au_mu_0 / s->area_2d) * (s->V_ee * J_l_y_ind);

        if (s->phi_ind.size() != s->bonds.size())
            s->phi_ind.resize(s->bonds.size());
        if (s->combined_phase.size() != s->bonds.size())
            s->combined_phase.resize(s->bonds.size());

        for (size_t b = 0; b < s->bonds.size(); ++b) {
            const int i = s->bonds[b].first;
            const int j = s->bonds[b].second;
            double dx = s->points_2d[j][0] - s->points_2d[i][0];
            double dy = s->points_2d[j][1] - s->points_2d[i][1];
            double A_avg_x = (s->A_ind_x(i) + s->A_ind_x(j)) * 0.5;
            double A_avg_y = (s->A_ind_y(i) + s->A_ind_y(j)) * 0.5;
            s->phi_ind[b] = (s->e_charge / s->hbar) * (A_avg_x * dx + A_avg_y * dy);
            s->combined_phase[b] = s->phi_ext[b] + s->phi_ind[b];
        }

        H_out = TB_hamiltonian_from_points_with_phases(
            s->points_2d, s->a_bond, s->t_hop, s->bonds, s->combined_phase, spin, 1e-3);
    } else if (can_use_phases_1d) {
        // 1D SSH: induced phase only (no external B). Phase from A_ind_x and x_pos.
        MatrixC J_x(N_mat, N_mat), J_y(N_mat, N_mat);
        compute_current_operators(s->H_prev, s->P_x, s->P_y, eoh, J_x, J_y);

        Eigen::VectorXd J_l_x(N_sites), J_l_y(N_sites);
        compute_bond_currents(rho, J_x, J_y, s->bonds, spin, N_sites, J_l_x, J_l_y);

        ensure_J_l_eq_computed(s);

        Eigen::VectorXd J_l_x_ind = J_l_x - s->J_l_x_eq;
        s->A_ind_x.noalias() = (s->au_mu_0 / s->area_2d) * (s->V_ee * J_l_x_ind);
        // A_ind_y unused for 1D; leave zeroed

        if (s->phi_ind.size() != s->bonds.size())
            s->phi_ind.resize(s->bonds.size());
        if (s->combined_phase.size() != s->bonds.size())
            s->combined_phase.resize(s->bonds.size());

        for (size_t b = 0; b < s->bonds.size(); ++b) {
            const int i = s->bonds[b].first;
            const int j = s->bonds[b].second;
            double dx = s->x_pos(j) - s->x_pos(i);
            double A_avg_x = (s->A_ind_x(i) + s->A_ind_x(j)) * 0.5;
            s->phi_ind[b] = (s->e_charge / s->hbar) * A_avg_x * dx;
            s->combined_phase[b] = s->phi_ext[b] + s->phi_ind[b];  // phi_ext = 0 for SSH
        }

        MatrixC H_ssh = TB_hamiltonian_SSH_with_phases(
            N_sites, s->t_hop, s->t_hop2, s->bonds, s->combined_phase);
        if (spin) {
            H_out.resize(N_mat, N_mat);
            H_out.block(0, 0, N_sites, N_sites) = H_ssh;
            H_out.block(N_sites, N_sites, N_sites, N_sites) = H_ssh;
        } else {
            H_out = H_ssh;
        }
    } else {
        H_out = s->H0;
    }

    add_Vext_to_H(H_out, Vext, spin, N_sites);

    if (s->coulomb_on) {
        Eigen::VectorXd rho_diag = rho_diag_from_rho(rho, N_sites, spin);
        const double hartree_factor = spin ? 1.0 : 2.0;
        add_Hartree_to_H(H_out, rho_diag, s->rho0_diag, s->V_ee, s->e_charge, hartree_factor, spin, N_sites);
    }

    // Zeeman diagonal: μ_B σ·B; B = (zeeman_use_external ? B_ext : 0) + (zeeman_use_induced ? B_ind : 0)
    if (spin) {
        Eigen::VectorXd B_total_z = Eigen::VectorXd::Zero(N_sites);
        if (s->zeeman_use_external && s->B_ext_on)
            B_total_z.array() += s->B_ext_z;
        if (s->zeeman_use_induced && s->self_consistent_on && !s->use_ssh_phases && static_cast<int>(s->points_2d.size()) == N_sites)
            B_total_z += compute_B_ind_z_2d(s);
        add_Zeeman_to_H(H_out, B_total_z, true, N_sites, s->au_mu_B);
    }
}

} // namespace

void add_Zeeman_diagonal(MatrixC& H, double B_z, int N_sites, bool spin_on, double au_mu_B) {
    const double zeeman = au_mu_B * B_z;
    for (int i = 0; i < N_sites; ++i) {
        H(i, i) += std::complex<double>(zeeman, 0.0);
        if (spin_on)
            H(i + N_sites, i + N_sites) += std::complex<double>(-zeeman, 0.0);
    }
}

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
void RhoObserver::operator()(
    const std::vector<std::complex<double>> &rho_vec,
    double t)
{
    counter++;
    const int N_sites = solver->N_sites;
    const int N_mat   = solver->N_mat;
    const bool spin   = solver->spin_on;

    hist.time.push_back(t);
    hist.diag.emplace_back(N_sites);

    for (int i = 0; i < N_sites; ++i) {
        double occ = std::real(rho_vec[i * N_mat + i]);
        if (spin)
            occ += std::real(rho_vec[(i + N_sites) * N_mat + (i + N_sites)]);
        hist.diag.back()[i] = occ;
    }

    MatrixC rho_k(N_mat, N_mat);
    for (int i = 0; i < N_mat; ++i)
        for (int j = 0; j < N_mat; ++j)
            rho_k(i, j) = rho_vec[i * N_mat + j];
    hist.rho_full.push_back(rho_k);

    // H(t, rho) same as RHS so stored J and A_ind match dynamics
    MatrixC H_t(N_mat, N_mat);
    build_H_for_time(solver, rho_k, t, H_t);

    const double eoh = solver->e_charge / solver->hbar;
    MatrixC J_x(N_mat, N_mat), J_y(N_mat, N_mat);
    compute_current_operators(H_t, solver->P_x, solver->P_y, eoh, J_x, J_y);

    hist.J_x.push_back(std::real((rho_k * J_x).trace()));
    hist.J_y.push_back(std::real((rho_k * J_y).trace()));

    if (solver->self_consistent_on && !solver->bonds.empty() &&
        solver->bonds.size() == solver->phi_ext.size()) {
        ensure_J_l_eq_computed(solver);  // in case build_H_for_time didn't (e.g. first step)
        Eigen::VectorXd J_l_x(N_sites), J_l_y(N_sites);
        compute_bond_currents(rho_k, J_x, J_y, solver->bonds, spin, N_sites, J_l_x, J_l_y);
        Eigen::VectorXd A_x = (solver->au_mu_0 / solver->area_2d) * (solver->V_ee * (J_l_x - solver->J_l_x_eq));
        Eigen::VectorXd A_y = (solver->au_mu_0 / solver->area_2d) * (solver->V_ee * (J_l_y - solver->J_l_y_eq));
        hist.A_ind_x.push_back(std::vector<double>(A_x.data(), A_x.data() + N_sites));
        hist.A_ind_y.push_back(std::vector<double>(A_y.data(), A_y.data() + N_sites));
    } else if (solver->self_consistent_on) {
        hist.A_ind_x.push_back(std::vector<double>(static_cast<size_t>(N_sites), 0.0));
        hist.A_ind_y.push_back(std::vector<double>(static_cast<size_t>(N_sites), 0.0));
    }

    if (counter % 5000 == 0)
        std::cout << "\r t = " << (t * AU_TIME_TO_FS) << " fs" << std::flush;
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
    const int N_mat_local = N_mat;

    for (int i = 0, k = 0; i < N_mat_local; ++i)
        for (int j = 0; j < N_mat_local; ++j, ++k)
            rho_tmp(i, j) = rho_vec[k];

    if (self_consistent_on && H_prev.rows() == 0)
        H_prev = H0;

    build_H_for_time(this, rho_tmp, t, H_tmp);

    if (self_consistent_on && H_tmp.rows() > 0)
        H_prev = H_tmp;

    comm_tmp.noalias() = H_tmp * rho_tmp;
    comm_tmp.noalias() -= rho_tmp * H_tmp;

    const std::complex<double> minus_i(0.0, -1.0);
    for (int i = 0, k = 0; i < N_mat_local; ++i)
        for (int j = 0; j < N_mat_local; ++j, ++k) {
            drho_dt_tmp(i, j) = minus_i * comm_tmp(i, j) - (gamma * 0.5) * (rho_tmp(i, j) - rho0(i, j));
            drho_dt_vec[k] = drho_dt_tmp(i, j);
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

    const bool self_consistent_2d = p.self_consistent_phase && p.two_dim && !p.xl_2D.empty();
    const bool self_consistent_1d_ssh = p.self_consistent_phase && !p.two_dim &&
        (p.lattice == "ssh" || p.lattice == "chain") && !p.xl_1D.empty();

    solver.self_consistent_on = self_consistent_2d || self_consistent_1d_ssh;

    if (solver.self_consistent_on) {
        solver.bonds = pot.get_bonds();
        solver.phi_ext.resize(solver.bonds.size(), 0.0);
        solver.t_hop = p.t1;
        solver.a_bond = p.a;
        solver.au_mu_0 = p.au_mu_0;
        solver.area_2d = p.area_2d;  // 1.0 for non-graphene (params)
        solver.J_l_x.resize(N_sites);
        solver.J_l_y.resize(N_sites);
        solver.A_ind_x.resize(N_sites);
        solver.A_ind_y.resize(N_sites);
        solver.J_l_x.setZero();
        solver.J_l_y.setZero();

        if (self_consistent_1d_ssh) {
            solver.use_ssh_phases = true;
            solver.t_hop2 = p.t2;
            // no points_2d; phi_ext stays 0 (no external B on SSH)
        } else {
            solver.use_ssh_phases = false;
            solver.points_2d = p.xl_2D;
            if (p.B_ext) {
                auto peierls = pot.build_peierls_phases(pot.compute_Bz());
                for (size_t k = 0; k < solver.bonds.size() && k < peierls.size(); ++k)
                    solver.phi_ext[k] = peierls[k].phi;
            }
        }
    }

    if (solver.coulomb_on) {
        solver.rho0_diag = rho_diag_from_rho(rho_initial_l, N_sites, solver.spin_on);
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

    solver.au_mu_B = 0.5;  // Bohr magneton in a.u.
    solver.B_ext_on = p.B_ext;
    solver.B_ext_z = solver.B_ext_on ? pot.compute_Bz() : 0.0;
    solver.zeeman_use_external = p.zeeman_external;
    solver.zeeman_use_induced = p.zeeman_induced;

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
    const int N_mat  = Hc.rows();
    if (history.rho_full.empty() || history.time.size() != history.rho_full.size()) {
        std::cerr << "compute_current_from_history: empty or mismatched history, skipping.\n";
        return;
    }

    const double e_over_hbar = static_cast<double>(p.e) / static_cast<double>(p.au_hbar);
    Eigen::VectorXd rho0_diag = rho_diag_from_rho(history.rho_full[0], N_sites, p.spin_on);

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

    std::ofstream fout(out_dir + "/current_time_evolution.txt");
    if (!fout) {
        std::cerr << "compute_current_from_history: could not open current_time_evolution.txt\n";
        return;
    }
    fout << "# t  Jx  Jy\n";

    MatrixC H_t(N_mat, N_mat);
    MatrixC J_x(N_mat, N_mat), J_y(N_mat, N_mat);
    const double hartree_factor = p.spin_on ? 1.0 : 2.0;

    for (size_t k = 0; k < history.time.size(); ++k) {
        const double t = history.time[k];
        const MatrixC &rho_k = history.rho_full[k];

        H_t = Hc;
        add_Vext_to_H(H_t, pot.get_potential(t, mode), p.spin_on, N_sites);
        if (p.coulomb_on) {
            Eigen::VectorXd rho_diag = rho_diag_from_rho(rho_k, N_sites, p.spin_on);
            add_Hartree_to_H(H_t, rho_diag, rho0_diag, p.V_ee, p.e, hartree_factor, p.spin_on, N_sites);
        }

        compute_current_operators(H_t, P_x, P_y, e_over_hbar, J_x, J_y);
        fout << t << " " << std::real((rho_k * J_x).trace()) << " " << std::real((rho_k * J_y).trace()) << "\n";
    }
}
