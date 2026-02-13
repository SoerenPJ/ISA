#include "hamiltonian.hpp"
#include "../DensityMatrix/Density.hpp"
#include <cmath>

// Build complex tight-binding Hamiltonian (RowMajor)
MatrixC TB_hamiltonian(int N, double t1, double t2)
{
    MatrixC H = MatrixC::Zero(N, N);

    for (int i = 0; i < N - 1; ++i)
    {
        double hopping = (i % 2 == 0) ? t1 : t2;

        H(i, i + 1) = std::complex<double>(hopping, 0.0);
        H(i + 1, i) = std::complex<double>(hopping, 0.0);
    }

    return H;
}

MatrixC TB_hamiltonian_from_points(
    const std::vector<std::array<double,2>>& points,
    double bond_length,
    double t,
    double tolerance_rel
)
{
    const int n = static_cast<int>(points.size());
    MatrixC H = MatrixC::Zero(n, n);

    const double tol = std::abs(bond_length) * std::abs(tolerance_rel);
    const double cutoff = bond_length + tol;
    const double cutoff2 = cutoff * cutoff;

    for (int i = 0; i < n; ++i) {
        const double xi = points[i][0];
        const double yi = points[i][1];
        for (int j = i + 1; j < n; ++j) {
            const double dx = xi - points[j][0];
            const double dy = yi - points[j][1];
            const double r2 = dx*dx + dy*dy;
            if (r2 <= cutoff2) {
                H(i, j) = std::complex<double>(t, 0.0);
                H(j, i) = std::complex<double>(t, 0.0);
            }
        }
    }

    return H;
}

MatrixC TB_hamiltonian_from_points_with_phases(
    const std::vector<std::array<double,2>>& points,
    double bond_length,
    double t,
    const std::vector<Bond>& bonds,
    const std::vector<double>& phase_per_bond,
    bool spinful,
    double tolerance_rel
)
{
    const int n = static_cast<int>(points.size());
    const double tol = std::abs(bond_length) * std::abs(tolerance_rel);
    const double cutoff = bond_length + tol;
    const double cutoff2 = cutoff * cutoff;

    if (phase_per_bond.size() != bonds.size())
        return MatrixC::Zero(spinful ? 2*n : n, spinful ? 2*n : n);

    if (!spinful) {
        MatrixC H = MatrixC::Zero(n, n);
        for (size_t idx = 0; idx < bonds.size(); ++idx) {
            int i = bonds[idx].first;
            int j = bonds[idx].second;
            if (i < 0 || j < 0 || i >= n || j >= n) continue;
            double dx = points[i][0] - points[j][0];
            double dy = points[i][1] - points[j][1];
            if (dx*dx + dy*dy > cutoff2) continue;
            double phi = phase_per_bond[idx];
            std::complex<double> phase = std::exp(std::complex<double>(0.0, phi));
            std::complex<double> hop = t * phase;
            H(i, j) = hop;
            H(j, i) = std::conj(hop);
        }
        return H;
    }

    // Spinful: up block +phi, down block -phi
    MatrixC H_spin = MatrixC::Zero(2*n, 2*n);
    for (size_t idx = 0; idx < bonds.size(); ++idx) {
        int i = bonds[idx].first;
        int j = bonds[idx].second;
        if (i < 0 || j < 0 || i >= n || j >= n) continue;
        double dx = points[i][0] - points[j][0];
        double dy = points[i][1] - points[j][1];
        if (dx*dx + dy*dy > cutoff2) continue;
        double phi = phase_per_bond[idx];
        std::complex<double> hop_up  = t * std::exp( 1i * phi);
        std::complex<double> hop_dn  = t * std::exp(-1i * phi);
        H_spin(i, j) = hop_up;
        H_spin(j, i) = std::conj(hop_up);
        H_spin(n + i, n + j) = hop_dn;
        H_spin(n + j, n + i) = std::conj(hop_dn);
    }
    return H_spin;
}

MatrixC spin_tonian(const MatrixC& H)
{
    const int n = H.rows();
    MatrixC H_spin = MatrixC::Zero(2*n, 2*n);

    H_spin.block(0,0,n,n) = H;
    H_spin.block(n,n,n,n) = H;

    return H_spin;
}
