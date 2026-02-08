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
