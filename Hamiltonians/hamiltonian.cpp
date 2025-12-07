#include "hamiltonian.hpp"
#include "../DensityMatrix/Density.hpp"

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
