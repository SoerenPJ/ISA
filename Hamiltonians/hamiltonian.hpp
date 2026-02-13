#pragma once

#include <eigen3/Eigen/Dense>
#include <complex>
#include <vector>
#include <array>
#include <utility>

#include "../DensityMatrix/Density.hpp"

using namespace std;
using namespace Eigen;

// Tight-binding Hamiltonian returning complex RowMajor matrix
MatrixC TB_hamiltonian(int N, double t1, double t2);

// Build nearest-neighbor tight-binding Hamiltonian from 2D points (graphene etc.)
MatrixC TB_hamiltonian_from_points(
    const std::vector<std::array<double,2>>& points,
    double bond_length,
    double t,
    double tolerance_rel = 1e-3
);

// Bond index pair (i, j), i < j
using Bond = std::pair<int, int>;

// Build TB from points with Peierls phases per bond. phase_per_bond.size() == bonds.size().
// For spinful: spin_up gets +phi, spin_down gets -phi (2*N_sites x 2*N_sites).
MatrixC TB_hamiltonian_from_points_with_phases(
    const std::vector<std::array<double,2>>& points,
    double bond_length,
    double t,
    const std::vector<Bond>& bonds,
    const std::vector<double>& phase_per_bond,
    bool spinful,
    double tolerance_rel = 1e-3
);

MatrixC spin_tonian(const MatrixC& H);