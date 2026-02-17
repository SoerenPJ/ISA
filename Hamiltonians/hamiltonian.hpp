#pragma once

#include <eigen3/Eigen/Dense>
#include <complex>
#include <vector>
#include <array>
#include <utility>

#include "../DensityMatrix/Density.hpp"

using namespace std;
using namespace Eigen;

// Bond index pair (i, j), i < j
using Bond = std::pair<int, int>;

// Tight-binding Hamiltonian returning complex RowMajor matrix
MatrixC TB_hamiltonian(int N, double t1, double t2);

// SSH chain with Peierls phases per bond. bonds ordered (0,1),(1,2),...; phase_per_bond[b] for bonds[b].
// No external B: use for induced phase only (phi_ext = 0).
MatrixC TB_hamiltonian_SSH_with_phases(
    int N,
    double t1,
    double t2,
    const std::vector<Bond>& bonds,
    const std::vector<double>& phase_per_bond
);

// Build nearest-neighbor tight-binding Hamiltonian from 2D points (graphene etc.)
MatrixC TB_hamiltonian_from_points(
    const std::vector<std::array<double,2>>& points,
    double bond_length,
    double t,
    double tolerance_rel = 1e-3
);

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