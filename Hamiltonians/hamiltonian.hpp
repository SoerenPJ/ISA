#pragma once

#include <eigen3/Eigen/Dense>
#include <complex>
#include <vector>
#include <array>

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

MatrixC spin_tonian(const MatrixC& H);