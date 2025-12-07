#pragma once

#include <eigen3/Eigen/Dense>
#include <complex>
#include <vector>

// We need MatrixC available here
#include "../DensityMatrix/Density.hpp"

using namespace std;
using namespace Eigen;

// Tight-binding Hamiltonian returning complex RowMajor matrix
MatrixC TB_hamiltonian(int N, double t1, double t2);
