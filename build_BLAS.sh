#!/bin/bash

echo "Building sim_blas with OpenBLAS..."

# ---- Thread configuration ----
# Eigen is compiled with EIGEN_DONT_PARALLELIZE, so all BLAS-level threading is
# handled by OpenBLAS itself. Tune OPENBLAS_NUM_THREADS to the target machine.
export OPENBLAS_NUM_THREADS=6
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

# ---- Compile ----
# Same sources and optimization flags as the MKL build, but Eigen is pointed at
# a generic BLAS/LAPACKE backend (OpenBLAS) instead of MKL:
#   EIGEN_USE_BLAS     -> matrix products (GEMM etc.) dispatch to OpenBLAS
#   EIGEN_USE_LAPACKE  -> dense decompositions dispatch to LAPACKE/OpenBLAS
g++ \
  -std=c++20 \
  -O3 \
  -march=native \
  -funroll-loops \
  -ftree-vectorize \
  -frename-registers \
  -fopenmp \
  -DEIGEN_USE_BLAS \
  -DEIGEN_USE_LAPACKE \
  -DEIGEN_DONT_PARALLELIZE \
  -I/usr/include/eigen3 \
  -I. \
  -Iparams \
  -IDensityMatrix \
  -IHamiltonians \
  -IObservables \
  -Iexternal/toml++ \
  main.cpp \
  DensityMatrix/Density.cpp \
  Hamiltonians/hamiltonian.cpp \
  Hamiltonians/hubbard.cpp \
  Hamiltonians/potential.cpp \
  Observables/observables.cpp \
  params/params.cpp \
  -llapacke \
  -lopenblas \
  -lgfortran \
  -lpthread \
  -lm \
  -ldl \
  -o sim_blas

if [ $? -eq 0 ]; then
    echo "✅ OpenBLAS build successful."
    echo "Run with: ./sim_blas config.toml"
else
    echo "❌ OpenBLAS build failed."
fi
