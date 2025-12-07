#!/bin/bash

# ==========================================================
#   BUILD SCRIPT — Eigen + Intel MKL + OpenMP + O3
# ==========================================================

MKLROOT=/opt/intel/oneapi/mkl/latest

echo "Building sim_mkl with MKL..."
echo "MKLROOT = $MKLROOT"

g++ \
  -O3 -march=native -funroll-loops -fopenmp \
  -DEIGEN_USE_MKL_ALL \
  -I/usr/include/eigen3 \
  -I. \
  -Iparams \
  -IDensityMatrix \
  -IHamiltonians \
  -IObservables \
  main.cpp \
  DensityMatrix/Density.cpp \
  Hamiltonians/hamiltonian.cpp \
  Hamiltonians/potential.cpp \
  Observables/observables.cpp \
  params/params.cpp \
  -L${MKLROOT}/lib/intel64 \
  -lmkl_intel_lp64 \
  -lmkl_core \
  -lmkl_gnu_thread \
  -lgomp \
  -lpthread \
  -lm \
  -ldl \
  -o sim_mkl

if [ $? -eq 0 ]; then
    echo "✅ MKL build successful. Run with: ./sim_mkl"
else
    echo "❌ MKL build failed."
fi

