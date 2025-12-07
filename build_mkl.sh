#!/bin/bash

# ==========================================================
#   BUILD SCRIPT — Eigen + Intel MKL + OpenMP + O3
# ==========================================================

# MKL root directory
MKLROOT=/opt/intel/oneapi/mkl/latest
ONEAPIROOT=/opt/intel/oneapi

echo "⚡ Building ISA SIM with MKL acceleration..."
echo "MKLROOT = $MKLROOT"

g++ \
  -O3 -march=native -funroll-loops -fopenmp \
  -I/usr/include/eigen3 \
  -I. \
  -Iparams \
  -IObservables \
  -IHamiltonians \
  -IDensityMatrix \
  -I${MKLROOT}/include \
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
    echo "✅ MKL build successful!"
    echo "Run with: ./sim_mkl"
else
    echo "❌ MKL build FAILED."
fi
