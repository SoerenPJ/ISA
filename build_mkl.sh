#!/bin/bash

MKLROOT=/opt/intel/oneapi/mkl/latest

echo "Building sim_mkl with MKL..."
echo "MKLROOT = $MKLROOT"

export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8

g++ \
  -std=c++20 \
  -O3 -march=native -funroll-loops \
  -ftree-vectorize -frename-registers \
  -fopenmp \
  -DEIGEN_USE_MKL_ALL \
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
