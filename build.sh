#!/bin/bash

# ============================
#   BUILD SCRIPT FOR ISA SIM
# ============================

echo "🚀 Building project with full optimization..."

g++ \
  -O3 -march=native -funroll-loops -fopenmp \
  -I /usr/include/eigen3 \
  -I . \
  -I DensityMatrix \
  -I Hamiltonians \
  -I Observables \
  -I params \
  main.cpp \
  DensityMatrix/Density.cpp \
  Hamiltonians/hamiltonian.cpp \
  Hamiltonians/potential.cpp \
  Observables/observables.cpp \
  params/params.cpp \
  -o sim

if [ $? -eq 0 ]; then
    echo "✅ Build successful! Run with: ./sim"
else
    echo "❌ Build failed."
fi
