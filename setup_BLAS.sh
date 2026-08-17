#!/bin/bash
set -e  # stop on any error
echo ""
echo "================================================"
echo "   TBTimeEvolution - OpenBLAS Dependency Setup"
echo "   (AMD-optimized alternative to the MKL setup)"
echo "================================================"
echo ""

# ---- Core build tools + apt packages ----
echo ">>> [1/5] Installing core build tools and apt packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    cmake \
    g++ \
    build-essential \
    libeigen3-dev \
    libgsl-dev \
    libcli11-dev \
    zlib1g-dev \
    git \
    wget \
    ca-certificates \
    apt-transport-https
echo "    Done."

# ---- OpenBLAS + LAPACKE ----
# The Ubuntu OpenBLAS packages are built with DYNAMIC_ARCH, so at runtime they
# auto-detect the CPU and dispatch the Zen/Zen2/Zen3/Zen4 kernels on AMD chips.
# We also install liblapacke-dev because Eigen's EIGEN_USE_LAPACKE path needs
# the LAPACKE C interface headers/libs.
echo ""
echo ">>> [2/5] Installing OpenBLAS + LAPACKE..."
sudo apt-get install -y \
    libopenblas-dev \
    liblapacke-dev \
    libgfortran5

# Prefer the OpenMP-threaded OpenBLAS variant when update-alternatives offers it,
# so BLAS threading cooperates with the app's own OpenMP. Non-fatal if absent.
if update-alternatives --list libopenblas.so.0-x86_64-linux-gnu >/dev/null 2>&1; then
    OMP_VARIANT=$(update-alternatives --list libopenblas.so.0-x86_64-linux-gnu \
        | grep -i openmp | head -n1 || true)
    if [ -n "$OMP_VARIANT" ]; then
        sudo update-alternatives --set libopenblas.so.0-x86_64-linux-gnu "$OMP_VARIANT" || true
        echo "    Selected OpenMP OpenBLAS variant: $OMP_VARIANT"
    fi
fi
echo "    Done."

# ---- cnpy ----
echo ""
echo ">>> [3/5] Installing cnpy (from source)..."
if [ ! -d /tmp/cnpy ]; then
    git clone https://github.com/rogersce/cnpy.git /tmp/cnpy
fi
cd /tmp/cnpy && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
cd - >/dev/null
echo "    Done."

# ---- toml++ (header only) ----
echo ""
echo ">>> [4/5] Installing toml++ headers..."
# Try apt first, fall back to direct header download
if sudo apt-get install -y libtomlplusplus-dev 2>/dev/null; then
    echo "    Installed via apt."
else
    echo "    apt failed, installing single header manually..."
    sudo mkdir -p /usr/local/include/toml++
    sudo wget -q https://raw.githubusercontent.com/marzer/tomlplusplus/master/toml.hpp \
        -O /usr/local/include/toml++/toml.hpp
    sudo cp /usr/local/include/toml++/toml.hpp /usr/local/include/toml.hpp
    echo "    Done (installed to /usr/local/include)."
fi

# ---- OpenMP (bundled with g++, just verify) ----
echo ""
echo ">>> [5/5] Verifying OpenMP..."
echo '#include <omp.h>
int main() { return 0; }' > /tmp/omp_test.cpp
g++ -fopenmp /tmp/omp_test.cpp -o /tmp/omp_test && echo "    OpenMP OK." || echo "    WARNING: OpenMP not found!"
rm -f /tmp/omp_test.cpp /tmp/omp_test

echo ""
echo "================================================"
echo "   All OpenBLAS dependencies installed!"
echo "================================================"
echo ""
echo "To build the OpenBLAS variant:"
echo ""
echo "  ./build_BLAS.sh"
echo ""
echo "This produces the 'sim_blas' binary. Run with:"
echo ""
echo "  ./sim_blas <config.toml>"
echo ""
echo "The original MKL workflow (setup.sh / build_mkl.sh -> sim_mkl)"
echo "is unchanged and still available."
echo ""
