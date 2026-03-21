#!/bin/bash
set -e  # stop on any error
echo ""
echo "================================================"
echo "   TBTimeEvolution - UCloud Dependency Setup"
echo "================================================"
echo ""

# ---- Core build tools + apt packages ----
echo ">>> [1/6] Installing core build tools and apt packages..."
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

# ---- Intel MKL ----
echo ""
echo ">>> [2/6] Installing Intel OneAPI MKL..."
wget -q -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor \
    | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt-get update -qq
sudo apt-get install -y intel-oneapi-mkl-devel
source /opt/intel/oneapi/setvars.sh --force
echo "    Done."

# ---- cnpy ----
echo ""
echo ">>> [3/6] Installing cnpy (from source)..."
git clone https://github.com/rogersce/cnpy.git /tmp/cnpy
cd /tmp/cnpy && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
cd /work
echo "    Done."

# ---- toml++ (header only) ----
echo ""
echo ">>> [4/6] Installing toml++ headers..."
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
echo ">>> [5/6] Verifying OpenMP..."
echo '#include <omp.h>
int main() { return 0; }' > /tmp/omp_test.cpp
g++ -fopenmp /tmp/omp_test.cpp -o /tmp/omp_test && echo "    OpenMP OK." || echo "    WARNING: OpenMP not found!"
rm -f /tmp/omp_test.cpp /tmp/omp_test

# ---- Summary ----
echo ""
echo ">>> [6/6] Sourcing MKL environment..."
source /opt/intel/oneapi/setvars.sh --force
echo "    MKL environment active."

echo ""
echo "================================================"
echo "   All dependencies installed successfully!"
echo "================================================"
echo ""
echo "To build your project:"
echo ""
echo "  source /opt/intel/oneapi/setvars.sh"
echo "  cd /work/TB_simulation"
echo "  mkdir -p build && cd build"
echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "  make -j\$(nproc)"
echo ""
echo "NOTE: LTO is enabled in your CMakeLists.txt."
echo "If you have less than 16GB RAM allocated, disable it"
echo "by changing the lto_ok block to: if(FALSE)"
echo ""
apt.repos.intel.com
