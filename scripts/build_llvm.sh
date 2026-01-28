#!/bin/bash
#===----------------------------------------------------------------------===//
# LLVM/MLIR Build Script for Ascend-MLIR
#===----------------------------------------------------------------------===//

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLVM_SRC="${PROJECT_ROOT}/externals/llvm-project"
LLVM_BUILD="${LLVM_SRC}/build"

BUILD_TYPE="${BUILD_TYPE:-Release}"
NUM_JOBS="${NUM_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

echo "=== Building LLVM/MLIR ==="
echo "Source:     ${LLVM_SRC}"
echo "Build:      ${LLVM_BUILD}"
echo "Build Type: ${BUILD_TYPE}"
echo "Jobs:       ${NUM_JOBS}"

# Initialize submodule if needed
if [ ! -d "${LLVM_SRC}/llvm" ]; then
    echo "Initializing LLVM submodule..."
    cd "${PROJECT_ROOT}/externals"
    git clone -n https://github.com/llvm/llvm-project.git
    cd llvm-project && git checkout 2078da43e25a4623cab2d0d60decddf709aaea28 && cd ..  # llvm 21.1.8
fi

# Create build directory
mkdir -p "${LLVM_BUILD}"
cd "${LLVM_BUILD}"

# Configure
cmake -G Ninja ../llvm \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMLIR_PYTHON_BINDINGS_LIBRARY=nanobind \
    -DPython3_EXECUTABLE="$(which python3)" \
    -DLLVM_ENABLE_LIBEDIT=OFF

# Build
BUILD_START_TIME=$(date +%s)
echo "Building LLVM/MLIR..."
cmake --build . --target all -j${NUM_JOBS}
BUILD_END_TIME=$(date +%s)
BUILD_DURATION=$((BUILD_END_TIME - BUILD_START_TIME))
echo "Build completed in ${BUILD_DURATION}s ($(printf '%02d:%02d:%02d' $((BUILD_DURATION/3600)) $((BUILD_DURATION%3600/60)) $((BUILD_DURATION%60))))"

echo "=== LLVM/MLIR build completed ==="
echo "LLVM_DIR: ${LLVM_BUILD}/lib/cmake/llvm"
echo "MLIR_DIR: ${LLVM_BUILD}/lib/cmake/mlir"
