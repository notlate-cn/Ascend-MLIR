#!/bin/bash
#===----------------------------------------------------------------------===//
# Ascend-MLIR Build Script
#===----------------------------------------------------------------------===//

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default configuration
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${PROJECT_ROOT}/build"
INSTALL_DIR="${PROJECT_ROOT}/install"
LLVM_BUILD_DIR="${PROJECT_ROOT}/externals/llvm-project/build"
NUM_JOBS="${NUM_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Ascend-MLIR project.

Options:
    --build-llvm        Build LLVM/MLIR from source
    --build-stablehlo   Build StableHLO from source
    --build-pyasc       Build PyAsc from source
    --build-deps        Build all dependencies
    --build-project     Build Ascend-MLIR project only
    --build-all         Build dependencies and project
    --build-tests       Build and run tests
    --clean             Clean build directory
    --release           Build in Release mode (default)
    --debug             Build in Debug mode
    --jobs N            Number of parallel jobs (default: auto)
    --help              Show this help message

Environment Variables:
    BUILD_TYPE          Build type: Release or Debug
    BUILD_DIR           Build directory path
    NUM_JOBS            Number of parallel jobs

Examples:
    $0 --build-all           # Build everything
    $0 --build-llvm          # Build LLVM/MLIR only
    $0 --build-project       # Build Ascend-MLIR only
    $0 --clean --build-all   # Clean and rebuild everything
EOF
}

init_submodules() {
    print_info "Initializing git submodules..."
    cd "${PROJECT_ROOT}"
    git submodule update --init --recursive
}

build_llvm() {
    print_info "Building LLVM/MLIR..."

    local LLVM_SRC="${PROJECT_ROOT}/externals/llvm-project"

    if [ ! -d "${LLVM_SRC}" ]; then
        print_error "LLVM source not found. Please run: git submodule update --init"
        exit 1
    fi

    mkdir -p "${LLVM_BUILD_DIR}"
    cd "${LLVM_BUILD_DIR}"

    cmake -G Ninja ../llvm \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF

    cmake --build . --target all -j${NUM_JOBS}

    print_info "LLVM/MLIR build completed."
}

build_stablehlo() {
    print_info "Building StableHLO..."

    local STABLEHLO_SRC="${PROJECT_ROOT}/externals/stablehlo"
    local STABLEHLO_BUILD="${STABLEHLO_SRC}/build"

    if [ ! -d "${STABLEHLO_SRC}" ]; then
        print_error "StableHLO source not found. Please run: git submodule update --init"
        exit 1
    fi

    mkdir -p "${STABLEHLO_BUILD}"
    cd "${STABLEHLO_BUILD}"

    cmake -G Ninja .. \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm" \
        -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir"

    cmake --build . --target all -j${NUM_JOBS}

    print_info "StableHLO build completed."
}

build_pyasc() {
    print_info "Building PyAsc..."

    local PYASC_SRC="${PROJECT_ROOT}/externals/pyasc"

    if [ ! -d "${PYASC_SRC}" ]; then
        print_error "PyAsc source not found. Please run: git submodule update --init"
        exit 1
    fi

    # PyAsc build instructions depend on its actual build system
    # This is a placeholder that should be updated based on PyAsc's build system
    print_warn "PyAsc build: Please check PyAsc documentation for build instructions"

    print_info "PyAsc setup completed."
}

build_project() {
    print_info "Building Ascend-MLIR..."

    if [ ! -d "${LLVM_BUILD_DIR}" ]; then
        print_error "LLVM build not found. Please build LLVM first with --build-llvm"
        exit 1
    fi

    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    cmake -G Ninja "${PROJECT_ROOT}" \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm" \
        -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir"

    cmake --build . --target all -j${NUM_JOBS}

    print_info "Ascend-MLIR build completed."
}

build_tests() {
    print_info "Building and running tests..."

    cd "${BUILD_DIR}"
    cmake --build . --target check-afir -j${NUM_JOBS}

    print_info "Tests completed."
}

clean_build() {
    print_info "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    print_info "Clean completed."
}

# Parse command line arguments
BUILD_LLVM=false
BUILD_STABLEHLO=false
BUILD_PYASC=false
BUILD_PROJECT=false
BUILD_TESTS=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-llvm)
            BUILD_LLVM=true
            shift
            ;;
        --build-stablehlo)
            BUILD_STABLEHLO=true
            shift
            ;;
        --build-pyasc)
            BUILD_PYASC=true
            shift
            ;;
        --build-deps)
            BUILD_LLVM=true
            BUILD_STABLEHLO=true
            BUILD_PYASC=true
            shift
            ;;
        --build-project)
            BUILD_PROJECT=true
            shift
            ;;
        --build-all)
            BUILD_LLVM=true
            BUILD_STABLEHLO=true
            BUILD_PYASC=true
            BUILD_PROJECT=true
            shift
            ;;
        --build-tests)
            BUILD_TESTS=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute build steps
print_info "Build configuration:"
print_info "  Build Type: ${BUILD_TYPE}"
print_info "  Build Dir:  ${BUILD_DIR}"
print_info "  Jobs:       ${NUM_JOBS}"

if $CLEAN; then
    clean_build
fi

# Initialize submodules if any build is requested
if $BUILD_LLVM || $BUILD_STABLEHLO || $BUILD_PYASC; then
    init_submodules
fi

if $BUILD_LLVM; then
    build_llvm
fi

if $BUILD_STABLEHLO; then
    build_stablehlo
fi

if $BUILD_PYASC; then
    build_pyasc
fi

if $BUILD_PROJECT; then
    build_project
fi

if $BUILD_TESTS; then
    build_tests
fi

# If no options specified, show usage
if ! $BUILD_LLVM && ! $BUILD_STABLEHLO && ! $BUILD_PYASC && ! $BUILD_PROJECT && ! $BUILD_TESTS && ! $CLEAN; then
    print_warn "No build target specified."
    usage
fi

print_info "Done!"
