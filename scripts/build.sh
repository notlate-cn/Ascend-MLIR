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
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${PROJECT_ROOT}/externals/llvm-project/build}"
NUM_JOBS="${NUM_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
COVERAGE="${COVERAGE:-false}"
OUTPUT_DIR="${PROJECT_ROOT}/output"

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
    --coverage          Enable code coverage collection
    --release           Build in Release mode (default)
    --debug             Build in Debug mode
    --llvm-build-dir    Path to LLVM build directory (default: externals/llvm-project/build)
    --jobs N            Number of parallel jobs (default: auto)
    --help              Show this help message

Environment Variables:
    BUILD_TYPE          Build type: Release or Debug
    BUILD_DIR           Build directory path
    LLVM_BUILD_DIR      LLVM build directory path
    NUM_JOBS            Number of parallel jobs
    COVERAGE            Enable code coverage collection (true/false)

Examples:
    $0 --build-all                           # Build everything
    $0 --build-llvm                          # Build LLVM/MLIR only
    $0 --build-project                       # Build Ascend-MLIR only
    $0 --llvm-build-dir /path/to/llvm/build  # Use external LLVM build
    $0 --clean --build-all                   # Clean and rebuild everything
    $0 --build-tests --coverage              # Build and run tests with coverage
EOF
}

init_submodules() {
    print_info "Initializing git submodules..."
    cd "${PROJECT_ROOT}"
    git submodule update --init --recursive
}

build_llvm() {
    print_info "Building LLVM/MLIR..."

    # Call the dedicated build_llvm.sh script
    "${SCRIPT_DIR}/build_llvm.sh"
}

build_stablehlo() {
    local start_time=$(date +%s)
    print_info "Building StableHLO..."

    local STABLEHLO_SRC="${PROJECT_ROOT}/externals/stablehlo"
    local STABLEHLO_BUILD="${STABLEHLO_SRC}/build"

    if [ ! -d "${STABLEHLO_SRC}" ]; then
        print_error "StableHLO source not found. Please run: git submodule update --init"
        exit 1
    fi

    # Derive MLIR_DIR from LLVM_BUILD_DIR
    local MLIR_CMAKE_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir"

    mkdir -p "${STABLEHLO_BUILD}"
    cd "${STABLEHLO_BUILD}"

    # Only need to pass MLIR_DIR, LLVM_DIR will be automatically discovered
    cmake -G Ninja .. \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DMLIR_DIR="${MLIR_CMAKE_DIR}"

    cmake --build . --target all -j${NUM_JOBS}

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    print_info "StableHLO build completed in ${duration}s ($(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60))))"
}

build_pyasc() {
    local start_time=$(date +%s)
    print_info "Building PyAsc..."

    local PYASC_SRC="${PROJECT_ROOT}/externals/pyasc"

    if [ ! -d "${PYASC_SRC}" ]; then
        print_error "PyAsc source not found. Please run: git submodule update --init"
        exit 1
    fi

    # PyAsc build instructions depend on its actual build system
    # This is a placeholder that should be updated based on PyAsc's build system
    print_warn "PyAsc build: Please check PyAsc documentation for build instructions"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    print_info "PyAsc setup completed in ${duration}s"
}

build_project() {
    local start_time=$(date +%s)
    print_info "Building Ascend-MLIR..."

    if [ ! -d "${LLVM_BUILD_DIR}" ]; then
        print_error "LLVM build not found at: ${LLVM_BUILD_DIR}"
        print_error "Please build LLVM first with --build-llvm or specify path with --llvm-build-dir"
        exit 1
    fi

    # Verify LLVM build directory has the expected structure
    local MLIR_CMAKE_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir"
    if [ ! -d "${MLIR_CMAKE_DIR}" ]; then
        print_error "MLIR CMake config not found at: ${MLIR_CMAKE_DIR}"
        print_error "Please ensure MLIR was built correctly (LLVM_ENABLE_PROJECTS must include mlir)"
        exit 1
    fi

    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    print_info "Using LLVM from: ${LLVM_BUILD_DIR}"

    # Pass LLVM_BUILD_DIR to cmake, it will automatically derive MLIR_DIR
    cmake -G Ninja "${PROJECT_ROOT}" \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DLLVM_BUILD_DIR="${LLVM_BUILD_DIR}"

    cmake --build . --target all -j${NUM_JOBS}

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    print_info "Ascend-MLIR build completed in ${duration}s ($(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60))))"
}

build_tests() {
    local start_time=$(date +%s)
    print_info "Building and running tests..."

    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    if $COVERAGE; then
        print_info "Building with coverage enabled..."
        cmake -G Ninja "${PROJECT_ROOT}" \
            -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
            -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
            -DLLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
            -DCMAKE_C_COMPILER="${LLVM_BUILD_DIR}/bin/clang" \
            -DCMAKE_CXX_COMPILER="${LLVM_BUILD_DIR}/bin/clang++" \
            -DCMAKE_C_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
            -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping"

        cmake --build . --target all -j${NUM_JOBS}
        mkdir -p "${BUILD_DIR}/coverage"
        
        print_info "Cleaning old coverage data..."
        find "${BUILD_DIR}" -name "*.profraw" -type f -delete 2>/dev/null || true
        find "${BUILD_DIR}/coverage" -name "*.profdata" -type f -delete 2>/dev/null || true
        
        cmake --build . --target check-afir-coverage -j${NUM_JOBS}

        collect_coverage
    else
        cmake --build . --target check-afir -j${NUM_JOBS}
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    print_info "Tests completed in ${duration}s ($(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60))))"
}

collect_coverage() {
    print_info "Collecting coverage data..."

    local LLVM_COV="${LLVM_BUILD_DIR}/bin/llvm-cov"
    local LLVM_PROFDATA="${LLVM_BUILD_DIR}/bin/llvm-profdata"
    local COVERAGE_DIR="${BUILD_DIR}/coverage"
    local BINARY="${BUILD_DIR}/bin/afir-opt"
    local IGNORE_REGEX=".*externals.*|.*build/.*|.*test/.*|.*unittest.*"

    if [ ! -f "$LLVM_COV" ] || [ ! -f "$LLVM_PROFDATA" ]; then
        print_warn "llvm-cov or llvm-profdata not found in system LLVM"
        return 0
    fi

    if [ ! -f "$BINARY" ]; then
        print_warn "Binary not found: $BINARY"
        return 0
    fi

    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    find coverage -name "*.profdata" -type f -delete 2>/dev/null || true
    mkdir -p "$COVERAGE_DIR"

    local PROFDATA_FILES=$(find . -name "*.profraw" 2>/dev/null)
    if [ -z "$PROFDATA_FILES" ]; then
        print_warn "No .profraw files found"
        return 0
    fi

    print_info "Found $(echo "$PROFDATA_FILES" | wc -l) .profraw files"

    print_info "Merging profile data..."
    local valid_files=""
    for profraw_file in $PROFDATA_FILES; do
        if "$LLVM_PROFDATA" show "$profraw_file" >/dev/null 2>&1; then
            valid_files="$valid_files $profraw_file"
        fi
    done

    if [ -z "$valid_files" ]; then
        print_warn "No valid .profraw files found"
        return 0
    fi

    "$LLVM_PROFDATA" merge -sparse -output "${COVERAGE_DIR}/coverage.profdata" $valid_files

    if [ ! -f "${COVERAGE_DIR}/coverage.profdata" ]; then
        print_warn "Failed to merge profile data"
        return 0
    fi

    print_info "Generating coverage report..."
    local report_dir="${COVERAGE_DIR}/report"
    mkdir -p "$report_dir"

    "$LLVM_COV" show -format=html -output-dir="$report_dir" \
        -instr-profile="${COVERAGE_DIR}/coverage.profdata" \
        "$BINARY" \
        -ignore-filename-regex="$IGNORE_REGEX"

    "$LLVM_COV" report -instr-profile="${COVERAGE_DIR}/coverage.profdata" \
        "$BINARY" \
        -ignore-filename-regex="$IGNORE_REGEX" \
        > "${COVERAGE_DIR}/summary.txt"

    print_info "Coverage report generated at: ${report_dir}/index.html"
    print_info "Coverage summary saved at: ${COVERAGE_DIR}/summary.txt"
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
        --coverage)
            COVERAGE=true
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
        --llvm-build-dir)
            LLVM_BUILD_DIR="$2"
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
if $BUILD_STABLEHLO || $BUILD_PYASC; then
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
