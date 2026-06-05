#!/bin/bash
#===----------------------------------------------------------------------===//
# Ascend-MLIR Build Script
#===----------------------------------------------------------------------===//

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/resolve_llvm_env.sh"
source "${SCRIPT_DIR}/resolve_ascend_env.sh"

ASCEND_HOME="$(resolve_ascend_home || true)"
if [ -n "${ASCEND_HOME}" ]; then
    export ASCEND_HOME_PATH="${ASCEND_HOME}"
    export ASCEND_TOOLKIT_HOME="${ASCEND_HOME}"
fi

# Default configuration
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${PROJECT_ROOT}/build"
INSTALL_DIR="${PROJECT_ROOT}/install"
LLVM_BUILD_DIR="$(resolve_llvm_build_dir || true)"
NUM_JOBS="${NUM_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
PROJECT_PROFILE="full"
PROJECT_ENABLE_AFIR_OVERRIDE=""
PROJECT_ENABLE_TESTS_OVERRIDE=""
PROJECT_ENABLE_PYTHON_OVERRIDE=""

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

# Check LLVM_BUILD_DIR exists and provide helpful error message
check_llvm_build_dir() {
    LLVM_BUILD_DIR="$(require_llvm_build_dir || true)"
    if [ -z "${LLVM_BUILD_DIR}" ]; then
        print_error "LLVM build is not configured."
        print_error "Solutions:"
        print_error "  1. If LLVM is not built yet, run: $0 --build-llvm"
        print_error "  2. If LLVM is already built elsewhere:"
        print_error "     - Set environment variable: export LLVM_BUILD_DIR=<path>"
        print_error "     - Or specify: $0 --llvm-build-dir <path>"
        exit 1
    fi

    # Verify MLIR CMake directory exists
    local mlir_cmake_dir="${LLVM_BUILD_DIR}/lib/cmake/mlir"
    if [ ! -d "${mlir_cmake_dir}" ]; then
        print_error "MLIR CMake config not found at: ${mlir_cmake_dir}"
        print_error "Please ensure MLIR was built correctly (LLVM_ENABLE_PROJECTS must include mlir)"
        exit 1
    fi
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
    --build-ascend      Build Ascend conversion and ascend-prefixed tools only
    --build-all         Build dependencies and project
    --build-tests       Build and run tests
    --build-coverage    Build with coverage instrumentation and generate coverage report
    --clean             Clean build directory
    --release           Build in Release mode (default)
    --debug             Build in Debug mode
    --enable-afir       Enable AFIR dialect, AFIR-only tools, and AFIR C API
    --disable-afir      Disable AFIR dialect, AFIR-only tools, and AFIR C API
    --enable-tests      Configure lit and unit-test targets
    --disable-tests     Skip configuring lit and unit-test targets
    --enable-python-bindings
                         Enable AFIR Python bindings
    --disable-python-bindings
                         Disable AFIR Python bindings
    --llvm-build-dir    Path to LLVM build directory (default: externals/llvm-project/build)
    --jobs N            Number of parallel jobs (default: auto)
    --help              Show this help message

Environment Variables:
    BUILD_TYPE          Build type: Release or Debug
    BUILD_DIR           Build directory path
    LLVM_BUILD_DIR      LLVM build directory path
    NUM_JOBS            Number of parallel jobs

Examples:
    $0 --build-all                           # Build everything
    $0 --build-llvm                          # Build LLVM/MLIR only
    $0 --build-project                       # Build Ascend-MLIR only
    $0 --build-ascend                        # Build Ascend conversion/tool profile
    $0 --build-tests                         # Build and run tests
    $0 --build-coverage                      # Run test coverage analysis
    $0 --llvm-build-dir /path/to/llvm/build  # Use external LLVM build
    $0 --clean --build-all                   # Clean and rebuild everything
EOF
}

init_submodules() {
    print_info "Initializing git submodules..."
    cd "${PROJECT_ROOT}"
    
    if [ "$1" = "stablehlo" ]; then
        git submodule update --init --recursive -- externals/stablehlo
    elif [ "$1" = "pyasc" ]; then
        git submodule update --init --recursive -- externals/pyasc
    else
        git submodule update --init --recursive
    fi
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

    # Check LLVM build directory
    check_llvm_build_dir

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
    print_info "Building Ascend-MLIR (${PROJECT_PROFILE} profile)..."

    # Check LLVM build directory
    check_llvm_build_dir

    local project_enable_afir="ON"
    local project_enable_tests="ON"
    local project_enable_python="ON"
    local project_targets=("all")
    local project_extra_targets=("ascir-translate")

    if [ "${PROJECT_PROFILE}" = "ascend" ]; then
        project_enable_afir="OFF"
        project_enable_tests="OFF"
        project_enable_python="OFF"
        project_targets=(
            AscendConversion
            ascend-mlir-opt
            ascend-mlir-translate
            ascend-debug
            runtime-session
            mix-compiler
            mix-tiling-helper
        )
        project_extra_targets=()
    fi

    if [ -n "${PROJECT_ENABLE_AFIR_OVERRIDE}" ]; then
        project_enable_afir="${PROJECT_ENABLE_AFIR_OVERRIDE}"
    fi
    if [ -n "${PROJECT_ENABLE_TESTS_OVERRIDE}" ]; then
        project_enable_tests="${PROJECT_ENABLE_TESTS_OVERRIDE}"
    fi
    if [ -n "${PROJECT_ENABLE_PYTHON_OVERRIDE}" ]; then
        project_enable_python="${PROJECT_ENABLE_PYTHON_OVERRIDE}"
    fi

    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    print_info "Using LLVM from: ${LLVM_BUILD_DIR}"
    print_info "CMake options:"
    print_info "  ASCEND_ENABLE_AFIR:          ${project_enable_afir}"
    print_info "  ASCEND_ENABLE_TESTS:         ${project_enable_tests}"
    print_info "  AFIR_ENABLE_BINDING_PYTHON:  ${project_enable_python}"
    print_info "Build targets: ${project_targets[*]}${project_extra_targets[*]:+ ${project_extra_targets[*]}}"

    configure_project() {
        local launcher_args=()
        if [ -n "${CMAKE_C_COMPILER_LAUNCHER:-}" ]; then
            launcher_args+=("-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}")
        fi
        if [ -n "${CMAKE_CXX_COMPILER_LAUNCHER:-}" ]; then
            launcher_args+=("-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}")
        fi

        cmake -G Ninja "${PROJECT_ROOT}" \
            -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
            -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
            -DASCEND_ENABLE_AFIR="${project_enable_afir}" \
            -DASCEND_ENABLE_TESTS="${project_enable_tests}" \
            -DAFIR_ENABLE_BINDING_PYTHON="${project_enable_python}" \
            -DPython3_EXECUTABLE="$(which python3)" \
            -DLLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
            "${launcher_args[@]}"
    }

    configure_project_with_retry() {
        if configure_project; then
            return 0
        fi
        print_warn "CMake configure failed; recreating build directory and retrying..."
        cd "${PROJECT_ROOT}"
        rm -rf "${BUILD_DIR}"
        mkdir -p "${BUILD_DIR}"
        cd "${BUILD_DIR}"
        configure_project
    }

    cache_value() {
        local key="$1"
        sed -n "s/^${key}:[^=]*=//p" CMakeCache.txt 2>/dev/null | tail -n 1
    }

    cache_requires_reconfigure() {
        [ -f CMakeCache.txt ] || return 0

        local actual
        for entry in \
            "CMAKE_BUILD_TYPE=${BUILD_TYPE}" \
            "LLVM_BUILD_DIR=${LLVM_BUILD_DIR}" \
            "ASCEND_ENABLE_AFIR=${project_enable_afir}" \
            "ASCEND_ENABLE_TESTS=${project_enable_tests}" \
            "AFIR_ENABLE_BINDING_PYTHON=${project_enable_python}"; do
            local key="${entry%%=*}"
            local expected="${entry#*=}"
            actual="$(cache_value "${key}")"
            if [ "${actual}" != "${expected}" ]; then
                print_info "CMake cache mismatch for ${key}: '${actual}' -> '${expected}'"
                return 0
            fi
        done

        actual="$(cache_value CMAKE_C_COMPILER_LAUNCHER)"
        if [ "${actual}" != "${CMAKE_C_COMPILER_LAUNCHER:-}" ]; then
            print_info "CMake cache mismatch for CMAKE_C_COMPILER_LAUNCHER: '${actual}' -> '${CMAKE_C_COMPILER_LAUNCHER:-}'"
            return 0
        fi

        actual="$(cache_value CMAKE_CXX_COMPILER_LAUNCHER)"
        if [ "${actual}" != "${CMAKE_CXX_COMPILER_LAUNCHER:-}" ]; then
            print_info "CMake cache mismatch for CMAKE_CXX_COMPILER_LAUNCHER: '${actual}' -> '${CMAKE_CXX_COMPILER_LAUNCHER:-}'"
            return 0
        fi

        return 1
    }

    build_selected_targets() {
        ninja -j${NUM_JOBS} "${project_targets[@]}"
        if [ "${#project_extra_targets[@]}" -gt 0 ]; then
            ninja -j${NUM_JOBS} "${project_extra_targets[@]}"
        fi
    }

    # Check if this is an incremental build (build.ninja exists)
    if [ -f "build.ninja" ]; then
        print_info "Incremental build detected (build.ninja exists)"
        if cache_requires_reconfigure; then
            print_info "CMake configuration changed; reconfiguring..."
            configure_project_with_retry
        else
            print_info "Skipping CMake configuration, running ninja directly..."
        fi
        if ! build_selected_targets; then
            print_warn "Incremental build failed; reconfiguring once and retrying..."
            configure_project_with_retry
            build_selected_targets
        fi
    else
        print_info "First-time build or CMake configuration needed"
        # Pass LLVM_BUILD_DIR to cmake, it will automatically derive MLIR_DIR
        configure_project_with_retry
        build_selected_targets
    fi

    touch "${BUILD_DIR}/.last_build_time"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    print_info "Ascend-MLIR build completed in ${duration}s ($(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60))))"
}

build_tests() {
    local start_time=$(date +%s)

    # Check if sources have changed since last build via ninja dry-run
    local needs_build=false
    if [ ! -f "${BUILD_DIR}/build.ninja" ]; then
        print_info "No previous build found, building first..."
        needs_build=true
    else
        cd "${BUILD_DIR}"
        if ! ninja -n all 2>&1 | grep -q "^ninja: no work to do\.$"; then
            needs_build=true
        fi
    fi

    if $needs_build; then
        print_info "Source changes detected, building..."
        cd "${BUILD_DIR}"
        ninja -j${NUM_JOBS}
    else
        print_info "No source changes detected, skipping build."
    fi

    # Run lit-based MLIR tests
    print_info "Running lit-based MLIR tests..."
    cd "${BUILD_DIR}"
    ninja check-afir -j${NUM_JOBS}

    # Run tool integration tests
    print_info "Running tool integration tests..."
    local test_passed=0
    local test_failed=0

    # Export LLVM_BUILD_DIR for test scripts
    export LLVM_BUILD_DIR

    for test_script in test/tools/*/run_*.sh; do
        if [ -f "$test_script" ]; then
            local test_name=$(basename "$test_script")
            print_info "Running $test_name..."
            if bash "$test_script" > /tmp/${test_name}.log 2>&1; then
                print_info "  ✓ $test_name PASSED"
                ((test_passed++))
            else
                print_error "  ✗ $test_name FAILED"
                print_error "    Log: /tmp/${test_name}.log"
                cat /tmp/${test_name}.log | tail -20
                ((test_failed++))
            fi
        fi
    done

    # Summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    print_info "Tests completed in ${duration}s ($(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60))))"
    print_info "Tool tests: $test_passed passed, $test_failed failed"

    if [ $test_failed -gt 0 ]; then
        return 1
    fi
}

build_coverage() {
    print_info "Running test coverage analysis..."

    # Call the dedicated coverage script with proper arguments
    "${SCRIPT_DIR}/run_coverage.sh" \
        --llvm-build-dir "${LLVM_BUILD_DIR}" \
        --jobs "${NUM_JOBS}"
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
BUILD_ASCEND=false
BUILD_TESTS=false
BUILD_COVERAGE=false
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
            PROJECT_PROFILE="full"
            shift
            ;;
        --build-ascend)
            BUILD_ASCEND=true
            BUILD_PROJECT=true
            PROJECT_PROFILE="ascend"
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
        --build-coverage)
            BUILD_COVERAGE=true
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
        --enable-afir)
            PROJECT_ENABLE_AFIR_OVERRIDE="ON"
            shift
            ;;
        --disable-afir)
            PROJECT_ENABLE_AFIR_OVERRIDE="OFF"
            shift
            ;;
        --enable-tests)
            PROJECT_ENABLE_TESTS_OVERRIDE="ON"
            shift
            ;;
        --disable-tests)
            PROJECT_ENABLE_TESTS_OVERRIDE="OFF"
            shift
            ;;
        --enable-python-bindings)
            PROJECT_ENABLE_PYTHON_OVERRIDE="ON"
            shift
            ;;
        --disable-python-bindings)
            PROJECT_ENABLE_PYTHON_OVERRIDE="OFF"
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
    exit 0
fi

if $BUILD_STABLEHLO && $BUILD_PYASC; then
    init_submodules
elif $BUILD_STABLEHLO; then
    init_submodules stablehlo
elif $BUILD_PYASC; then
    init_submodules pyasc
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

if $BUILD_COVERAGE; then
    build_coverage
fi

# If no options specified, show usage
if ! $BUILD_LLVM && ! $BUILD_STABLEHLO && ! $BUILD_PYASC && ! $BUILD_PROJECT && ! $BUILD_ASCEND && ! $BUILD_TESTS && ! $BUILD_COVERAGE && ! $CLEAN; then
    print_warn "No build target specified."
    usage
fi

print_info "Done!"
