#!/bin/bash
#===----------------------------------------------------------------------===//
# Ascend-MLIR Test Coverage Script
#===----------------------------------------------------------------------===//
# This script builds the project with coverage instrumentation, runs tests,
# and generates coverage reports using lcov.
#===----------------------------------------------------------------------===//

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default configuration
BUILD_DIR="${PROJECT_ROOT}/build-coverage"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${PROJECT_ROOT}/externals/llvm-project/build}"
NUM_JOBS="${NUM_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_section() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}\n"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Generate test coverage reports for Ascend-MLIR project.

Options:
    --build-dir DIR         Coverage build directory (default: build-coverage)
    --llvm-build-dir DIR    Path to LLVM build directory
    --jobs N                Number of parallel jobs (default: auto)
    --skip-build            Skip building, only generate coverage report
    --skip-tests            Skip running tests, only collect existing coverage
    --clean                 Clean coverage build directory before starting
    --help                  Show this help message

Examples:
    $0                                           # Run full coverage analysis
    $0 --clean                                   # Clean and run coverage
    $0 --skip-build                              # Only collect coverage from existing build
    $0 --llvm-build-dir /path/to/llvm/build     # Use external LLVM build

Output:
    Coverage reports will be generated in: ${BUILD_DIR}/coverage/
    - HTML report: coverage/lcov_report/index.html
    - XML report:  coverage/lcov_report/coverage.xml
    - Info file:   coverage/total.info

EOF
}

# Check if lcov is available
check_lcov() {
    if ! command -v lcov &> /dev/null; then
        print_error "lcov is not installed. Please install it first:"
        echo "  macOS:  brew install lcov"
        echo "  Ubuntu: sudo apt-get install lcov"
        echo "  CentOS: sudo yum install lcov"
        exit 1
    fi
    print_info "Found lcov: $(which lcov)"
}

# Build project with coverage flags
build_with_coverage() {
    print_section "Building Ascend-MLIR with Coverage Instrumentation"

    if [ ! -d "${LLVM_BUILD_DIR}" ]; then
        print_error "LLVM build not found at: ${LLVM_BUILD_DIR}"
        print_error "Please build LLVM first or specify path with --llvm-build-dir"
        exit 1
    fi

    # Verify LLVM build directory has the expected structure
    local MLIR_CMAKE_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir"
    if [ ! -d "${MLIR_CMAKE_DIR}" ]; then
        print_error "MLIR CMake config not found at: ${MLIR_CMAKE_DIR}"
        print_error "Please ensure MLIR was built correctly"
        exit 1
    fi

    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    print_info "Using LLVM from: ${LLVM_BUILD_DIR}"
    print_info "Build directory: ${BUILD_DIR}"

    # Detect C and C++ compilers
    local CC_COMPILER="${CC:-gcc}"
    local CXX_COMPILER="${CXX:-g++}"

    print_info "Using C compiler: ${CC_COMPILER}"
    print_info "Using C++ compiler: ${CXX_COMPILER}"

    # Get compiler version
    local GCC_VERSION=$(${CXX_COMPILER} -dumpversion 2>/dev/null || echo "unknown")
    print_info "Compiler version: ${GCC_VERSION}"

    # Configure with coverage flags
    cmake -G Ninja "${PROJECT_ROOT}" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
        -DCMAKE_C_COMPILER="${CC_COMPILER}" \
        -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
        -DAFIR_ENABLE_BINDING_PYTHON=true \
        -DPython3_EXECUTABLE="$(which python3)" \
        -DCMAKE_CXX_FLAGS="-fprofile-arcs -ftest-coverage" \
        -DCMAKE_C_FLAGS="-fprofile-arcs -ftest-coverage" \
        -DCMAKE_EXE_LINKER_FLAGS="-fprofile-arcs -ftest-coverage"

    print_info "Building project with ${NUM_JOBS} parallel jobs..."
    cmake --build . --target all -j${NUM_JOBS}

    print_info "Build completed successfully"
}

# Run tests to generate coverage data
run_tests() {
    print_section "Running Tests to Generate Coverage Data"

    cd "${BUILD_DIR}"

    # Clean old coverage data
    print_info "Cleaning old coverage data (.gcda files)..."
    find . -name "*.gcda" -delete

    # Run lit tests
    print_info "Running AFIR regression tests..."
    cmake --build . --target check-afir

    print_info "Tests completed successfully"
}

# Collect and process coverage data
collect_coverage() {
    print_section "Collecting and Processing Coverage Data"

    cd "${BUILD_DIR}"

    # Create coverage output directory
    rm -rf coverage
    mkdir -p coverage

    print_info "Searching for coverage data files..."
    local GCDA_COUNT=$(find . -name "*.gcda" | wc -l)
    local GCNO_COUNT=$(find . -name "*.gcno" | wc -l)

    print_info "Found ${GCDA_COUNT} .gcda files and ${GCNO_COUNT} .gcno files"

    if [ ${GCDA_COUNT} -eq 0 ]; then
        print_error "No coverage data found. Please run tests first."
        exit 1
    fi

    # Detect the gcov tool that matches the compiler used for building
    local GCOV_TOOL="gcov"
    local CXX_COMPILER="${CXX:-g++}"

    # Try to find the matching gcov version
    # If using g++-11, look for gcov-11; if using g++, use gcov
    if [[ "${CXX_COMPILER}" =~ g\+\+(-[0-9]+)$ ]]; then
        local GCC_SUFFIX="${BASH_REMATCH[1]}"
        local VERSIONED_GCOV="gcov${GCC_SUFFIX}"
        if command -v "${VERSIONED_GCOV}" &> /dev/null; then
            GCOV_TOOL="${VERSIONED_GCOV}"
            print_info "Found matching gcov tool: ${GCOV_TOOL}"
        fi
    fi

    # Verify gcov version matches compiler version
    local GCOV_VERSION=$(${GCOV_TOOL} --version 2>/dev/null | head -1 | grep -oP '\d+\.\d+' | head -1)
    local GCC_VERSION=$(${CXX_COMPILER} -dumpversion 2>/dev/null)

    print_info "Compiler version: ${GCC_VERSION}"
    print_info "Using gcov tool: ${GCOV_TOOL} (version ${GCOV_VERSION})"

    if [ "${GCOV_VERSION}" != "${GCC_VERSION}" ]; then
        print_warn "gcov version (${GCOV_VERSION}) does not match compiler version (${GCC_VERSION})"
        print_warn "This may cause compatibility issues. Trying to proceed anyway..."
    fi

    # Copy coverage files to coverage directory
    print_info "Copying coverage data files..."
    find . -name "*.gcda" -exec cp {} coverage/ \;
    find . -name "*.gcno" -exec cp {} coverage/ \;

    cd coverage

    # Detect lcov version to determine supported options
    local LCOV_VERSION=$(lcov --version 2>/dev/null | grep -oP 'version \K[0-9.]+' | head -1)
    local LCOV_MAJOR=$(echo "${LCOV_VERSION}" | cut -d. -f1)

    print_info "Using lcov version: ${LCOV_VERSION}"
    print_info "Using gcov tool: ${GCOV_TOOL}"

    # Determine which error types to ignore based on lcov version
    # lcov 1.x: supports source, graph, gcov, unused
    # lcov 2.x: adds support for mismatch, version, negative, empty, corrupt, format, parallel
    local IGNORE_ERRORS_CAPTURE="source,gcov"
    local IGNORE_ERRORS_REMOVE="unused,source"
    local IGNORE_ERRORS_GENHTML="source"
    local GENINFO_RC=""

    if [ "${LCOV_MAJOR}" -ge 2 ]; then
        print_info "Detected lcov 2.x - using extended error handling"
        # negative: ignore negative counts from race conditions in multi-threaded programs
        # mismatch: ignore version mismatches
        # version: ignore version compatibility warnings
        IGNORE_ERRORS_CAPTURE="source,gcov,mismatch,version,negative"
        IGNORE_ERRORS_REMOVE="unused,source,mismatch"
        # Set geninfo_unexecuted_blocks=0 to treat unexecuted blocks as warnings not errors
        GENINFO_RC="--rc geninfo_unexecuted_blocks=0"
    else
        print_info "Detected lcov 1.x - using basic error handling"
    fi

    # Generate coverage info with lcov
    # --gcov-tool specifies which gcov binary to use
    print_info "Generating initial coverage data..."
    lcov -c -i -d ./ -o cpp_init.info \
        --rc branch_coverage=1 \
        --gcov-tool ${GCOV_TOOL} \
        ${GENINFO_RC} \
        --ignore-errors ${IGNORE_ERRORS_CAPTURE} > /dev/null 2>&1

    print_info "Capturing test coverage data..."
    lcov -c -d ./ -o cpp_cover.info \
        --rc branch_coverage=1 \
        --gcov-tool ${GCOV_TOOL} \
        ${GENINFO_RC} \
        --ignore-errors ${IGNORE_ERRORS_CAPTURE} > /dev/null 2>&1

    print_info "Combining coverage data..."
    lcov -a cpp_init.info -a cpp_cover.info -o cpp_total.info \
        --rc branch_coverage=1 > /dev/null 2>&1

    # Remove unwanted files from coverage report
    print_info "Filtering coverage data..."
    lcov --ignore-errors ${IGNORE_ERRORS_REMOVE} \
         --remove cpp_total.info \
         '*/build/*' \
         '*/build-coverage/*' \
         '*/externals/*' \
         '/usr/include/*' \
         '*/llvm/*' \
         '*/mlir/*' \
         '*/test/*' \
         -o total.info --rc branch_coverage=1 > /dev/null 2>&1

    print_info "Generating HTML coverage report..."
    genhtml -o lcov_report total.info -q \
        --rc branch_coverage=1 \
        --ignore-errors ${IGNORE_ERRORS_GENHTML}

    # Generate XML report if lcov_cobertura is available
    if command -v lcov_cobertura &> /dev/null; then
        print_info "Generating XML coverage report..."
        lcov_cobertura total.info --output lcov_report/coverage.xml --demangle
    else
        print_warn "lcov_cobertura not found. Skipping XML report generation."
        print_warn "Install with: pip install lcov_cobertura"
    fi

    print_section "Coverage Report Generated Successfully"

    # Display coverage summary
    echo ""
    lcov --summary total.info --rc branch_coverage=1 2>&1 | grep -E "lines\.*:|functions\.*:|branches\.*:" || true
    echo ""

    print_info "Coverage reports generated in: ${BUILD_DIR}/coverage/"
    print_info "  HTML Report: ${BUILD_DIR}/coverage/lcov_report/index.html"
    if [ -f "lcov_report/coverage.xml" ]; then
        print_info "  XML Report:  ${BUILD_DIR}/coverage/lcov_report/coverage.xml"
    fi
    print_info ""
    print_info "To view the HTML report, open in browser:"
    echo -e "  ${YELLOW}open ${BUILD_DIR}/coverage/lcov_report/index.html${NC}"
}

# Parse command line arguments
SKIP_BUILD=false
SKIP_TESTS=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --llvm-build-dir)
            LLVM_BUILD_DIR="$2"
            shift 2
            ;;
        --jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
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

# Main execution
print_section "Ascend-MLIR Test Coverage Analysis"

if $CLEAN; then
    print_info "Cleaning coverage build directory..."
    rm -rf "${BUILD_DIR}"
    print_info "Done! 🎉"
    exit 0
fi

# Check for lcov
check_lcov

# Build with coverage
if ! $SKIP_BUILD; then
    build_with_coverage
fi

# Run tests
if ! $SKIP_TESTS; then
    run_tests
fi

# Collect coverage
collect_coverage

print_info "Done! 🎉"
