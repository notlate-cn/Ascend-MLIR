# Ascend-MLIR

Ascend-MLIR is an MLIR-based compiler infrastructure for Ascend hardware, featuring the AFIR (Ascend Frontend IR) dialect.

## Features

- **AFIR Dialect**: A high-level dialect for representing computational operations targeting Ascend hardware
  - `afir.add` - Element-wise addition
  - `afir.sub` - Element-wise subtraction
  - `afir.mul` - Element-wise multiplication
  - `afir.div` - Element-wise division

- **AFIRDialectBuilder**: Convenient builder interface for creating AFIR operations (inspired by onnx-mlir)

- **ShapeHelperOpInterface**: Interface for shape inference on AFIR operations (inspired by onnx-mlir)

- **Conversion Passes**:
  - StableHLO to AFIR conversion
  - AFIR to ASC-IR (PyAsc) conversion

## Project Structure

```
Ascend-MLIR/
├── externals/              # Git submodules
│   ├── llvm-project/       # LLVM/MLIR source
│   ├── stablehlo/          # StableHLO source
│   └── pyasc/              # PyAsc (ASC-IR) source
├── include/               # Header files
│   ├── Conversion/         # Conversion pass headers
│   │   └── AFIRToASCIR/    # AFIR to ASC-IR conversion
│   ├── Dialect/AFIR/       # AFIR dialect definitions
│   ├── Interface/          # Op interface definitions
│   └── Utils/              # Utility functions
├── lib/                   # Implementation files
│   ├── Conversion/         # Conversion pass implementations
│   │   └── AFIRToASCIR/    # AFIR to ASC-IR conversion
│   │       └── Math/       # Math operator conversions (Elementwise)
│   ├── Dialect/AFIR/       # AFIR dialect implementations
│   └── Utils/              # Utility implementations
├── tools/afir-opt/         # MLIR optimizer tool
├── test/                  # Test cases
│   ├── Conversion/         # Conversion tests
│   ├── Dialect/           # Dialect tests
│   └── Interface/         # Interface tests
├── scripts/               # Build and utility scripts
│   ├── build.sh           # Main build script
│   ├── build_llvm.sh      # LLVM build script
│   ├── run_tests.sh       # Test runner
│   └── check_clang_format.sh  # Code format checker
├── version/               # Version information
├── CMakeLists.txt         # Main CMake configuration
└── .clang-format          # Clang-format configuration
```

## Building

### Prerequisites

- CMake >= 3.20
- Ninja build system
- C++17 compatible compiler (GCC >= 9 or Clang >= 10)
- Python 3 (optional, for Python bindings)
- LLVM/MLIR (see LLVM setup below)

### LLVM/MLIR Setup

You have two options for setting up LLVM/MLIR:

#### Option 1: Build LLVM from provided script (Recommended)

If you don't have LLVM/MLIR installed, use the provided script to download and build it:

```bash
./scripts/build_llvm.sh
```

This will download LLVM 21.1.8 and build it in `externals/llvm-project/build`.

#### Option 2: Use your own LLVM build

If you already have LLVM/MLIR built elsewhere, you can specify the path to your LLVM build directory:

```bash
# Using build.sh
./scripts/build.sh --build-project --llvm-build-dir /path/to/your/llvm/build

# Or using cmake directly
mkdir build && cd build
cmake -G Ninja .. \
    -DLLVM_BUILD_DIR=/path/to/your/llvm/build
ninja
```

**Note:** Specify the LLVM **build root** directory (e.g., `/path/to/llvm-project/build`), not the cmake subdirectory

### Build Steps

#### Quick Start (Build everything)

```bash
# Clone the repository
git clone <repository-url>
cd Ascend-MLIR

# Build LLVM/MLIR and Ascend-MLIR
./scripts/build.sh --build-all
```

#### Step-by-step Build

1. Clone the repository:

```bash
git clone <repository-url>
cd Ascend-MLIR
```

2. Build or specify LLVM/MLIR (choose one):

```bash
# Option A: Build LLVM using provided script
./scripts/build_llvm.sh

# Option B: Use your own LLVM (skip if using Option A)
# Just note the path to your LLVM build directory
```

3. Build Ascend-MLIR:

```bash
# If using default LLVM location (externals/llvm-project/build)
./scripts/build.sh --build-project

# If using custom LLVM location
./scripts/build.sh --build-project --llvm-build-dir /path/to/llvm/build
```

### Build Options

```bash
./scripts/build.sh --help
```

## Usage

### afir-opt Tool

```bash
# Run AFIR optimization passes
./build/bin/afir-opt --afir-shape-inference input.mlir

# Run canonicalization
./build/bin/afir-opt --afir-canonicalize input.mlir
```

### Code Formatting

The project uses clang-format for code formatting. You can format the code using the following commands:

```bash
# Format all files (excluding externals/)
bash scripts/check_clang_format.sh -a -f

# Format files changed in working directory (compared to last commit)
bash scripts/check_clang_format.sh -c -f

# Format files in the last commit
bash scripts/check_clang_format.sh -l -f
```

To check code format without modifying files, use the check script without the `-f` flag:

```bash
# Check all files
bash scripts/check_clang_format.sh -a

# Check files changed in working directory (compared to last commit)
# Note: This checks uncommitted changes, no need to git add first
bash scripts/check_clang_format.sh -c

# Check files in the last commit
bash scripts/check_clang_format.sh -l
```

Note: The `-f` flag enables format mode (modifies files), while without it the script only verifies format.

## Example

```mlir
// example.mlir
func.func @example(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = afir.mul %0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
```

## Testing

Run the test suite:

```bash
./scripts/run_tests.sh
```

Or using cmake:

```bash
cd build
cmake --build . --target check-afir
```

## Test Coverage

Generate test coverage reports to analyze code coverage of the test suite.

### Prerequisites

Install lcov for coverage report generation:

```bash
# macOS
brew install lcov

# Ubuntu/Debian
sudo apt-get install lcov

# CentOS/RHEL
sudo yum install lcov
```

Optional: Install lcov_cobertura for XML coverage reports:

```bash
pip install lcov_cobertura
```

### Quick Start

Run the full coverage analysis:

```bash
# Using build.sh
./scripts/build.sh --build-coverage

# Or using the dedicated coverage script
./scripts/run_coverage.sh
```

This will:
1. Build the project with coverage instrumentation in `build-coverage/`
2. Run the test suite to generate coverage data
3. Generate coverage reports

### Coverage Script Options

```bash
./scripts/run_coverage.sh [OPTIONS]

Options:
  --build-dir DIR         Coverage build directory (default: build-coverage)
  --llvm-build-dir DIR    Path to LLVM build directory
  --jobs N                Number of parallel jobs (default: auto)
  --skip-build            Skip building, only generate coverage report
  --skip-tests            Skip running tests, only collect existing coverage
  --clean                 Clean coverage build directory before starting
  --help                  Show help message
```

### Usage Examples

```bash
# Run full coverage analysis
./scripts/run_coverage.sh

# Clean and run coverage
./scripts/run_coverage.sh --clean
./scripts/run_coverage.sh

# Only regenerate report from existing coverage data
./scripts/run_coverage.sh --skip-build --skip-tests

# Use custom LLVM build location
./scripts/run_coverage.sh --llvm-build-dir /path/to/llvm/build

# Specify number of parallel jobs
./scripts/run_coverage.sh --jobs 8
```

### Viewing Coverage Reports

After running the coverage script, reports are generated in `build-coverage/coverage/`:

```bash
# Open HTML report in browser
open build-coverage/coverage/lcov_report/index.html

# View coverage summary
lcov --summary build-coverage/coverage/total.info --rc branch_coverage=1
```

Generated files:
- **HTML Report**: `build-coverage/coverage/lcov_report/index.html` - Interactive coverage browser
- **XML Report**: `build-coverage/coverage/lcov_report/coverage.xml` - Cobertura format (if lcov_cobertura is installed)
- **Info File**: `build-coverage/coverage/total.info` - Raw coverage data

## Dependencies

| Dependency | Version | Description |
|------------|---------|-------------|
| LLVM/MLIR  | 21.1.8  | Core compiler infrastructure |
| StableHLO  | main    | StableHLO dialect for ML frameworks |
| PyAsc      | main    | ASC-IR dialect for Ascend hardware |

## License

Apache License 2.0
