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
├── include/mlir/
│   ├── Dialect/AFIR/       # AFIR dialect definitions
│   ├── Conversion/         # Conversion pass headers
│   ├── Interface/          # Op interface definitions
│   └── Utils/              # Utility functions
├── lib/                    # Implementation files
├── tools/afir-opt/         # MLIR optimizer tool
├── test/                   # Test cases
├── scripts/                # Build scripts
└── version/                # Version information
```

## Building

### Prerequisites

- CMake >= 3.20
- Ninja build system
- C++17 compatible compiler (GCC >= 9 or Clang >= 10)
- Python 3 (optional, for Python bindings)

### Build Steps

1. Clone the repository and initialize submodules:

```bash
git clone <repository-url>
cd Ascend-MLIR
git submodule update --init --recursive
```

2. Build LLVM/MLIR:

```bash
./scripts/build_llvm.sh
```

3. Build Ascend-MLIR:

```bash
./scripts/build.sh --build-project
```

Or build everything at once:

```bash
./scripts/build.sh --build-all
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

## Dependencies

| Dependency | Version | Description |
|------------|---------|-------------|
| LLVM/MLIR  | 21.1    | Core compiler infrastructure |
| StableHLO  | main    | StableHLO dialect for ML frameworks |
| PyAsc      | main    | ASC-IR dialect for Ascend hardware |

## License

Apache License 2.0
