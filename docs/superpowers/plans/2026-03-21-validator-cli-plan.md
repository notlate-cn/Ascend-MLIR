# Validator CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `validator` CLI tool that loads a pre-compiled `.bin`, executes it via the simulator, and compares output against an expected `.npy`.

**Architecture:** `tools/validator/validator_main.cpp` is a thin wrapper around `Executor` (load + run `.bin`) and `SimValidator::ValidateBinary()` (accuracy comparison). It does NOT compile — the `.bin` must already exist. Follows the same CMakeLists pattern as `tools/sim-validator/`. The `--npu` flag is reserved but not implemented (exits 4).

**Tech Stack:** C++17, LLVM CommandLine, `mlir::runtime::Executor`, `mlir::runtime::SimValidator`, `mlir::runtime::NpyIO`.

---

## Context for Implementors

### Existing library interfaces

**`Executor`** — `include/Runtime/Executor.h`:
```cpp
class Executor {
public:
  static constexpr uint32_t MAGIC_ELF_AIVEC  = 0x41415246u;
  static constexpr uint32_t MAGIC_ELF_AICUBE = 0x41494343u;

  explicit Executor(BackendMode mode = BackendMode::Simulation);
  llvm::Error Initialize(int device_id = 0);
  llvm::Expected<void*> RegisterBinary(const std::string& binary_path,
                                       const std::string& function_name,
                                       uint32_t magic = MAGIC_ELF_AIVEC);
  llvm::Error RunWithHandle(void* func_handle, RunArgs& args);
};
```

**`SimValidator::ValidateBinary()`** — `include/Runtime/SimValidator.h`:
```cpp
struct Result {
  bool   passed;
  double max_abs_diff;
  double mean_abs_diff;
  int64_t cycle_count;   // -1 if unavailable
  std::string error_msg;
};

Result ValidateBinary(void*                       func_handle,
                      Executor&                   executor,
                      RunArgs&                    args,
                      const std::vector<NDArray>& expected,
                      double atol,
                      double rtol);
```

**`RunArgs`** — `include/Runtime/Types.h`:
```cpp
struct RunArgs {
  std::vector<NDArray> inputs;
  std::vector<NDArray> outputs;
  std::vector<uint8_t> tiling;
  int                  block_dim = 1;
};
```

**`LoadNpy` / `NDArray`** — `include/Runtime/NpyIO.h`:
```cpp
struct NDArray { void* data; std::vector<int64_t> shape; DType dtype; size_t nbytes() const; };
llvm::Expected<NDArray> LoadNpy(const std::string& path);
```

### Magic constant mapping (kernel_type → magic)
```
"vec"  → Executor::MAGIC_ELF_AIVEC  (0x41415246)
"cube" → Executor::MAGIC_ELF_AICUBE (0x41494343)
"mix"  → Executor::MAGIC_ELF_AIVEC  (conservative default)
```

### Tiling bytes packing (same as sim-validator)

```cpp
static bool buildTiling(const std::string& params, const std::string& layout,
                        std::vector<uint8_t>& bytes) {
  auto pvec = splitComma(params);
  auto lvec = splitComma(layout);
  for (size_t i = 0; i < pvec.size(); ++i) {
    auto eq = pvec[i].find('=');
    if (eq == std::string::npos) {
      llvm::errs() << "Error: --tiling-params token missing '=': " << pvec[i] << "\n";
      return false;
    }
    int64_t val = std::stoll(pvec[i].substr(eq + 1));
    std::string type = i < lvec.size() ? lvec[i] : "int64";
    if (type == "int64" || type == "int64_t") {
      uint8_t buf[8]; std::memcpy(buf, &val, 8);
      bytes.insert(bytes.end(), buf, buf + 8);
    } else if (type == "int32" || type == "int32_t") {
      int32_t v = static_cast<int32_t>(val);
      uint8_t buf[4]; std::memcpy(buf, &v, 4);
      bytes.insert(bytes.end(), buf, buf + 4);
    } else {
      llvm::errs() << "Error: unknown tiling type '" << type << "'\n";
      return false;
    }
  }
  return true;
}
```

### Exit codes (from spec)
| Code | Meaning |
|------|---------|
| 0 | accuracy PASS |
| 1 | accuracy FAIL |
| 3 | execution error (rtKernelLaunch or runtime failure) |
| 4 | input error (bad args, missing files, --npu not implemented) |

### stdout format (from spec)
```
max_abs_diff:  0.031
mean_abs_diff: 0.008
PASS
```
or
```
max_abs_diff:  1.234
mean_abs_diff: 0.567
FAIL
```

### _Exit pattern (IMPORTANT)

The simulator (`libruntime_camodel.so`) leaves background threads running after `rtDeviceSynchronize`. Calling normal `exit()` races those threads and segfaults. Always exit with `_Exit(exit_code)` — same pattern as `sim-validator`:
```cpp
_Exit(exit_code);
```

### Memory management

`LoadNpy` allocates `NDArray::data` with `new uint8_t[...]`. Caller must free with `delete[] static_cast<uint8_t*>(arr.data)`. Free all inputs, expected, and output buffers before `_Exit()`.

### Validator CLI interface (from spec)

```
validator --bin           ./build/my_kernel.bin \
          --name          my_kernel \
          --inputs        a.npy,b.npy \
          --expected      out.npy \
          --tiling-params "TB_M=16,TB_N=4,M=32,N=32" \
          --tiling-layout "int64,int64,int64,int64" \
          --block-dim     2 \
          [--sim | --npu] \   # default --sim; --npu prints error, exits 4
          --atol          0.1 \
          --rtol          1e-2
```

`--sim` is the default; omitting `--sim`/`--npu` means `--sim`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `tools/validator/validator_main.cpp` | Create | CLI entry point: parse args, Executor + ValidateBinary |
| `tools/validator/CMakeLists.txt` | Create | `add_llvm_executable(validator ...)` |
| `CMakeLists.txt` (root) | Modify | Add `add_subdirectory(tools/validator)` |
| `test/tools/validator/run_validator.sh` | Create | Integration test shell script |

---

## Task 1: CMakeLists

**Files:**
- Create: `tools/validator/CMakeLists.txt`
- Modify: root `CMakeLists.txt`

- [ ] **Step 1: Create `tools/validator/CMakeLists.txt`**

```cmake
# tools/validator/CMakeLists.txt
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(validator
  validator_main.cpp
)

target_link_libraries(validator PRIVATE
  AscendCRuntime
)

target_include_directories(validator PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
```

- [ ] **Step 2: Wire into root CMakeLists.txt**

**Dependency**: If the compiler plan (Plan 2) has already been applied, line 97 is `add_subdirectory(tools/compiler)`. If not, line 96 is `add_subdirectory(tools/autotuner)`. Either way, add `add_subdirectory(tools/validator)` immediately after whichever is last in the tools block.

After both plans applied, the block looks like:
```cmake
add_subdirectory(tools/afir-opt)
add_subdirectory(tools/sim-validator)
add_subdirectory(tools/autotuner)
add_subdirectory(tools/compiler)   # from Plan 2; may not exist yet
add_subdirectory(tools/validator)
```
If Plan 2 is not yet applied, add after `tools/autotuner` instead.

- [ ] **Step 3: Verify CMake configures (inside xvm)**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && sleep 1 && bash scripts/build.sh --build-project --llvm-build-dir ~/code/llvm-project/llvm/build 2>&1 | tail -10"
```

Expected: "validator_main.cpp: No such file or directory" (expected; .cpp doesn't exist yet), not a CMake config error.

- [ ] **Step 4: Commit**

```bash
git add tools/validator/CMakeLists.txt CMakeLists.txt
git commit -m "build: add validator CLI CMakeLists skeleton"
```

---

## Task 2: validator_main.cpp implementation

**Files:**
- Create: `tools/validator/validator_main.cpp`

- [ ] **Step 1: Write `tools/validator/validator_main.cpp`**

```cpp
// tools/validator/validator_main.cpp
#include "Runtime/Executor.h"
#include "Runtime/NpyIO.h"
#include "Runtime/SimValidator.h"
#include "Runtime/Types.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using namespace mlir::runtime;
using namespace llvm;

static cl::opt<std::string> BinFile("bin",
    cl::desc("Pre-compiled kernel .bin file"), cl::Required);
static cl::opt<std::string> KernelName("name",
    cl::desc("Kernel function name"), cl::Required);
static cl::opt<std::string> Inputs("inputs",
    cl::desc("Comma-separated input .npy files"), cl::Required);
static cl::opt<std::string> ExpectedFile("expected",
    cl::desc("Expected output .npy file"), cl::Required);
static cl::opt<std::string> TilingParams("tiling-params",
    cl::desc("Comma-separated KEY=VALUE tiling params: TB_M=16,TB_N=4,..."),
    cl::init(""));
static cl::opt<std::string> TilingLayout("tiling-layout",
    cl::desc("Comma-separated tiling param types: int64,int64,..."), cl::init(""));
static cl::opt<int> BlockDim("block-dim",
    cl::desc("Number of AiCore blocks"), cl::init(1));
static cl::opt<std::string> KernelType("kernel-type",
    cl::desc("Kernel type: vec | cube | mix (default: vec)"), cl::init("vec"));
static cl::opt<bool> SimMode("sim",
    cl::desc("Use CPU simulator backend (default)"), cl::init(false));
static cl::opt<bool> NpuMode("npu",
    cl::desc("Use NPU backend (not implemented)"), cl::init(false));
static cl::opt<double> Atol("atol",
    cl::desc("Absolute tolerance"), cl::init(1.0));
static cl::opt<double> Rtol("rtol",
    cl::desc("Relative tolerance"), cl::init(1e-2));

static std::vector<std::string> splitComma(const std::string& s) {
  std::vector<std::string> parts;
  std::istringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) parts.push_back(tok);
  return parts;
}

static bool buildTiling(const std::string& params, const std::string& layout,
                        std::vector<uint8_t>& bytes) {
  auto pvec = splitComma(params);
  auto lvec = splitComma(layout);
  for (size_t i = 0; i < pvec.size(); ++i) {
    auto eq = pvec[i].find('=');
    if (eq == std::string::npos) {
      llvm::errs() << "Error: --tiling-params token missing '=': " << pvec[i] << "\n";
      return false;
    }
    int64_t val = std::stoll(pvec[i].substr(eq + 1));
    std::string type = i < lvec.size() ? lvec[i] : "int64";
    if (type == "int64" || type == "int64_t") {
      uint8_t buf[8]; std::memcpy(buf, &val, 8);
      bytes.insert(bytes.end(), buf, buf + 8);
    } else if (type == "int32" || type == "int32_t") {
      int32_t v = static_cast<int32_t>(val);
      uint8_t buf[4]; std::memcpy(buf, &v, 4);
      bytes.insert(bytes.end(), buf, buf + 4);
    } else {
      llvm::errs() << "Error: unknown tiling type '" << type << "'\n";
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv, "AscendC Kernel Validator\n");

  // --npu is reserved but not implemented
  if (NpuMode) {
    llvm::errs() << "Error: --npu not implemented in this version\n";
    return 4;
  }

  // Build tiling bytes
  // Note: pass TilingLayout only if non-empty; buildTiling handles empty layout
  // by defaulting each param to "int64".
  std::vector<uint8_t> tiling;
  if (!TilingParams.empty()) {
    if (!buildTiling(TilingParams, TilingLayout, tiling)) return 4;
  }

  // Load inputs
  RunArgs args;
  args.tiling    = tiling;
  args.block_dim = BlockDim;

  for (auto& path : splitComma(Inputs)) {
    auto arr_or = LoadNpy(path);
    if (!arr_or) {
      for (auto& inp : args.inputs) delete[] static_cast<uint8_t*>(inp.data);
      llvm::errs() << "Error loading input " << path << ": "
                   << llvm::toString(arr_or.takeError()) << "\n";
      return 4;
    }
    args.inputs.push_back(*arr_or);
  }

  // Load expected output
  auto exp_or = LoadNpy(ExpectedFile);
  if (!exp_or) {
    for (auto& inp : args.inputs) delete[] static_cast<uint8_t*>(inp.data);
    llvm::errs() << "Error loading expected: "
                 << llvm::toString(exp_or.takeError()) << "\n";
    return 4;
  }
  NDArray exp_arr = *exp_or;
  std::vector<NDArray> expected_arrs = {exp_arr};

  // Pre-alloc output buffer (same shape/dtype as expected)
  NDArray out_buf;
  out_buf.shape = exp_arr.shape;
  out_buf.dtype = exp_arr.dtype;
  out_buf.data  = new uint8_t[out_buf.nbytes()]();
  args.outputs.push_back(out_buf);

  // Map kernel_type → magic
  uint32_t magic = Executor::MAGIC_ELF_AIVEC;
  if (KernelType.getValue() == "cube") magic = Executor::MAGIC_ELF_AICUBE;

  // Initialize executor
  Executor executor(BackendMode::Simulation);
  if (auto err = executor.Initialize()) {
    for (auto& inp : args.inputs) delete[] static_cast<uint8_t*>(inp.data);
    delete[] static_cast<uint8_t*>(args.outputs[0].data);
    delete[] static_cast<uint8_t*>(exp_arr.data);
    llvm::errs() << "Error: executor init failed: "
                 << llvm::toString(std::move(err)) << "\n";
    return 3;
  }

  // Register binary once
  auto handle_or = executor.RegisterBinary(BinFile, KernelName, magic);
  if (!handle_or) {
    for (auto& inp : args.inputs) delete[] static_cast<uint8_t*>(inp.data);
    delete[] static_cast<uint8_t*>(args.outputs[0].data);
    delete[] static_cast<uint8_t*>(exp_arr.data);
    llvm::errs() << "Error: RegisterBinary failed: "
                 << llvm::toString(handle_or.takeError()) << "\n";
    return 3;
  }

  // Run and compare
  SimValidator validator;
  auto result = validator.ValidateBinary(*handle_or, executor, args,
                                         expected_arrs, Atol, Rtol);

  // Free host-side allocations
  delete[] static_cast<uint8_t*>(args.outputs[0].data);
  for (auto& inp : args.inputs) delete[] static_cast<uint8_t*>(inp.data);
  delete[] static_cast<uint8_t*>(exp_arr.data);

  // Determine exit code first.
  // ValidateBinary sets error_msg (non-empty) for runtime/executor failures,
  // and passed=false with zero diffs for those paths. Use error_msg as the
  // sole indicator for exit 3 (runtime error) vs 0/1 (accuracy pass/fail).
  int exit_code;
  if (!result.error_msg.empty()) {
    // Runtime error: executor failed to run kernel. Do NOT print PASS/FAIL.
    llvm::errs() << "Error: " << result.error_msg << "\n";
    exit_code = 3;
  } else {
    // Accuracy comparison completed. Print diff stats + PASS/FAIL.
    llvm::outs() << "max_abs_diff:  " << result.max_abs_diff  << "\n"
                 << "mean_abs_diff: " << result.mean_abs_diff << "\n";
    llvm::outs() << (result.passed ? "PASS\n" : "FAIL\n");
    exit_code = result.passed ? 0 : 1;
  }

  llvm::outs().flush();
  llvm::errs().flush();
  // _Exit: skip destructors; simulator leaves background threads running.
  _Exit(exit_code);
}
```

- [ ] **Step 2: Copy to xvm (if file sync unreliable)**

```bash
scp tools/validator/validator_main.cpp xvm@orb:/home/niu/code/Ascend-MLIR/tools/validator/validator_main.cpp
```

- [ ] **Step 3: Build inside xvm**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && ninja -C build validator 2>&1 | tail -20"
```

Expected: `[N/N] Linking CXX executable bin/validator` — success.

- [ ] **Step 4: Commit**

```bash
git add tools/validator/validator_main.cpp
git commit -m "feat(tools): add validator CLI (.bin + .npy → accuracy check)"
```

---

## Task 3: Integration test

**Files:**
- Create: `test/tools/validator/run_validator.sh`

The integration test verifies argument parsing and error paths. It does NOT require the CANN simulator (we avoid loading `.bin` files that need camodel). We test:
1. `--help` exits 0.
2. Missing `--bin` file → exit 3 (executor init + RegisterBinary fails).
3. `--npu` → exit 4 (not implemented).

- [ ] **Step 1: Create `test/tools/validator/run_validator.sh`**

```bash
#!/usr/bin/env bash
set -e
cd /home/niu/code/Ascend-MLIR
source examples/env.sh

VALIDATOR=./build/bin/validator

# Test 1: --help exits 0
echo "--- Test 1: --help ---"
$VALIDATOR --help > /dev/null
echo "PASS: --help exits 0"

# Test 2: --npu → exit 4 (not implemented)
echo "--- Test 2: --npu not implemented ---"
# Create dummy npy files to pass arg parsing (actual loading happens later)
python3 -c "
import numpy as np
np.save('/tmp/dummy_in.npy', np.zeros((1,), dtype=np.float16))
np.save('/tmp/dummy_exp.npy', np.zeros((1,), dtype=np.float16))
"
set +e
$VALIDATOR --bin /tmp/nonexistent.bin \
           --name dummy \
           --inputs /tmp/dummy_in.npy \
           --expected /tmp/dummy_exp.npy \
           --npu 2>/dev/null
rc=$?
set -e
if [ "$rc" -eq 4 ]; then
  echo "PASS: --npu → exit 4"
else
  echo "FAIL: expected exit 4, got $rc"
  exit 1
fi

# Test 3: missing .bin file → exit 3 (RegisterBinary fails)
echo "--- Test 3: missing .bin file ---"
set +e
$VALIDATOR --bin /tmp/nonexistent.bin \
           --name dummy \
           --inputs /tmp/dummy_in.npy \
           --expected /tmp/dummy_exp.npy \
           2>/dev/null
rc=$?
set -e
if [ "$rc" -eq 3 ]; then
  echo "PASS: missing .bin → exit 3"
else
  echo "FAIL: expected exit 3, got $rc"
  exit 1
fi

echo "ALL TESTS PASSED"
```

- [ ] **Step 2: Make executable and copy to xvm**

```bash
chmod +x test/tools/validator/run_validator.sh
mkdir -p test/tools/validator
scp test/tools/validator/run_validator.sh xvm@orb:/home/niu/code/Ascend-MLIR/test/tools/validator/run_validator.sh
ssh xvm@orb "chmod +x /home/niu/code/Ascend-MLIR/test/tools/validator/run_validator.sh"
```

- [ ] **Step 3: Run test inside xvm**

```bash
ssh xvm@orb "bash /home/niu/code/Ascend-MLIR/test/tools/validator/run_validator.sh"
```

Expected:
```
--- Test 1: --help ---
PASS: --help exits 0
--- Test 2: --npu not implemented ---
PASS: --npu → exit 4
--- Test 3: missing .bin file ---
PASS: missing .bin → exit 3
ALL TESTS PASSED
```

- [ ] **Step 4: Commit**

```bash
git add test/tools/validator/run_validator.sh
git commit -m "test(tools): add validator CLI integration test"
```

---

## Notes for Implementors

- **`_Exit` not `exit`**: The simulator's background threads segfault on normal `exit()`. Always use `_Exit(code)`. This is the same pattern as `tools/sim-validator/sim_validator_main.cpp:138`.
- **Exit code 3 vs 4**: Exit 3 = runtime/executor failure (can't even run the kernel). Exit 4 = input error (bad args, missing files, --npu). The split: if the executor can't register the binary, that's exit 3. If `--npu` is passed, that's exit 4 (input error = unsupported flag).
- **`ValidateBinary` vs `Validate`**: `Validate()` does compilation + execution + compare. `ValidateBinary()` skips compilation — it uses an already-registered handle. Use `ValidateBinary` here (no compilation in this tool).
- **Executor initialization**: Call `executor.Initialize()` before `RegisterBinary()`. The executor loads `libruntime_camodel.so` and calls `rtSetDevice(0)` in `Initialize()`.
- **File sync to xvm**: The project is on `/Volumes/GM9/` (external drive) and OrbStack sync may lag. Use `scp` to copy changed files after writing them locally.
