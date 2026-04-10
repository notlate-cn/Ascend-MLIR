# Compiler CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `compiler` CLI tool that compiles an AscendC kernel `.cpp` → `.o` + `.bin` + `runner` executable in one command.

**Architecture:** `tools/compiler/compiler_main.cpp` is a thin wrapper around `Compiler::Compile()` (bisheng → .o + .bin) and `HostRunnerGen::Generate()` (g++ → runner). Follows the exact same CMakeLists pattern as `tools/autotuner/` and `tools/sim-validator/`. No new library code — both library classes already exist.

**Tech Stack:** C++17, LLVM CommandLine (`llvm::cl`), `mlir::runtime::Compiler`, `mlir::runtime::HostRunnerGen`, CMake `add_llvm_executable`.

---

## Context for Implementors

### Existing library interfaces

**`Compiler::Compile(src_file, output_dir, kernel_name)`** — in `include/Runtime/Compiler.h`:
```cpp
struct CompilerConfig {
  std::string soc_version = "Ascend910B1";
  std::string arch        = "dav-c220-vec";
  int         opt_level   = 3;
  bool        verbose     = false;
};
class Compiler {
public:
  explicit Compiler(const Config& cfg = Config{});
  llvm::Expected<std::string> Compile(const std::string& src_file,
                                      const std::string& output_dir,
                                      const std::string& kernel_name);
};
```
`Compile()` runs bisheng (→ .o) and ld.lld (→ .bin). Returns path to `.bin` on success. The `.o` is also written to `output_dir/<kernel_name>.o`.

**`HostRunnerGen::Generate(cfg, output_dir)`** — in `include/Runtime/HostRunnerGen.h`:
```cpp
class HostRunnerGen {
public:
  struct Config {
    std::string kernel_name;
    std::string kernel_type;  // "vec" | "cube" | "mix"
    std::string soc_version = "Ascend910B1";
    int         num_inputs  = 1;
    int         num_outputs = 1;  // only 1 supported
    std::vector<std::string> tiling_layout;  // e.g. {"int64","int64","int32"}
    bool        verbose = false;
  };
  llvm::Expected<std::string> Generate(const Config& cfg,
                                        const std::string& output_dir);
};
```
`Generate()` writes `runner.cpp` and compiles it with g++. Returns path to `runner` on success.

### Existing CMakeLists pattern (copy from autotuner)

```cmake
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(compiler
  compiler_main.cpp
)

target_link_libraries(compiler PRIVATE
  AscendCRuntime
)

target_include_directories(compiler PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
```

### Root CMakeLists wiring

The root `CMakeLists.txt` has a `tools/` section. Add `add_subdirectory(compiler)` there (alongside `add_subdirectory(autotuner)` etc.). Check the file to find the exact insertion point.

### Compiler CLI interface (from spec)

```
compiler --kernel        step8_kernel.cpp \
         --output        ./build/ \
         --name          my_kernel \        # optional; default: stem of --kernel filename
         --soc           Ascend910B1 \
         --arch          dav-c220-vec \
         --num-inputs    2 \
         --num-outputs   1 \
         --tiling-layout "int64,int64,int64,int64" \
         --kernel-type   vec
```

Outputs:
```
build/<name>.o
build/<name>.bin
build/runner
```

### Exit codes (from spec)
| Code | Meaning |
|------|---------|
| 0 | success |
| 2 | compilation error (bisheng/lld failed) |
| 4 | input error (bad args, missing files) |

### Deriving kernel name from filename

If `--name` is not given, derive it from the stem of `--kernel`:
- `step8_kernel.cpp` → `step8_kernel`
- `./path/to/foo.cpp` → `foo`

Use `llvm::sys::path::stem(kernel_file)` to extract the stem.

### splitComma helper

```cpp
static std::vector<std::string> splitComma(const std::string& s) {
  std::vector<std::string> parts;
  std::istringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) parts.push_back(tok);
  return parts;
}
```

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `tools/compiler/compiler_main.cpp` | Create | CLI entry point: parse args, call Compiler + HostRunnerGen |
| `tools/compiler/CMakeLists.txt` | Create | `add_llvm_executable(compiler ...)` |
| `CMakeLists.txt` (root) | Modify | Add `add_subdirectory(tools/compiler)` |
| `test/tools/compiler/run_compiler.sh` | Create | Integration test shell script |

---

## Task 1: CMakeLists

**Files:**
- Create: `tools/compiler/CMakeLists.txt`
- Modify: root `CMakeLists.txt`

- [ ] **Step 1: Create `tools/compiler/CMakeLists.txt`**

```cmake
# tools/compiler/CMakeLists.txt
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(compiler
  compiler_main.cpp
)

target_link_libraries(compiler PRIVATE
  AscendCRuntime
)

target_include_directories(compiler PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
```

- [ ] **Step 2: Wire into root CMakeLists.txt**

In the root `CMakeLists.txt`, find the tools block (currently lines 94-96):
```cmake
add_subdirectory(tools/afir-opt)
add_subdirectory(tools/sim-validator)
add_subdirectory(tools/autotuner)
```
Add `add_subdirectory(tools/compiler)` after `add_subdirectory(tools/autotuner)`:
```cmake
add_subdirectory(tools/afir-opt)
add_subdirectory(tools/sim-validator)
add_subdirectory(tools/autotuner)
add_subdirectory(tools/compiler)
```

- [ ] **Step 3: Verify CMake configures without error (inside xvm)**

```bash
ssh xvm@orb
cd /home/niu/code/Ascend-MLIR
sleep 1  # wait for file sync; if sync unreliable, scp files first
bash scripts/build.sh --build-project --llvm-build-dir ~/code/llvm-project/llvm/build
```

Expected: CMake configures, ninja starts building. The build will fail because `compiler_main.cpp` doesn't exist yet — that's fine. Look for "compiler_main.cpp: No such file or directory" (expected) rather than a CMake config error.

- [ ] **Step 4: Commit**

```bash
git add tools/compiler/CMakeLists.txt CMakeLists.txt
git commit -m "build: add compiler CLI CMakeLists skeleton"
```

---

## Task 2: compiler_main.cpp implementation

**Files:**
- Create: `tools/compiler/compiler_main.cpp`

- [ ] **Step 1: Write `tools/compiler/compiler_main.cpp`**

```cpp
// tools/compiler/compiler_main.cpp
#include "Runtime/Compiler.h"
#include "Runtime/HostRunnerGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>
#include <string>
#include <vector>

using namespace mlir::runtime;
using namespace llvm;

static cl::opt<std::string> KernelFile("kernel",
    cl::desc("Kernel .cpp source file"), cl::Required);
static cl::opt<std::string> OutputDir("output",
    cl::desc("Output directory for .o, .bin, runner"), cl::init("./build"));
static cl::opt<std::string> KernelName("name",
    cl::desc("Kernel name (default: stem of --kernel filename)"), cl::init(""));
static cl::opt<std::string> SocVersion("soc",
    cl::desc("SoC version (default: Ascend910B1)"), cl::init("Ascend910B1"));
static cl::opt<std::string> Arch("arch",
    cl::desc("bisheng arch (default: dav-c220-vec)"), cl::init("dav-c220-vec"));
static cl::opt<int> NumInputs("num-inputs",
    cl::desc("Number of kernel inputs (for runner generation)"), cl::init(1));
static cl::opt<int> NumOutputs("num-outputs",
    cl::desc("Number of kernel outputs (for runner generation; only 1 supported)"),
    cl::init(1));
static cl::opt<std::string> TilingLayout("tiling-layout",
    cl::desc("Comma-separated tiling param types: int64,int64,int32,..."), cl::init(""));
static cl::opt<std::string> KernelType("kernel-type",
    cl::desc("Kernel type: vec | cube | mix (default: vec)"), cl::init("vec"));
static cl::opt<bool> Verbose("verbose",
    cl::desc("Print compilation commands"), cl::init(false));

static std::vector<std::string> splitComma(const std::string& s) {
  std::vector<std::string> parts;
  std::istringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) parts.push_back(tok);
  return parts;
}

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv, "AscendC Kernel Compiler\n");

  // Derive kernel name from filename if not given
  std::string kernel_name = KernelName.getValue();
  if (kernel_name.empty())
    kernel_name = llvm::sys::path::stem(KernelFile.getValue()).str();

  if (kernel_name.empty()) {
    llvm::errs() << "Error: cannot derive kernel name from --kernel; use --name\n";
    return 4;
  }

  // Validate numeric args
  if (NumInputs <= 0) {
    llvm::errs() << "Error: --num-inputs must be >= 1\n"; return 4;
  }
  if (NumOutputs <= 0) {
    llvm::errs() << "Error: --num-outputs must be >= 1\n"; return 4;
  }
  if (NumOutputs != 1) {
    llvm::errs() << "Error: --num-outputs " << NumOutputs
                 << " not supported (only 1 is implemented)\n";
    return 4;
  }

  // Validate input file exists (exit 4 = input error, per spec §5)
  if (!llvm::sys::fs::exists(KernelFile.getValue())) {
    llvm::errs() << "Error: kernel file not found: " << KernelFile << "\n";
    return 4;
  }

  // Step 1: compile kernel.cpp → .o + .bin
  Compiler::Config cc;
  cc.soc_version = SocVersion;
  cc.arch        = Arch;
  cc.verbose     = Verbose;

  Compiler compiler(cc);
  auto bin_or = compiler.Compile(KernelFile, OutputDir, kernel_name);
  if (!bin_or) {
    llvm::errs() << "Compilation error: "
                 << llvm::toString(bin_or.takeError()) << "\n";
    return 2;
  }
  llvm::outs() << "Compiled: " << *bin_or << "\n";

  // Step 2: generate runner executable
  HostRunnerGen::Config hcfg;
  hcfg.kernel_name   = kernel_name;
  hcfg.kernel_type   = KernelType;
  hcfg.soc_version   = SocVersion;
  hcfg.num_inputs    = NumInputs;
  hcfg.num_outputs   = NumOutputs;
  // splitComma("") → {""} which is wrong; guard for empty
  if (!TilingLayout.getValue().empty())
    hcfg.tiling_layout = splitComma(TilingLayout);
  hcfg.verbose       = Verbose;

  HostRunnerGen gen;
  auto runner_or = gen.Generate(hcfg, OutputDir);
  if (!runner_or) {
    llvm::errs() << "Runner generation error: "
                 << llvm::toString(runner_or.takeError()) << "\n";
    // g++ compile failure → exit 2; unsupported config (num_outputs) → exit 4.
    // num_outputs is pre-validated above (exit 4 if != 1), so any Generate()
    // error here is a genuine g++ failure → exit 2.
    return 2;
  }
  llvm::outs() << "Runner:   " << *runner_or << "\n";

  return 0;
}
```

- [ ] **Step 2: Copy to xvm (if file sync unreliable)**

```bash
scp tools/compiler/compiler_main.cpp xvm@orb:/home/niu/code/Ascend-MLIR/tools/compiler/compiler_main.cpp
```

- [ ] **Step 3: Build inside xvm**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && ninja -C build compiler 2>&1 | tail -20"
```

Expected: `[N/N] Linking CXX executable bin/compiler` — success.

- [ ] **Step 4: Commit**

```bash
git add tools/compiler/compiler_main.cpp
git commit -m "feat(tools): add compiler CLI (kernel.cpp → .o + .bin + runner)"
```

---

## Task 3: Integration test

**Files:**
- Create: `test/tools/compiler/run_compiler.sh`

The integration test verifies:
1. `compiler` exits 0 on a valid (but fake) `.cpp` — actually, bisheng won't be available in all CI environments, so the test verifies **argument parsing** and **exit codes** instead of a full compile.
2. `compiler` exits non-zero on missing `--kernel` file.
3. The binary exists and `--help` works.

> **Note:** Full bisheng compilation requires the CANN toolkit in the xvm container. The integration test uses `--help` and missing-file error paths to avoid needing bisheng. A separate manual test (documented in comments) covers the full pipeline when bisheng is available.

- [ ] **Step 1: Create `test/tools/compiler/run_compiler.sh`**

```bash
#!/usr/bin/env bash
set -e
cd /home/niu/code/Ascend-MLIR
source examples/env.sh

COMPILER=./build/bin/compiler

# Test 1: binary exists and --help exits 0
echo "--- Test 1: --help ---"
set +e
$COMPILER --help > /dev/null 2>&1
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "PASS: --help exits 0"
else
  echo "FAIL: --help exited $rc"
  exit 1
fi

# Test 2: missing --kernel file → exit 4 (input error: file not found)
echo "--- Test 2: missing kernel file ---"
set +e
$COMPILER --kernel /nonexistent/kernel.cpp --output /tmp/compiler_test 2>/dev/null
rc=$?
set -e
if [ "$rc" -eq 4 ]; then
  echo "PASS: missing kernel file → exit 4"
else
  echo "FAIL: expected exit 4, got $rc"
  exit 1
fi

# Test 3: --num-outputs 2 → exit 4 (unsupported, input error per spec §5)
echo "--- Test 3: --num-outputs 2 rejected ---"
echo "// dummy" > /tmp/dummy_kernel.cpp
set +e
$COMPILER --kernel /tmp/dummy_kernel.cpp \
          --output /tmp/compiler_test \
          --num-outputs 2 2>/dev/null
rc=$?
set -e
if [ "$rc" -eq 4 ]; then
  echo "PASS: --num-outputs 2 → exit 4"
else
  echo "FAIL: expected exit 4, got $rc"
  exit 1
fi

echo "ALL TESTS PASSED"
```

- [ ] **Step 2: Make executable and copy to xvm**

```bash
chmod +x test/tools/compiler/run_compiler.sh
mkdir -p test/tools/compiler
scp test/tools/compiler/run_compiler.sh xvm@orb:/home/niu/code/Ascend-MLIR/test/tools/compiler/run_compiler.sh
ssh xvm@orb "chmod +x /home/niu/code/Ascend-MLIR/test/tools/compiler/run_compiler.sh"
```

- [ ] **Step 3: Run test inside xvm**

```bash
ssh xvm@orb "bash /home/niu/code/Ascend-MLIR/test/tools/compiler/run_compiler.sh"
```

Expected:
```
--- Test 1: --help ---
PASS: --help exits 0
--- Test 2: missing kernel file ---
PASS: missing kernel file → exit 4
ALL TESTS PASSED
```

- [ ] **Step 4: Commit**

```bash
git add test/tools/compiler/run_compiler.sh
git commit -m "test(tools): add compiler CLI integration test"
```
