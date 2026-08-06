# RuntimeMix AscendC Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `RuntimeMix`-local AscendC CMake backend and a dedicated `mix-validator` so mix kernels can be compiled and executed in sim from source without touching `lib/Runtime`.

**Architecture:** Build on the isolated `RuntimeMix` scaffold by adding a small backend that materializes a temporary AscendC-style CMake project, invokes configure/build/install, and returns a stable artifact description. Keep execution separate in a `mix-validator` tool that runs the sample-style ACL sim path for mix kernels and compares outputs against golden data.

**Tech Stack:** C++, CMake, LLVM Support, AscendC sample toolchain, ACL simulator

---

### Task 1: Define The RuntimeMix Artifact Boundary

**Files:**
- Create: `include/RuntimeMix/MixArtifact.h`
- Modify: `tools/mix-compiler/mix_compiler_main.cpp`
- Test: xvm build of `mix-compiler`

- [ ] **Step 1: Add the shared artifact model header**

Create `include/RuntimeMix/MixArtifact.h` with a minimal stable result struct:

```cpp
#pragma once
#include <string>

namespace mlir::runtime {

struct MixArtifact {
  std::string kernel_name;
  std::string soc_version;
  std::string work_dir;
  std::string build_dir;
  std::string install_dir;
  std::string kernel_so_path;
  std::string launcher_header_dir;
  std::string host_runner_path;
  std::string manifest_path;
};

} // namespace mlir::runtime
```

- [ ] **Step 2: Include the artifact header in the compiler tool**

Modify `tools/mix-compiler/mix_compiler_main.cpp` so it includes the new header even before the backend exists:

```cpp
#include "RuntimeMix/MixArtifact.h"
```

Expected: this is a no-op behavior change, but it locks the include boundary before implementation grows.

- [ ] **Step 3: Build `mix-compiler` in xvm**

Run:

```bash
orb -m xvm sh -lc 'cd /home/niu/code/Ascend-MLIR && cmake --build build/runtime-mix-bootstrap --target mix-compiler -j4'
```

Expected: `mix-compiler` still builds successfully.

- [ ] **Step 4: Commit**

```bash
git add include/RuntimeMix/MixArtifact.h tools/mix-compiler/mix_compiler_main.cpp
git commit -m "feat: add runtime mix artifact model"
```

### Task 2: Add The AscendC CMake Backend

**Files:**
- Create: `include/RuntimeMix/AscendCMixCompiler.h`
- Create: `lib/RuntimeMix/AscendCMixCompiler.cpp`
- Modify: `lib/RuntimeMix/CMakeLists.txt`
- Test: xvm build of `AscendCRuntimeMix`

- [ ] **Step 1: Add the backend interface**

Create `include/RuntimeMix/AscendCMixCompiler.h`:

```cpp
#pragma once
#include "RuntimeMix/MixArtifact.h"
#include "llvm/Support/Error.h"
#include <string>

namespace mlir::runtime {

struct AscendCMixCompileConfig {
  std::string kernel_src;
  std::string kernel_name;
  std::string soc_version = "Ascend910B1";
  std::string output_dir;
  std::string ascend_cmake_dir;
};

class AscendCMixCompiler {
public:
  llvm::Expected<MixArtifact> Compile(const AscendCMixCompileConfig &cfg);
};

} // namespace mlir::runtime
```

- [ ] **Step 2: Implement the minimal backend**

Create `lib/RuntimeMix/AscendCMixCompiler.cpp` with these responsibilities:

```cpp
#include "RuntimeMix/AscendCMixCompiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include <fstream>
#include <optional>
#include <vector>
```

The implementation must:

```text
1. Create output_dir/work, output_dir/build, output_dir/out
2. Write a temporary CMakeLists.txt that mirrors examples/baremix-test/CMakeLists.txt
3. Copy or reference the kernel source file directly
4. Reuse examples/baremix-test/cmake/npu_lib.cmake
5. Run:
   cmake -S <work> -B <build> -DSOC_VERSION=... -DCMAKE_INSTALL_PREFIX=<out>
   cmake --build <build> --target install -j4
6. Verify these outputs exist:
   out/lib/libascendc_kernels_sim.so
   out/include/ascendc_kernels_sim
7. Return a MixArtifact with stable absolute paths
```

- [ ] **Step 3: Register the backend in the library target**

Modify `lib/RuntimeMix/CMakeLists.txt` and add:

```cmake
  AscendCMixCompiler.cpp
```

- [ ] **Step 4: Build `AscendCRuntimeMix` in xvm**

Run:

```bash
orb -m xvm sh -lc 'cd /home/niu/code/Ascend-MLIR && cmake --build build/runtime-mix-bootstrap --target AscendCRuntimeMix -j4'
```

Expected: library builds successfully with the new backend.

- [ ] **Step 5: Commit**

```bash
git add include/RuntimeMix/AscendCMixCompiler.h lib/RuntimeMix/AscendCMixCompiler.cpp lib/RuntimeMix/CMakeLists.txt
git commit -m "feat: add runtime mix AscendC CMake backend"
```

### Task 3: Upgrade `mix-compiler` Into A Real Compiler Tool

**Files:**
- Modify: `tools/mix-compiler/mix_compiler_main.cpp`
- Test: xvm run of `mix-compiler`

- [ ] **Step 1: Replace the smoke-test CLI with compile arguments**

Modify `tools/mix-compiler/mix_compiler_main.cpp` to parse:

```cpp
static llvm::cl::opt<std::string> KernelFile("kernel", llvm::cl::Required);
static llvm::cl::opt<std::string> OutputDir("output", llvm::cl::init("./build/mix"));
static llvm::cl::opt<std::string> KernelName("name", llvm::cl::init(""));
static llvm::cl::opt<std::string> SocVersion("soc", llvm::cl::init("Ascend910B1"));
```

- [ ] **Step 2: Call the new backend and print artifact paths**

Main flow must:

```cpp
1. Derive kernel name from --kernel if --name is empty
2. Fill AscendCMixCompileConfig
3. Call AscendCMixCompiler::Compile
4. Print:
   kernel_name=...
   kernel_so=...
   launcher_header_dir=...
   install_dir=...
```

On error, print `llvm::toString(...)` and exit non-zero.

- [ ] **Step 3: Validate compile flow against baremix in xvm**

Run:

```bash
orb -m xvm sh -lc 'cd /home/niu/code/Ascend-MLIR && ./build/runtime-mix-bootstrap/bin/mix-compiler --kernel examples/baremix-test/baremix_custom.cpp --name baremix_custom --output build/runtime-mix-baremix --soc Ascend910B1'
```

Expected: command succeeds and prints a valid `kernel_so=` path under `build/runtime-mix-baremix/out/lib/`.

- [ ] **Step 4: Commit**

```bash
git add tools/mix-compiler/mix_compiler_main.cpp
git commit -m "feat: turn mix-compiler into a real compile driver"
```

### Task 4: Add A Dedicated Mix Validator Tool

**Files:**
- Create: `tools/mix-validator/CMakeLists.txt`
- Create: `tools/mix-validator/mix_validator_main.cpp`
- Modify: `CMakeLists.txt`
- Test: xvm build of `mix-validator`

- [ ] **Step 1: Add the validator executable target**

Create `tools/mix-validator/CMakeLists.txt`:

```cmake
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(mix-validator
  mix_validator_main.cpp
)

target_link_libraries(mix-validator PRIVATE
  AscendCRuntimeMix
)

target_include_directories(mix-validator PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
```

- [ ] **Step 2: Register the new tool**

Modify the top-level `CMakeLists.txt` and add:

```cmake
add_subdirectory(tools/mix-validator)
```

- [ ] **Step 3: Implement a minimal validator entrypoint**

Create `tools/mix-validator/mix_validator_main.cpp` that parses:

```cpp
--artifact-root
--input-dir
--golden
--runner
```

The first implementation may shell out to the verified `examples/baremix-test` style host runner binary inside the artifact install tree rather than re-implement ACL invocation immediately.

- [ ] **Step 4: Build the new tool in xvm**

Run:

```bash
orb -m xvm sh -lc 'cd /home/niu/code/Ascend-MLIR && cmake --build build/runtime-mix-bootstrap --target mix-validator -j4'
```

Expected: `mix-validator` builds successfully.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt tools/mix-validator
git commit -m "feat: add mix validator tool"
```

### Task 5: Close The Baremix End-To-End Loop In XVM

**Files:**
- Modify: `lib/RuntimeMix/AscendCMixCompiler.cpp`
- Modify: `tools/mix-validator/mix_validator_main.cpp`
- Test: xvm end-to-end compile and validate commands

- [ ] **Step 1: Make the backend emit or preserve a runnable host entry**

If the first backend version only produced `libascendc_kernels_sim.so`, extend it so the install tree also contains a runnable host executable or wrapper script path and record it in `MixArtifact::host_runner_path`.

- [ ] **Step 2: Wire the validator to baremix inputs**

The validator must support this workflow:

```text
1. read baremix artifact root
2. use existing baremix input binaries or generated data
3. run the host entry in sim mode
4. compare output against golden
5. print PASS/FAIL plus max/mean diff
```

- [ ] **Step 3: Run end-to-end compile**

Run:

```bash
orb -m xvm sh -lc 'cd /home/niu/code/Ascend-MLIR && ./build/runtime-mix-bootstrap/bin/mix-compiler --kernel examples/baremix-test/baremix_custom.cpp --name baremix_custom --output build/runtime-mix-baremix --soc Ascend910B1'
```

Expected: compile succeeds.

- [ ] **Step 4: Run end-to-end validation**

Run:

```bash
orb -m xvm sh -lc 'cd /home/niu/code/Ascend-MLIR/examples/baremix-test && python3 scripts/gen_data.py && cd /home/niu/code/Ascend-MLIR && ./build/runtime-mix-bootstrap/bin/mix-validator --artifact-root build/runtime-mix-baremix --input-dir examples/baremix-test/input --golden examples/baremix-test/output/golden.bin'
```

Expected: validator prints PASS and zero-diff or equivalent successful verification.

- [ ] **Step 5: Commit**

```bash
git add include/RuntimeMix lib/RuntimeMix tools/mix-compiler tools/mix-validator
git commit -m "feat: validate runtime mix AscendC wrapper end to end"
```

## Self-Review

- Spec coverage: the plan adds the backend, stable artifact model, `mix-compiler`, `mix-validator`, and xvm baremix end-to-end validation.
- Placeholder scan: no `TODO`, `TBD`, or implied later work appears inside execution steps.
- Type consistency: `MixArtifact`, `AscendCMixCompileConfig`, `AscendCMixCompiler`, `mix-compiler`, and `mix-validator` names are used consistently throughout.
