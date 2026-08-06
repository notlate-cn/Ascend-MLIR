# RuntimeMix Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an isolated `RuntimeMix` fork beside `Runtime` so mix-kernel toolchain work can be developed and debugged without touching `lib/Runtime`.

**Architecture:** Copy the current `Runtime` library into parallel `include/RuntimeMix` and `lib/RuntimeMix` trees, rename the exported library target to `AscendCRuntimeMix`, and add a tiny standalone smoke-test tool that links only against the new library. Keep the copied implementation behaviorally identical for now; later mix-specific backend work can happen entirely inside `RuntimeMix` without destabilizing the main runtime.

**Tech Stack:** C++, CMake, LLVM Support, existing MLIR build macros

---

### Task 1: Freeze The Bootstrap Scope

**Files:**
- Create: `docs/superpowers/plans/2026-03-31-runtime-mix-bootstrap.md`
- Modify: none
- Test: none

- [ ] **Step 1: Record the source file inventory to fork**

Use this exact inventory from the current tree:

```text
include/Runtime/Compiler.h
include/Runtime/Executor.h
include/Runtime/HostRunnerGen.h
include/Runtime/NpyIO.h
include/Runtime/SimValidator.h
include/Runtime/TilingSchema.h
include/Runtime/Types.h
lib/Runtime/Compiler.cpp
lib/Runtime/Executor.cpp
lib/Runtime/HostRunnerGen.cpp
lib/Runtime/NpyIO.cpp
lib/Runtime/SimValidator.cpp
lib/Runtime/TilingSchema.cpp
lib/Runtime/CMakeLists.txt
```

- [ ] **Step 2: Define the isolation rules**

The bootstrap must obey these constraints:

```text
1. Do not modify files under lib/Runtime or include/Runtime.
2. Do not rename or relink existing AscendCRuntime users.
3. Only add new RuntimeMix paths and the smallest required top-level CMake hook.
4. New target names must be unique: AscendCRuntimeMix and a new smoke-test executable.
5. Bootstrap should compile before any mix-specific backend redesign starts.
```

- [ ] **Step 3: Confirm the first deliverable**

The first milestone is:

```text
cmake build can compile a new RuntimeMix library and a tiny executable that links it,
without changing existing Runtime library behavior or tool targets.
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-03-31-runtime-mix-bootstrap.md
git commit -m "docs: add runtime mix bootstrap plan"
```

Expected: skip this commit during bootstrap if code and docs are being landed together in one changeset.

### Task 2: Fork The Runtime Library Skeleton

**Files:**
- Create: `include/RuntimeMix/Compiler.h`
- Create: `include/RuntimeMix/Executor.h`
- Create: `include/RuntimeMix/HostRunnerGen.h`
- Create: `include/RuntimeMix/NpyIO.h`
- Create: `include/RuntimeMix/SimValidator.h`
- Create: `include/RuntimeMix/TilingSchema.h`
- Create: `include/RuntimeMix/Types.h`
- Create: `lib/RuntimeMix/Compiler.cpp`
- Create: `lib/RuntimeMix/Executor.cpp`
- Create: `lib/RuntimeMix/HostRunnerGen.cpp`
- Create: `lib/RuntimeMix/NpyIO.cpp`
- Create: `lib/RuntimeMix/SimValidator.cpp`
- Create: `lib/RuntimeMix/TilingSchema.cpp`
- Create: `lib/RuntimeMix/CMakeLists.txt`
- Modify: `CMakeLists.txt`
- Test: configure/build output for the new library target

- [ ] **Step 1: Copy the public headers into the new include tree**

Create `include/RuntimeMix/` and copy each `include/Runtime/*.h` file into it. Update include guards or include paths only where necessary so the copied headers include `RuntimeMix/...` peers rather than `Runtime/...`.

- [ ] **Step 2: Copy the library sources into the new implementation tree**

Create `lib/RuntimeMix/` and copy each `lib/Runtime/*.cpp` file into it. Update the includes at the top of each file from `Runtime/...` to `RuntimeMix/...`.

- [ ] **Step 3: Add an independent CMake target**

Create `lib/RuntimeMix/CMakeLists.txt` with the same source list as `lib/Runtime/CMakeLists.txt`, but export a different library name:

```cmake
add_mlir_library(AscendCRuntimeMix
  Compiler.cpp
  Executor.cpp
  HostRunnerGen.cpp
  SimValidator.cpp
  NpyIO.cpp
  TilingSchema.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/RuntimeMix

  LINK_LIBS PUBLIC
  LLVMSupport
)
```

- [ ] **Step 4: Register the new subdirectory**

Modify the top-level `CMakeLists.txt` and add exactly one new line beside the existing runtime entry:

```cmake
add_subdirectory(lib/Runtime)
add_subdirectory(lib/RuntimeMix)
```

- [ ] **Step 5: Run a focused configure/build**

Run:

```bash
cmake -S . -B build/runtime-mix-bootstrap
cmake --build build/runtime-mix-bootstrap --target AscendCRuntimeMix -j
```

Expected: `AscendCRuntimeMix` builds successfully without rebuilding errors in unrelated targets.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt include/RuntimeMix lib/RuntimeMix
git commit -m "feat: add isolated runtime mix library scaffold"
```

### Task 3: Add A Standalone Smoke-Test Entry Point

**Files:**
- Create: `tools/mix-compiler/CMakeLists.txt`
- Create: `tools/mix-compiler/mix-compiler.cpp`
- Modify: `tools/CMakeLists.txt`
- Test: build output for `mix-compiler`

- [ ] **Step 1: Add the tool source**

Create `tools/mix-compiler/mix-compiler.cpp` as a minimal executable that proves the new library links. Keep it intentionally small:

```cpp
#include "RuntimeMix/Compiler.h"
#include "llvm/Support/raw_ostream.h"

int main() {
  mlir::runtime::CompilerConfig cfg;
  cfg.kernel_type = "mix";
  llvm::outs() << "RuntimeMix smoke test: kernel_type=" << cfg.kernel_type << "\n";
  return 0;
}
```

- [ ] **Step 2: Add the tool CMake target**

Create `tools/mix-compiler/CMakeLists.txt`:

```cmake
add_llvm_executable(mix-compiler mix-compiler.cpp)

llvm_update_compile_flags(mix-compiler)

target_link_libraries(mix-compiler PRIVATE
  AscendCRuntimeMix
)
```

- [ ] **Step 3: Register the tool subdirectory**

Modify `tools/CMakeLists.txt` and add:

```cmake
add_subdirectory(mix-compiler)
```

- [ ] **Step 4: Build and run the smoke test**

Run:

```bash
cmake --build build/runtime-mix-bootstrap --target mix-compiler -j
./build/runtime-mix-bootstrap/bin/mix-compiler
```

Expected output:

```text
RuntimeMix smoke test: kernel_type=mix
```

- [ ] **Step 5: Commit**

```bash
git add tools/CMakeLists.txt tools/mix-compiler
git commit -m "feat: add runtime mix smoke test tool"
```

### Task 4: Validate In The XVM Environment

**Files:**
- Modify: none
- Test: xvm configure/build/run logs

- [ ] **Step 1: Sync the repository state into xvm**

Run the repository validation in the xvm workspace, not on the host:

```bash
orb -m xvm sh -lc 'cd /home/niu/code/Ascend-MLIR && cmake -S . -B build/runtime-mix-bootstrap && cmake --build build/runtime-mix-bootstrap --target AscendCRuntimeMix mix-compiler -j'
```

Expected: both targets build successfully.

- [ ] **Step 2: Run the smoke test inside xvm**

Run:

```bash
orb -m xvm sh -lc 'cd /home/niu/code/Ascend-MLIR && ./build/runtime-mix-bootstrap/bin/mix-compiler'
```

Expected output:

```text
RuntimeMix smoke test: kernel_type=mix
```

- [ ] **Step 3: Record the non-goals of this bootstrap**

Bootstrap is successful even if these are still unimplemented:

```text
- No mix-specific CMake backend wrapper yet
- No dedicated pack/launch flow yet
- No integration into existing compiler/validator tools yet
```

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "test: validate runtime mix bootstrap in xvm"
```

## Self-Review

- Spec coverage: the plan creates an isolated `RuntimeMix` fork, keeps `lib/Runtime` untouched, adds an independent build target, adds a dedicated test entrypoint, and validates in xvm.
- Placeholder scan: no `TODO` or unspecified "appropriate handling" steps remain.
- Type consistency: all planned code uses `mlir::runtime::CompilerConfig` from the copied headers and the new CMake target name `AscendCRuntimeMix`.
