# Single Runtime CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `compiler`, `validator`, and `mix-validator` so `runtime-session` becomes the only general-purpose runtime CLI.

**Architecture:** The implementation deletes the three compatibility CLIs, rewires examples and regression harnesses to call `runtime-session` directly, then removes wrapper-only tests and build targets. `mix-compiler` and `autotuner` remain in place for now, and low-level runtime classes are left untouched unless they become dead as a direct result of the CLI cutover.

**Tech Stack:** C++, CMake, bash, xvm CPU simulation via Ascend/CANN simulator, LLVM/MLIR toolchain.

---

### Task 1: Remove compat CLI build targets and sources

**Files:**
- Delete: `tools/compiler/compiler_main.cpp`
- Delete: `tools/compiler/CMakeLists.txt`
- Delete: `tools/validator/validator_main.cpp`
- Delete: `tools/validator/CMakeLists.txt`
- Delete: `tools/mix-validator/mix_validator_main.cpp`
- Delete: `tools/mix-validator/CMakeLists.txt`
- Modify: `tools/CMakeLists.txt`
- Test: `cmake --build build --target runtime-session mix-compiler autotuner -j2`

- [ ] **Step 1: Write the failing test expectation**

The failure to create is a build/configuration failure if the deleted targets are still referenced.

```text
Expected failure before cleanup:
- CMake or ninja still tries to configure/build compiler
- CMake or ninja still tries to configure/build validator
- CMake or ninja still tries to configure/build mix-validator
```

- [ ] **Step 2: Verify the old targets are still wired**

Run:

```bash
cd /Volumes/GM9/code/Codex-Ascend-MLIR
rg -n "add_subdirectory\\((compiler|validator|mix-validator)\\)" tools/CMakeLists.txt
```

Expected: the three `add_subdirectory(...)` lines are present.

- [ ] **Step 3: Delete the three wrapper CLIs and drop their CMake entries**

Apply this exact edit to `tools/CMakeLists.txt`:

```cmake
add_subdirectory(afir-opt)
add_subdirectory(afir-translate)
add_subdirectory(autotuner)
add_subdirectory(mix-compiler)
add_subdirectory(runtime-session)
```

Then delete these files:

```text
tools/compiler/compiler_main.cpp
tools/compiler/CMakeLists.txt
tools/validator/validator_main.cpp
tools/validator/CMakeLists.txt
tools/mix-validator/mix_validator_main.cpp
tools/mix-validator/CMakeLists.txt
```

- [ ] **Step 4: Build the remaining CLI targets**

Run:

```bash
cd /Volumes/GM9/code/Codex-Ascend-MLIR
source examples/env.sh >/dev/null
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
cmake -S . -B build -DLLVM_BUILD_DIR="$LLVM_BUILD_DIR" >/dev/null
cmake --build build --target runtime-session mix-compiler autotuner -j2
```

Expected: exit code `0`.

- [ ] **Step 5: Commit**

```bash
git add tools/CMakeLists.txt tools/compiler tools/validator tools/mix-validator
git commit -m "refactor: remove compat runtime clis"
```

### Task 2: Rewire examples and runtime harnesses to `runtime-session`

**Files:**
- Modify: `examples/relu-broadcast-transpose/run.sh`
- Modify: `examples/add-broadcast-concat/run.sh`
- Modify: `examples/broadcast-add-reduce/run.sh`
- Modify: `examples/gather-elementwise-fusion/run.sh`
- Modify: `examples/split-relu-brc-add-mul/run.sh`
- Modify: `examples/matmul-add-leakyrelu/run.sh`
- Modify: `test/tools/runtime/run_simbackend_examples.sh`
- Modify: `test/tools/runtime/run_runtime.sh`
- Test: `bash test/tools/runtime/run_simbackend_examples.sh`

- [ ] **Step 1: Write the failing test expectation**

The failure to create is any remaining dependency on deleted wrapper binaries.

```bash
cd /Volumes/GM9/code/Codex-Ascend-MLIR
rg -n "build/bin/(compiler|validator|mix-validator)|\\bcompiler\\b|\\bvalidator\\b|mix-validator" \
  examples test/tools/runtime --glob 'run.sh' --glob '*.sh'
```

Expected before implementation: matches still exist.

- [ ] **Step 2: Replace wrapper invocations with `runtime-session` flows**

For each vec example, preserve the existing MLIR/codegen/test-data stages, but replace compile/run stages with:

```bash
build/bin/runtime-session \
  --kernel "${KERNEL_CPP}" \
  --kernel-kind vec \
  --name "${KERNEL_NAME}" \
  --output "${ARTIFACT_DIR}"

build/bin/runtime-session \
  --run-manifest "${RUN_MANIFEST}" \
  --run
```

For the mix example, keep `mix-compiler` as the compile entry and preserve:

```bash
build/bin/runtime-session \
  --run-manifest "${RUN_MANIFEST}" \
  --run
```

Update `test/tools/runtime/run_simbackend_examples.sh` and `test/tools/runtime/run_runtime.sh` so they only build/check:

```text
runtime-session
mix-compiler
```

and never reference deleted wrapper binaries.

- [ ] **Step 3: Verify no deleted CLI references remain in examples/harness**

Run:

```bash
cd /Volumes/GM9/code/Codex-Ascend-MLIR
rg -n "build/bin/(compiler|validator|mix-validator)|\\bcompiler\\b|\\bvalidator\\b|mix-validator" \
  examples test/tools/runtime --glob 'run.sh' --glob '*.sh'
```

Expected: no matches that reference the deleted runtime wrapper CLIs. `mix-compiler` matches are allowed.

- [ ] **Step 4: Run the xvm CPU simulation baseline**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && \
  source examples/env.sh >/dev/null && \
  export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && \
  bash test/tools/runtime/run_simbackend_examples.sh'
```

Expected: exit code `0` and summary ending with:

```text
SimBackend example baseline passed
```

- [ ] **Step 5: Commit**

```bash
git add examples test/tools/runtime
git commit -m "refactor: route example flows through runtime session"
```

### Task 3: Remove wrapper-only tests and assertions

**Files:**
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Modify: `test/tools/runtime/run_runtime.sh`
- Test: `g++ ... test/tools/runtime/test_taskgraph_runtime.cpp ...`

- [ ] **Step 1: Write the failing test expectation**

The failure to create is any focused runtime test that still assumes the deleted CLIs exist.

Run:

```bash
cd /Volumes/GM9/code/Codex-Ascend-MLIR
rg -n "build/bin/compiler|compat compiler|validator_main|mix-validator|mix validator" \
  test/tools/runtime/test_taskgraph_runtime.cpp test/tools/runtime/run_runtime.sh
```

Expected before implementation: matches exist.

- [ ] **Step 2: Delete wrapper-only assertions and preserve runtime-core coverage**

Remove test coverage that only verifies:

```text
compiler CLI argument parsing
compiler CLI runner warning behavior
validator CLI-specific logging
mix-validator CLI shell behavior
```

Keep coverage that still validates:

```text
CompatRuntime request/manifest construction
runtime-session planning/run behavior
ExecutionSession/Backend semantics
profile output normalization
```

If a test currently shells out to `build/bin/compiler`, delete it rather than rewriting it into a second CLI.

- [ ] **Step 3: Compile the focused runtime test binary**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && \
  source examples/env.sh >/dev/null && \
  export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && \
  bash test/tools/runtime/run_runtime.sh'
```

Expected: exit code `0`.

- [ ] **Step 4: Commit**

```bash
git add test/tools/runtime
git commit -m "test: remove compat cli runtime coverage"
```

### Task 4: Final xvm verification and documentation cleanup

**Files:**
- Modify: `docs/superpowers/specs/2026-04-11-runtime-compat-replacement-design.md`
- Modify: any repo docs that still recommend `compiler`, `validator`, or `mix-validator`
- Test: xvm full focused runtime verification

- [ ] **Step 1: Write the failing test expectation**

The failure to create is stale documentation that still presents deleted wrapper CLIs as valid runtime entrypoints.

Run:

```bash
cd /Volumes/GM9/code/Codex-Ascend-MLIR
rg -n "\\bcompiler\\b|\\bvalidator\\b|mix-validator" docs examples README* --glob '!build/**'
```

Expected before implementation: matches remain.

- [ ] **Step 2: Rewrite docs to reflect the final CLI architecture**

Update docs so they say:

```text
runtime-session is the only general-purpose runtime CLI
mix-compiler remains a specialized compile tool
autotuner remains separate
compiler / validator / mix-validator have been removed
```

Do not rewrite historical design docs beyond adding a short note that the
compatibility-shell phase was superseded by the single-runtime-CLI cutover.

- [ ] **Step 3: Run the final verification set on xvm**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && \
  source examples/env.sh >/dev/null && \
  export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && \
  bash test/tools/runtime/run_runtime.sh && \
  bash test/tools/runtime/run_simbackend_examples.sh'
```

Expected:

```text
run_runtime.sh exits 0
run_simbackend_examples.sh exits 0
SimBackend example baseline passed
```

- [ ] **Step 4: Commit**

```bash
git add docs
git commit -m "docs: finalize single runtime cli architecture"
```

## Self-Review

Spec coverage check:

- single-center CLI architecture: covered by Tasks 1 and 4
- removing `compiler` / `validator` / `mix-validator`: covered by Task 1
- examples and harnesses moved to `runtime-session`: covered by Task 2
- wrapper-only tests removed: covered by Task 3
- xvm CPU simulation acceptance: covered by Tasks 2 and 4
- `mix-compiler` and `autotuner` intentionally retained: reflected in Tasks 1 and 4

Placeholder scan:

- No `TODO`/`TBD` placeholders remain
- Every task includes concrete files, commands, and expected outcomes

Type/interface consistency:

- The plan consistently treats `runtime-session` as the only general-purpose CLI
- `mix-compiler` is consistently retained as a specialized tool
- `autotuner` is consistently deferred
