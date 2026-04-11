# Runtime Compat Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old top-level runtime entrypoints (`examples/run.sh`, `compiler`/`validator`/`mix-validator`, and `lib/CAPI/Runtime`) with adapters over the new `lib/Runtime` task-graph runtime while preserving existing external interfaces during migration.

**Architecture:** Keep the current public surfaces and script flow, but move compile and execution orchestration behind `ArtifactCompiler`, `ExecutionSession`, `SimBackend`, and `NpuBackend`. Implement this as a thin compatibility layer rather than a second execution stack. Validate the migration through focused runtime tests, xvm regression, and real `SimBackend` CPU simulation of the 6 example pipelines.

**Tech Stack:** C++, LLVM support libraries, existing AFIR tools, Bash example pipelines, C API shim layer, Catch2 runtime tests, xvm-based build/test flow.

---

## File Map

### Core runtime-facing files

- Modify: `tools/compiler/compiler_main.cpp`
  - Replace direct compile orchestration with an adapter over `ArtifactCompiler`.
- Modify: `tools/validator/validator_main.cpp`
  - Replace direct `Executor + SimValidator` orchestration with a single-task runtime-session style execution path.
- Modify: `tools/mix-validator/mix_validator_main.cpp`
  - Replace standalone mix execution orchestration with runtime-manifest/session adaptation.
- Modify: `lib/CAPI/Runtime/Runtime.cpp`
  - Retarget C API internals to new runtime-backed compile/run behavior while preserving exported ABI.

### Shared helper layer

- Create: `include/Runtime/CompatRuntime.h`
  - Shared adapter-facing helper declarations for compile/run manifest construction and compatibility execution.
- Create: `lib/Runtime/CompatRuntime.cpp`
  - Shared implementation used by CLI shims and C API.

### Example pipeline files

- Modify: `examples/relu-broadcast-transpose/run.sh`
- Modify: `examples/add-broadcast-concat/run.sh`
- Modify: `examples/broadcast-add-reduce/run.sh`
- Modify: `examples/gather-elementwise-fusion/run.sh`
- Modify: `examples/split-relu-brc-add-mul/run.sh`
- Modify: `examples/matmul-add-leakyrelu/run.sh`
  - Switch the default compile/validate path to the new runtime-backed flow while preserving lowering/codegen stages.

### Tests and verification

- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
  - Add focused compatibility tests for compiler/validator/mix-validator/C API shims.
- Modify: `test/tools/runtime/run_runtime.sh`
  - Keep focused xvm regression aligned with new compatibility surfaces.
- Modify: `test/tools/runtime/run_simbackend_examples.sh`
  - Ensure the 6 example baselines exercise the default runtime-backed path.
- Create: `test/tools/runtime/test_runtime_compat_cli.sh`
  - Focused shell-level regression for legacy CLI entrypoints if needed.

## Task 1: Add Shared Compatibility Runtime Helpers

**Files:**
- Create: `include/Runtime/CompatRuntime.h`
- Create: `lib/Runtime/CompatRuntime.cpp`
- Modify: `include/Runtime/ArtifactCompiler.h`
- Modify: `lib/Runtime/ArtifactCompiler.cpp`
- Modify: `lib/Runtime/CMakeLists.txt`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing focused runtime tests**

Add tests that describe the helper layer behavior before implementation:

```cpp
TEST_CASE("compat compile request preserves requested kernel name") {
  auto request = buildCompatCompileRequest("/tmp/kernel.cpp", "/tmp/out",
                                           "legacy_name", "Ascend910B1",
                                           "dav-c220-vec", "vec", false);
  REQUIRE(request.kernelSourcePath == "/tmp/kernel.cpp");
  REQUIRE(request.outputRoot == "/tmp/out");
  REQUIRE(request.requestedKernelName == "legacy_name");
}

TEST_CASE("compat validator request builds single-task manifest inputs") {
  CompatValidateOptions options;
  options.artifactRoot = "/tmp/artifact";
  options.inputPaths = {"/tmp/a.npy", "/tmp/b.npy"};
  options.expectedOutputPath = "/tmp/out.npy";
  options.blockDim = 8;
  options.atol = 1e-2;
  options.rtol = 1e-2;

  auto manifest = buildCompatSingleTaskRunManifest(options);
  REQUIRE(manifest.tasks.size() == 1);
  REQUIRE(manifest.tasks[0].inputs.size() == 2);
  REQUIRE(manifest.tasks[0].expectedOutputs.size() == 1);
  REQUIRE(manifest.tasks[0].blockDim == 8);
}
```

- [ ] **Step 2: Run focused tests to verify they fail**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:
- compile failure because `buildCompatCompileRequest` / `buildCompatSingleTaskRunManifest` do not exist yet

- [ ] **Step 3: Implement the helper declarations**

Create `include/Runtime/CompatRuntime.h` with focused adapter types and function declarations:

```cpp
struct CompatCompileOptions {
  std::string kernelSourcePath;
  std::string outputRoot;
  std::string requestedKernelName;
  std::string socVersion;
  std::string arch;
  std::string kernelType;
  bool verbose = false;
};

struct CompatValidateOptions {
  std::string artifactRoot;
  std::vector<std::string> inputPaths;
  std::string expectedOutputPath;
  std::string actualOutputPath;
  std::string tilingSchemaPath;
  std::string tilingParams;
  std::string tilingBinaryPath;
  std::optional<std::vector<int64_t>> actualOutputShape;
  std::optional<DType> actualOutputDType;
  int blockDim = 1;
  double atol = 1.0;
  double rtol = 1e-2;
};

llvm::Expected<ArtifactCompileRequest>
buildCompatCompileRequest(const CompatCompileOptions &);
llvm::Expected<RunManifestSpec>
buildCompatSingleTaskRunManifest(const CompatValidateOptions &);
```

- [ ] **Step 4: Implement the helper definitions**

Create `lib/Runtime/CompatRuntime.cpp` with minimal conversion logic:

```cpp
llvm::Expected<ArtifactCompileRequest> buildCompatCompileRequest(
    const CompatCompileOptions &options) {
  if (options.kernelType != "vec" && options.kernelType != "cube" &&
      options.kernelType != "mix")
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported kernel type: %s",
                                   options.kernelType.c_str());

  ArtifactCompileRequest request;
  request.kernelSource = options.kernelSourcePath;
  request.kernelName = options.requestedKernelName;
  request.socVersion = options.socVersion;
  request.arch = options.arch;
  request.kernelKind = options.kernelType == "cube" ? KernelKind::Cube
                    : options.kernelType == "mix"  ? KernelKind::Mix
                                                   : KernelKind::Vec;
  request.outputDir = options.outputRoot;
  request.verbose = options.verbose;
  return request;
}
```

and

```cpp
llvm::Expected<RunManifestSpec> buildCompatSingleTaskRunManifest(
    const CompatValidateOptions &options) {
  RunManifestSpec manifest;
  manifest.backendKind = ExecutionBackendKind::Simulation;

  RunTaskSpec task;
  task.taskId = "main";
  task.artifactRoot = options.artifactRoot;
  task.invocation.blockDim = options.blockDim;
  task.invocation.atol = options.atol;
  task.invocation.rtol = options.rtol;
  // Build file-backed input and output bindings here.
  manifest.tasks.push_back(std::move(task));
  return manifest;
}
```

If helper hardening reveals that `ArtifactCompileRequest` must carry `arch` and
`verbose` to avoid dead compatibility fields, this task is allowed to make the
smallest required changes in:

```cpp
include/Runtime/ArtifactCompiler.h
lib/Runtime/ArtifactCompiler.cpp
```

That extension remains part of Task 1 rather than being deferred to Task 2.

- [ ] **Step 5: Wire the helper into the runtime library**

Update `lib/Runtime/CMakeLists.txt` to compile the new compatibility helper:

```cmake
add_mlir_library(AscendCRuntime
  ...
  CompatRuntime.cpp
)
```

- [ ] **Step 6: Run focused regression to verify the helper passes**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:
- `test_taskgraph_runtime` passes with the new helper tests included

- [ ] **Step 7: Commit**

```bash
git add include/Runtime/CompatRuntime.h lib/Runtime/CompatRuntime.cpp \
  include/Runtime/ArtifactCompiler.h lib/Runtime/ArtifactCompiler.cpp \
  lib/Runtime/CMakeLists.txt test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "feat: add runtime compatibility adapter helpers"
```

## Task 2: Retarget `compiler` to `ArtifactCompiler`

**Files:**
- Modify: `tools/compiler/compiler_main.cpp`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Test: `test/tools/runtime/run_runtime.sh`

- [ ] **Step 1: Write the failing compiler adapter test**

Add a focused test that asserts the compatibility compiler path uses normalized artifacts:

```cpp
TEST_CASE("compat compiler emits artifact manifest path") {
  auto result = runCompatCompilerForTest(...);
  REQUIRE(result.manifestPath.endswith("/manifest.txt"));
  REQUIRE(result.kernelArtifact.kernelName == "legacy_name");
}
```

- [ ] **Step 2: Run focused regression to verify it fails**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:
- test failure because `compiler_main.cpp` still uses `Compiler`

- [ ] **Step 3: Replace direct compile orchestration in `compiler_main.cpp`**

Refactor the tool to:

```cpp
CompatCompileOptions options;
options.kernelSourcePath = KernelFile;
options.outputRoot = OutputDir;
options.requestedKernelName = kernel_name;
options.socVersion = resolvedSocVersion;
options.arch = Arch;
options.kernelType = KernelType;
options.verbose = Verbose;

ArtifactCompiler compiler;
auto artifactOr = compiler.compile(buildCompatCompileRequest(options));
```

Then print compatibility output using artifact fields instead of old `Compiler::Compile` return values.

- [ ] **Step 4: Preserve optional runner-generation compatibility**

Keep the existing runner-generation path as compatibility output only. In this phase,
it is acceptable to continue attempting runner generation for current CLI kernel kinds
(`vec`, `cube`, and `mix`) while old scripts may still consume runner artifacts:

```cpp
if (shouldAttemptLegacyRunnerCompatibility(KernelType)) {
  auto runnerOr = generateLegacyRunnerCompatibility(...);
  ...
}
```

Requirements:

- runner generation must not remain the main compile success criterion
- runner failure must degrade to warning/non-fatal compatibility output
- runner-specific argument validation must happen before compile/artifact emission
  on paths where runner compatibility is attempted

- [ ] **Step 5: Run focused regression**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:
- focused runtime tests pass
- compiler CLI planning/compile checks still pass

- [ ] **Step 6: Commit**

```bash
git add tools/compiler/compiler_main.cpp test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "refactor: route compiler through artifact compiler"
```

## Task 3: Retarget `validator` to `ExecutionSession`

**Files:**
- Modify: `tools/validator/validator_main.cpp`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Test: `test/tools/runtime/run_runtime.sh`

- [ ] **Step 1: Write the failing validator adapter test**

Add a test that asserts the validator compatibility layer constructs a runtime manifest and returns a session result:

```cpp
TEST_CASE("compat validator builds runtime session request") {
  CompatValidateOptions options;
  ...
  auto sessionInput = buildCompatSingleTaskRunManifest(options);
  REQUIRE(sessionInput.tasks.size() == 1);
  REQUIRE(sessionInput.tasks[0].backend == "sim");
}
```

- [ ] **Step 2: Run focused regression to verify it fails if adaptation is incomplete**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:
- current validator path still bypasses runtime-session style execution

- [ ] **Step 3: Replace `Executor + SimValidator` orchestration**

Refactor `validator_main.cpp` to:

```cpp
CompatValidateOptions options;
...
auto manifest = buildCompatSingleTaskRunManifest(options);
ExecutionSession session(makeExecutionBackend("sim"));
auto result = session.run(buildTaskGraphFromManifest(manifest));
```

Keep:
- `--dump-actual`
- `--dump-expected`
- tolerance comparison behavior
- block-dim and tiling handling

- [ ] **Step 4: Preserve legacy argument surface and exit behavior**

Keep existing arguments and continue to fail with user-input-style exit codes for malformed input:

```cpp
if (invalidUserInput)
  return 4;
if (runtimeExecutionFailed)
  return 2;
```

- [ ] **Step 5: Run focused regression**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:
- `validator` compatibility checks pass
- smoke sim examples still pass

- [ ] **Step 6: Commit**

```bash
git add tools/validator/validator_main.cpp test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "refactor: route validator through execution session"
```

## Task 4: Retarget `mix-validator` to Runtime Session

**Files:**
- Modify: `tools/mix-validator/mix_validator_main.cpp`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Test: `test/tools/runtime/run_simbackend_examples.sh`

- [ ] **Step 1: Write the failing mix compatibility test**

Add a focused test that asserts mix validation can be represented as a runtime-manifest execution request:

```cpp
TEST_CASE("compat mix validator resolves artifact-root into runtime task") {
  auto request = buildCompatMixValidationRequest("/tmp/artifact", "/tmp/input");
  REQUIRE(request.tasks.size() == 1);
  REQUIRE(request.tasks[0].artifactRoot == "/tmp/artifact");
}
```

- [ ] **Step 2: Run focused regression to verify it fails**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:
- mix validator path is still standalone and not runtime-session-backed

- [ ] **Step 3: Replace standalone mix execution orchestration**

Refactor `mix_validator_main.cpp` so that:

```cpp
auto manifest = buildCompatMixRunManifest(...);
ExecutionSession session(makeExecutionBackend("sim"));
auto result = session.run(buildTaskGraphFromManifest(manifest));
```

Keep artifact-root, input-dir, golden, and direct-packed compatibility semantics where required by existing mix flow.

- [ ] **Step 4: Run real SimBackend mix regression**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_simbackend_examples.sh
```

Expected:
- `matmul-add-leakyrelu` still passes real `SimBackend` CPU simulation

- [ ] **Step 5: Commit**

```bash
git add tools/mix-validator/mix_validator_main.cpp \
  test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "refactor: route mix validator through execution session"
```

## Task 5: Switch Vec Example Defaults to Runtime-Backed Path

**Files:**
- Modify: `examples/relu-broadcast-transpose/run.sh`
- Modify: `examples/add-broadcast-concat/run.sh`
- Modify: `examples/broadcast-add-reduce/run.sh`
- Modify: `examples/gather-elementwise-fusion/run.sh`
- Modify: `examples/split-relu-brc-add-mul/run.sh`
- Modify: `tools/validator/validator_main.cpp`
- Modify: `test/tools/runtime/run_simbackend_examples.sh`

- [ ] **Step 1: Write the failing example regression expectation**

Update `run_simbackend_examples.sh` expectations so that vec examples are required to run through the runtime-backed default path.

Add shell checks such as:

```bash
grep -q "session.backend=sim" "$LOG_FILE"
grep -q "session.result=success" "$LOG_FILE"
```

If `validator` does not yet emit stable runtime-backed session markers, this
task is allowed to make the smallest required change in:

```cpp
tools/validator/validator_main.cpp
```

to print stable success/error `session.*` lines so the example default path can
be asserted without forcing the examples to switch to direct `runtime-session`
invocation in this phase.

- [ ] **Step 2: Run the example regression to verify current defaults are not yet using the new path**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_simbackend_examples.sh
```

Expected:
- regression fails because the vec scripts still invoke legacy validate flow directly

- [ ] **Step 3: Update each vec example to use the runtime-backed compile/validate path**

After `step8_kernel.cpp` generation, keep the script structure but switch the execution half to the new path. For example:

```bash
"$COMPILER" \
  --kernel "$DIR/step8_kernel.cpp" \
  --output "$BUILD_DIR" \
  --name relu_transpose_broadcast_add \
  --num-inputs 2

"$VALIDATOR" \
  --bin "$BUILD_DIR/relu_transpose_broadcast_add.bin" \
  --name relu_transpose_broadcast_add \
  --inputs "$DIR/input_data0.npy,$DIR/input_data1.npy" \
  --expected "$DIR/output_expected.npy" \
  ...
```

The visible commands may stay the same if `compiler`/`validator` have already been retargeted. The requirement is that the default path is now new-runtime-backed.

- [ ] **Step 4: Run full vec SimBackend real-execution regression**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_simbackend_examples.sh
```

Expected:
- all 5 vec examples pass real `SimBackend` CPU simulation

- [ ] **Step 5: Commit**

```bash
git add examples/relu-broadcast-transpose/run.sh \
  examples/add-broadcast-concat/run.sh \
  examples/broadcast-add-reduce/run.sh \
  examples/gather-elementwise-fusion/run.sh \
  examples/split-relu-brc-add-mul/run.sh \
  test/tools/runtime/run_simbackend_examples.sh
git commit -m "test: switch vec examples to runtime-backed default path"
```

## Task 6: Switch Mix Example Default to Runtime-Backed Path

**Files:**
- Modify: `examples/matmul-add-leakyrelu/run.sh`
- Modify: `test/tools/runtime/run_simbackend_examples.sh`

- [ ] **Step 1: Write the failing mix example regression expectation**

Require the mix example log to show the runtime-backed execution summary:

```bash
grep -q "session.backend=sim" "$MIX_LOG"
grep -q "session.result=success" "$MIX_LOG"
```

- [ ] **Step 2: Run the mix baseline to verify the script has not fully switched yet**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_simbackend_examples.sh
```

Expected:
- regression fails until the mix example default path goes through the retargeted compatibility flow

- [ ] **Step 3: Update the mix example default path**

Keep the MLIR/codegen/bootstrap parts that are still necessary, but make the default validation path depend on the retargeted `mix-validator` compatibility layer rather than a standalone execution stack.

- [ ] **Step 4: Run full 6-example real-execution regression**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_simbackend_examples.sh
```

Expected:
- all 6 examples pass
- `matmul-add-leakyrelu` still passes real `SimBackend` CPU simulation

- [ ] **Step 5: Commit**

```bash
git add examples/matmul-add-leakyrelu/run.sh \
  test/tools/runtime/run_simbackend_examples.sh
git commit -m "test: switch mix example to runtime-backed default path"
```

## Task 7: Retarget the C API Internals

**Files:**
- Modify: `lib/CAPI/Runtime/Runtime.cpp`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Test: `test/tools/runtime/run_runtime.sh`

- [ ] **Step 1: Write the failing C API compatibility tests**

Add focused tests that describe runtime-backed behavior behind the existing C ABI:

```cpp
TEST_CASE("c api compiler handle routes through artifact compiler") {
  auto *handle = createCompatCompilerHandleForTest();
  auto result = compatCompilerCompileForTest(handle, ...);
  REQUIRE(result.manifestPath.endswith("/manifest.txt"));
}

TEST_CASE("c api executor handle routes through execution session") {
  auto *handle = createCompatExecutorHandleForTest();
  auto result = compatExecutorRunForTest(handle, ...);
  REQUIRE(result.success);
}
```

- [ ] **Step 2: Run focused runtime regression to verify it fails**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:
- failure because C API still directly instantiates `Compiler` / `Executor`

- [ ] **Step 3: Replace C API internal handles**

Refactor `Runtime.cpp` so that the opaque handles wrap compatibility runtime state rather than old direct orchestration objects:

```cpp
struct CompatCompilerHandle {
  ArtifactCompiler compiler;
  CompatCompileOptions defaults;
};

struct CompatExecutorHandle {
  std::unique_ptr<ExecutionBackend> backend;
};
```

- [ ] **Step 4: Reimplement `compile`, `run`, and `run_file` through the new runtime**

Translate the flat C ABI inputs into compatibility requests and runtime bindings:

```cpp
auto manifest = buildCompatSingleTaskRunManifest(options);
ExecutionSession session(std::move(backend));
auto result = session.run(buildTaskGraphFromManifest(manifest));
```

Preserve the current exported function names and error-buffer behavior.

- [ ] **Step 5: Run focused runtime regression**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:
- focused runtime tests pass
- C API compatibility tests pass

- [ ] **Step 6: Commit**

```bash
git add lib/CAPI/Runtime/Runtime.cpp test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "refactor: route runtime c api through compatibility runtime"
```

## Task 8: Demote the Old Direct Orchestration Path

**Files:**
- Modify: `tools/compiler/compiler_main.cpp`
- Modify: `tools/validator/validator_main.cpp`
- Modify: `tools/mix-validator/mix_validator_main.cpp`
- Modify: `lib/CAPI/Runtime/Runtime.cpp`
- Modify: `test/tools/runtime/run_runtime.sh`

- [ ] **Step 1: Write the failing regression checks for default-path usage**

Add focused checks that the public entrypoints now report or exercise runtime-backed execution rather than the old direct path.

Examples:

```bash
grep -q "artifact.manifest=" "$LOG"
grep -q "session.backend=sim" "$LOG"
```

- [ ] **Step 2: Run focused runtime regression to verify the checks fail before cleanup**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:
- failure until all public entrypoints are using the compatibility runtime flow consistently

- [ ] **Step 3: Remove default-path dependence on old orchestration**

Clean up any remaining direct default-path references so that:

- old entrypoint names remain
- old orchestration is no longer the default flow
- the new runtime is the only public orchestration center

- [ ] **Step 4: Run full verification**

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
bash test/tools/runtime/run_simbackend_examples.sh
```

Expected:
- focused xvm runtime regression passes
- all 6 examples pass real `SimBackend` CPU simulation

- [ ] **Step 5: Commit**

```bash
git add tools/compiler/compiler_main.cpp tools/validator/validator_main.cpp \
  tools/mix-validator/mix_validator_main.cpp lib/CAPI/Runtime/Runtime.cpp \
  test/tools/runtime/run_runtime.sh
git commit -m "refactor: demote old runtime orchestration to compatibility layer"
```

## Self-Review

Spec coverage:
- CLI replacement is covered by Tasks 2, 3, and 4.
- Example default-path replacement is covered by Tasks 5 and 6.
- C API replacement is covered by Task 7.
- Old-path demotion and final replacement criteria are covered by Task 8.

Placeholder scan:
- No `TBD`, `TODO`, or deferred placeholder steps remain.

Type consistency:
- The plan consistently uses `CompatCompileOptions`, `CompatValidateOptions`, `ArtifactCompileRequest`, `RunManifest`, and `ExecutionSession` as the adapter vocabulary across tasks.
