# Runtime Profile Schema V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade simulator profiling artifacts to schema v1 with richer task metadata while keeping runtime-session and autotuner behavior compatible.

**Architecture:** Keep the existing `trace.json` artifact location and `ProfileTrace` model unchanged. Enrich the JSON emitted by `SimBackend`, keep autotuner score extraction backward-compatible, and prove the rollout with focused unit coverage plus xvm runtime/autotuner verification.

**Tech Stack:** C++17, LLVM JSON/support libraries, `SimBackend`, `ProfileTrace`, `ExecutionSession`, xvm simulator verification.

---

### Task 1: Encode schema v1 fields in simulator profile artifacts

**Files:**
- Modify: `lib/Runtime/SimBackend.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Add failing field-presence assertions for schema v1**

In `test/tools/runtime/test_taskgraph_runtime.cpp`, extend the existing simulator profile artifact test coverage so it checks the emitted JSON content, not just the artifact path. Add assertions for required fields:

```cpp
EXPECT(object->getInteger("schema_version") &&
           *object->getInteger("schema_version") == 1,
       "sim profile trace has schema_version=1");
EXPECT(object->getString("session_id") &&
           *object->getString("session_id") == "session-profile",
       "sim profile trace carries session_id");
EXPECT(object->getString("task_id") &&
           *object->getString("task_id") == "task0",
       "sim profile trace carries task_id");
EXPECT(object->getString("kernel_name") &&
           *object->getString("kernel_name") == "kernel0",
       "sim profile trace carries kernel_name");
EXPECT(object->getString("kernel_kind") &&
           *object->getString("kernel_kind") == "vec",
       "sim profile trace carries kernel_kind");
EXPECT(object->getString("soc_version") &&
           *object->getString("soc_version") == "Ascend910B1",
       "sim profile trace carries soc_version");
EXPECT(object->getInteger("block_dim") &&
           *object->getInteger("block_dim") == 8,
       "sim profile trace carries block_dim");
EXPECT(object->getInteger("workspace_size") &&
           *object->getInteger("workspace_size") == 8192,
       "sim profile trace carries workspace_size");
EXPECT(object->getBoolean("validation_passed") &&
           *object->getBoolean("validation_passed"),
       "sim profile trace marks validation_passed");
```

- [ ] **Step 2: Add failing assertions for tensor and tiling metadata**

In the same test, assert `inputs`, `outputs`, and `tiling` exist and have the expected shape:

```cpp
auto *inputs = object->getArray("inputs");
EXPECT(inputs && inputs->size() == 1, "sim profile trace emits one input");
auto *input0 = (*inputs)[0].getAsObject();
EXPECT(input0 && input0->getString("name") &&
           *input0->getString("name") == "input0",
       "sim profile trace input name");
EXPECT(input0 && input0->getString("dtype") &&
           *input0->getString("dtype") == "f16",
       "sim profile trace input dtype");

auto *tiling = object->getObject("tiling");
EXPECT(tiling && tiling->getBoolean("present") &&
           *tiling->getBoolean("present"),
       "sim profile trace marks tiling present");
EXPECT(tiling && tiling->getInteger("bytes") &&
           *tiling->getInteger("bytes") > 0,
       "sim profile trace records tiling bytes");
```

- [ ] **Step 3: Implement schema v1 emission in `SimBackend.cpp`**

Refactor `materializeSimulatorProfileArtifact(...)` in `lib/Runtime/SimBackend.cpp` so it writes the richer schema from `ExecutionRequest` data. Emit the full object shape:

```cpp
llvm::json::Object root;
root["schema_version"] = 1;
root["backend"] = "simulation";
root["session_id"] = request.sessionId;
root["task_id"] = request.task.taskId;
root["kernel_name"] = request.task.artifact.kernelName;
root["kernel_kind"] =
    std::string(kernelKindToString(request.task.artifact.kernelKind));
root["soc_version"] = request.task.artifact.socVersion;
root["block_dim"] = request.task.invocation.blockDim;
root["workspace_size"] =
    static_cast<int64_t>(request.task.invocation.workspaceSize);
root["cycle_count"] = cycleCount;
root["elapsed_us"] = cycleCount;
root["score"] = cycleCount;
root["validation_passed"] = true;
root["artifact_root"] = request.task.artifact.artifactRoot;
root["inputs"] = buildProfileTensorArray(request.task.invocation.inputs);
root["outputs"] = buildProfileTensorArray(request.task.invocation.outputs);
root["tiling"] = buildProfileTilingObject(request.task.invocation.tiling);
```

Add small local helpers in the same file for:

```cpp
static llvm::StringRef kernelKindToString(KernelKind kind);
static std::string dtypeToShortName(DType dtype);
static llvm::json::Array toJsonShape(llvm::ArrayRef<int64_t> shape);
static llvm::json::Array buildProfileTensorArray(llvm::ArrayRef<TensorBinding> bindings);
static llvm::json::Object buildProfileTilingObject(const std::optional<TilingBinding> &tiling);
```

- [ ] **Step 4: Run the focused runtime test binary on xvm**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && \
  export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && \
  bash test/tools/runtime/run_runtime.sh'
```

Expected:
- `test_taskgraph_runtime` passes
- `run_runtime.sh` still exits `0`

- [ ] **Step 5: Commit**

```bash
git add lib/Runtime/SimBackend.cpp test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "feat: emit runtime profile schema v1"
```

### Task 2: Keep autotuner score extraction compatible with schema v1

**Files:**
- Modify: `tools/autotuner/autotuner_main.cpp`
- Test: `tools/autotuner/autotuner_main.cpp` xvm smoke

- [ ] **Step 1: Tighten score extraction preference order**

In `tools/autotuner/autotuner_main.cpp`, update `extractRuntimeScore(...)` so it first prefers explicit top-level score fields when present, before recursive fallback scanning:

```cpp
if (const auto *object = parsedOr->getAsObject()) {
  if (auto score = object->getInteger("score"))
    return *score;
  if (auto cycles = object->getInteger("cycle_count"))
    return *cycles;
}
```

Keep the existing recursive fallback logic after this fast path so older trace JSON remains readable.

- [ ] **Step 2: Keep failure text precise**

If neither top-level nor fallback score-like fields exist, preserve a clear error:

```cpp
return llvm::createStringError(
    llvm::inconvertibleErrorCode(),
    "runtime profile JSON does not contain a score-like integer field: %s",
    profilePath.c_str());
```

Do not change autotuner candidate ranking or best-config semantics in this task.

- [ ] **Step 3: Run the real autotuner vec smoke on xvm**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && \
  source examples/env.sh && \
  export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && \
  cmake --build build --target autotuner -j2 && \
  build/bin/autotuner \
    --space examples/relu-broadcast-transpose/tiling_space.json \
    --kernel examples/relu-broadcast-transpose/step8_kernel.cpp \
    --kernel-kind vec \
    --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy \
    --expected examples/relu-broadcast-transpose/output_expected.npy \
    --shape M=640,N=500 \
    --output /tmp/autotuner-best.json && \
  cat /tmp/autotuner-best.json'
```

Expected:
- command exits `0`
- JSON contains non-zero `score`
- JSON contains non-zero `cycle_count`

- [ ] **Step 4: Commit**

```bash
git add tools/autotuner/autotuner_main.cpp
git commit -m "fix: prefer schema v1 scores in autotuner"
```

### Task 3: Prove schema v1 in real xvm artifacts

**Files:**
- Test-only: xvm runtime and autotuner outputs

- [ ] **Step 1: Re-run the focused runtime verification on xvm**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && \
  export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && \
  bash test/tools/runtime/run_runtime.sh'
```

Expected:
- exit `0`
- `test_taskgraph_runtime`, `test_capi_runtime`, `test_runtime` all pass
- `SimBackend` vec and mix smoke both pass

- [ ] **Step 2: Inspect a real generated schema v1 trace**

After the script run, inspect one real trace artifact:

```bash
ssh xvm@orb 'LATEST=$(find /tmp/ascendc-runtime -path \"*/opprof/simulator/trace.json\" | tail -n 1) && \
  echo \"$LATEST\" && cat \"$LATEST\"'
```

Expected JSON includes:
- `schema_version`
- `session_id`
- `kernel_name`
- `kernel_kind`
- `inputs`
- `outputs`
- `tiling`

- [ ] **Step 3: Commit if test-only helpers/scripts needed changes**

If no code/test files changed in this task, skip commit.

### Task 4: Final regression sweep and handoff

**Files:**
- No new production files expected

- [ ] **Step 1: Re-read the spec and confirm coverage**

Check the implemented changes against:

- `docs/superpowers/specs/2026-04-13-runtime-profile-schema-v1-design.md`

Confirm each spec section is covered:
- richer task-level schema
- same artifact path
- autotuner compatibility
- xvm verification

- [ ] **Step 2: Record final evidence**

Capture the exact commands and key outputs used to support the final report:

```bash
git log --oneline -4
```

Expected:
- shows the schema v1 implementation commits in order

- [ ] **Step 3: Final branch state check**

Run:

```bash
git status --short
```

Expected:
- no unexpected tracked modifications
- the known untracked file `docs/superpowers/plans/2026-04-10-runtime-taskgraph-mix.md` may remain

