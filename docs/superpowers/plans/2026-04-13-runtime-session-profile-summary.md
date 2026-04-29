# Runtime Session Profile Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stable runtime `session_summary.json` contract, retain task profiles under deterministic task-based paths, and make `autotuner` consume and report those runtime profiling artifacts.

**Architecture:** Extend the retained profile stage to materialize a session-oriented directory layout and summary JSON while preserving task profile schema v1. Keep `runtime-session` as the reporting surface for profile artifact locations, and keep `autotuner` thin by reading runtime-produced task/session artifacts instead of creating a parallel profile format.

**Tech Stack:** C++17, LLVM support utilities, existing runtime profile/session code, shell regression scripts, xvm simulator verification.

---

### Task 1: Add Failing Coverage For Session Summary Retention

**Files:**
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test for retained session summary output**

Add a focused test near the existing retained-profile coverage that constructs a `ProfileTrace` with two task artifacts, calls `retainProfileArtifactsForCli(...)`, and expects:
- `session_summary.json` exists under the retained session dir
- `tasks/main.json` and `tasks/consumer.json` exist
- returned `ProfileTrace::profileArtifactPaths()` points at the retained task paths

Use a pattern like:

```cpp
void testRetainProfileArtifactsCreatesSessionSummary() {
  auto tempRoot = makeTempDir("retain-summary");
  std::filesystem::create_directories(tempRoot / "work");

  ProfileTrace trace;
  trace.sessionId = "runtime-session--summary";
  addProfileArtifact(trace, "main", ExecutionBackendKind::Simulation,
                     (tempRoot / "work" / "main.json").string());
  addProfileArtifact(trace, "consumer", ExecutionBackendKind::Simulation,
                     (tempRoot / "work" / "consumer.json").string());
  std::ofstream(tempRoot / "work" / "main.json") << R"({"score":10,"cycle_count":10})";
  std::ofstream(tempRoot / "work" / "consumer.json") << R"({"score":20,"cycle_count":20})";

  auto retainedOr = retainProfileArtifactsForCli(trace, tempRoot.string());
  EXPECT(static_cast<bool>(retainedOr),
         "retainProfileArtifactsForCli should succeed for session summary test");
  if (!retainedOr)
    return;

  const auto summaryPath = tempRoot / "runtime-session--summary" / "session_summary.json";
  EXPECT(std::filesystem::exists(summaryPath),
         "retained profiles should include session_summary.json");
  EXPECT(std::filesystem::exists(tempRoot / "runtime-session--summary" / "tasks" / "main.json"),
         "retained profiles should include task-based main.json");
  EXPECT(std::filesystem::exists(tempRoot / "runtime-session--summary" / "tasks" / "consumer.json"),
         "retained profiles should include task-based consumer.json");
}
```

- [ ] **Step 2: Write the failing test for summary contents**

In the same file, add a test that reads `session_summary.json` via `llvm::json::parse` and expects:
- `schema_version == 1`
- `session_id == "runtime-session--summary"`
- `task_count == 2`
- `total_score == 30`
- `total_cycle_count == 30`
- `tasks[0].profile_path` points to `tasks/main.json`

Use a pattern like:

```cpp
void testRetainedSessionSummaryAggregatesTaskScores() {
  // arrange like previous test
  auto summaryText = readTextFile(summaryPath.string());
  auto parsed = llvm::json::parse(summaryText);
  EXPECT(static_cast<bool>(parsed), "session summary json should parse");
  if (!parsed)
    return;
  auto *obj = parsed->getAsObject();
  EXPECT(obj != nullptr, "session summary should be a json object");
  EXPECT(obj->getInteger("schema_version").value_or(-1) == 1,
         "session summary should set schema_version=1");
  EXPECT(obj->getString("session_id").value_or("") == "runtime-session--summary",
         "session summary should preserve session_id");
  EXPECT(obj->getInteger("task_count").value_or(-1) == 2,
         "session summary should count retained tasks");
}
```

- [ ] **Step 3: Run the focused test binary to verify failure**

Run on xvm:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
source scripts/resolve_llvm_env.sh
LLVM_BUILD=$(require_llvm_build_dir)
LLVM_SOURCE_INCLUDE=$(cd "${LLVM_BUILD}/.." && pwd)/include
cmake --build build --target AscendCRuntime -j2
g++ -std=c++17 -I include/ -I "$LLVM_BUILD/include" -I "$LLVM_SOURCE_INCLUDE" \
  test/tools/runtime/test_taskgraph_runtime.cpp build/lib/libAscendCRuntime.a \
  $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support --system-libs) -ldl \
  -o /tmp/test_taskgraph_runtime.profile_summary
/tmp/test_taskgraph_runtime.profile_summary
```

Expected: FAIL because retained profile code does not yet create `session_summary.json` or `tasks/<task_id>.json`.

- [ ] **Step 4: Commit the red test**

```bash
git add test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "test: add runtime session profile summary coverage"
```

### Task 2: Implement Session-Oriented Retained Profile Layout

**Files:**
- Modify: `include/Runtime/ProfileTrace.h`
- Modify: `lib/Runtime/ProfileTrace.cpp`
- Modify: `lib/Runtime/ProfileUtils.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Add minimal helpers for stable retained task naming**

If needed in `ProfileTrace.h/.cpp`, add small helpers that preserve task id ordering and make task artifact iteration explicit, without changing the public meaning of existing `profileArtifactPaths()`.

The implementation should stay minimal and support deterministic summary generation.

- [ ] **Step 2: Implement retained task relocation into `tasks/<task_id>.json`**

Update `retainProfileArtifactsForCli(...)` in `lib/Runtime/ProfileUtils.cpp` so retained session dirs use this layout:

```text
<dest-root>/<session_id>/
  session_summary.json
  tasks/
    <task_id>.json
```

When copying retained task profiles:
- preserve file contents
- use task id as the filename stem
- if multiple artifacts ever exist for one task, keep the current round simple by using `<task_id>.json` and failing clearly on duplicate retained task ids

- [ ] **Step 3: Implement `session_summary.json` generation**

Inside `retainProfileArtifactsForCli(...)`, after task profiles are copied, emit a summary document with:

```json
{
  "schema_version": 1,
  "session_id": "...",
  "backend": "simulation",
  "task_count": 2,
  "successful_task_count": 2,
  "failed_task_count": 0,
  "tasks": [
    {
      "task_id": "main",
      "profile_path": ".../tasks/main.json",
      "score": 10,
      "cycle_count": 10
    }
  ],
  "total_score": 10,
  "total_cycle_count": 10
}
```

Populate `score` and `cycle_count` by parsing the retained task JSON and reading top-level numeric fields, with this policy:
- if `score` exists and is numeric, use it
- if `cycle_count` exists and is numeric, use it
- if one is missing, fall back to the other
- if both are missing, fail clearly

- [ ] **Step 4: Return retained task artifact paths in the rewritten `ProfileTrace`**

Ensure `retainProfileArtifactsForCli(...)` returns a `ProfileTrace` whose task artifact paths now point at the retained `tasks/<task_id>.json` files, not the old anonymous `<task>-<index>.json` layout.

- [ ] **Step 5: Run the focused test binary to verify pass**

Run:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
source scripts/resolve_llvm_env.sh
LLVM_BUILD=$(require_llvm_build_dir)
LLVM_SOURCE_INCLUDE=$(cd "${LLVM_BUILD}/.." && pwd)/include
cmake --build build --target AscendCRuntime -j2
g++ -std=c++17 -I include/ -I "$LLVM_BUILD/include" -I "$LLVM_SOURCE_INCLUDE" \
  test/tools/runtime/test_taskgraph_runtime.cpp build/lib/libAscendCRuntime.a \
  $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support --system-libs) -ldl \
  -o /tmp/test_taskgraph_runtime.profile_summary
/tmp/test_taskgraph_runtime.profile_summary
```

Expected: PASS, with the new retained summary tests green.

- [ ] **Step 6: Commit the runtime retained-profile implementation**

```bash
git add include/Runtime/ProfileTrace.h lib/Runtime/ProfileTrace.cpp lib/Runtime/ProfileUtils.cpp test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "feat: retain runtime session profile summaries"
```

### Task 3: Surface Session Summary Paths In Runtime CLI And Scripts

**Files:**
- Modify: `tools/runtime-session/runtime_session_main.cpp`
- Modify: `test/tools/runtime/run_runtime.sh`
- Modify: `test/tools/runtime/run_simbackend_examples.sh`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing CLI coverage**

In `test/tools/runtime/test_taskgraph_runtime.cpp`, add a focused unit-level helper test around retained profiles that expects a stable summary path line to be derivable, or if a direct unit test is awkward, add a regression assertion to the shell scripts first and let it fail in xvm.

The required new CLI line is:

```text
session.profile.summary=/tmp/ascendc-runtime-profiles/<session_id>/session_summary.json
```

- [ ] **Step 2: Print `session.profile.summary=` from `runtime-session`**

After sim retained profiles are created in `tools/runtime-session/runtime_session_main.cpp`, print:

```cpp
llvm::outs() << "session.profile.summary=" << summaryPath << "\n";
```

Only print it when a retained session summary exists.

- [ ] **Step 3: Update runtime shell scripts to surface and check summary paths**

In `test/tools/runtime/run_runtime.sh` and `test/tools/runtime/run_simbackend_examples.sh`:
- extract `session.profile.summary=` from `runtime-session` output
- include it in final summary output
- assert that the referenced file exists

Keep the existing `session.profile[i]=...` checks intact.

- [ ] **Step 4: Run xvm runtime verification**

Run:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
bash test/tools/runtime/run_simbackend_examples.sh relu-broadcast-transpose matmul-add-leakyrelu
```

Expected:
- PASS
- summaries now include real `session_summary.json` paths

- [ ] **Step 5: Commit the CLI/script surfacing changes**

```bash
git add tools/runtime-session/runtime_session_main.cpp test/tools/runtime/run_runtime.sh test/tools/runtime/run_simbackend_examples.sh test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "feat: surface runtime session profile summaries"
```

### Task 4: Make Autotuner Record Runtime Session Summary Outputs

**Files:**
- Modify: `tools/autotuner/autotuner_main.cpp`
- Test: `test/tools/runtime/run_runtime.sh` or focused autotuner smoke support if needed

- [ ] **Step 1: Write the failing autotuner smoke expectation**

Use the existing xvm autotuner vec smoke path and define the required new JSON fields in `/tmp/autotuner-best.json`:
- `best.profile_path`
- `best.session_summary_path`

The smoke command remains:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
source examples/env.sh
cmake --build build --target autotuner -j2
build/bin/autotuner \
  --space examples/relu-broadcast-transpose/tiling_space.json \
  --kernel examples/relu-broadcast-transpose/step8_kernel.cpp \
  --kernel-kind vec \
  --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy \
  --expected examples/relu-broadcast-transpose/output_expected.npy \
  --shape M=640,N=500 \
  --output /tmp/autotuner-best.json
```

Expected before implementation: JSON lacks `best.session_summary_path`.

- [ ] **Step 2: Record runtime profile and summary paths in autotuner output**

In `tools/autotuner/autotuner_main.cpp`, when a candidate succeeds:
- continue extracting score from the task profile artifact
- also locate the matching session summary path from runtime profile outputs
- carry both into the selected best-candidate record

The best-config JSON should include:

```json
"best": {
  "block_dim": 8,
  "score": 12345,
  "cycle_count": 12345,
  "profile_path": "/tmp/ascendc-runtime-profiles/runtime-session--abcd12/tasks/main.json",
  "session_summary_path": "/tmp/ascendc-runtime-profiles/runtime-session--abcd12/session_summary.json"
}
```

- [ ] **Step 3: Re-run the autotuner smoke and inspect JSON**

Run on xvm:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
source examples/env.sh
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
cmake --build build --target autotuner -j2
build/bin/autotuner \
  --space examples/relu-broadcast-transpose/tiling_space.json \
  --kernel examples/relu-broadcast-transpose/step8_kernel.cpp \
  --kernel-kind vec \
  --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy \
  --expected examples/relu-broadcast-transpose/output_expected.npy \
  --shape M=640,N=500 \
  --output /tmp/autotuner-best.json
cat /tmp/autotuner-best.json
```

Expected:
- command exits 0
- `score` and `cycle_count` remain non-zero
- `best.profile_path` exists
- `best.session_summary_path` exists

- [ ] **Step 4: Commit the autotuner output enhancement**

```bash
git add tools/autotuner/autotuner_main.cpp
git commit -m "feat: record runtime session summaries in autotuner"
```

### Task 5: Final XVM Regression Verification

**Files:**
- No code changes required

- [ ] **Step 1: Re-run runtime focused verification**

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:
- PASS
- retained profiles include `session_summary.json`
- smoke summary shows task profile and session summary paths

- [ ] **Step 2: Re-run full SimBackend example baseline**

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
source examples/env.sh
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_simbackend_examples.sh
```

Expected:
- all 6 examples pass
- vec and mix examples still validate correctly

- [ ] **Step 3: Re-run autotuner vec smoke**

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
source examples/env.sh
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
cmake --build build --target autotuner -j2
build/bin/autotuner \
  --space examples/relu-broadcast-transpose/tiling_space.json \
  --kernel examples/relu-broadcast-transpose/step8_kernel.cpp \
  --kernel-kind vec \
  --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy \
  --expected examples/relu-broadcast-transpose/output_expected.npy \
  --shape M=640,N=500 \
  --output /tmp/autotuner-best.json
```

Expected:
- PASS
- non-zero `score`
- both runtime profile path fields present in JSON

- [ ] **Step 4: Commit verification-only updates if any were needed**

```bash
git status --short
```

Expected: no unexpected file changes beyond tracked implementation files.
