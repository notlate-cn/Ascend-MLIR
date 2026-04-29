# Legacy Executor Deletion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `Legacy/Executor` entirely from the repository, including public shims and retained legacy tests, while keeping the runtime-native execution path green on xvm.

**Architecture:** The runtime main execution path already runs through `NativeExecutionRunner` via `DefaultExecutionRunner`, so this round is a physical deletion and cleanup pass, not a behavior redesign. The change should proceed in narrow TDD slices: first lock the deletion contract in tests and seam checks, then remove the headers/implementation and retained legacy probe, and finally sync audits and `AGENTS.md` to the new reality.

**Tech Stack:** C++17, LLVM Support (`llvm::Error`, `MemoryBuffer`), existing runtime xvm verification scripts, bash, CMake.

---

## File Map

### Delete files
- `include/Runtime/Executor.h` — public shim that currently re-exports the legacy executor header.
- `include/Runtime/Legacy/Executor.h` — legacy executor declaration.
- `lib/Runtime/Legacy/Executor.cpp` — retained legacy executor implementation.

### Modify files
- `lib/Runtime/CMakeLists.txt` — remove `Legacy/Executor.cpp` from the runtime library build.
- `test/tools/runtime/test_runtime.cpp` — remove the retained legacy executor probe and tighten focused runtime coverage around the runtime-native path.
- `AGENTS.md` — update progress/decisions/TODO after deletion.
- `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md` — reclassify `Legacy/Executor` from retained unit to deleted.
- `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md` — add a note that the legacy executor audit findings are historical and the file has been removed.
- `docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md` — update retained legacy unit text if it still mentions `Legacy/Executor` as present.

### Reference-only files
- `include/Runtime/Execution/ExecutionRunner.h`
- `include/Runtime/Execution/DefaultExecutionRunner.h`
- `include/Runtime/Execution/NativeExecutionRunner.h`
- `lib/Runtime/Execution/DefaultExecutionRunner.cpp`
- `lib/Runtime/Execution/NativeExecutionRunner.cpp`
- `test/tools/runtime/run_runtime.sh`
- `test/tools/runtime/run_mix_repeat.sh`

### Expected final state
- No in-repo file includes `Runtime/Executor.h`.
- No runtime library source file named `Legacy/Executor.cpp` remains.
- No retained legacy executor probe remains in `test_runtime.cpp`.
- Focused runtime verification continues to pass on xvm through runtime-native paths only.

---

### Task 1: Lock the executor deletion contract with focused tests and seam checks

**Files:**
- Modify: `test/tools/runtime/test_runtime.cpp`
- Test: `test/tools/runtime/run_runtime.sh`

- [ ] **Step 1: Replace the retained legacy executor probe with a runtime-native negative-path assertion in `test_runtime.cpp`**

Delete the existing retained legacy executor probe section and replace it with a subprocess-based runtime-native mix launch failure check that uses `runtime-session` rather than `Executor` directly. The replacement test should keep coverage for the missing packed mix failure stage without keeping the legacy type alive.

```cpp
SECTION("Runtime-native packed mix error path") {
  llvm::outs() << "\n[Runtime-native packed mix error path]\n";

  TempDir temp("rt_native_mix_error");
  REQUIRE(!temp.path.empty());

  const std::string manifestPath = temp.path + "/runtime-manifest.json";
  const std::string outputPath = temp.path + "/actual.npy";
  const std::string missingSo = temp.path + "/missing.so";

  std::ofstream manifest(manifestPath);
  manifest << R"JSON({
    "backend": "sim",
    "tasks": [{
      "task_id": "main",
      "artifact": {
        "kernel_name": "fc_relu",
        "kernel_kind": "mix",
        "mix_resource_type": "mix_1c1v",
        "packed_shared_object_path": ")JSON"
           << missingSo << R"JSON("
      },
      "outputs": [
        { "name": "out", "path": ")JSON"
           << outputPath << R"JSON(", "shape": [1], "dtype": "f16" }
      ]
    }]
  })JSON";
  manifest.close();

  const std::string command =
      std::string("build/bin/runtime-session --run-manifest ") + manifestPath +
      " --run >/tmp/runtime_native_mix_error.log 2>&1";
  int rc = std::system(command.c_str());
  REQUIRE(rc != 0);

  auto log = readFileOrEmpty("/tmp/runtime_native_mix_error.log");
  CHECK(log.find("session.error_stage=kernel_launch") != std::string::npos);
  CHECK(log.find("packed mix") != std::string::npos);
}
```

- [ ] **Step 2: Add a focused seam check command to the plan notes in `test_runtime.cpp` comments**

Update the file header comment block so the focused manual compile notes no longer mention `Runtime/Executor.h` expectations and instead mention the runtime-native mix failure coverage.

```cpp
// Coverage note:
//   This file should only retain runtime-native execution probes plus explicitly
//   retained legacy units that are still present in-tree. It must not keep
//   Executor-only compatibility coverage once Legacy/Executor is deleted.
```

- [ ] **Step 3: Run the focused runtime script to prove the current tree still passes before deletion**

Run:
```bash
bash test/tools/runtime/run_runtime.sh
```

Expected:
- PASS on xvm before the actual file deletion
- mix repeat baseline still passes

- [ ] **Step 4: Commit the test-side contract change**

```bash
git add test/tools/runtime/test_runtime.cpp
git commit -m "test: replace retained legacy executor probe"
```

---

### Task 2: Remove the legacy executor headers and implementation from the build

**Files:**
- Delete: `include/Runtime/Executor.h`
- Delete: `include/Runtime/Legacy/Executor.h`
- Delete: `lib/Runtime/Legacy/Executor.cpp`
- Modify: `lib/Runtime/CMakeLists.txt`

- [ ] **Step 1: Remove the legacy executor build entry from `lib/Runtime/CMakeLists.txt`**

Delete the `Legacy/Executor.cpp` line from the runtime library source list.

```cmake
  Legacy/CompatRuntime.cpp
  Legacy/Compiler.cpp
  Legacy/HostRunnerGen.cpp
```

- [ ] **Step 2: Delete the legacy executor public shim and implementation files**

Remove these files completely:

```text
include/Runtime/Executor.h
include/Runtime/Legacy/Executor.h
lib/Runtime/Legacy/Executor.cpp
```

- [ ] **Step 3: Run a repository seam scan to confirm no code still references the deleted surface**

Run:
```bash
rg -n "Runtime/Executor.h|\bExecutor\b" include lib test -g '!docs/**'
```

Expected after deletion:
- no matches for `Runtime/Executor.h`
- no remaining runtime-owned `Executor ex(` or `Executor executor(` use sites
- the only possible remaining hits should be documentation text if you accidentally included `docs/**`; with the command above there should be no code hits

- [ ] **Step 4: Build the runtime-native targets to catch compile fallout immediately**

Run:
```bash
cmake --build build --target AscendCRuntime runtime-session autotuner -j1
```

Expected:
- build succeeds
- no unresolved references to deleted executor symbols

- [ ] **Step 5: Commit the physical deletion**

```bash
git add lib/Runtime/CMakeLists.txt
git rm include/Runtime/Executor.h include/Runtime/Legacy/Executor.h lib/Runtime/Legacy/Executor.cpp
git commit -m "refactor: delete legacy executor"
```

---

### Task 3: Re-run focused xvm verification on runtime-native execution paths only

**Files:**
- Test: `test/tools/runtime/run_runtime.sh`
- Test: `test/tools/runtime/run_mix_repeat.sh`
- Test: autotuner vec smoke command

- [ ] **Step 1: Run the focused runtime verification script on xvm**

Run:
```bash
bash test/tools/runtime/run_runtime.sh
```

Expected:
- runtime-focused verification passes
- vec/mix smoke pass
- repeated mix simulation baseline passes

If the xvm machine OOMs in non-runtime outer targets, immediately rerun the reduced runtime-only build instead of treating that as an executor deletion regression.

- [ ] **Step 2: Run the reduced runtime-only xvm build and autotuner vec smoke if needed**

Run:
```bash
cmake --build build --target AscendCRuntime runtime-session autotuner -j1
source examples/env.sh >/dev/null
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
- build succeeds
- autotuner prints `PASS score=...`
- `/tmp/autotuner-best.json` contains non-zero `score` / `cycle_count`

- [ ] **Step 3: Run the repeated mix simulation baseline directly if `run_runtime.sh` had outer-build noise**

Run:
```bash
source examples/env.sh >/dev/null
bash test/tools/runtime/run_mix_repeat.sh
```

Expected:
- `mix runtime-session repeat passed`

- [ ] **Step 4: Commit only if verification is green**

No new code commit in this task. This task is the verification gate for the prior deletion commit.

---

### Task 4: Sync audits and AGENTS to the post-deletion state

**Files:**
- Modify: `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`
- Modify: `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md`
- Modify: `docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update cleanup candidates to mark `Legacy/Executor` as deleted**

Edit the cleanup candidates doc so `Legacy/Executor` no longer appears as a retained unit. Replace its bullets with a short historical note.

```md
- `Legacy/Executor`
  - Deleted after the runtime-native execution runner cutover.
  - Remaining execution cleanup work should focus on retained compatibility files, not executor seams.
```

- [ ] **Step 2: Add a historical note to the dependency audit instead of rewriting its original findings away**

Append a short update note near the `Legacy/Executor` section:

```md
> Update (2026-04-14): `Legacy/Executor` and its public shim have now been
> removed. This section is retained as historical audit context for the pre-
> deletion state.
```

- [ ] **Step 3: Update the autotuner normalization audit to remove present-tense executor retention language**

Replace stale present-tense wording with a short note that autotuner now runs on the runtime-native execution path and no longer inherits a retained executor unit.

```md
Autotuner no longer inherits a retained `Legacy/Executor` unit. Remaining
legacy cleanup around autotuner is limited to non-executor retained surfaces.
```

- [ ] **Step 4: Update `AGENTS.md` progress and TODO**

Update `AGENTS.md` so it states:
- `Legacy/Executor` has been deleted
- runtime default execution path is fully runtime-native
- TODO no longer mentions retained executor cleanup
- next cleanup priority moves to `Legacy/Compiler`, `Legacy/CompatRuntime`, and `Legacy/HostRunnerGen`

Use wording like:

```md
- `Legacy/Executor` has been deleted after the runtime-native execution runner cutover.
- The default runtime execution path is now fully runtime-native through `NativeExecutionRunner`.
```

and

```md
- Plan the next-round `Legacy` cleanup in risk-ordered slices:
  - retained compiler/compat surface cleanup first
  - host-runner retained surface second
  - deeper legacy implementation deletion last
```

- [ ] **Step 5: Commit the doc sync**

```bash
git add AGENTS.md \
  docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md \
  docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md \
  docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md
git commit -m "docs: sync status after legacy executor deletion"
```

---

## Self-Review

- **Spec coverage:** The plan covers full executor deletion, runtime-focused verification, and post-deletion audit/AGENTS sync. No spec requirement is missing.
- **Placeholder scan:** No `TODO`/`TBD` placeholders remain. Each task has exact files, commands, and expected results.
- **Type consistency:** The plan consistently refers to `NativeExecutionRunner`, `DefaultExecutionRunner`, and the deleted `Legacy/Executor` surface without inventing new type names.
