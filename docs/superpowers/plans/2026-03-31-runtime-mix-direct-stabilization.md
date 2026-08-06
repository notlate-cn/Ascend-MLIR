# RuntimeMix Direct Backend Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the `RuntimeMix` direct backend so the current baremix end-to-end path is repeatable, contract-driven, easier to debug, and ready to merge back into `lib/Runtime` later.

**Architecture:** Keep the current direct backend pipeline, but tighten it around four explicit concerns: stable artifact contract, stage-oriented compile pipeline, explicit mix host ABI metadata, and merge-readiness documentation. All core behavior remains in `RuntimeMix` library code; tools and example scripts stay as thin entrypoints.

**Tech Stack:** C++17, LLVM Support, Ascend toolkit (`bisheng`, toolkit Python scripts, `ascendc_pack_kernel`), bash, xvm-based simulator verification.

---

## File Map

- Modify: `include/RuntimeMix/MixArtifact.h`
  Purpose: define the stable direct-backend artifact contract, including which fields are required and which may remain empty.

- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
  Purpose: make the compile pipeline explicitly stage-oriented, improve failure context, and ensure manifest/debug output reflects real paths and stage outputs.

- Modify: `lib/RuntimeMix/MixCommandBuilder.cpp`
  Purpose: keep all external command construction in one place and avoid command fragments leaking into tools or example scripts.

- Modify: `tools/mix-validator/mix_validator_main.cpp`
  Purpose: make runner-first execution explicit and keep the direct packed path as a constrained fallback only.

- Modify: `examples/baremix-test/README.md`
  Purpose: document the stabilized acceptance flow and the expected artifact layout, without adding new behavior.

- Verify: `examples/baremix-test/run.sh`
  Purpose: remain the xvm acceptance entrypoint; verify behavior only unless a spec-defined cleanup is necessary.

- Create: `docs/superpowers/specs/2026-03-31-runtime-mix-direct-stabilization-design.md`
  Purpose: already written spec; implementation must stay aligned with it.

- Create: `docs/superpowers/plans/2026-03-31-runtime-mix-direct-stabilization.md`
  Purpose: this execution plan.

## Task 1: Stabilize Artifact Contract

**Files:**
- Modify: `include/RuntimeMix/MixArtifact.h`
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
- Test: xvm compile-only invocation through `mix-compiler`

- [ ] **Step 1: Write the expected artifact contract into `MixArtifact`**

Update `include/RuntimeMix/MixArtifact.h` comments so each field has an explicit contract:

```cpp
struct MixArtifact {
  // Always set.
  std::string kernel_name;
  std::string soc_version;
  std::string work_dir;
  std::string build_dir;
  std::string install_dir;
  std::string kernel_so_path;
  std::string launcher_header_dir;
  std::string host_runner_path;
  std::string manifest_path;

  // Optional until a backend produces them.
  std::string device_object_path;
  std::string host_stub_source_path;
};
```

The implementation may keep the same fields, but the comments must make “required vs optional” unambiguous.

- [ ] **Step 2: Make manifest writing match the artifact contract**

In `lib/RuntimeMix/MixDirectBackend.cpp`, ensure the manifest always writes the required fields from `MixArtifact`, and only writes optional fields when non-empty:

```cpp
manifest << "kernel_name=" << artifact.kernel_name << "\n";
manifest << "soc_version=" << artifact.soc_version << "\n";
manifest << "work_dir=" << artifact.work_dir << "\n";
manifest << "build_dir=" << artifact.build_dir << "\n";
manifest << "install_dir=" << artifact.install_dir << "\n";
manifest << "kernel_so_path=" << artifact.kernel_so_path << "\n";
manifest << "launcher_header_dir=" << artifact.launcher_header_dir << "\n";
manifest << "host_runner_path=" << artifact.host_runner_path << "\n";
manifest << "manifest_path=" << artifact.manifest_path << "\n";
if (!artifact.device_object_path.empty())
  manifest << "device_object_path=" << artifact.device_object_path << "\n";
if (!artifact.host_stub_source_path.empty())
  manifest << "host_stub_source_path=" << artifact.host_stub_source_path << "\n";
```

Keep existing useful preprocess/debug keys, but do not let required keys depend on backend branch conditions.

- [ ] **Step 3: Rebuild and run compile-only contract verification on xvm**

Run:

```bash
sleep 2 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  clang++ tools/mix-compiler/mix_compiler_main.cpp \
    lib/RuntimeMix/MixDirectBackend.cpp \
    lib/RuntimeMix/MixCommandBuilder.cpp \
    lib/RuntimeMix/MixSourceAnalyzer.cpp \
    lib/RuntimeMix/MixStubTemplate.cpp \
    $(llvm-config --cxxflags --ldflags --libs support --system-libs) \
    -std=c++17 -Iinclude -o /tmp/mix-compiler-stable-contract &&
  rm -rf build/runtime-mix-baremix-contract &&
  /tmp/mix-compiler-stable-contract \
    --kernel examples/baremix-test/baremix_custom.cpp \
    --name baremix_custom \
    --output build/runtime-mix-baremix-contract \
    --soc Ascend910B1 &&
  sed -n "1,80p" build/runtime-mix-baremix-contract/out/manifest.txt
'
```

Expected:
- compile succeeds
- `manifest.txt` contains all required fields
- `kernel_so_path` and `host_runner_path` point to existing paths under `build/runtime-mix-baremix-contract/out`

- [ ] **Step 4: Commit**

```bash
git add include/RuntimeMix/MixArtifact.h \
        lib/RuntimeMix/MixDirectBackend.cpp
git commit -m "refactor: stabilize RuntimeMix artifact contract"
```

## Task 2: Stabilize Pipeline Stages and Error Context

**Files:**
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
- Modify: `lib/RuntimeMix/MixCommandBuilder.cpp`
- Test: xvm compile failure and success paths

- [ ] **Step 1: Introduce explicit stage labels for every external tool boundary**

In `lib/RuntimeMix/MixDirectBackend.cpp`, define stage labels as constants or named local variables and pass them consistently to process execution helpers:

```cpp
static constexpr llvm::StringLiteral kStageAnalyze = "analyze source";
static constexpr llvm::StringLiteral kStagePreprocess = "preprocess source";
static constexpr llvm::StringLiteral kStageExtractStub = "extract host stub";
static constexpr llvm::StringLiteral kStageCompileAic = "compile AIC object";
static constexpr llvm::StringLiteral kStageCompileAiv = "compile AIV object";
static constexpr llvm::StringLiteral kStageMerge = "merge device objects";
static constexpr llvm::StringLiteral kStageFinalizeStub = "finalize generated host stub";
static constexpr llvm::StringLiteral kStagePack = "pack mix kernel";
static constexpr llvm::StringLiteral kStageRunner = "build host runner";
```

Use the same labels in error creation, `ensureFileExists(...)`, and manifest/debug output.

- [ ] **Step 2: Centralize command-to-string rendering in `MixCommandBuilder.cpp`**

If `MixDirectBackend.cpp` still formats shell/debug command strings directly, move that logic into `lib/RuntimeMix/MixCommandBuilder.cpp` behind a small helper like:

```cpp
std::string renderCommandForDebug(llvm::ArrayRef<std::string> args);
```

Then use it in `MixDirectBackend.cpp` so manifest/debug output always comes from the same renderer that the command builder uses.

- [ ] **Step 3: Add stage-specific failure context**

When a tool fails, wrap the error so it names:
- stage
- primary command/program
- critical path inputs

For example:

```cpp
return llvm::createStringError(
    llvm::inconvertibleErrorCode(),
    "[%s] failed while processing kernel=%s generated_dir=%s: %s",
    kStageCompileAic.data(), cfg.kernel_name.c_str(),
    prep.generatedDir.c_str(), llvm::toString(std::move(err)).c_str());
```

Do this for preprocess, AIC/AIV compile, merge, finalize-stub, pack, and runner build.

- [ ] **Step 4: Verify both success and stage-context failure on xvm**

Run success path:

```bash
sleep 2 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  /tmp/mix-compiler-stable-contract \
    --kernel examples/baremix-test/baremix_custom.cpp \
    --name baremix_custom \
    --output build/runtime-mix-baremix-pipeline \
    --soc Ascend910B1
'
```

Then run a forced failure path:

```bash
sleep 2 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  /tmp/mix-compiler-stable-contract \
    --kernel examples/baremix-test/does_not_exist.cpp \
    --name baremix_custom \
    --output build/runtime-mix-baremix-fail \
    --soc Ascend910B1
'
```

Expected:
- success path still produces the packed artifact
- failure path mentions a concrete stage name instead of a vague process failure

- [ ] **Step 5: Commit**

```bash
git add lib/RuntimeMix/MixDirectBackend.cpp \
        lib/RuntimeMix/MixCommandBuilder.cpp
git commit -m "refactor: add stage-oriented RuntimeMix pipeline diagnostics"
```

## Task 3: Stabilize Runner-First Execution and ABI Metadata

**Files:**
- Modify: `tools/mix-validator/mix_validator_main.cpp`
- Modify: `include/RuntimeMix/MixArtifact.h`
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
- Modify: `examples/baremix-test/README.md`
- Test: xvm full end-to-end run

- [ ] **Step 1: Make runner-first behavior explicit in validator logic**

In `tools/mix-validator/mix_validator_main.cpp`, keep this ordering explicit:

```cpp
const bool canUseRunner = llvm::sys::fs::exists(runnerPath);
const bool canUseDirectPacked = !manifestKernelName.empty() &&
                                !manifestKernelSo.empty() &&
                                llvm::sys::fs::exists(manifestKernelSo) &&
                                !canUseRunner;
```

If the file already has this logic, tighten the surrounding comments and error messages so the intended priority is obvious:

```cpp
// Direct packed execution is only a constrained fallback.
// When a generated host runner exists, it is always the primary path.
```

- [ ] **Step 2: Surface current ABI assumptions in artifact metadata or manifest comments**

Without adding new user CLI flags, add stable metadata emission in `lib/RuntimeMix/MixDirectBackend.cpp` so the manifest makes the current ABI assumptions visible, for example:

```cpp
manifest << "abi_kind=mix_gm_workspace_tiling\n";
manifest << "abi_inputs=3\n";
manifest << "abi_outputs=1\n";
manifest << "abi_workspace_mode=fixed\n";
manifest << "abi_tiling_mode=fixed_bytes\n";
```

These values may still reflect the current baremix acceptance path, but they must be emitted as explicit metadata rather than hidden assumptions.

- [ ] **Step 3: Document the stabilized ABI in the example README**

Update `examples/baremix-test/README.md` with a short “Current ABI” section:

```md
## Current ABI

This stabilized direct-backend validation path currently assumes the AscendC mix host ABI:

- GM inputs
- GM outputs
- workspace
- tiling

For the current baremix acceptance case, the generated runner and validator expect 3 inputs and 1 output.
```

Do not over-promise generality; this README should describe the current stabilized behavior honestly.

- [ ] **Step 4: Verify full runner-first execution on xvm**

Run:

```bash
sleep 2 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  bash examples/baremix-test/run.sh &&
  sed -n "1,120p" build/runtime-mix-baremix/out/manifest.txt
'
```

Expected:
- `run.sh` passes
- `mix-validator` prints `PASS`
- `verify_result.py` prints `test pass`
- manifest shows the explicit ABI metadata

- [ ] **Step 5: Commit**

```bash
git add tools/mix-validator/mix_validator_main.cpp \
        include/RuntimeMix/MixArtifact.h \
        lib/RuntimeMix/MixDirectBackend.cpp \
        examples/baremix-test/README.md
git commit -m "docs: make RuntimeMix runner-first ABI assumptions explicit"
```

## Task 4: Record Merge-Readiness and Keep Tools Thin

**Files:**
- Modify: `examples/baremix-test/README.md`
- Modify: `docs/superpowers/specs/2026-03-31-runtime-mix-direct-stabilization-design.md`
- Test: repository inspection plus xvm acceptance rerun

- [ ] **Step 1: Add future merge mapping to the README or spec**

Extend the spec or README with a short mapping table like:

```md
## Future Merge Mapping

- `RuntimeMix/MixDirectBackend.*` -> `Runtime/Compiler.*` mix branch
- `RuntimeMix` mix artifact metadata -> `Runtime` artifact/runner config model
- `tools/mix-validator` runner-first path -> `Runtime/SimValidator.*` and `tools/validator`
```

Put this in the spec if you want it as engineering intent, or in the README if you want it closer to the example workflow. Do not duplicate it in both places unless the wording differs by audience.

- [ ] **Step 2: Verify tools remain thin shells**

Inspect:

```bash
rg -n "buildPreprocessCommand|extract_host_stub|update_host_stub|ascendc_pack_kernel|merge_mix_obj|abi_" \
  tools/mix-compiler tools/mix-validator examples/baremix-test
```

Expected:
- toolkit orchestration remains in `lib/RuntimeMix`
- tools and example scripts only invoke library behavior or document usage

If the search finds new core orchestration logic in tools/scripts, move it into library code before proceeding.

- [ ] **Step 3: Run final xvm acceptance**

Run:

```bash
sleep 2 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  bash examples/baremix-test/run.sh &&
  md5sum build/runtime-mix-baremix/testdata/output/golden.bin \
         build/runtime-mix-baremix/testdata/output/actual.bin
'
```

Expected:
- `PASS`
- `test pass`
- identical MD5 values for `golden.bin` and `actual.bin`

- [ ] **Step 4: Commit**

```bash
git add examples/baremix-test/README.md \
        docs/superpowers/specs/2026-03-31-runtime-mix-direct-stabilization-design.md
git commit -m "docs: mark RuntimeMix direct backend merge readiness"
```

## Self-Review

- Spec coverage:
  - Contract stabilization is covered by Task 1.
  - Pipeline stabilization is covered by Task 2.
  - ABI stabilization is covered by Task 3.
  - Merge readiness is covered by Task 4.

- Placeholder scan:
  - No `TODO`, `TBD`, or “implement later” placeholders remain.
  - Every task includes exact files and concrete xvm commands.

- Type consistency:
  - The plan reuses the current `MixArtifact`, `MixDirectBackend`, `mix-validator`, and `run.sh` names directly.
  - No new CLI options are introduced.
