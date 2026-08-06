# RuntimeMix Direct Preprocess Baremix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire toolkit preprocess/stub generation into `RuntimeMix` direct backend so `examples/baremix-test` passes on xvm with matching output and precision.

**Architecture:** Add a `MixPreprocessStage` inside `MixDirectBackend` that runs `bisheng -E`, emits a minimal `compile_commands.json`, invokes `extract_host_stub.py`, then consumes generated `auto_gen_*.cpp`, `host_stub.cpp`, and `aic/aiv_config.cmake` in the existing direct compile/merge/pack pipeline. Keep the external `mix-compiler` / `mix-validator` CLI unchanged.

**Tech Stack:** C++, LLVM Support, Ascend toolkit `bisheng`, toolkit Python scripts (`extract_host_stub.py`, `update_host_stub.py`), bash/xvm validation.

---

## File Map

- Modify: `include/RuntimeMix/MixCommandBuilder.h`
  Purpose: declare preprocess/script command builders and small helpers needed by direct backend.

- Modify: `lib/RuntimeMix/MixCommandBuilder.cpp`
  Purpose: construct `bisheng -E`, toolkit Python script, and merge/link command vectors with absolute toolkit paths.

- Modify: `include/RuntimeMix/MixDirectBackend.h`
  Purpose: add any small internal structs required by preprocess and config parsing if they need to be shared.

- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
  Purpose: add preprocess stage, parse generated config files, switch compile inputs to `auto_gen_*.cpp`, switch host stub source to toolkit generated `host_stub.cpp`, keep manifest/debug output current.

- Modify: `tools/mix-compiler/mix_compiler_main.cpp`
  Purpose: keep output stable if new artifact fields or paths need to be printed, without changing CLI semantics.

- Modify: `tools/mix-validator/mix_validator_main.cpp`
  Purpose: adapt only if manifest/path expectations need to change; do not change CLI.

- Verify: `examples/baremix-test/run.sh`
  Purpose: acceptance path only; do not redesign unless required for xvm verification.

- Verify only: `examples/baremix-test/scripts/gen_data.py`
  Purpose: existing golden generation for acceptance.

- Verify only: `examples/baremix-test/scripts/verify_result.py`
  Purpose: final precision check.

## Task 1: Add Toolkit Preprocess Command Support

**Files:**
- Modify: `include/RuntimeMix/MixCommandBuilder.h`
- Modify: `lib/RuntimeMix/MixCommandBuilder.cpp`
- Test: xvm one-off command probes via `clang++` + temporary binary

- [ ] **Step 1: Add preprocess/stub command builder declarations**

Add declarations for the new command builders in `include/RuntimeMix/MixCommandBuilder.h`:

```cpp
std::vector<std::string>
buildPreprocessCommand(llvm::StringRef src, llvm::StringRef outputPath);

std::vector<std::string>
buildExtractHostStubCommand(llvm::StringRef preprocessedPath,
                            llvm::StringRef dstDir,
                            llvm::StringRef headerDir,
                            llvm::ArrayRef<std::string> aivObjects,
                            llvm::ArrayRef<std::string> aicObjects,
                            llvm::StringRef compileCommandsPath,
                            llvm::StringRef buildMode,
                            llvm::StringRef runMode);

std::vector<std::string>
buildUpdateHostStubCommand(llvm::StringRef codeDir, llvm::StringRef objDir,
                           llvm::StringRef lowerSocVersion,
                           llvm::StringRef targetName);
```

- [ ] **Step 2: Implement minimal preprocess command builder**

Implement `buildPreprocessCommand(...)` in `lib/RuntimeMix/MixCommandBuilder.cpp` using the working xvm prototype flags:

```cpp
std::vector<std::string>
buildPreprocessCommand(llvm::StringRef src, llvm::StringRef outputPath) {
  return {"/bin/bash", "-lc",
          shellQuote(getBishengPath()) + " -E -includestdio.h -x cce -O3 "
          "--cce-aicore-lang -std=c++17 -DTILING_KEY_VAR=0 "
          "-I " + shellQuote(getTikcppRoot() + "/tikcfw") + " "
          "-I " + shellQuote(getTikcppRoot() + "/tikcfw/interface") + " "
          "-I " + shellQuote(getTikcppRoot() + "/tikcfw/impl") + " "
          "-D__CHECK_FEATURE_AT_PRECOMPILE " + shellQuote(src.str()) + " > " +
          shellQuote(outputPath.str())};
}
```

Notes for implementation:
- reuse existing toolkit path helpers
- add a local `shellQuote()` helper in this file if needed
- do not pass `-o` to `bisheng -E`; redirect stdout instead

- [ ] **Step 3: Implement toolkit script command builders**

Implement the two Python script command builders in `lib/RuntimeMix/MixCommandBuilder.cpp`:

```cpp
std::vector<std::string> buildExtractHostStubCommand(...);
std::vector<std::string> buildUpdateHostStubCommand(...);
```

Required behavior:
- use absolute script paths under `/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util`
- pass `--dynamic-mode`
- pass `--generate-definition`
- pass `--build-mode c220`
- pass `--run-mode sim`
- pass all AIC/AIV object paths exactly as produced by the backend

- [ ] **Step 4: Rebuild command-builder consumers on xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
clang++ tools/mix-compiler/mix_compiler_main.cpp \
  lib/RuntimeMix/MixDirectBackend.cpp \
  lib/RuntimeMix/MixCommandBuilder.cpp \
  lib/RuntimeMix/MixSourceAnalyzer.cpp \
  lib/RuntimeMix/MixStubTemplate.cpp \
  $(llvm-config --cxxflags --ldflags --libs support --system-libs) \
  -std=c++17 -Iinclude -o /tmp/mix-compiler-cmds'
```

Expected:
- compile succeeds
- no undefined references from newly added declarations

- [ ] **Step 5: Commit**

```bash
git add include/RuntimeMix/MixCommandBuilder.h \
        lib/RuntimeMix/MixCommandBuilder.cpp
git commit -m "feat: add RuntimeMix preprocess command builders"
```

## Task 2: Add RuntimeMix Preprocess Stage and Generated-Config Parsing

**Files:**
- Modify: `include/RuntimeMix/MixDirectBackend.h`
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
- Test: xvm `mix-compiler` artifact generation only

- [ ] **Step 1: Add small internal structs for preprocess outputs**

In `lib/RuntimeMix/MixDirectBackend.cpp` or `include/RuntimeMix/MixDirectBackend.h` if sharing is cleaner, add focused structs like:

```cpp
struct MixGeneratedConfig {
  std::vector<std::string> mixSources;
  llvm::StringMap<std::vector<std::string>> definitionsBySource;
};

struct MixPreprocessOutputs {
  std::string preprocessedSourcePath;
  std::string compileCommandsPath;
  std::string generatedDir;
  std::string includeDir;
  std::string hostStubPath;
  std::string launcherHeaderPath;
  std::string aicConfigPath;
  std::string aivConfigPath;
};
```

Keep these internal to `RuntimeMix`; do not expose them through `MixArtifact`.

- [ ] **Step 2: Add a minimal `compile_commands.json` writer**

Add a helper in `lib/RuntimeMix/MixDirectBackend.cpp` that writes a single-entry JSON file matching the working prototype:

```cpp
static llvm::Error writeCompileCommandsJson(llvm::StringRef path,
                                            llvm::StringRef directory,
                                            llvm::StringRef command,
                                            llvm::StringRef file);
```

Requirements:
- `command` must include a fake `-o <preprocessedPath>` token even though the actual preprocess command uses shell redirection
- `directory` should be the repo-root-or-current build cwd used for the command

- [ ] **Step 3: Add a minimal parser for generated `aic_config.cmake` / `aiv_config.cmake`**

Add a small parser helper in `lib/RuntimeMix/MixDirectBackend.cpp`:

```cpp
static llvm::Expected<MixGeneratedConfig>
parseGeneratedConfig(llvm::StringRef path);
```

Parser scope is intentionally narrow:
- parse `set(MIX_SOURCES ... )`
- parse `set(AIC_SOURCES ...)` / `set(AIV_SOURCES ...)` if present
- parse `set_source_files_properties(<source> PROPERTIES COMPILE_DEFINITIONS "...")`
- split definition strings on `;`
- drop empty definitions

Do not implement a generic CMake parser.

- [ ] **Step 4: Add the preprocess stage before device compilation**

In `MixDirectBackend::compile(...)`, add a preprocess block before AIC/AIV compile:

```cpp
MixPreprocessOutputs prep = runPreprocessStage(...);
auto aicConfig = parseGeneratedConfig(prep.aicConfigPath);
auto aivConfig = parseGeneratedConfig(prep.aivConfigPath);
```

Required side effects:
- create a dedicated subdir under `work/`, for example:
  - `work/preprocessed/`
  - `work/generated/`
  - `work/generated/include/`
- run `buildPreprocessCommand(...)`
- write `compile_commands.json`
- run `buildExtractHostStubCommand(...)`
- run `buildUpdateHostStubCommand(...)`

Required validation:
- `auto_gen_baremix_custom.cpp` exists
- generated `host_stub.cpp` exists
- generated `aclrtlaunch_baremix_custom.h` exists
- `aic_config.cmake` and `aiv_config.cmake` exist

- [ ] **Step 5: Verify preprocess artifacts on xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
rm -rf build/runtime-mix-baremix-pre && \
/tmp/mix-compiler-cmds --kernel examples/baremix-test/baremix_custom.cpp \
  --name baremix_custom --output build/runtime-mix-baremix-pre --soc Ascend910B1 || true && \
find build/runtime-mix-baremix-pre/work -maxdepth 3 -type f | sort | sed -n "1,80p"'
```

Expected files include:
- `work/preprocessed/...`
- `work/generated/auto_gen_baremix_custom.cpp`
- `work/generated/host_stub.cpp`
- `work/generated/aic_config.cmake`
- `work/generated/aiv_config.cmake`
- `work/generated/include/aclrtlaunch_baremix_custom.h`

- [ ] **Step 6: Commit**

```bash
git add include/RuntimeMix/MixDirectBackend.h \
        lib/RuntimeMix/MixDirectBackend.cpp
git commit -m "feat: add RuntimeMix preprocess stage for baremix"
```

## Task 3: Compile Generated Sources Instead of Raw Kernel Source

**Files:**
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
- Possibly modify: `lib/RuntimeMix/MixCommandBuilder.cpp`
- Test: xvm `mix-compiler` object/merge generation

- [ ] **Step 1: Replace raw-source AIC/AIV compile inputs**

In `MixDirectBackend::compile(...)`, stop building AIC/AIV commands from `sourcePath`. Replace with generated source paths from parsed configs:

```cpp
const std::string generatedSource =
    findOnlyMixSourceOrErr(*aicConfig, *aivConfig);
```

For baremix in this stage, enforce a single generated mix source:
- if no source found, return error
- if more than one mix source found, return explicit error saying this stage only supports single-source baremix

- [ ] **Step 2: Apply compile definitions from generated config**

Add helper logic that converts parsed config definitions to command arguments:

```cpp
static std::vector<std::string>
definitionsToFlags(llvm::ArrayRef<std::string> defs) {
  std::vector<std::string> out;
  for (const auto &def : defs)
    if (!def.empty())
      out.push_back("-D" + def);
  return out;
}
```

Update command construction so AIC and AIV compile commands use:
- generated source path
- definition list from `aic_config.cmake` / `aiv_config.cmake`

Do not also keep the old hand-written `auto_gen_<kernel>_kernel=...` defines on the same path.

- [ ] **Step 3: Switch host stub source to toolkit-generated `host_stub.cpp`**

Replace the current host stub source path assignment:

```cpp
const std::string hostStubSourcePath = prep.hostStubPath;
const std::string launcherHeaderPath = prep.launcherHeaderPath;
```

Do not call `writeMixStubTemplate(...)` for the baremix main path anymore.

If needed, keep `MixStubTemplate` code compiled but unused.

- [ ] **Step 4: Keep manifest/debug info truthful**

Update manifest generation in `lib/RuntimeMix/MixDirectBackend.cpp` so it records:
- preprocess command
- generated source path
- generated host stub path
- actual config-derived AIC/AIV definitions

The existing keys can remain, but their values must match the new pipeline. Remove stale references to the old manual stub path if they are no longer true.

- [ ] **Step 5: Verify object symbol shape on xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
rm -rf build/runtime-mix-baremix-obj && \
/tmp/mix-compiler-cmds --kernel examples/baremix-test/baremix_custom.cpp \
  --name baremix_custom --output build/runtime-mix-baremix-obj --soc Ascend910B1 || true && \
nm build/runtime-mix-baremix-obj/objects/baremix_custom_aic.o | egrep "baremix_custom|mix_aic|mix_aiv" && \
echo ====AIV==== && \
nm build/runtime-mix-baremix-obj/objects/baremix_custom_aiv.o | egrep "baremix_custom|mix_aic|mix_aiv"'
```

Expected:
- no longer see both AIC and AIV exporting the same plain `T baremix_custom`
- symbol naming reflects generated split behavior

- [ ] **Step 6: Commit**

```bash
git add lib/RuntimeMix/MixDirectBackend.cpp \
        lib/RuntimeMix/MixCommandBuilder.cpp
git commit -m "feat: compile RuntimeMix baremix from generated sources"
```

## Task 4: Reconnect Merge, Pack, and Host Runner to Generated Outputs

**Files:**
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`
- Modify: `tools/mix-compiler/mix_compiler_main.cpp`
- Modify: `tools/mix-validator/mix_validator_main.cpp` if path handling needs adjustment
- Test: xvm end-to-end compile and validator

- [ ] **Step 1: Re-run official merge scripts using generated-object outputs**

Keep the current official merge path:

```cpp
buildDeviceMergeCommand(...)
buildMixFinalMergeCommand(...)
```

But ensure the objects feeding it are the newly generated-source AIC/AIV outputs. Do not fall back to the old raw-source compile path.

- [ ] **Step 2: Keep host link inputs aligned with generated host stub**

Ensure host stub compile and pack now use:

```cpp
hostStubSourcePath = prep.hostStubPath;
launcherDir = prep.includeDir;
```

and continue to produce:
- packed `libbaremix_custom_packed.so`
- launcher header in manifest
- `mix_runner`

- [ ] **Step 3: Keep runner generation unchanged unless broken by header path changes**

If the runner compile only needs include path updates, limit changes to that. Do not redesign `main.cpp` generation in this task.

If the launcher include moved, update the include path inputs only:

```cpp
runnerCompileCmd += " -I\"" + prep.includeDir + "\"";
```

- [ ] **Step 4: Build current source tools on xvm and run compiler only**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
clang++ tools/mix-compiler/mix_compiler_main.cpp \
  lib/RuntimeMix/MixDirectBackend.cpp \
  lib/RuntimeMix/MixCommandBuilder.cpp \
  lib/RuntimeMix/MixSourceAnalyzer.cpp \
  lib/RuntimeMix/MixStubTemplate.cpp \
  $(llvm-config --cxxflags --ldflags --libs support --system-libs) \
  -std=c++17 -Iinclude -o /tmp/mix-compiler-stage4 && \
rm -rf build/runtime-mix-baremix-stage4 && \
/tmp/mix-compiler-stage4 --kernel examples/baremix-test/baremix_custom.cpp \
  --name baremix_custom --output build/runtime-mix-baremix-stage4 --soc Ascend910B1 && \
find build/runtime-mix-baremix-stage4/out -maxdepth 3 -type f | sort'
```

Expected outputs:
- `out/libbaremix_custom_packed.so`
- `out/bin/mix_runner`
- `out/manifest.txt`

- [ ] **Step 5: Commit**

```bash
git add lib/RuntimeMix/MixDirectBackend.cpp \
        tools/mix-compiler/mix_compiler_main.cpp \
        tools/mix-validator/mix_validator_main.cpp
git commit -m "feat: reconnect RuntimeMix merge and pack to generated baremix artifacts"
```

## Task 5: Verify Baremix End-to-End on xvm

**Files:**
- Verify: `examples/baremix-test/run.sh`
- Verify: `tools/mix-validator/mix_validator_main.cpp`
- Verify: `lib/RuntimeMix/MixDirectBackend.cpp`

- [ ] **Step 1: Build current source tools on xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
clang++ tools/mix-compiler/mix_compiler_main.cpp \
  lib/RuntimeMix/MixDirectBackend.cpp \
  lib/RuntimeMix/MixCommandBuilder.cpp \
  lib/RuntimeMix/MixSourceAnalyzer.cpp \
  lib/RuntimeMix/MixStubTemplate.cpp \
  $(llvm-config --cxxflags --ldflags --libs support --system-libs) \
  -std=c++17 -Iinclude -o /tmp/mix-compiler-final && \
clang++ tools/mix-validator/mix_validator_main.cpp \
  lib/RuntimeMix/Executor.cpp \
  lib/RuntimeMix/MixDirectBackend.cpp \
  lib/RuntimeMix/MixCommandBuilder.cpp \
  lib/RuntimeMix/MixSourceAnalyzer.cpp \
  lib/RuntimeMix/MixStubTemplate.cpp \
  $(llvm-config --cxxflags --ldflags --libs support --system-libs) \
  -std=c++17 -Iinclude -ldl -o /tmp/mix-validator-final'
```

Expected:
- both binaries compile cleanly

- [ ] **Step 2: Run the direct backend baremix flow**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
rm -rf build/runtime-mix-baremix-final && \
/tmp/mix-compiler-final --kernel examples/baremix-test/baremix_custom.cpp \
  --name baremix_custom --output build/runtime-mix-baremix-final --soc Ascend910B1 && \
mkdir -p build/runtime-mix-baremix-final/testdata && \
cd build/runtime-mix-baremix-final/testdata && \
python3 ../../../examples/baremix-test/scripts/gen_data.py && \
cd /home/niu/code/Ascend-MLIR && \
/tmp/mix-validator-final --artifact-root build/runtime-mix-baremix-final \
  --input-dir build/runtime-mix-baremix-final/testdata/input \
  --golden build/runtime-mix-baremix-final/testdata/output/golden.bin \
  --output-file build/runtime-mix-baremix-final/testdata/output/actual.bin \
  --soc Ascend910B1'
```

Expected:
- validator returns exit code `0`
- output includes `PASS`

- [ ] **Step 3: Run repository precision check**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR/build/runtime-mix-baremix-final/testdata && \
python3 ../../../examples/baremix-test/scripts/verify_result.py \
  output/actual.bin output/golden.bin'
```

Expected:
- `test pass`

- [ ] **Step 4: Run the user-facing acceptance entrypoint**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
bash examples/baremix-test/run.sh'
```

Expected:
- full script succeeds
- final output includes `test pass`

- [ ] **Step 5: Record final evidence in manifest or notes**

Capture and inspect:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
sed -n "1,120p" build/runtime-mix-baremix-final/out/manifest.txt && \
md5sum build/runtime-mix-baremix-final/testdata/output/golden.bin \
       build/runtime-mix-baremix-final/testdata/output/actual.bin'
```

Expected:
- manifest points at generated preprocess artifacts
- MD5 hashes match

- [ ] **Step 6: Commit**

```bash
git add lib/RuntimeMix/MixDirectBackend.cpp \
        lib/RuntimeMix/MixCommandBuilder.cpp \
        tools/mix-compiler/mix_compiler_main.cpp \
        tools/mix-validator/mix_validator_main.cpp
git commit -m "fix: make RuntimeMix direct baremix pass with toolkit preprocess"
```

## Self-Review

- Spec coverage:
  - preprocess stage: covered by Tasks 1-2
  - generated-source compile path: covered by Task 3
  - merge/pack/runner integration: covered by Task 4
  - xvm baremix pass + precision: covered by Task 5

- Placeholder scan:
  - no `TBD`, `TODO`, or “similar to” references remain
  - each verification step includes exact commands

- Type consistency:
  - all newly introduced helpers use consistent names across tasks:
    - `buildPreprocessCommand`
    - `buildExtractHostStubCommand`
    - `buildUpdateHostStubCommand`
    - `MixPreprocessOutputs`
    - `MixGeneratedConfig`

