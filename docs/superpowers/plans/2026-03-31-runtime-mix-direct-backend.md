# RuntimeMix Direct Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current AscendC CMake wrapper in `RuntimeMix` with a direct mix backend that orchestrates `bisheng + ld.lld + merge + pack + host stub/launcher` while keeping `mix-compiler` input shape and `mix-validator` CLI behavior stable.

**Architecture:** Add a staged direct backend under `RuntimeMix` with a stable `MixArtifact` output, keep the existing wrapper backend as a temporary baseline, and switch `mix-compiler` to use the direct path for end-to-end validation on `examples/baremix-test/baremix_custom.cpp`. Keep validator semantics unchanged so the future merge into `lib/Runtime` is mostly code motion, not redesign.

**Tech Stack:** C++17, LLVM Support, Ascend `bisheng`, `ld.lld`, `ascendc_pack_kernel`, generated host stub/runner sources, xvm simulator on Ascend910B1.

---

## File Map

- Create: `include/RuntimeMix/MixDirectBackend.h`
- Create: `include/RuntimeMix/MixSourceAnalyzer.h`
- Create: `include/RuntimeMix/MixCommandBuilder.h`
- Create: `include/RuntimeMix/MixStubTemplate.h`
- Modify: `include/RuntimeMix/MixArtifact.h`
- Modify: `lib/RuntimeMix/CMakeLists.txt`
- Create: `lib/RuntimeMix/MixDirectBackend.cpp`
- Create: `lib/RuntimeMix/MixSourceAnalyzer.cpp`
- Create: `lib/RuntimeMix/MixCommandBuilder.cpp`
- Create: `lib/RuntimeMix/MixStubTemplate.cpp`
- Modify: `tools/mix-compiler/mix_compiler_main.cpp`
- Modify: `tools/mix-validator/mix_validator_main.cpp`
- Test/verify in xvm using:
  - `build/runtime-mix-bootstrap`
  - `build/runtime-mix-baremix`
  - `examples/baremix-test/run.sh`

## Task 1: Stabilize the Direct Backend Public Surface

**Files:**
- Create: `include/RuntimeMix/MixDirectBackend.h`
- Create: `include/RuntimeMix/MixSourceAnalyzer.h`
- Create: `include/RuntimeMix/MixCommandBuilder.h`
- Modify: `include/RuntimeMix/MixArtifact.h`
- Modify: `lib/RuntimeMix/CMakeLists.txt`

- [ ] **Step 1: Add a direct-backend config/result surface**

```cpp
// include/RuntimeMix/MixSourceAnalyzer.h
#pragma once

#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace mlir::runtime {

struct MixAnalyzedKernel {
  std::string kernelName;
  std::string socVersion;
  std::string launcherSymbol;
  std::string aicEntry;
  std::string aivEntry;
  std::vector<std::string> commonFlags;
  std::vector<std::string> aicDefines;
  std::vector<std::string> aivDefines;
};

llvm::Expected<MixAnalyzedKernel>
analyzeMixKernel(llvm::StringRef kernelPath, llvm::StringRef kernelName,
                 llvm::StringRef socVersion);

} // namespace mlir::runtime
```

- [ ] **Step 2: Add a direct-backend orchestration entrypoint**

```cpp
// include/RuntimeMix/MixDirectBackend.h
#pragma once

#include "RuntimeMix/MixArtifact.h"
#include "llvm/Support/Error.h"
#include <string>

namespace mlir::runtime {

struct MixDirectCompileConfig {
  std::string kernelSrc;
  std::string kernelName;
  std::string socVersion;
  std::string outputDir;
};

class MixDirectBackend {
public:
  llvm::Expected<MixArtifact> compile(const MixDirectCompileConfig &cfg);
};

} // namespace mlir::runtime
```

- [ ] **Step 3: Extend `MixArtifact` only with fields needed by both backends**

```cpp
// include/RuntimeMix/MixArtifact.h
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
  std::string device_object_path;
  std::string host_stub_source_path;
};
```

- [ ] **Step 4: Register the new implementation units in the library target**

```cmake
# lib/RuntimeMix/CMakeLists.txt
add_mlir_library(AscendCRuntimeMix
  AscendCMixCompiler.cpp
  MixDirectBackend.cpp
  MixSourceAnalyzer.cpp
  MixCommandBuilder.cpp
  MixStubTemplate.cpp
  LINK_LIBS PUBLIC
  LLVMSupport
)
```

- [ ] **Step 5: Build to catch interface errors early**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build/runtime-mix-bootstrap --target AscendCRuntimeMix mix-compiler -j4'
```

Expected:

- compile succeeds
- no undefined type or header errors

- [ ] **Step 6: Commit**

```bash
git add include/RuntimeMix/MixDirectBackend.h \
  include/RuntimeMix/MixSourceAnalyzer.h \
  include/RuntimeMix/MixCommandBuilder.h \
  include/RuntimeMix/MixArtifact.h \
  lib/RuntimeMix/CMakeLists.txt
git commit -m "feat: add RuntimeMix direct backend interfaces"
```

## Task 2: Implement Source Analysis and Command Construction

**Files:**
- Create: `lib/RuntimeMix/MixSourceAnalyzer.cpp`
- Create: `lib/RuntimeMix/MixCommandBuilder.cpp`
- Possibly modify: `include/RuntimeMix/MixCommandBuilder.h`

- [ ] **Step 1: Implement standard mix naming analysis**

```cpp
llvm::Expected<MixAnalyzedKernel>
analyzeMixKernel(llvm::StringRef kernelPath, llvm::StringRef kernelName,
                 llvm::StringRef socVersion) {
  if (kernelPath.empty() || kernelName.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "kernelPath and kernelName are required");
  MixAnalyzedKernel out;
  out.kernelName = kernelName.str();
  out.socVersion = socVersion.str();
  out.launcherSymbol = ("aclrtlaunch_" + kernelName).str();
  out.aicEntry = (kernelName + "_0_mix_aic").str();
  out.aivEntry = (kernelName + "_0_mix_aiv").str();
  out.aicDefines = {
      "__MIX_CORE_MACRO__=1",
      ("auto_gen_" + kernelName + "_kernel=" + out.aicEntry),
      "__DAV_C220_CUBE__",
  };
  out.aivDefines = {
      "__MIX_CORE_MACRO__=1",
      ("auto_gen_" + kernelName + "_kernel=" + out.aivEntry),
      "__DAV_C220_VEC__",
  };
  return out;
}
```

- [ ] **Step 2: Add reusable command-build helpers for `bisheng` and `ld.lld`**

```cpp
std::vector<std::string> buildBishengCommand(const MixAnalyzedKernel &info,
                                             llvm::StringRef src,
                                             llvm::StringRef obj,
                                             llvm::ArrayRef<std::string> defs);

std::vector<std::string> buildLldRelocCommand(llvm::StringRef inputObj,
                                              llvm::StringRef outputObj);

std::vector<std::string> buildLldMergeCommand(llvm::StringRef aicObj,
                                              llvm::StringRef aivObj,
                                              llvm::StringRef outputObj);
```

- [ ] **Step 3: Encode tool lookup without sample CMake dependencies**

```cpp
static std::string getBishengPath() {
  return getAscendHome() + "/compiler/ccec_compiler/bin/bisheng";
}

static std::string getLldPath() {
  return getAscendHome() + "/aarch64-linux/ccec_compiler/bin/ld.lld";
}
```

- [ ] **Step 4: Build and run a compile-only smoke test**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build/runtime-mix-bootstrap --target AscendCRuntimeMix -j4'
```

Expected:

- library builds successfully

- [ ] **Step 5: Commit**

```bash
git add include/RuntimeMix/MixCommandBuilder.h \
  lib/RuntimeMix/MixSourceAnalyzer.cpp \
  lib/RuntimeMix/MixCommandBuilder.cpp
git commit -m "feat: add RuntimeMix direct mix source analysis"
```

## Task 3: Implement Direct AIC/AIV Compile and Device Merge

**Files:**
- Create: `lib/RuntimeMix/MixDirectBackend.cpp`
- Modify: `include/RuntimeMix/MixDirectBackend.h`

- [ ] **Step 1: Add workspace layout creation for direct backend outputs**

```cpp
// Inside MixDirectBackend::compile
llvm::SmallString<256> workDir(cfg.outputDir);
llvm::sys::path::append(workDir, "work");
llvm::SmallString<256> objDir(cfg.outputDir);
llvm::sys::path::append(objDir, "objects");
llvm::SmallString<256> outDir(cfg.outputDir);
llvm::sys::path::append(outDir, "out");
```

- [ ] **Step 2: Run two `bisheng` invocations and capture AIC/AIV objects**

```cpp
auto analyzed = analyzeMixKernel(cfg.kernelSrc, cfg.kernelName, cfg.socVersion);
if (!analyzed)
  return analyzed.takeError();

if (auto err = runProcess(buildBishengCommand(*analyzed, cfg.kernelSrc,
                                              aicObj, analyzed->aicDefines)))
  return err;
if (auto err = runProcess(buildBishengCommand(*analyzed, cfg.kernelSrc,
                                              aivObj, analyzed->aivDefines)))
  return err;
```

- [ ] **Step 3: Run `ld.lld` reloc and merge to produce a final device object**

```cpp
if (auto err = runProcess(buildLldRelocCommand(aicObj, aicRelocObj)))
  return err;
if (auto err = runProcess(buildLldRelocCommand(aivObj, aivRelocObj)))
  return err;
if (auto err = runProcess(buildLldMergeCommand(aicRelocObj, aivRelocObj,
                                               mergedDeviceObj)))
  return err;
```

- [ ] **Step 4: Persist enough metadata to debug direct backend outputs**

```cpp
std::string manifest =
    "kernel_name=" + cfg.kernelName + "\n" +
    "soc_version=" + cfg.socVersion + "\n" +
    "aic_object=" + aicObj + "\n" +
    "aiv_object=" + aivObj + "\n" +
    "device_object=" + mergedDeviceObj + "\n";
```

- [ ] **Step 5: Verify Milestone 1 in xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
  cmake --build build/runtime-mix-bootstrap --target mix-compiler -j4'
```

Then:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
  ./build/runtime-mix-bootstrap/bin/mix-compiler \
    --kernel examples/baremix-test/baremix_custom.cpp \
    --name baremix_custom \
    --output build/runtime-mix-baremix \
    --soc Ascend910B1'
```

Expected:

- `build/runtime-mix-baremix/objects/*.o` exist
- merged device object exists
- compile still fails later because stub/pack is not added yet

- [ ] **Step 6: Commit**

```bash
git add lib/RuntimeMix/MixDirectBackend.cpp \
  include/RuntimeMix/MixDirectBackend.h
git commit -m "feat: add RuntimeMix direct device compile pipeline"
```

## Task 4: Implement Generic Host Stub and Pack Stage

**Files:**
- Create: `include/RuntimeMix/MixStubTemplate.h`
- Create: `lib/RuntimeMix/MixStubTemplate.cpp`
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`

- [ ] **Step 1: Add a reusable stub template generator**

```cpp
struct MixStubTemplateArgs {
  std::string kernelName;
  std::string launcherSymbol;
  std::string outputHeaderPath;
  std::string outputSourcePath;
};

llvm::Error writeMixStubTemplate(const MixStubTemplateArgs &args);
```

- [ ] **Step 2: Generate a host stub source that exports `aclrtlaunch_<kernel>`**

```cpp
return R"cpp(
#include "acl/acl.h"
extern "C" aclError ACLRT_LAUNCH_KERNEL_IMPL(blockDim_t, aclrtStream,
                                             void*, void*, void*, void*,
                                             void*, void*);
extern "C" aclError )cpp" + args.launcherSymbol + R"cpp((
    uint32_t blockDim, aclrtStream stream, void* arg0, void* arg1, void* arg2,
    void* arg3, void* workspace, void* tiling) {
  return ACLRT_LAUNCH_KERNEL_IMPL(blockDim, stream, arg0, arg1, arg2, arg3,
                                  workspace, tiling);
}
)cpp";
```

- [ ] **Step 3: Add the pack-tool invocation**

```cpp
std::vector<std::string> buildPackCommand(llvm::StringRef hostStubObj,
                                          llvm::StringRef deviceObj,
                                          llvm::StringRef packedSo);
```

- [ ] **Step 4: Extend backend compile flow to build stub object and pack `.so`**

```cpp
if (auto err = writeMixStubTemplate(stubArgs))
  return err;
if (auto err = runProcess(buildHostCompileCommand(stubCpp, stubObj)))
  return err;
if (auto err = runProcess(buildPackCommand(stubObj, mergedDeviceObj, packedSo)))
  return err;
```

- [ ] **Step 5: Verify Milestone 2 in xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
  ./build/runtime-mix-bootstrap/bin/mix-compiler \
    --kernel examples/baremix-test/baremix_custom.cpp \
    --name baremix_custom \
    --output build/runtime-mix-baremix \
    --soc Ascend910B1'
```

Expected:

- launcher header exists
- packed `.so` exists
- artifact manifest prints non-empty `kernel_so` and `launcher_header_dir`

- [ ] **Step 6: Commit**

```bash
git add include/RuntimeMix/MixStubTemplate.h \
  lib/RuntimeMix/MixStubTemplate.cpp \
  lib/RuntimeMix/MixDirectBackend.cpp
git commit -m "feat: add RuntimeMix direct stub and pack stages"
```

## Task 5: Switch `mix-compiler` to the Direct Backend and Preserve Artifact Shape

**Files:**
- Modify: `tools/mix-compiler/mix_compiler_main.cpp`
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`

- [ ] **Step 1: Wire `mix-compiler` to call the direct backend**

```cpp
mlir::runtime::MixDirectCompileConfig cfg;
cfg.kernelSrc = KernelFile;
cfg.kernelName = kernelName;
cfg.socVersion = SocVersion;
cfg.outputDir = OutputDir;

mlir::runtime::MixDirectBackend backend;
auto artifact = backend.compile(cfg);
```

- [ ] **Step 2: Keep printed artifact fields identical to the current CLI**

```cpp
llvm::outs() << "kernel_name=" << artifact->kernel_name << "\n";
llvm::outs() << "kernel_so=" << artifact->kernel_so_path << "\n";
llvm::outs() << "launcher_header_dir=" << artifact->launcher_header_dir << "\n";
llvm::outs() << "install_dir=" << artifact->install_dir << "\n";
llvm::outs() << "host_runner=" << artifact->host_runner_path << "\n";
```

- [ ] **Step 3: Keep the old wrapper backend source intact for temporary comparison**

```cpp
// Do not delete AscendCMixCompiler in this task.
// Only stop routing mix-compiler through it.
```

- [ ] **Step 4: Rebuild and run compile-only validation in xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
  cmake --build build/runtime-mix-bootstrap --target mix-compiler -j4 && \
  ./build/runtime-mix-bootstrap/bin/mix-compiler \
    --kernel examples/baremix-test/baremix_custom.cpp \
    --name baremix_custom \
    --output build/runtime-mix-baremix \
    --soc Ascend910B1'
```

Expected:

- CLI output still uses the same keys
- packed artifact paths are valid

- [ ] **Step 5: Commit**

```bash
git add tools/mix-compiler/mix_compiler_main.cpp \
  lib/RuntimeMix/MixDirectBackend.cpp
git commit -m "feat: switch mix-compiler to RuntimeMix direct backend"
```

## Task 6: Reuse Current Validator Semantics Against Direct Artifacts

**Files:**
- Modify: `tools/mix-validator/mix_validator_main.cpp`
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`

- [ ] **Step 1: Ensure direct backend writes the same artifact layout validator expects**

```cpp
artifact.install_dir = outDir.str().str();
artifact.kernel_so_path = packedSo;
artifact.launcher_header_dir = includeDir;
artifact.host_runner_path = runnerPath;
```

- [ ] **Step 2: Only adjust validator if layout details changed, not CLI shape**

```cpp
// Preserve:
// --artifact-root
// --input-dir
// --golden
// --output-file
// --soc
```

- [ ] **Step 3: End-to-end validate in xvm with repo scripts**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && bash examples/baremix-test/run.sh'
```

Expected:

- direct backend compiles baremix
- `mix-validator` prints `PASS`
- `verify_result.py` prints `test pass`

- [ ] **Step 4: Commit**

```bash
git add tools/mix-validator/mix_validator_main.cpp \
  lib/RuntimeMix/MixDirectBackend.cpp
git commit -m "feat: validate RuntimeMix direct artifacts in simulator"
```

## Task 7: Generality Check with a Second Mix Example

**Files:**
- Reuse: existing second mix example under `examples/matmul-add-relu-sum/`
- Modify only if needed: `examples/baremix-test/README.md` or a new runtime-mix note

- [ ] **Step 1: Select one non-baremix mix kernel already present in the repo**

```bash
rg -n "MIX|mix" examples/matmul-add-relu-sum -g'*.cpp'
```

- [ ] **Step 2: Compile the second example with the same direct backend CLI**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
  ./build/runtime-mix-bootstrap/bin/mix-compiler \
    --kernel <second-kernel>.cpp \
    --name <second-kernel-name> \
    --output build/runtime-mix-second \
    --soc Ascend910B1'
```

Expected:

- AIC/AIV compile succeeds
- pack succeeds

- [ ] **Step 3: If ready-to-run data exists, execute validator; otherwise at minimum record compile success**

```bash
# Preferred if data/golden exist:
./build/runtime-mix-bootstrap/bin/mix-validator ...
```

- [ ] **Step 4: Document the generality result**

```md
- Example 1: `baremix_custom.cpp` compile + sim + accuracy PASS
- Example 2: `<second-kernel>.cpp` compile PASS
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-03-31-runtime-mix-direct-backend-design.md \
  examples/baremix-test/README.md
git commit -m "docs: record RuntimeMix direct backend validation"
```

## Self-Review

- Spec coverage:
  - direct backend stages: Tasks 1-4
  - stable artifact and CLI boundary: Tasks 1, 5, 6
  - xvm validation: Tasks 3-7
  - second example generality check: Task 7
- Placeholder scan:
  - no `TBD` or `TODO`
  - one intentionally flexible input remains in Task 7 (`<second-kernel>.cpp`) because the exact second sample should be chosen from live repo contents immediately before execution
- Type consistency:
  - `MixDirectCompileConfig`, `MixAnalyzedKernel`, `MixArtifact` names are consistent across tasks

