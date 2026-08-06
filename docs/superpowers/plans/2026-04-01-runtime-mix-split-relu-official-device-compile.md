# RuntimeMix Split-ReLU Official-Style Device Compile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 split-relu wrapperless 的 preprocess-generated source 新增 official-style device compile command，并在 xvm 上验证 device object 尺寸与注册行为是否向官方基线收敛。

**Architecture:** 在 `MixCommandBuilder` 中增加一条独立的 preprocess-device builder，只给 split-relu wrapperless 的 preprocess-generated path 使用。`RuntimeMix` 其余 manual mix 路径、host stub/pack/runner/validator 逻辑保持不变；验证先看 object size，再看 `RegisterAscendBinary failed: 107000` 是否消失。

**Tech Stack:** C++, LLVM Support, RuntimeMix, bisheng, AscendC simulator, Bash

---

### Task 1: Add Official-Style Preprocessed Device Compile Builder

**Files:**
- Modify: `include/RuntimeMix/MixCommandBuilder.h`
- Modify: `lib/RuntimeMix/MixCommandBuilder.cpp`

- [ ] **Step 1: Declare a dedicated builder for preprocess-generated source**

Add a new declaration to `include/RuntimeMix/MixCommandBuilder.h`:

```cpp
std::vector<std::string>
buildPreprocessedDeviceCompileCommand(llvm::StringRef src,
                                      llvm::StringRef obj,
                                      MixCoreType coreType);
```

Do not change existing builder signatures.

- [ ] **Step 2: Implement the new builder with official-style flags**

In `lib/RuntimeMix/MixCommandBuilder.cpp`, implement the builder with this structure:

```cpp
std::vector<std::string>
buildPreprocessedDeviceCompileCommand(llvm::StringRef src,
                                      llvm::StringRef obj,
                                      MixCoreType coreType) {
  std::vector<std::string> args;
  args.push_back(getBishengPath());
  args.push_back("-DHAVE_TILING");
  args.push_back("-DHAVE_WORKSPACE");
  args.push_back("-DTILING_KEY_VAR=0");
  appendAscIncludes(args);
  appendTikcppIncludes(args);
  args.push_back("-g");
  args.push_back("--cce-disable-kernel-global-attr-check");
  args.push_back("--cce-aicore-arch=" + getArchForCore(coreType));
  args.push_back("--cce-aicore-only");
  args.push_back("--cce-auto-sync");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-stack-size=0x8000");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-function-stack-size=0x8000");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-record-overflow=true");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-addr-transform");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-dcci-insert-for-scalar=false");
  args.push_back("-O3");
  args.push_back("-std=c++17");
  args.push_back("--cce-aicore-lang");
  args.push_back("-include");
  args.push_back(getVersionHeader());
  args.push_back("-o");
  args.push_back(obj.str());
  args.push_back("-c");
  args.push_back(src.str());
  return args;
}
```

Do not add:
- `__MIX_CORE_MACRO__`
- `auto_gen_*=` macros
- `__DAV_C220_*__`

- [ ] **Step 3: Re-read builder declarations and implementation**

Run:

```bash
sed -n '1,220p' include/RuntimeMix/MixCommandBuilder.h
sed -n '1,260p' lib/RuntimeMix/MixCommandBuilder.cpp
```

Expected:
- new builder is declared once
- implementation is separate from `buildBishengCommand(...)`
- old manual mix builder remains intact

- [ ] **Step 4: Static grep for forbidden manual macros in the new path**

Run:

```bash
rg -n "buildPreprocessedDeviceCompileCommand|__MIX_CORE_MACRO__|auto_gen_.*=|__DAV_C220_" include/RuntimeMix/MixCommandBuilder.h lib/RuntimeMix/MixCommandBuilder.cpp
```

Expected:
- the new builder exists
- forbidden manual macros only appear in old manual mix code paths, not inside the new builder body

- [ ] **Step 5: Commit builder-only changes**

```bash
git add include/RuntimeMix/MixCommandBuilder.h \
        lib/RuntimeMix/MixCommandBuilder.cpp
git commit -m "feat: add official-style device compile builder for preprocess path"
```

Expected:
- commit contains only command-builder changes for this task

### Task 2: Use The New Builder Only For Split-ReLU Preprocess Path

**Files:**
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`

- [ ] **Step 1: Route only split-relu preprocess path to the new builder**

In `lib/RuntimeMix/MixDirectBackend.cpp`, keep this policy:

- manual mix path still uses `buildBishengCommand(...)`
- split-relu wrapperless preprocess-generated path uses `buildPreprocessedDeviceCompileCommand(...)`

The decision point should look like:

```cpp
const bool useOfficialPreprocessedCompile =
    useSplitReluSample && !useManualGeneratedPath;

const std::vector<std::string> aicCmd =
    useOfficialPreprocessedCompile
        ? buildPreprocessedDeviceCompileCommand(generatedSourcePath, aicObj,
                                               MixCoreType::AIC)
        : buildBishengCommand(deviceAnalyzed, generatedSourcePath, aicObj,
                              MixCoreType::AIC);

const std::vector<std::string> aivCmd =
    useOfficialPreprocessedCompile
        ? buildPreprocessedDeviceCompileCommand(generatedSourcePath, aivObj,
                                               MixCoreType::AIV)
        : buildBishengCommand(deviceAnalyzed, generatedSourcePath, aivObj,
                              MixCoreType::AIV);
```

- [ ] **Step 2: Preserve existing split-relu fallback logic**

Do not remove the current split-relu preprocess fallback that:
- tolerates missing launcher header from preprocess output
- falls back to manual stub template when needed
- falls back to analyzed AIC/AIV definitions if generated config omits them

This task is only about changing device compile semantics.

- [ ] **Step 3: Re-read the touched backend section**

Run:

```bash
sed -n '1330,1605p' lib/RuntimeMix/MixDirectBackend.cpp
```

Expected:
- only split-relu preprocess path uses the new builder
- baremix and current manual mix paths are unchanged

- [ ] **Step 4: Commit backend routing change**

```bash
git add lib/RuntimeMix/MixDirectBackend.cpp
git commit -m "feat: use official-style device compile for split relu preprocess path"
```

Expected:
- commit contains only backend routing logic for this task

### Task 3: Validate Device Object Convergence And Runtime Behavior On xvm

**Files:**
- Verify: `lib/RuntimeMix/MixCommandBuilder.cpp`
- Verify: `lib/RuntimeMix/MixDirectBackend.cpp`
- Verify: `build/runtime-mix-relu-split/out/manifest.txt`
- Verify: `build/runtime-mix-relu-split/work/merge_obj/device.o`

- [ ] **Step 1: Rebuild tool bootstrap and rerun the split-relu example on xvm**

Run:

```bash
sleep 4 && ssh xvm@orb 'set -euo pipefail
cd /home/niu/code/Ascend-MLIR
rm -f build/runtime-mix-bootstrap/bin/mix-compiler
rm -f build/runtime-mix-bootstrap/bin/mix-validator
rm -rf build/runtime-mix-relu-split
bash examples/relu-split-mix-test/run.sh'
```

Expected:
- `mix-compiler` and `mix-validator` rebuild successfully
- split-relu artifact rebuild completes far enough to emit manifest and device object

- [ ] **Step 2: Compare device object sizes against the current failure baseline**

Run:

```bash
sleep 4 && ssh xvm@orb 'set -euo pipefail
cd /home/niu/code/Ascend-MLIR
stat -c "AIC_SIZE %s" build/runtime-mix-relu-split/objects/fc_relu_split_aic.o
stat -c "AIV_SIZE %s" build/runtime-mix-relu-split/objects/fc_relu_split_aiv.o
stat -c "MERGE_DEVICE_SIZE %s" build/runtime-mix-relu-split/work/merge_obj/device.o'
```

Expected:
- sizes are no longer the old `1120 / 1120 / 1520`
- sizes move materially toward the official wrapperless baseline

- [ ] **Step 3: Inspect manifest compile commands**

Run:

```bash
sleep 4 && ssh xvm@orb 'set -euo pipefail
cd /home/niu/code/Ascend-MLIR
sed -n "1,120p" build/runtime-mix-relu-split/out/manifest.txt'
```

Expected manifest evidence:
- `source_path=.../fc_relu_split_wrapperless.cpp`
- `generated_source_path=.../auto_gen_fc_relu_split_wrapperless.cpp`
- `bisheng_aic=` and `bisheng_aiv=` lines no longer contain manual mix macros for this path

- [ ] **Step 4: Record runtime result without guessing a fix**

If `bash examples/relu-split-mix-test/run.sh` still fails, record the first concrete execution boundary:

```bash
sleep 4 && ssh xvm@orb 'set -euo pipefail
cd /home/niu/code/Ascend-MLIR
./build/runtime-mix-bootstrap/bin/mix-validator \
  --artifact-root build/runtime-mix-relu-split \
  --input-dir build/runtime-mix-relu-split/testdata/input \
  --golden build/runtime-mix-relu-split/testdata/output/golden.bin \
  --output-file build/runtime-mix-relu-split/testdata/output/actual.bin \
  --soc Ascend910B1'
```

Expected:
- either PASS, or a single narrower runtime failure than the old `107000` registration error
- do not patch further in this task; capture evidence only

- [ ] **Step 5: Commit only if convergence is real**

```bash
git add include/RuntimeMix/MixCommandBuilder.h \
        lib/RuntimeMix/MixCommandBuilder.cpp \
        lib/RuntimeMix/MixDirectBackend.cpp
git commit -m "fix: align split relu device compile with official preprocess path"
```

Expected:
- commit only if xvm evidence shows real convergence
- if device sizes do not move or runtime regresses ambiguously, stop and report evidence instead of committing
