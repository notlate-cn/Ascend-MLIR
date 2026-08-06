# RuntimeMix Split-ReLU Dual-Source Host Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 split-relu 第二样例在保持当前 device 路径稳定的前提下，补回官方 host launcher 生成链，并在 xvm 上验证最终 `.so` 含 `aclrtlaunch_fc_relu_split` 且运行通过。

**Architecture:** split-relu 采用 dual-source policy。device 侧继续使用 wrapperless 源码，host 侧单独使用原始 mix 源码走官方等价的 `host_bisheng` 编译，再通过 `recompile_binary.py` 把 host objects 重新链接进最终 `.so`。现有 baremix 和其他 RuntimeMix 路径保持不变。

**Tech Stack:** C++, LLVM Support, RuntimeMix, bisheng, AscendC legacy util scripts, Bash, xvm

---

### Task 1: Add Host Bisheng / Recompile Command Builders

**Files:**
- Modify: `include/RuntimeMix/MixCommandBuilder.h`
- Modify: `lib/RuntimeMix/MixCommandBuilder.cpp`

- [ ] **Step 1: Declare host-bisheng and recompile builders**

Add these declarations to `include/RuntimeMix/MixCommandBuilder.h`:

```cpp
std::vector<std::string>
buildHostBishengCommand(llvm::StringRef src, llvm::StringRef obj,
                        llvm::StringRef tripleChevronHeader);

std::vector<std::string>
buildRecompileBinaryCommand(llvm::StringRef rootDir, llvm::StringRef targetName,
                            llvm::StringRef addDir);
```

Do not change existing builder signatures.

- [ ] **Step 2: Implement host-bisheng builder using official flags**

In `lib/RuntimeMix/MixCommandBuilder.cpp`, implement `buildHostBishengCommand(...)` using the official sample-style host flags observed in xvm:

```cpp
std::vector<std::string>
buildHostBishengCommand(llvm::StringRef src, llvm::StringRef obj,
                        llvm::StringRef tripleChevronHeader) {
  std::vector<std::string> args;
  args.push_back(getBishengPath());
  args.push_back("-DTILING_KEY_VAR=0");
  appendAscIncludes(args);
  args.push_back("-I");
  args.push_back(getTikcppRoot() + "/tikcfw");
  args.push_back("-I");
  args.push_back(getTikcppRoot() + "/tikcfw/interface");
  args.push_back("-I");
  args.push_back(getTikcppRoot() + "/tikcfw/impl");
  args.push_back("-g");
  args.push_back("-include");
  args.push_back(tripleChevronHeader.str());
  args.push_back("-O3");
  args.push_back("-std=c++17");
  args.push_back("--cce-aicore-lang");
  args.push_back("-include");
  args.push_back(getVersionHeader());
  args.push_back("-fPIC");
  args.push_back("--cce-host-only");
  args.push_back("-fcce-kernel-launch-custom");
  args.push_back("-DONE_CORE_DUMP_SIZE=1048576");
  args.push_back("-o");
  args.push_back(obj.str());
  args.push_back("-c");
  args.push_back(src.str());
  return args;
}
```

Do not route this builder through normal C++ host compile logic.

- [ ] **Step 3: Implement recompile command builder**

Add `buildRecompileBinaryCommand(...)` in `lib/RuntimeMix/MixCommandBuilder.cpp`:

```cpp
std::vector<std::string>
buildRecompileBinaryCommand(llvm::StringRef rootDir, llvm::StringRef targetName,
                            llvm::StringRef addDir) {
  return {
      "python3",
      getAscendHome() +
          "/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/"
          "recompile_binary.py",
      "--root-dir",
      rootDir.str(),
      "--target-name",
      targetName.str(),
      "--add-dir",
      addDir.str(),
  };
}
```

- [ ] **Step 4: Static inspection**

Run:

```bash
sed -n '1,220p' include/RuntimeMix/MixCommandBuilder.h
sed -n '1,520p' lib/RuntimeMix/MixCommandBuilder.cpp
```

Expected:
- declarations appear once
- host-bisheng builder is separate from normal host stub compile builder
- recompile builder is separate from pack/link builders

- [ ] **Step 5: Commit command-builder changes**

```bash
git add include/RuntimeMix/MixCommandBuilder.h \
        lib/RuntimeMix/MixCommandBuilder.cpp
git commit -m "feat: add split relu host bisheng and recompile builders"
```

Expected:
- commit only contains command-builder changes

### Task 2: Route Split-ReLU To Dual-Source Host Pipeline

**Files:**
- Modify: `lib/RuntimeMix/MixDirectBackend.cpp`

- [ ] **Step 1: Add explicit host source selection for split-relu**

In `lib/RuntimeMix/MixDirectBackend.cpp`, add a helper path policy for split-relu:

- device source path remains `generatedSourcePath` from wrapperless preprocess
- host source path becomes original mix source:

```cpp
const bool useSplitReluDualSourceHost = useSplitReluSample;
std::string hostSourcePath = sourcePath.str().str();
if (useSplitReluDualSourceHost)
  hostSourcePath = sourcePath.str().str().find("fc_relu_split_wrapperless.cpp") != std::string::npos
      ? sourcePath.str().str().substr(0, sourcePath.str().str().find("fc_relu_split_wrapperless.cpp")) +
            "fc_relu_split_mix.cpp"
      : sourcePath.str().str();
```

Refine path construction cleanly; do not leave brittle substring logic in final code. Prefer a small helper that maps wrapperless -> original split mix explicitly.

- [ ] **Step 2: Add host_dir layout**

Still in `MixDirectBackend.cpp`, create a RuntimeMix-local host object directory matching the official purpose:

```cpp
llvm::SmallString<256> hostDir(outputRoot);
llvm::sys::path::append(hostDir, "host_dir");
llvm::SmallString<256> hostObjectsDir(hostDir);
llvm::sys::path::append(hostObjectsDir, "objects-Debug", "host_bisheng_obj");
```

Ensure directories are created before host compilation.

- [ ] **Step 3: Use host-bisheng for split-relu**

For split-relu only:
- compile `hostSourcePath` with `buildHostBishengCommand(...)`
- use `build/include/.../aclrtlaunch_triple_chevrons_func.h` equivalent from preprocess output
- write resulting object into the RuntimeMix host_dir tree

Do not change:
- current wrapperless device compile path
- current pack command
- current host stub compile path

This task adds host objects; it does not replace host_stub.o.

- [ ] **Step 4: Add recompile step after host link**

After current `pack + host link` succeeds, add:

```cpp
const std::vector<std::string> recompileCmd =
    buildRecompileBinaryCommand(outputRoot, "ascendc_kernels_sim", hostDir);
```

Run it only for split-relu dual-source path.

Important:
- `recompile_binary.py` expects a `link.txt` under `CMakeFiles/<target>.dir/link.txt`
- if RuntimeMix layout does not yet provide that path, create the minimal compatible layout and write the link command text there before running recompile

Do not guess. Mirror only what the script actually reads:
- root dir
- target name
- add dir
- link.txt

- [ ] **Step 5: Extend manifest/debug output**

Add these manifest fields:

```text
host_source_path=...
host_bisheng_object=...
host_bisheng_cmd=...
host_object_dir=...
recompile_cmd=...
```

Leave existing fields untouched.

- [ ] **Step 6: Static inspection**

Run:

```bash
sed -n '1280,1810p' lib/RuntimeMix/MixDirectBackend.cpp
```

Expected:
- split-relu clearly uses device wrapperless + host original mix source
- host_bisheng and recompile are isolated to split-relu path
- baremix path is unchanged

- [ ] **Step 7: Commit backend routing changes**

```bash
git add lib/RuntimeMix/MixDirectBackend.cpp
git commit -m "feat: add dual-source host pipeline for split relu"
```

Expected:
- commit contains only backend-side dual-source host pipeline changes

### Task 3: Validate Symbol Recovery And Runtime Behavior On xvm

**Files:**
- Verify: `lib/RuntimeMix/MixDirectBackend.cpp`
- Verify: `build/runtime-mix-relu-split/out/manifest.txt`
- Verify: `build/runtime-mix-relu-split/out/libfc_relu_split_packed.so`

- [ ] **Step 1: Rebuild RuntimeMix tools and rerun split-relu example**

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
- artifact rebuild completes
- manifest is emitted
- final `.so` exists

- [ ] **Step 2: Verify host_dir and host object are no longer empty**

Run:

```bash
sleep 4 && ssh xvm@orb 'set -euo pipefail
cd /home/niu/code/Ascend-MLIR
find build/runtime-mix-relu-split/host_dir -type f | sort'
```

Expected:
- host_dir contains at least one compiled host object
- not just empty directories

- [ ] **Step 3: Verify final `.so` exports launcher symbol**

Run:

```bash
sleep 4 && ssh xvm@orb 'set -euo pipefail
nm -C /home/niu/code/Ascend-MLIR/build/runtime-mix-relu-split/out/libfc_relu_split_packed.so | \
  grep -E "aclrtlaunch_fc_relu_split|fc_relu_split" || true'
```

Expected:
- `aclrtlaunch_fc_relu_split` appears

- [ ] **Step 4: Verify manifest records dual-source host pipeline**

Run:

```bash
sleep 4 && ssh xvm@orb 'set -euo pipefail
grep -nE "^(source_path|generated_source_path|host_source_path|host_bisheng_object|host_bisheng_cmd|host_object_dir|recompile_cmd)=" \
  /home/niu/code/Ascend-MLIR/build/runtime-mix-relu-split/out/manifest.txt'
```

Expected:
- `source_path` still points to wrapperless compile input
- `host_source_path` points to original `fc_relu_split_mix.cpp`
- host/recompile metadata is present

- [ ] **Step 5: Record runtime result without speculative fixes**

If the rerun still fails, record exact output again:

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
- either PASS
- or a narrower failure than current “missing launcher / missing host object” gap

Do not patch further in this task. Capture evidence only.

- [ ] **Step 6: Commit only if symbol recovery is real**

```bash
git add include/RuntimeMix/MixCommandBuilder.h \
        lib/RuntimeMix/MixCommandBuilder.cpp \
        lib/RuntimeMix/MixDirectBackend.cpp
git commit -m "feat: restore split relu host launcher through dual-source host pipeline"
```

Expected:
- only commit if xvm evidence shows:
  - non-empty host_dir
  - launcher symbol present in final `.so`

If runtime still fails, that is acceptable for this task as long as host launcher recovery is proven.

### Task 4: Update Example Documentation

**Files:**
- Modify: `examples/relu-split-mix-test/README.md`

- [ ] **Step 1: Document dual-source behavior**

Update the README to state clearly:
- split-relu currently uses wrapperless source for device path
- original mix source is retained for host launcher generation
- this is an interim validation strategy, not the final general solution

- [ ] **Step 2: Document current acceptance criteria**

Add a concise note covering:
- host_dir must be non-empty
- final `.so` should contain `aclrtlaunch_fc_relu_split`
- runtime pass is still the final gate, but launcher recovery is the first acceptance milestone

- [ ] **Step 3: Re-read README**

Run:

```bash
sed -n '1,220p' examples/relu-split-mix-test/README.md
```

Expected:
- wording matches current implementation reality
- no stale “single-source wrapperless end-to-end” claims remain

- [ ] **Step 4: Commit docs change**

```bash
git add examples/relu-split-mix-test/README.md
git commit -m "docs: describe split relu dual-source host pipeline"
```

Expected:
- commit contains only README update
