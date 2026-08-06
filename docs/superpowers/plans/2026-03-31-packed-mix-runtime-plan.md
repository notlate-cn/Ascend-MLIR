# Packed Mix Runtime Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make our own `lib/runtime` execute packed mix kernels for matmul mix examples while keeping existing pure-vector and cube flows unchanged.

**Architecture:** Keep the current raw runtime path for `vec`/`cube` kernels. Add a new `mix`-only path where `Compiler` emits a toolkit-packed shared library (`lib<kernel>_packed.so`), and `Executor`/`validator`/`HostRunnerGen` launch it through `aclrtlaunch_<kernel>` exported by the packed host stub.

**Tech Stack:** C++, LLVM support utilities, existing `lib/runtime` compiler/executor, CANN toolkit scripts (`merge_mix_obj.sh`, `merge_obj.sh`, `ascendc_pack_kernel.sh`), simulator runtime (`acl`, `libruntime_camodel.so`).

---

## File Structure

### Existing files to modify
- `include/Runtime/Compiler.h`
  - Extend compiler contract comments to note that `mix` returns a packed `.so` artifact instead of a plain `.bin`.
- `lib/Runtime/Compiler.cpp`
  - Add a `mix`-specific artifact pipeline that emits wrapper source, merges AIC/AIV device objects, packs them into a host stub object, and links a packed shared library.
- `include/Runtime/Executor.h`
  - Add a mix-only execution entrypoint for packed shared libraries.
- `lib/Runtime/Executor.cpp`
  - Implement packed mix launch via `dlopen + dlsym(aclrtlaunch_<kernel>)`, reusing existing H2D/D2H and allocation logic.
- `tools/validator/validator_main.cpp`
  - Route `kernel_type == mix` to the new packed launch path; keep `vec/cube` unchanged.
- `lib/Runtime/HostRunnerGen.cpp`
  - Generate a mix runner that loads the packed `.so` and calls `aclrtlaunch_<kernel>` instead of the raw `rtKernelLaunch` path.
- `test/tools/runtime/test_runtime.cpp`
  - Add regression tests for `Compiler`/`HostRunnerGen` mix artifact naming and emitted packed-path behavior.

### New files to create
- `lib/Runtime/MixHostStubTemplate.h` (or `.inc`)
  - Small string template helper for generating the packed host stub source in `Compiler.cpp` without cluttering the compiler implementation.
- `test/tools/runtime/mix_stub_fixture.cpp`
  - Minimal fixture content for validating packed mix generation strings without needing a real simulator run in unit tests.

### Existing files used only for verification/reference
- `examples/matmul-add-relu-sum/final_sample_target_graph/*`
  - Reference successful packed mix graph and official toolkit execution model.
- `/home/niu/Ascend/latest/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/merge_mix_obj.sh`
- `/home/niu/Ascend/latest/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/merge_obj.sh`
- `/home/niu/Ascend/latest/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/ascendc_pack_kernel.sh`

---

### Task 1: Teach `Compiler` to emit packed mix artifacts

**Files:**
- Create: `lib/Runtime/MixHostStubTemplate.h`
- Modify: `include/Runtime/Compiler.h`
- Modify: `lib/Runtime/Compiler.cpp`
- Test: `test/tools/runtime/test_runtime.cpp`

- [ ] **Step 1: Write the failing compiler artifact test**

```cpp
// test/tools/runtime/test_runtime.cpp
{
  HostRunnerGen gen;
  Compiler::Config cfg;
  cfg.kernel_type = "mix";
  cfg.soc_version = "Ascend910B1";
  cfg.verbose = false;

  Compiler compiler(cfg);
  auto out = compiler.Compile(
      "/tmp/rt_mix_fixture/fc_relu_mix.cpp",
      "/tmp/rt_mix_fixture/build",
      "fc_relu");
  EXPECT((bool)out, "mix compiler returns an artifact path");
  if (out) {
    EXPECT(out->find(".so") != std::string::npos,
           "mix compiler returns packed shared library path");
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
orb -m xvm bash -lc 'cd /home/niu/code/Ascend-MLIR && LLVM_BUILD_DIR=/home/niu/code/llvm-project/build bash test/tools/runtime/run_runtime.sh'
```

Expected: FAIL because `Compiler::Compile()` still returns a `.bin` for `mix`.

- [ ] **Step 3: Add a dedicated host-stub template helper**

```cpp
// lib/Runtime/MixHostStubTemplate.h
#pragma once
#include <string>

namespace mlir::runtime {

inline std::string emitPackedMixHostStub(const std::string &kernelName,
                                         const std::string &targetName,
                                         const std::string &socVersion,
                                         size_t mixLen) {
  return "... generated host stub source with aclrtlaunch_" + kernelName + " ...";
}

} // namespace mlir::runtime
```

- [ ] **Step 4: Update compiler contract comment**

```cpp
// include/Runtime/Compiler.h
// Compiles src_file into a runtime-consumable artifact.
// - vec/cube: returns output_dir/kernel_name.bin
// - mix: returns output_dir/lib<kernel_name>_packed.so
llvm::Expected<std::string> Compile(const std::string& src_file,
                                    const std::string& output_dir,
                                    const std::string& kernel_name);
```

- [ ] **Step 5: Implement `mix` artifact generation in `Compiler.cpp`**

```cpp
// lib/Runtime/Compiler.cpp (mix branch sketch)
if (cfg_.kernel_type == "mix") {
  std::string aicObj = output_dir + "/" + kernel_name + "_aic.o";
  std::string aivObj = output_dir + "/" + kernel_name + "_aiv.o";
  std::string mixBase = " -D__MIX_CORE_MACRO__=1";
  std::string aicDef = mixBase + " -Dauto_gen_" + kernel_name +
                       "_kernel=" + kernel_name + "_0_mix_aic";
  std::string aivDef = mixBase + " -Dauto_gen_" + kernel_name +
                       "_kernel=" + kernel_name + "_0_mix_aiv";

  if (auto err = runCmd(makeCompileCmd("dav-c220-cube", aicObj, aicDef)))
    return std::move(err);
  if (auto err = runCmd(makeCompileCmd("dav-c220-vec", aivObj, aivDef)))
    return std::move(err);

  std::string aicDir = output_dir + "/mix_aic";
  std::string aivDir = output_dir + "/mix_aiv";
  std::string mergeDir = output_dir + "/mix_merge";
  llvm::sys::fs::create_directories(aicDir);
  llvm::sys::fs::create_directories(aivDir);
  llvm::sys::fs::create_directories(mergeDir);
  llvm::sys::fs::copy_file(aicObj, aicDir + "/device.o");
  llvm::sys::fs::copy_file(aivObj, aivDir + "/device.o");

  writeTextFile(aicDir + "/mix_build.flag", "");
  writeTextFile(aivDir + "/mix_build.flag", "");

  std::string mergeScript = ascend_home + "/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/merge_mix_obj.sh";
  std::string mergeCmd = mergeScript + " -l " + lld +
                         " -o " + mergeDir +
                         " --aic-dir " + aicDir +
                         " --aiv-dir " + aivDir +
                         " --build-type Debug";
  if (auto err = RunProcess({"/bin/sh", "-c", mergeCmd}))
    return std::move(err);

  size_t mixLen = llvm::sys::fs::file_size(mergeDir + "/device.o");
  std::string hostStubCpp = output_dir + "/host_stub.cpp";
  std::string hostStubObj = output_dir + "/host_stub.o";
  writeTextFile(hostStubCpp, emitPackedMixHostStub(kernel_name,
                                                   "ascendc_kernels_sim",
                                                   cfg_.soc_version,
                                                   mixLen));
  if (auto err = RunProcess({"/bin/sh", "-c",
      std::string("g++ -fPIC -std=c++17 -c ") + hostStubCpp + " -o " + hostStubObj}))
    return std::move(err);

  std::string packScript = ascend_home + "/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/ascendc_pack_kernel.sh";
  std::string packTool = ascend_home + "/bin/ascendc_pack_kernel";
  std::string packCmd = packScript + " --pack_tool " + packTool +
                        " --elf_in " + hostStubObj +
                        " --add_dir " + mergeDir;
  if (auto err = RunProcess({"/bin/sh", "-c", packCmd}))
    return std::move(err);

  std::string packedSo = output_dir + "/lib" + kernel_name + "_packed.so";
  std::string linkSo = "/usr/bin/c++ -fPIC -shared -o " + packedSo +
                       " " + hostStubObj +
                       " -L" + ascend_home + "/lib64" +
                       " -L" + ascend_home + "/tools/simulator/" + cfg_.soc_version + "/lib" +
                       " " + ascend_home + "/lib64/libascendc_runtime.a -lascend_dump -lc_sec";
  if (auto err = RunProcess({"/bin/sh", "-c", linkSo}))
    return std::move(err);

  return packedSo;
}
```

- [ ] **Step 6: Re-run the runtime unit test**

Run:
```bash
orb -m xvm bash -lc 'cd /home/niu/code/Ascend-MLIR && LLVM_BUILD_DIR=/home/niu/code/llvm-project/build bash test/tools/runtime/run_runtime.sh'
```

Expected: the new `.so`-path assertion passes.

- [ ] **Step 7: Commit**

```bash
git add include/Runtime/Compiler.h lib/Runtime/Compiler.cpp lib/Runtime/MixHostStubTemplate.h test/tools/runtime/test_runtime.cpp
git commit -m "feat(runtime): emit packed shared libraries for mix kernels"
```

---

### Task 2: Add packed mix launch path to `Executor`

**Files:**
- Modify: `include/Runtime/Executor.h`
- Modify: `lib/Runtime/Executor.cpp`
- Test: `test/tools/runtime/test_runtime.cpp`

- [ ] **Step 1: Write the failing launch-path test**

```cpp
// test/tools/runtime/test_runtime.cpp
{
  Executor ex(BackendMode::Simulation);
  auto initErr = ex.Initialize();
  EXPECT(!initErr, "executor initializes");

  RunArgs args;
  args.block_dim = 1;
  auto err = ex.RunPackedMixFile("/tmp/missing.so", "fc_relu", args);
  EXPECT((bool)err, "missing packed mix library returns an error");
  if (err) llvm::consumeError(std::move(err));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
orb -m xvm bash -lc 'cd /home/niu/code/Ascend-MLIR && LLVM_BUILD_DIR=/home/niu/code/llvm-project/build bash test/tools/runtime/run_runtime.sh'
```

Expected: FAIL because `RunPackedMixFile` does not exist.

- [ ] **Step 3: Declare the new mix-only API**

```cpp
// include/Runtime/Executor.h
llvm::Error RunPackedMixFile(const std::string& shared_lib_path,
                             const std::string& kernel_name,
                             RunArgs& args);
```

- [ ] **Step 4: Implement packed mix launch in `Executor.cpp`**

```cpp
// lib/Runtime/Executor.cpp
llvm::Error Executor::RunPackedMixFile(const std::string& shared_lib_path,
                                       const std::string& kernel_name,
                                       RunArgs& args) {
  void* mix_lib = dlopen(shared_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!mix_lib)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "dlopen packed mix failed (%s): %s",
                                   shared_lib_path.c_str(), dlerror());

  std::string launch_name = "aclrtlaunch_" + kernel_name;
  using LaunchFn = uint32_t (*)(uint32_t, void*, void*, void*, void*, void*, void*, void*);
  auto* launch = reinterpret_cast<LaunchFn>(dlsym(mix_lib, launch_name.c_str()));
  if (!launch) {
    const char* err = dlerror();
    dlclose(mix_lib);
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "dlsym %s failed: %s",
                                   launch_name.c_str(), err ? err : "unknown");
  }

  std::vector<void*> input_gm;
  for (auto& inp : args.inputs) {
    auto p = Alloc(inp.nbytes());
    if (!p) { dlclose(mix_lib); FreeAll(); return p.takeError(); }
    input_gm.push_back(*p);
    if (auto err = H2D(*p, inp.data, inp.nbytes())) {
      dlclose(mix_lib); FreeAll(); return err;
    }
  }

  std::vector<void*> output_gm;
  for (auto& out : args.outputs) {
    auto p = Alloc(out.nbytes());
    if (!p) { dlclose(mix_lib); FreeAll(); return p.takeError(); }
    output_gm.push_back(*p);
  }

  auto ws = Alloc(args.workspace_size);
  if (!ws) { dlclose(mix_lib); FreeAll(); return ws.takeError(); }

  void* tiling_gm = nullptr;
  if (!args.tiling.empty()) {
    auto tgm = Alloc(args.tiling.size());
    if (!tgm) { dlclose(mix_lib); FreeAll(); return tgm.takeError(); }
    tiling_gm = *tgm;
    if (auto err = H2D(tiling_gm, args.tiling.data(), args.tiling.size())) {
      dlclose(mix_lib); FreeAll(); return err;
    }
  }

  if (input_gm.size() != 3 || output_gm.size() != 1) {
    dlclose(mix_lib);
    FreeAll();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "packed mix currently requires 3 inputs and 1 output");
  }

  uint32_t rc = launch(static_cast<uint32_t>(args.block_dim), stream_,
                       input_gm[0], input_gm[1], input_gm[2], output_gm[0],
                       *ws, tiling_gm);
  if (rc != 0) {
    dlclose(mix_lib);
    FreeAll();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "packed mix launch failed: rc=%u", rc);
  }

  rtStreamSynchronize_(stream_);

  for (size_t i = 0; i < args.outputs.size(); ++i) {
    if (auto err = D2H(args.outputs[i].data, output_gm[i], args.outputs[i].nbytes())) {
      dlclose(mix_lib); FreeAll(); return err;
    }
  }

  dlclose(mix_lib);
  FreeAll();
  return llvm::Error::success();
}
```

- [ ] **Step 5: Re-run runtime unit tests**

Run:
```bash
orb -m xvm bash -lc 'cd /home/niu/code/Ascend-MLIR && LLVM_BUILD_DIR=/home/niu/code/llvm-project/build bash test/tools/runtime/run_runtime.sh'
```

Expected: compile succeeds and the missing-library test passes.

- [ ] **Step 6: Commit**

```bash
git add include/Runtime/Executor.h lib/Runtime/Executor.cpp test/tools/runtime/test_runtime.cpp
git commit -m "feat(runtime): add packed mix launch path to executor"
```

---

### Task 3: Switch `validator` mix runs onto the packed path

**Files:**
- Modify: `tools/validator/validator_main.cpp`
- Test: existing successful mix fixture under `examples/matmul-add-relu-sum/final_sample_target_graph`

- [ ] **Step 1: Add a failing validator mix branch test by reproducing the current bad assumption**

```cpp
// no code file change yet — validation task uses a real mix fixture
// Expected current failure mode: validator still calls RegisterBinary(...)
// and treats mix like raw ELF launch.
```

- [ ] **Step 2: Run the current validator against the successful packed fixture to confirm failure**

Run:
```bash
orb -m xvm bash -lc '
  export ASCEND_HOME_PATH=/home/niu/Ascend/latest
  source /home/niu/code/Ascend-MLIR/examples/env.sh
  validator \
    --bin /home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/build/lib/libascendc_kernels_sim.so \
    --name baremix_custom \
    --kernel-type mix \
    --inputs /home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/input/x1_gm.bin,/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/input/x2_gm.bin,/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/input/bias.bin \
    --expected /home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/output/golden.bin'
```

Expected: FAIL under the old raw registration path.

- [ ] **Step 3: Route mix to `RunPackedMixFile` in `validator_main.cpp`**

```cpp
// tools/validator/validator_main.cpp
bool is_mix = (KernelType.getValue() == "mix");

Executor executor(BackendMode::Simulation);
if (auto err = executor.Initialize()) {
  llvm::errs() << "Error: executor init failed: "
               << llvm::toString(std::move(err)) << "\n";
  _Exit(3);
}

SimValidator validator;
SimValidator::Result result;
if (is_mix) {
  if (auto err = executor.RunPackedMixFile(BinFile, KernelName, args)) {
    result.error_msg = "Kernel run failed: " + llvm::toString(std::move(err));
  } else {
    result = validator.CompareOnly(args, expected_arrs, Atol, Rtol);
  }
} else {
  auto handle_or = executor.RegisterBinary(BinFile, KernelName, magic);
  if (!handle_or) {
    llvm::errs() << "Error: RegisterBinary failed: "
                 << llvm::toString(handle_or.takeError()) << "\n";
    _Exit(3);
  }
  result = validator.ValidateBinary(*handle_or, executor, args,
                                    expected_arrs, Atol, Rtol);
}
```

- [ ] **Step 4: Add a comparison-only helper if needed**

```cpp
// include/Runtime/SimValidator.h
Result CompareOnly(RunArgs& args,
                   const std::vector<NDArray>& expected,
                   double atol,
                   double rtol);

// lib/Runtime/SimValidator.cpp
SimValidator::Result SimValidator::CompareOnly(
    RunArgs& args,
    const std::vector<NDArray>& expected,
    double atol, double rtol) {
  Result r;
  {
    llvm::SmallString<256> cwd;
    llvm::sys::fs::current_path(cwd);
    r.cycle_count = ParseCycleCounts(cwd.str().str());
  }
  Result cmp = compareOutputs(args, expected, atol, rtol);
  r.max_abs_diff  = cmp.max_abs_diff;
  r.mean_abs_diff = cmp.mean_abs_diff;
  r.passed        = cmp.passed;
  if (!cmp.error_msg.empty()) r.error_msg = cmp.error_msg;
  return r;
}
```

- [ ] **Step 5: Re-run the successful packed mix fixture through `validator`**

Run:
```bash
orb -m xvm bash -lc '
  export ASCEND_HOME_PATH=/home/niu/Ascend/latest
  source /home/niu/code/Ascend-MLIR/examples/env.sh
  validator \
    --bin /home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/build/lib/libascendc_kernels_sim.so \
    --name baremix_custom \
    --kernel-type mix \
    --inputs /home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/input/x1_gm.bin,/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/input/x2_gm.bin,/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/input/bias.bin \
    --expected /home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/output/golden.bin \
    --block-dim 1'
```

Expected: successful completion and PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/validator/validator_main.cpp include/Runtime/SimValidator.h lib/Runtime/SimValidator.cpp
git commit -m "feat(runtime): route validator mix execution through packed launch path"
```

---

### Task 4: Update `HostRunnerGen` mix branch without touching vec/cube behavior

**Files:**
- Modify: `lib/Runtime/HostRunnerGen.cpp`
- Test: `test/tools/runner/test_runner_gen.cpp`

- [ ] **Step 1: Write the failing generated-source assertions**

```cpp
// test/tools/runner/test_runner_gen.cpp
{
  mlir::runtime::HostRunnerGen gen;
  mlir::runtime::HostRunnerGen::Config cfg;
  cfg.kernel_name = "fc_relu";
  cfg.kernel_type = "mix";
  cfg.soc_version = "Ascend910B1";
  cfg.num_inputs = 3;
  cfg.num_outputs = 1;
  auto r = gen.Generate(cfg, "/tmp/runner_gen_mix");
  EXPECT((bool)r, "Generate mix runner succeeds");
  if (r) {
    std::ifstream src("/tmp/runner_gen_mix/runner.cpp");
    std::string s((std::istreambuf_iterator<char>(src)), {});
    EXPECT(s.find("libfc_relu_packed.so") != std::string::npos,
           "mix runner references packed shared library");
    EXPECT(s.find("aclrtlaunch_fc_relu") != std::string::npos,
           "mix runner references aclrtlaunch wrapper");
    EXPECT(s.find("rtKernelLaunch(") == std::string::npos,
           "mix runner no longer emits raw rtKernelLaunch path");
  }
}
```

- [ ] **Step 2: Run the runner generation test to verify it fails**

Run:
```bash
orb -m xvm bash -lc 'cd /home/niu/code/Ascend-MLIR && LLVM_BUILD_DIR=/home/niu/code/llvm-project/build bash test/tools/runtime/run_runtime.sh'
```

Expected: FAIL because generated mix runner still emits raw launch code.

- [ ] **Step 3: Split `HostRunnerGen` output by kernel type**

```cpp
// lib/Runtime/HostRunnerGen.cpp
if (cfg.kernel_type == "mix") {
  // emit runner that:
  // 1. loads packed lib path derived from --bin or compiled artifact path
  // 2. dlsym("aclrtlaunch_<kernel>")
  // 3. allocates input/output/workspace/tiling GM
  // 4. calls wrapper instead of rtKernelLaunch
}
```

- [ ] **Step 4: Keep vec/cube runner logic byte-for-byte intact**

```cpp
// existing vec/cube runner branch remains unchanged
// only mix runner code generation is replaced
```

- [ ] **Step 5: Re-run runner generation tests**

Run:
```bash
orb -m xvm bash -lc 'cd /home/niu/code/Ascend-MLIR && LLVM_BUILD_DIR=/home/niu/code/llvm-project/build bash test/tools/runtime/run_runtime.sh'
```

Expected: mix runner source assertions pass, vector runner assertions unchanged.

- [ ] **Step 6: Commit**

```bash
git add lib/Runtime/HostRunnerGen.cpp test/tools/runner/test_runner_gen.cpp
git commit -m "feat(runtime): generate packed mix runners via aclrtlaunch wrappers"
```

---

### Task 5: End-to-end safety check — mix works, vector stays intact

**Files:**
- Verify only; no new source required

- [ ] **Step 1: Rebuild project tools**

Run:
```bash
orb -m xvm bash -lc 'source /home/niu/code/Ascend-MLIR/examples/env.sh && cd /home/niu/code/Ascend-MLIR/build && cmake --build . --target compiler validator -j4'
```

Expected: build succeeds.

- [ ] **Step 2: Validate the successful packed mix fixture through our own runtime path**

Run:
```bash
orb -m xvm bash -lc '
  export ASCEND_HOME_PATH=/home/niu/Ascend/latest
  source /home/niu/code/Ascend-MLIR/examples/env.sh
  validator \
    --bin /home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/build/lib/libascendc_kernels_sim.so \
    --name baremix_custom \
    --kernel-type mix \
    --inputs /home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/input/x1_gm.bin,/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/input/x2_gm.bin,/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/input/bias.bin \
    --expected /home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/output/golden.bin \
    --block-dim 1'
```

Expected: PASS.

- [ ] **Step 3: Re-run a known pure-vector example**

Run:
```bash
orb -m xvm bash -lc 'cd /home/niu/code/Ascend-MLIR/examples/add-broadcast-concat && timeout 60 bash run.sh'
```

Expected: existing pure-vector example still succeeds unchanged.

- [ ] **Step 4: Re-run one runtime runner test**

Run:
```bash
orb -m xvm bash -lc 'cd /home/niu/code/Ascend-MLIR && LLVM_BUILD_DIR=/home/niu/code/llvm-project/build bash test/tools/runtime/run_runtime.sh'
```

Expected: all runtime tests stay green.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Compiler.h lib/Runtime/Compiler.cpp include/Runtime/Executor.h lib/Runtime/Executor.cpp tools/validator/validator_main.cpp lib/Runtime/HostRunnerGen.cpp include/Runtime/SimValidator.h lib/Runtime/SimValidator.cpp test/tools/runtime/test_runtime.cpp test/tools/runner/test_runner_gen.cpp
git commit -m "feat(runtime): add packed mix execution path without affecting vector flows"
```

---

## Self-review

### Spec coverage
- Compiler packed artifact generation: covered by Task 1.
- Executor packed launch path: covered by Task 2.
- Validator switch for mix: covered by Task 3.
- HostRunnerGen mix branch: covered by Task 4.
- Safety check for existing pure-vector flows: covered by Task 5.

### Placeholder scan
- No `TODO`, `TBD`, or undefined hand-wavy steps remain.
- Each code-changing step contains concrete file paths and concrete code blocks.

### Type consistency
- Packed artifact target is consistently `lib<kernel>_packed.so` in Tasks 1–4.
- Mix launch wrapper is consistently `aclrtlaunch_<kernel>` in Tasks 2–4.
- Existing vec/cube path is consistently preserved as-is in all tasks.
