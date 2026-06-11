# Network Runner: Mixed AscendC + aclnn on CPU Sim — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End-to-end runner that takes a mixed `network.mlir` (AscendC `@kernel_groupN` + aclnn `@__aclnn_xxx` calls), per-kernel autotunes the AscendC parts, generates a single `network_host.cpp` binary that runs the whole thing on the CANN CPU sim (camodel) without NPU hardware, and verifies the final output against `expected.npy`. v1 ships one example: `examples/mixed-attn-e2e` (kernel_group → flash_attention → kernel_group).

**Architecture:** Spec at `docs/superpowers/specs/2026-05-13-network-runner-mixed-cpu-sim-design.md` (commit `95ee9f9`). Five-phase Python orchestrator (`python/network_runner.py`); generated host C++ from `aclnn-backend` calls AscendC kernels via `rt*` (camodel) and aclnn ops via `AclnnOps.cpp`'s existing `g_host_mode` C++-reference branch (no `aclInit` dependency). Per-kernel best tilings discovered offline via the existing `autotuner` + `runtime-session` against intermediates dumped from a default-tilings build.

**Tech Stack:** MLIR/LLVM (C++17), CMake, lit, Python 3 (numpy), CANN 9.0.0 sim libs (`libruntime_camodel.so`), existing tools `afir-opt`, `afir-translate`, `runtime-session`, `autotuner`, `aclnn-backend`.

---

## File Map

**New:**
- `lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.h` — shared header (declares `emitNetworkJson(func, llvm::raw_ostream&)`).
- `lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.cpp` — walks a network `func::FuncOp` body, classifies each `call` by callee `aclnn.op` attr, writes `network.json`.
- `lib/Dialect/AFIR/Transforms/EmitNetworkJsonPass.cpp` — standalone `--emit-network-json=path=...` pass on a module.
- `python/network_runner.py` — 5-phase orchestrator.
- `python/runner_utils/__init__.py`, `python/runner_utils/network_json.py`, `python/runner_utils/manifest.py`, `python/runner_utils/run_subprocess.py` — small helpers used by the runner.
- `examples/mixed-attn-e2e/network.mlir` — handwritten mixed module (kernel_group → flash_attention → kernel_group).
- `examples/mixed-attn-e2e/kernel_group0.mlir` — handwritten linalg pre-norm.
- `examples/mixed-attn-e2e/kernel_group1.mlir` — handwritten linalg post-proj.
- `examples/mixed-attn-e2e/gen_inputs.py` — numpy: writes inputs + `expected.npy`.
- `examples/mixed-attn-e2e/run.sh` — thin driver invoking `network_runner.py`.
- `test/Conversion/Group/group-outline/network-json.mlir` — lit (mode 1: through group-outline).
- `test/Dialect/AFIR/Transforms/emit-network-json-mixed.mlir` — lit (mode 2: standalone, mixed callees).
- `test/Conversion/AclnnBackend/host-mode-fallback.mlir` — lit (generated host has aclInit-failure host-mode branch, not `exit(1)`).
- `test/Conversion/AclnnBackend/ascendc-launch.mlir` — lit (generated host has `rt*`-based launch sequence, not `// TODO: launch`).

**Modify:**
- `lib/Conversion/VectorPlan/GroupOutline/GroupOutlinePass.cpp:307–320` — call shared emitter when writing `output-dir`.
- `lib/Conversion/VectorPlan/GroupOutline/CMakeLists.txt` — add `NetworkJsonEmitter.cpp`.
- `include/Dialect/AFIR/Transforms/Passes.td` — add `AFIREmitNetworkJsonPass` def.
- `include/Dialect/AFIR/Transforms/Passes.h` — add `createAFIREmitNetworkJsonPass()` decl.
- `lib/Dialect/AFIR/Transforms/CMakeLists.txt` — add `EmitNetworkJsonPass.cpp` + `LINK_LIBS PUBLIC MLIRVectorPlanGroupOutline` (for the shared emitter helper).
- `include/Runtime/AclnnBackend.h` — extend `AclnnBackendConfig`: `std::string tilingsPath; std::string kernelBinariesDir;`.
- `tools/aclnn-backend/aclnn-backend.cpp` — add `--tilings`, `--kernel-binaries` CLI flags; thread to config.
- `lib/Runtime/AclnnBackend/AclnnBackend.cpp` — (1) emit aclInit-failure host-mode branch in generated `network` entry; (2) replace `// TODO: launch ...` with `rt*`-based launch using tilings + kernel binary path; (3) emit `--dump-intermediates DIR` runtime flag plumbing.

**Touched (test infra only):**
- `test/lit.cfg.py` (no edits expected; new tests should be picked up automatically).

---

## Required Reading Before Starting

The plan-executor must skim these three files end-to-end before Task 5:
1. `lib/Runtime/AclnnBackend/AclnnBackend.cpp` (full, 266 lines) — current emitter structure.
2. `lib/Runtime/Execution/NativeExecutionRunner.cpp:20–290` — the working `rt*`-launch sequence we are porting into the generated host.
3. `examples/relu-e2e/run.sh` and `examples/aclnn-attn-e2e/run.sh` — both end-to-end conventions; generated host gets g++-linked similarly.

---

## Build Conventions

All builds use the existing `build/` directory. After every C++ change:
```
source /home/gser/Ascend/ascend-toolkit/set_env.sh
cd /home/gser/code/Ascend-MLIR
ninja -C build
```

If ninja errors with `Could not find MLIRConfig.cmake`, the project was misconfigured with a relative `LLVM_BUILD_DIR`. Reconfigure with absolute paths:
```
cmake -B build -G Ninja -S . \
  -DLLVM_BUILD_DIR=/home/gser/code/Ascend-MLIR/externals/llvm-project/build \
  -DMLIR_DIR=/home/gser/code/Ascend-MLIR/externals/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/home/gser/code/Ascend-MLIR/externals/llvm-project/build/lib/cmake/llvm
```

Lit runs:
```
build/bin/llvm-lit -v test/Conversion/Group/group-outline/network-json.mlir
```
or a directory:
```
build/bin/llvm-lit -v test/Conversion/Group/group-outline/
```

---

## Task 1: `NetworkJsonEmitter` — the shared helper

**Files:**
- Create: `lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.h`
- Create: `lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.cpp`
- Modify: `lib/Conversion/VectorPlan/GroupOutline/CMakeLists.txt`
- Test: `test/Conversion/Group/group-outline/network-json.mlir`

The helper takes a `func::FuncOp` (the network coordinator) + its `ModuleOp`, walks the body's `call` ops in source order, and writes `network.json` per the spec §3.1 schema.

Body ops other than `func.call` and `func.return` are tolerated only if they're `tensor.cast` (alias, threads name through) — anything else fails with an error. v1 networks have only calls + return. (`tensor.empty`/`expand`/`collapse` glue is out of scope per spec §5.)

- [ ] **Step 1: Write the failing lit test**

Create `test/Conversion/Group/group-outline/network-json.mlir`:

```mlir
// RUN: rm -rf %t && mkdir -p %t
// RUN: afir-opt --vector-plan-group-analysis '--vector-plan-group-outline=output-dir=%t' %s
// RUN: FileCheck %s < %t/network.json

#map = affine_map<(d0) -> (d0)>

func.func @two(%a: tensor<8xf16>, %b: tensor<8xf16>,
                %c: tensor<8xf16>, %d: tensor<8xf16>,
                %i0: tensor<8xf16>, %i1: tensor<8xf16>)
    -> (tensor<8xf16>, tensor<8xf16>) {
  %x = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<8xf16>, tensor<8xf16>) outs(%i0 : tensor<8xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.addf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>
  %y = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel"]}
       ins(%c, %d : tensor<8xf16>, tensor<8xf16>) outs(%i1 : tensor<8xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.mulf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>
  return %x, %y : tensor<8xf16>, tensor<8xf16>
}

// CHECK:      "function": "two"
// CHECK:      "kernels":
// CHECK:        "id": "kernel_group0"
// CHECK:        "kind": "ascendc"
// CHECK:        "file": "kernel_group0.mlir"
// CHECK:        "id": "kernel_group1"
// CHECK:        "kind": "ascendc"
// CHECK:      "outputs":
// CHECK:        "from": "kernel"
// CHECK:        "kernel": "kernel_group0"
// CHECK:        "from": "kernel"
// CHECK:        "kernel": "kernel_group1"
```

- [ ] **Step 2: Run lit, expect FAIL**

```
build/bin/llvm-lit -v test/Conversion/Group/group-outline/network-json.mlir
```
Expected: FileCheck fails (no `network.json` produced; or produced as empty).

- [ ] **Step 3: Create the helper header**

Write `lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.h`:

```cpp
//===- NetworkJsonEmitter.h - Emit network.json for a coordinator func ----===//
//
// Walks a network func body's call ops, classifies each callee as either an
// AscendC kernel group or an aclnn op (presence of "aclnn.op" attr), and writes
// network.json per docs/superpowers/specs/2026-05-13-network-runner-mixed-cpu-sim-design.md §3.1.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::vector_plan {

// Writes JSON describing the network coordinator's call graph to `os`.
// `coord` is the public (non-private) func; `module` is its enclosing module
// (used to look up callees). Returns an error if the body contains an
// unsupported op other than func.call / func.return / tensor.cast.
llvm::Error emitNetworkJson(mlir::ModuleOp module,
                            mlir::func::FuncOp coord,
                            llvm::raw_ostream &os);

} // namespace mlir::vector_plan
```

- [ ] **Step 4: Create the helper implementation**

Write `lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.cpp`:

```cpp
//===- NetworkJsonEmitter.cpp ---------------------------------------------===//
#include "NetworkJsonEmitter.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"

using namespace mlir;

namespace mlir::vector_plan {

namespace {

// dtype → JSON-friendly short name (matches RunManifest.cpp's parser).
static llvm::StringRef dtypeName(Type t) {
  if (t.isF16())  return "f16";
  if (t.isBF16()) return "bf16";
  if (t.isF32())  return "f32";
  if (t.isInteger(8))  return "int8";
  if (t.isInteger(32)) return "int32";
  if (t.isInteger(64)) return "int64";
  return "unknown";
}

static llvm::json::Array shapeArray(RankedTensorType t) {
  llvm::json::Array a;
  for (int64_t d : t.getShape())
    a.push_back(static_cast<int64_t>(d));
  return a;
}

static llvm::json::Object tensorEntry(StringRef name, RankedTensorType t) {
  llvm::json::Object obj;
  obj["name"] = name.str();
  obj["shape"] = shapeArray(t);
  obj["dtype"] = dtypeName(t.getElementType()).str();
  return obj;
}

} // namespace

llvm::Error emitNetworkJson(ModuleOp module, func::FuncOp coord,
                            llvm::raw_ostream &os) {
  llvm::json::Object root;
  root["function"] = coord.getSymName().str();

  // 1) inputs
  llvm::json::Array inputs;
  llvm::DenseMap<Value, std::string> argName;
  for (auto [i, arg] : llvm::enumerate(coord.getArguments())) {
    auto t = dyn_cast<RankedTensorType>(arg.getType());
    if (!t)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "network arg %u is not a ranked tensor",
                                     unsigned(i));
    std::string name = "arg" + std::to_string(i);
    argName[arg] = name;
    inputs.push_back(tensorEntry(name, t));
  }
  root["inputs"] = std::move(inputs);

  // 2) walk body, build kernels[] and a (Value → "from" descriptor) map
  llvm::json::Array kernels;
  // valueSource[v] = JSON object { "from": "input"|"kernel", ... }
  llvm::DenseMap<Value, llvm::json::Object> valueSource;
  for (auto [v, n] : argName) {
    llvm::json::Object o;
    o["from"] = "input";
    o["name"] = n;
    valueSource[v] = std::move(o);
  }

  for (Operation &op : coord.front()) {
    if (auto castOp = dyn_cast<tensor::CastOp>(op)) {
      // alias: thread the source's descriptor onto the cast result
      auto it = valueSource.find(castOp.getSource());
      if (it == valueSource.end())
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "tensor.cast source has no descriptor");
      llvm::json::Object copy;
      for (auto &kv : it->second)
        copy[kv.first] = kv.second;
      valueSource[castOp.getResult()] = std::move(copy);
      continue;
    }
    if (isa<func::ReturnOp>(op))
      continue;
    auto callOp = dyn_cast<func::CallOp>(op);
    if (!callOp)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "unsupported op in network body: %s",
                                     op.getName().getStringRef().str().c_str());

    auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (!callee)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "callee not found: %s",
                                     callOp.getCallee().str().c_str());

    llvm::json::Object kernel;
    kernel["id"] = callOp.getCallee().str();

    if (auto opAttr = callee->getAttrOfType<StringAttr>("aclnn.op")) {
      kernel["kind"] = "aclnn";
      kernel["op"] = opAttr.getValue().str();
      if (auto layoutAttr = callee->getAttrOfType<StringAttr>("aclnn.layout"))
        kernel["layout"] = layoutAttr.getValue().str();
    } else {
      kernel["kind"] = "ascendc";
      kernel["file"] = (callOp.getCallee().str() + ".mlir");
    }

    // args
    llvm::json::Array args;
    for (Value arg : callOp.getOperands()) {
      auto it = valueSource.find(arg);
      if (it == valueSource.end())
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "call arg has no descriptor");
      llvm::json::Object copy;
      for (auto &kv : it->second)
        copy[kv.first] = kv.second;
      args.push_back(std::move(copy));
    }
    kernel["args"] = std::move(args);

    // results: register source descriptors for downstream consumers
    llvm::json::Array results;
    for (auto [ri, res] : llvm::enumerate(callOp.getResults())) {
      auto t = dyn_cast<RankedTensorType>(res.getType());
      if (!t)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "call result is not a ranked tensor");
      std::string name = callOp.getCallee().str() + "_r" + std::to_string(ri);
      results.push_back(tensorEntry(name, t));

      llvm::json::Object src;
      src["from"] = "kernel";
      src["kernel"] = callOp.getCallee().str();
      src["result"] = static_cast<int64_t>(ri);
      valueSource[res] = std::move(src);
    }
    kernel["results"] = std::move(results);

    kernels.push_back(std::move(kernel));
  }
  root["kernels"] = std::move(kernels);

  // 3) outputs from the func's return
  llvm::json::Array outputs;
  auto retOp = cast<func::ReturnOp>(coord.front().getTerminator());
  for (auto [oi, v] : llvm::enumerate(retOp.getOperands())) {
    auto it = valueSource.find(v);
    if (it == valueSource.end())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "return operand has no descriptor");
    llvm::json::Object o;
    o["name"] = "out" + std::to_string(oi);
    for (auto &kv : it->second)
      o[kv.first] = kv.second;
    outputs.push_back(std::move(o));
  }
  root["outputs"] = std::move(outputs);

  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
  return llvm::Error::success();
}

} // namespace mlir::vector_plan
```

- [ ] **Step 5: Wire it into `GroupOutlinePass`**

Open `lib/Conversion/VectorPlan/GroupOutline/GroupOutlinePass.cpp`. Find the file-emit section near line 307 (`std::string netFile = (outputDir + "/network.mlir").str();`). After writing `network.mlir`, write `network.json` next to it.

Add at the top of the file:
```cpp
#include "NetworkJsonEmitter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
```

After the `network.mlir` write site (search the function `emitFiles` for where it writes `netFile`), add:

```cpp
  // Also emit network.json describing the coordinator call graph.
  std::string jsonFile = (outputDir + "/network.json").str();
  std::error_code jec;
  llvm::raw_fd_ostream jsonOs(jsonFile, jec);
  if (jec)
    return moduleOp.emitError("cannot open network.json: ") << jec.message();
  func::FuncOp coord;
  netModule.walk([&](func::FuncOp f) {
    if (!f.isPrivate()) {
      coord = f;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (coord)
    if (auto err = mlir::vector_plan::emitNetworkJson(netModule, coord, jsonOs))
      return moduleOp.emitError("emitNetworkJson: ") << llvm::toString(std::move(err));
```

(Adjust `netModule` / `moduleOp` to whatever the surrounding code uses; read 30 lines of context before editing to match local naming.)

- [ ] **Step 6: Add helper to CMakeLists**

Edit `lib/Conversion/VectorPlan/GroupOutline/CMakeLists.txt`. Find the `add_mlir_library(...)` for the GroupOutline target and add `NetworkJsonEmitter.cpp` to its source list. Make sure its `LINK_LIBS` includes `MLIRTensorDialect MLIRSupport` (for `llvm::json`).

- [ ] **Step 7: Build**

```
ninja -C build
```
Expected: clean build.

- [ ] **Step 8: Run lit, expect PASS**

```
build/bin/llvm-lit -v test/Conversion/Group/group-outline/network-json.mlir
```
Expected: PASS. Inspect the produced JSON manually:
```
cat /tmp/lit-tmp-*/Output/network-json.mlir.tmp/network.json
```

- [ ] **Step 9: Sanity-check on twochain example from session memory**

```
cat > /tmp/twochain.mlir <<'EOF'
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @twochain(%a: tensor<128x64xf16>, %b: tensor<128x64xf16>,
                    %c: tensor<128x64xf16>, %d: tensor<128x64xf16>,
                    %i0: tensor<128x64xf16>, %i1: tensor<128x64xf16>)
    -> (tensor<128x64xf16>, tensor<128x64xf16>) {
  %x = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel","parallel"]}
       ins(%a, %b : tensor<128x64xf16>, tensor<128x64xf16>) outs(%i0 : tensor<128x64xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16): %v = arith.addf %p, %q : f16; linalg.yield %v : f16
  } -> tensor<128x64xf16>
  %y = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel","parallel"]}
       ins(%c, %d : tensor<128x64xf16>, tensor<128x64xf16>) outs(%i1 : tensor<128x64xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16): %v = arith.mulf %p, %q : f16; linalg.yield %v : f16
  } -> tensor<128x64xf16>
  return %x, %y : tensor<128x64xf16>, tensor<128x64xf16>
}
EOF
rm -rf /tmp/twg && mkdir /tmp/twg
build/bin/afir-opt --vector-plan-group-analysis '--vector-plan-group-outline=output-dir=/tmp/twg' /tmp/twochain.mlir
cat /tmp/twg/network.json
```
Expected: well-formed JSON with two `ascendc` kernels and two outputs.

- [ ] **Step 10: Commit**

```
git add lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.h \
        lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.cpp \
        lib/Conversion/VectorPlan/GroupOutline/GroupOutlinePass.cpp \
        lib/Conversion/VectorPlan/GroupOutline/CMakeLists.txt \
        test/Conversion/Group/group-outline/network-json.mlir
git commit -m "feat(vector-plan): emit network.json alongside outlined .mlir files

NetworkJsonEmitter walks the coordinator func body, classifies callees by
the presence of an aclnn.op attr (kind: ascendc | aclnn), and writes the
spec's network.json schema. GroupOutlinePass invokes it when
output-dir is set. lit: test/Conversion/Group/group-outline/network-json.mlir.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Standalone `--emit-network-json` pass (mode 2)

For hand-written `network.mlir` (the v1 mixed-attn case) the user invokes a standalone
pass: `afir-opt network.mlir --emit-network-json=path=...`.

**Files:**
- Create: `lib/Dialect/AFIR/Transforms/EmitNetworkJsonPass.cpp`
- Modify: `include/Dialect/AFIR/Transforms/Passes.td`
- Modify: `include/Dialect/AFIR/Transforms/Passes.h`
- Modify: `lib/Dialect/AFIR/Transforms/CMakeLists.txt`
- Test: `test/Dialect/AFIR/Transforms/emit-network-json-mixed.mlir`

- [ ] **Step 1: Write the failing lit test**

Create `test/Dialect/AFIR/Transforms/emit-network-json-mixed.mlir`:

```mlir
// RUN: rm -f %t.json && afir-opt %s --emit-network-json=path=%t.json
// RUN: FileCheck %s < %t.json

module {
  func.func private @kernel_group0(tensor<8xf16>) -> tensor<8xf16>
  func.func private @__aclnn_softmax(tensor<?xf16>) -> tensor<?xf16>
      attributes {aclnn.op = "Softmax"}
  func.func private @kernel_group1(tensor<8xf16>) -> tensor<8xf16>

  func.func @model(%x: tensor<8xf16>) -> tensor<8xf16> {
    %a = call @kernel_group0(%x) : (tensor<8xf16>) -> tensor<8xf16>
    %ac = tensor.cast %a : tensor<8xf16> to tensor<?xf16>
    %s  = call @__aclnn_softmax(%ac) : (tensor<?xf16>) -> tensor<?xf16>
    %sc = tensor.cast %s : tensor<?xf16> to tensor<8xf16>
    %y  = call @kernel_group1(%sc) : (tensor<8xf16>) -> tensor<8xf16>
    return %y : tensor<8xf16>
  }
}

// CHECK:       "function": "model"
// CHECK:       "kernel_group0"
// CHECK:       "ascendc"
// CHECK:       "__aclnn_softmax"
// CHECK:       "aclnn"
// CHECK:       "op": "Softmax"
// CHECK:       "kernel_group1"
// CHECK-DAG:   "from": "kernel"
// CHECK-DAG:   "kernel": "__aclnn_softmax"
```

- [ ] **Step 2: Run lit, expect FAIL**

```
build/bin/llvm-lit -v test/Dialect/AFIR/Transforms/emit-network-json-mixed.mlir
```
Expected: FAIL — pass not registered.

- [ ] **Step 3: Add pass declaration in `Passes.td`**

Edit `include/Dialect/AFIR/Transforms/Passes.td`. Append:

```tablegen
def AFIREmitNetworkJsonPass : Pass<"emit-network-json", "ModuleOp"> {
  let summary = "Emit network.json for the coordinator func in the module";
  let description = [{
    Walks the (non-private) coordinator func's body, classifies each call by
    callee `aclnn.op` attribute, and writes network.json (per the network-runner
    design spec) to the file given by `path`.
  }];
  let constructor = "mlir::createAFIREmitNetworkJsonPass()";
  let options = [
    Option<"path", "path", "std::string", /*default=*/"\"\"",
           "Output path for network.json (required)">,
  ];
}
```

- [ ] **Step 4: Add the create-fn declaration in `Passes.h`**

Edit `include/Dialect/AFIR/Transforms/Passes.h`. In the existing block of `createAFIRXxxPass()` declarations, add:

```cpp
std::unique_ptr<mlir::Pass> createAFIREmitNetworkJsonPass();
```

- [ ] **Step 5: Implement the pass**

Write `lib/Dialect/AFIR/Transforms/EmitNetworkJsonPass.cpp`:

```cpp
//===- EmitNetworkJsonPass.cpp --------------------------------------------===//
#include "Dialect/AFIR/Transforms/Passes.h"
#include "lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
#define GEN_PASS_DEF_AFIREMITNETWORKJSONPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct EmitNetworkJsonPass
    : public impl::AFIREmitNetworkJsonPassBase<EmitNetworkJsonPass> {
  using Base::Base;
  void runOnOperation() override {
    if (path.empty()) {
      getOperation()->emitError("emit-network-json: --emit-network-json=path=<file> is required");
      return signalPassFailure();
    }
    func::FuncOp coord;
    getOperation().walk([&](func::FuncOp f) {
      if (!f.isPrivate()) {
        coord = f;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!coord) {
      getOperation()->emitError("emit-network-json: no public coordinator func found");
      return signalPassFailure();
    }
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec) {
      getOperation()->emitError("emit-network-json: cannot open ") << path
          << ": " << ec.message();
      return signalPassFailure();
    }
    if (auto err = vector_plan::emitNetworkJson(getOperation(), coord, os)) {
      getOperation()->emitError("emit-network-json: ") << llvm::toString(std::move(err));
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createAFIREmitNetworkJsonPass() {
  return std::make_unique<EmitNetworkJsonPass>();
}
```

- [ ] **Step 6: Register source in CMakeLists**

Edit `lib/Dialect/AFIR/Transforms/CMakeLists.txt`. Find the `add_mlir_library(AFIRTransforms ...)` and add `EmitNetworkJsonPass.cpp` to its source list. Add `MLIRVectorPlanGroupOutline` to its `LINK_LIBS` (the helper lives there).

If the GroupOutline target is not already a library, ensure its `add_mlir_library` exports `NetworkJsonEmitter`'s symbols (it should, per Task 1 step 6 — same target).

- [ ] **Step 7: Build**

```
ninja -C build
```
Expected: clean build.

- [ ] **Step 8: Run lit, expect PASS**

```
build/bin/llvm-lit -v test/Dialect/AFIR/Transforms/emit-network-json-mixed.mlir
```
Expected: PASS.

- [ ] **Step 9: Manual mixed-callee sanity check**

```
cat > /tmp/mixed.mlir <<'EOF'
module {
  func.func private @kernel_group0(tensor<8xf16>) -> tensor<8xf16>
  func.func private @__aclnn_softmax(tensor<?xf16>) -> tensor<?xf16>
      attributes {aclnn.op = "Softmax"}
  func.func @model(%x: tensor<8xf16>) -> tensor<8xf16> {
    %a = call @kernel_group0(%x) : (tensor<8xf16>) -> tensor<8xf16>
    %ac = tensor.cast %a : tensor<8xf16> to tensor<?xf16>
    %s  = call @__aclnn_softmax(%ac) : (tensor<?xf16>) -> tensor<?xf16>
    %sc = tensor.cast %s : tensor<?xf16> to tensor<8xf16>
    return %sc : tensor<8xf16>
  }
}
EOF
build/bin/afir-opt /tmp/mixed.mlir --emit-network-json=path=/tmp/mixed.json
cat /tmp/mixed.json
```
Expected: JSON with one `ascendc` kernel, one `aclnn` kernel (op `Softmax`), and the output's `from: kernel` pointing at `__aclnn_softmax`.

- [ ] **Step 10: Commit**

```
git add lib/Dialect/AFIR/Transforms/EmitNetworkJsonPass.cpp \
        include/Dialect/AFIR/Transforms/Passes.td \
        include/Dialect/AFIR/Transforms/Passes.h \
        lib/Dialect/AFIR/Transforms/CMakeLists.txt \
        test/Dialect/AFIR/Transforms/emit-network-json-mixed.mlir
git commit -m "feat(afir): --emit-network-json pass for hand-written network.mlir

Standalone pass that runs NetworkJsonEmitter on a module's public coordinator
func, used when the user provides a hand-written mixed network.mlir
(kernel_group + aclnn calls) instead of running --vector-plan-group-outline.
Tolerates tensor.cast aliases. lit:
test/Dialect/AFIR/Transforms/emit-network-json-mixed.mlir.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: `aclnn-backend` CLI extensions (`--tilings`, `--kernel-binaries`)

These are config-only changes; no codegen behavior change yet — Tasks 4–6 will use the new fields. Splitting them out keeps each task small.

**Files:**
- Modify: `include/Runtime/AclnnBackend.h`
- Modify: `tools/aclnn-backend/aclnn-backend.cpp`
- Modify: `lib/Runtime/AclnnBackend/AclnnBackend.cpp` (config wiring + a debug echo line for the test)

- [ ] **Step 1: Write the failing test**

Create `test/Conversion/AclnnBackend/cli-tilings.mlir`:
```mlir
// RUN: rm -f %t.cpp && aclnn-backend --input %s --output %t.cpp \
// RUN:   --tilings %S/Inputs/tilings_default.json \
// RUN:   --kernel-binaries %S/Inputs/artifacts
// RUN: FileCheck %s < %t.cpp

// CHECK: // tilings_path: {{.*}}tilings_default.json
// CHECK: // kernel_binaries_dir: {{.*}}artifacts

module {
  func.func @model(%x: tensor<8xf16>) -> tensor<8xf16> {
    return %x : tensor<8xf16>
  }
}
```

Create the placeholder input files:
```
mkdir -p test/Conversion/AclnnBackend/Inputs/artifacts
echo '{}' > test/Conversion/AclnnBackend/Inputs/tilings_default.json
```

- [ ] **Step 2: Run lit, expect FAIL**

```
build/bin/llvm-lit -v test/Conversion/AclnnBackend/cli-tilings.mlir
```
Expected: FAIL (unknown CLI flags).

- [ ] **Step 3: Extend the config struct**

Edit `include/Runtime/AclnnBackend.h`:

```cpp
struct AclnnBackendConfig {
  std::string networkMlirPath;     // input:  path to network.mlir
  std::string outputCppPath;       // output: path to network_host.cpp
  std::string tilingsPath;         // input:  path to tilings JSON (per-kernel best params)
  std::string kernelBinariesDir;   // input:  dir of compiled AscendC kernel binaries
};
```

- [ ] **Step 4: Wire new flags in the CLI tool**

Edit `tools/aclnn-backend/aclnn-backend.cpp`. Below the existing `OutputFile`:

```cpp
static cl::opt<std::string> TilingsFile(
    "tilings", cl::init(""),
    cl::desc("Path to per-kernel tilings JSON (best-config map)"));

static cl::opt<std::string> KernelBinariesDir(
    "kernel-binaries", cl::init(""),
    cl::desc("Directory containing compiled AscendC kernel artifacts"));
```

In `main`, after assigning the existing config fields:
```cpp
  cfg.tilingsPath = TilingsFile;
  cfg.kernelBinariesDir = KernelBinariesDir;
```

- [ ] **Step 5: Echo the new config in generated output as a debug breadcrumb**

Edit `lib/Runtime/AclnnBackend/AclnnBackend.cpp`. Inside `buildNetworkHostCpp`, right after the existing `// Auto-generated by AclnnBackend. DO NOT EDIT.` line, add (taking `cfg` from a new param — see step 6):

```cpp
  if (!cfg.tilingsPath.empty())
    os << "// tilings_path: " << cfg.tilingsPath << "\n";
  if (!cfg.kernelBinariesDir.empty())
    os << "// kernel_binaries_dir: " << cfg.kernelBinariesDir << "\n";
```

- [ ] **Step 6: Thread `cfg` into `buildNetworkHostCpp`**

Change the helper signature in `AclnnBackend.cpp`:
```cpp
static std::string buildNetworkHostCpp(ModuleOp module,
                                       const AclnnBackendConfig &cfg);
```
Update its single caller in `AclnnBackend::generate` to pass `cfg`.

- [ ] **Step 7: Build**

```
ninja -C build
```
Expected: clean build.

- [ ] **Step 8: Run lit, expect PASS**

```
build/bin/llvm-lit -v test/Conversion/AclnnBackend/cli-tilings.mlir
```
Expected: PASS.

- [ ] **Step 9: Commit**

```
git add include/Runtime/AclnnBackend.h tools/aclnn-backend/aclnn-backend.cpp \
        lib/Runtime/AclnnBackend/AclnnBackend.cpp \
        test/Conversion/AclnnBackend/cli-tilings.mlir \
        test/Conversion/AclnnBackend/Inputs/
git commit -m "feat(aclnn-backend): --tilings / --kernel-binaries CLI flags

Config plumbing only. Generated network_host.cpp now carries the paths as
breadcrumb comments; subsequent commits will use them to bake tilings into
kernel launches and locate kernel binaries.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Generated host falls back to host-mode on `aclInit` failure

Today the existing example's `harness.cpp` does `setHostMode(true)` and continues. The
generated `network_host.cpp` from `aclnn-backend` should do the same in its `network`
entry point — so the binary keeps running on the CPU sim instead of needing real HW.

**Files:**
- Modify: `lib/Runtime/AclnnBackend/AclnnBackend.cpp`
- Test: `test/Conversion/AclnnBackend/host-mode-fallback.mlir`

- [ ] **Step 1: Write the failing test**

Create `test/Conversion/AclnnBackend/host-mode-fallback.mlir`:
```mlir
// RUN: rm -f %t.cpp && aclnn-backend --input %s --output %t.cpp
// RUN: FileCheck %s < %t.cpp

module {
  func.func private @__aclnn_softmax(tensor<?xf16>) -> tensor<?xf16>
      attributes {aclnn.op = "Softmax"}
  func.func @model(%x: tensor<?xf16>) -> tensor<?xf16> {
    %s = call @__aclnn_softmax(%x) : (tensor<?xf16>) -> tensor<?xf16>
    return %s : tensor<?xf16>
  }
}

// The generated entry must (a) try aclInit, (b) on failure call setHostMode(true)
// and continue, NOT exit.
// CHECK: extern "C" void network(
// CHECK:   if (aclInit(nullptr) != 0) {
// CHECK:     mlir::runtime::aclnn::setHostMode(true)
// CHECK-NOT: exit(
// CHECK:   network_impl(
```

- [ ] **Step 2: Run lit, expect FAIL**

```
build/bin/llvm-lit -v test/Conversion/AclnnBackend/host-mode-fallback.mlir
```
Expected: FAIL.

- [ ] **Step 3: Edit the entry-point emit**

In `lib/Runtime/AclnnBackend/AclnnBackend.cpp`, find the `// ④ Public entry point` block and replace its body. Final form:

```cpp
  // ④ Public entry point — extern "C" so harness.cpp can declare it without mangling
  os << "extern \"C\" void network(\n";
  os << "    TensorInfo inputs[], int numInputs,\n";
  os << "    TensorInfo outputs[], int numOutputs,\n";
  os << "    aclrtStream stream) {\n";
  os << "  // Try aclInit; if it fails (e.g. running on CPU sim with no NPU\n";
  os << "  // device available), fall through in host-mode so aclnn ops dispatch\n";
  os << "  // to AclnnOps.cpp's CPU-reference implementations.\n";
  os << "  static bool initialized = false;\n";
  os << "  if (!initialized) {\n";
  os << "    initialized = true;\n";
  os << "    if (aclInit(nullptr) != 0) {\n";
  os << "      mlir::runtime::aclnn::setHostMode(true);\n";
  os << "    }\n";
  os << "  }\n";
  os << "  network_impl(inputs, numInputs, outputs, numOutputs, stream);\n";
  os << "}\n";
```

Add the include for `aclInit` declaration to the generated header section (search `// ① File header + includes`):

```cpp
  os << "#include \"acl/acl.h\"\n";
```

- [ ] **Step 4: Build**

```
ninja -C build
```
Expected: clean build.

- [ ] **Step 5: Run lit, expect PASS**

```
build/bin/llvm-lit -v test/Conversion/AclnnBackend/host-mode-fallback.mlir
```
Expected: PASS.

- [ ] **Step 6: Manually re-generate and inspect**

```
build/bin/aclnn-backend --input examples/aclnn-attn-e2e/network.mlir \
  --output /tmp/net_host.cpp
grep -A 12 'extern "C" void network' /tmp/net_host.cpp
```
Expected: visible `if (aclInit(nullptr) != 0) { ... setHostMode(true); }` block; NOT a call to `exit(...)`.

- [ ] **Step 7: Commit**

```
git add lib/Runtime/AclnnBackend/AclnnBackend.cpp \
        test/Conversion/AclnnBackend/host-mode-fallback.mlir
git commit -m "feat(aclnn-backend): generated host falls back to host-mode on aclInit failure

When the binary runs on a CPU machine (no NPU device), aclInit fails. Instead
of exiting, set g_host_mode=true so aclnn ops dispatch to AclnnOps.cpp's
CPU-reference branch. AscendC kernel launches use rt* directly (added in
follow-up commit) and don't depend on aclInit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Generated host launches AscendC kernels via `rt*` (camodel-direct)

This is the largest task. It replaces the `// TODO: launch ...` stub with code that:

1. once-per-process: `dlopen("libruntime_camodel.so")` + resolve `rt*` symbols (or call into a tiny helper in `lib/Runtime` that's already linked into the host C++);
2. once-per-kernel: load kernel `.o`, `rtDevBinaryRegister`, `rtFunctionRegister`, cache function handle;
3. per-call: `rtMalloc` GM I/O buffers, `rtMemcpy` H2D, build `KernelArgs` (input ptrs + output ptrs + workspace + tiling blob + block_dim), `rtKernelLaunch`, `rtMemcpy` D2H.

To avoid duplicating the `rt*` plumbing in generated code, expose a thin C-callable
helper in `lib/Runtime/Execution/` that the generated host calls. Generated code stays
small and readable.

**Files:**
- Create: `include/Runtime/Execution/HostLaunchHelper.h` — public C-style API.
- Create: `lib/Runtime/Execution/HostLaunchHelper.cpp` — wraps `NativeExecutionRunner` for one-shot use from generated code.
- Modify: `lib/Runtime/Execution/CMakeLists.txt` — add `HostLaunchHelper.cpp`.
- Modify: `lib/Runtime/AclnnBackend/AclnnBackend.cpp` — emit calls into the helper.
- Modify: `examples/aclnn-attn-e2e/run.sh` (and any other shell driver) — link against the helper lib (deferred to Task 11; just note here).
- Test: `test/Conversion/AclnnBackend/ascendc-launch.mlir`

- [ ] **Step 1: Write the failing lit test**

Create `test/Conversion/AclnnBackend/ascendc-launch.mlir`:

```mlir
// RUN: rm -f %t.cpp && aclnn-backend --input %s --output %t.cpp \
// RUN:   --kernel-binaries /tmp/dummy_artifacts \
// RUN:   --tilings /tmp/dummy_tilings.json
// RUN: FileCheck %s < %t.cpp

module {
  func.func private @kernel_group0(tensor<8xf16>, tensor<8xf16>) -> tensor<8xf16>
  func.func @model(%a: tensor<8xf16>, %b: tensor<8xf16>) -> tensor<8xf16> {
    %r = call @kernel_group0(%a, %b) : (tensor<8xf16>, tensor<8xf16>) -> tensor<8xf16>
    return %r : tensor<8xf16>
  }
}

// CHECK-NOT: TODO: launch
// CHECK: hostLaunchAscendCKernel(
// CHECK:   "kernel_group0"
// CHECK:   /*kernelBinariesDir=*/
// CHECK:   /*tilingsPath=*/
```

- [ ] **Step 2: Run lit, expect FAIL**

```
touch /tmp/dummy_tilings.json && mkdir -p /tmp/dummy_artifacts
build/bin/llvm-lit -v test/Conversion/AclnnBackend/ascendc-launch.mlir
```
Expected: FAIL — current emit is `// TODO: launch ...`.

- [ ] **Step 3: Define the host helper API**

Write `include/Runtime/Execution/HostLaunchHelper.h`:

```cpp
//===- HostLaunchHelper.h - One-shot AscendC kernel launch from generated code -===//
//
// Used by the network_host.cpp emitted by AclnnBackend. Internally driven by
// NativeExecutionRunner in Simulation mode (libruntime_camodel.so), so the
// generated host can launch AscendC kernels on the CANN CPU sim without
// requiring aclInit.
//
// Thread-safety: process-wide singletons are lazy-initialized under a mutex.
//===-------------------------------------------------------------------------===//
#pragma once

#include "Runtime/AclnnOps.h"  // for TensorInfo

namespace mlir::runtime {

// Launch AscendC kernel `kernelName` once.
//
// `kernelBinariesDir` should contain `<kernelName>/<kernelName>.o` (matches
// runtime-session --kernel ... --output ... --name ... layout).
// `tilingsPath` is the JSON written by the runner: {kernelName: {param: value, ...}, ...}
//
// `inputs`/`outputs` are `TensorInfo`s already populated with host pointers,
// shape, dtype. The helper allocates GM buffers, copies H2D, launches, copies D2H,
// and frees GM. `outputs[i].data` must be a valid host buffer of the right size
// before calling.
//
// Returns 0 on success; non-zero on any underlying rt*/file/parse error (and
// writes a diagnostic to stderr).
extern "C" int hostLaunchAscendCKernel(
    const char *kernelName,
    const char *kernelBinariesDir,
    const char *tilingsPath,
    aclnn::TensorInfo *inputs, int numInputs,
    aclnn::TensorInfo *outputs, int numOutputs);

// Optional: when running with `--dump-intermediates DIR`, the runner-generated host
// passes the dir down so `hostLaunchAscendCKernel` can dump per-kernel inputs/outputs
// to `DIR/<kernelName>_in_<i>.npy` / `_out_<i>.npy`. Pass nullptr to disable.
extern "C" void hostLaunchSetDumpIntermediatesDir(const char *dir);

} // namespace mlir::runtime
```

- [ ] **Step 4: Implement the helper**

Write `lib/Runtime/Execution/HostLaunchHelper.cpp`. Read
`lib/Runtime/Execution/NativeExecutionRunner.cpp` first (lines 20–290) — this helper
wraps that class.

```cpp
//===- HostLaunchHelper.cpp -----------------------------------------------===//
#include "Runtime/Execution/HostLaunchHelper.h"
#include "Runtime/Execution/NativeExecutionRunner.h"
#include "Runtime/Execution/TaskGraph.h"           // ExecutionInvocation, KernelArtifact

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

using namespace mlir::runtime;

namespace {

struct HelperState {
  std::mutex mu;
  // process-wide sim runner; created lazily on first use
  std::unique_ptr<NativeExecutionRunner> runner;
  // cached per-kernel: tilings JSON parsed once
  std::map<std::string, std::map<std::string, int64_t>> tilingsByKernel;
  std::string tilingsPath;
  std::string dumpDir;  // empty = disabled
};

static HelperState &state() {
  static HelperState s;
  return s;
}

static void loadTilingsIfNeeded(const std::string &path) {
  auto &s = state();
  if (s.tilingsPath == path) return;
  s.tilingsPath = path;
  s.tilingsByKernel.clear();

  auto bufOr = llvm::MemoryBuffer::getFile(path, /*IsText=*/true);
  if (!bufOr) {
    fprintf(stderr, "[HostLaunchHelper] cannot read tilings: %s\n", path.c_str());
    return;
  }
  auto jsonOr = llvm::json::parse((*bufOr)->getBuffer());
  if (!jsonOr) {
    fprintf(stderr, "[HostLaunchHelper] invalid tilings JSON: %s\n", path.c_str());
    return;
  }
  auto *root = jsonOr->getAsObject();
  if (!root) return;
  for (auto &kv : *root) {
    auto *params = kv.second.getAsObject();
    if (!params) continue;
    auto &m = s.tilingsByKernel[kv.first.str()];
    for (auto &pkv : *params)
      if (auto v = pkv.second.getAsInteger())
        m[pkv.first.str()] = *v;
  }
}

// dtype id → bytes per element (matches AclnnOps.cpp::elemBytes)
static size_t elemBytes(int dtype) {
  switch (dtype) {
    case 0:  return 4;  // FLOAT
    case 1:  return 2;  // FLOAT16
    case 2:  return 1;  // INT8
    case 3:  return 4;  // INT32
    case 27: return 2;  // BF16
    default: return 2;
  }
}

static size_t tensorBytes(const aclnn::TensorInfo &t) {
  size_t n = 1;
  for (int i = 0; i < t.rank; ++i) n *= static_cast<size_t>(t.shape[i]);
  return n * elemBytes(t.dtype);
}

} // namespace

namespace mlir::runtime {

extern "C" void hostLaunchSetDumpIntermediatesDir(const char *dir) {
  std::lock_guard<std::mutex> lk(state().mu);
  state().dumpDir = dir ? dir : "";
}

extern "C" int hostLaunchAscendCKernel(
    const char *kernelName,
    const char *kernelBinariesDir,
    const char *tilingsPath,
    aclnn::TensorInfo *inputs, int numInputs,
    aclnn::TensorInfo *outputs, int numOutputs) {

  auto &s = state();
  std::lock_guard<std::mutex> lk(s.mu);

  if (!s.runner) {
    s.runner = std::make_unique<NativeExecutionRunner>(
        ExecutionRunnerMode::Simulation);
    if (auto err = s.runner->loadRuntimeLibraries()) {
      fprintf(stderr, "[HostLaunchHelper] loadRuntimeLibraries: %s\n",
              llvm::toString(std::move(err)).c_str());
      s.runner.reset();
      return 1;
    }
  }

  loadTilingsIfNeeded(tilingsPath);

  // Build ExecutionInvocation: copy host inputs into invocation.inputs as TensorBindings
  // pointing at in-memory buffers. NativeExecutionRunner handles GM alloc + H2D + launch + D2H.
  ExecutionInvocation invoc;
  for (int i = 0; i < numInputs; ++i) {
    TensorBinding b;
    b.name = std::string(kernelName) + "_in_" + std::to_string(i);
    b.sourceKind = BindingSourceKind::ExternalFile;  // reused as "in-memory" via inlineBytes
    b.shape = std::vector<int64_t>(inputs[i].shape, inputs[i].shape + inputs[i].rank);
    b.dtype = static_cast<DType>(inputs[i].dtype);
    b.inlineBytes.assign(static_cast<const char*>(inputs[i].data),
                         static_cast<const char*>(inputs[i].data) + tensorBytes(inputs[i]));
    invoc.inputs.push_back(std::move(b));
  }
  for (int i = 0; i < numOutputs; ++i) {
    TensorBinding b;
    b.name = std::string(kernelName) + "_out_" + std::to_string(i);
    b.shape = std::vector<int64_t>(outputs[i].shape, outputs[i].shape + outputs[i].rank);
    b.dtype = static_cast<DType>(outputs[i].dtype);
    invoc.outputs.push_back(std::move(b));
  }

  // Tiling: serialize the parsed map into a TilingBinding.params string
  // ("XBLOCK=128,XBLOCK_SUB=16,...") and use the tiling space JSON
  // for schema/blob layout.
  TilingBinding tb;
  tb.schemaPath = std::string(kernelBinariesDir) + "/" + kernelName + "/tiling_space.json";
  std::string params;
  auto it = s.tilingsByKernel.find(kernelName);
  if (it != s.tilingsByKernel.end()) {
    for (auto &kv : it->second) {
      if (!params.empty()) params.push_back(',');
      params += kv.first + "=" + std::to_string(kv.second);
    }
  }
  tb.params = std::move(params);
  invoc.tiling = std::move(tb);

  // Block_dim: read from the parsed best_config (the runner writes it as a "BLOCK_DIM" pseudo-param,
  // or computes from block_dim_expr — for v1 require the runner to put block_dim in tilings JSON
  // under the kernel's map as "_block_dim").
  if (it != s.tilingsByKernel.end()) {
    auto bdIt = it->second.find("_block_dim");
    if (bdIt != it->second.end())
      invoc.blockDim = static_cast<int>(bdIt->second);
  }
  if (invoc.blockDim <= 0)
    invoc.blockDim = 1;

  // Kernel artifact location: <kernelBinariesDir>/<kernelName>/
  KernelArtifact art;
  art.artifactRoot = std::string(kernelBinariesDir) + "/" + kernelName;
  art.kernelName = kernelName;

  // Run via NativeExecutionRunner.
  if (auto err = s.runner->runOne(art, invoc)) {
    fprintf(stderr, "[HostLaunchHelper] runOne(%s): %s\n", kernelName,
            llvm::toString(std::move(err)).c_str());
    return 2;
  }

  // Copy output bytes back into caller-provided host buffers.
  for (int i = 0; i < numOutputs; ++i) {
    if (i >= static_cast<int>(invoc.outputs.size())) break;
    const auto &outBytes = invoc.outputs[i].outBytes;
    std::memcpy(outputs[i].data, outBytes.data(),
                std::min(outBytes.size(), tensorBytes(outputs[i])));
  }

  // Optional: dump intermediates
  if (!s.dumpDir.empty()) {
    auto writeNpy = [&](const std::string &name, const aclnn::TensorInfo &t) {
      std::string path = s.dumpDir + "/" + name + ".npy";
      // Minimal .npy v1.0 writer: header for shape+dtype, then bytes.
      // Reuse python/numpy on the runner side for dtype validation; here we just
      // dump raw bytes into a file with a sidecar describing shape/dtype.
      // For v1 we write raw bytes; the runner side reads the sidecar JSON.
      FILE *fp = fopen(path.c_str(), "wb");
      if (!fp) return;
      fwrite(t.data, 1, tensorBytes(t), fp);
      fclose(fp);
      // sidecar: shape + dtype as JSON
      std::string side = path + ".meta.json";
      FILE *fm = fopen(side.c_str(), "w");
      if (!fm) return;
      fprintf(fm, "{\"shape\":[");
      for (int k = 0; k < t.rank; ++k)
        fprintf(fm, "%s%lld", k ? "," : "", (long long)t.shape[k]);
      fprintf(fm, "],\"dtype\":%d}", t.dtype);
      fclose(fm);
    };
    for (int i = 0; i < numInputs; ++i)
      writeNpy(std::string(kernelName) + "_in_" + std::to_string(i), inputs[i]);
    for (int i = 0; i < numOutputs; ++i)
      writeNpy(std::string(kernelName) + "_out_" + std::to_string(i), outputs[i]);
  }

  return 0;
}

} // namespace mlir::runtime
```

**Note:** if `NativeExecutionRunner` doesn't expose a `runOne(KernelArtifact, ExecutionInvocation)` signature, search for the equivalent (likely `runTask` or `runFromArtifact`) by:
```
grep -n 'runOne\|runTask\|::run\b' lib/Runtime/Execution/NativeExecutionRunner.cpp
```
and adjust the call. The semantics needed: load kernel binary from `art.artifactRoot`, alloc GM, H2D, launch, D2H, free.

If `TensorBinding` doesn't have `inlineBytes` / `outBytes`, extend it (the binding parser supports `ExternalFile`/`TaskOutput` — add an `InlineBuffer` source kind that holds bytes directly). Update `RunManifest.cpp` to ignore the new kind in JSON parsing (it's runtime-only).

This may require changes to `include/Runtime/Execution/TaskGraph.h` (add `InlineBuffer` to `BindingSourceKind`, add `inlineBytes`/`outBytes` to `TensorBinding`). If those changes feel large, the alternative is to write a temp-file shim: dump host inputs to `/tmp/<kernel>_in_<i>.bin`, point the binding at `ExternalFile`, read `/tmp/<kernel>_out_<i>.bin` after run. Choose temp-file shim if `TaskGraph.h` modifications grow beyond ~30 LoC.

- [ ] **Step 5: Register helper source in CMakeLists**

Edit `lib/Runtime/Execution/CMakeLists.txt`. Add `HostLaunchHelper.cpp` to the runtime execution library's source list. Make sure the library exports `hostLaunchAscendCKernel` (extern "C" already does this; no further work needed).

- [ ] **Step 6: Emit the helper call from `aclnn-backend`**

Edit `lib/Runtime/AclnnBackend/AclnnBackend.cpp`. Replace the `// TODO: launch ...` block in `CoordEmitter::emitCall` (around line 100):

```cpp
    // AscendC kernel group → call into the host launch helper.
    // numInputs = call operands, numOutputs = call results.
    std::string outsName = fresh();
    os_ << "  TensorInfo " << outsName << "[" << callOp.getResults().size() << "];\n";
    // Preallocate output host buffers using the result types (rank/shape/dtype).
    for (auto [ri, res] : llvm::enumerate(callOp.getResults())) {
      auto t = cast<RankedTensorType>(res.getType());
      os_ << "  " << outsName << "[" << ri << "].rank = " << t.getRank() << ";\n";
      for (int d = 0; d < t.getRank(); ++d)
        os_ << "  " << outsName << "[" << ri << "].shape[" << d << "] = "
            << t.getDimSize(d) << ";\n";
      // dtype: 1 = FLOAT16; tweak per element type
      int dt = t.getElementType().isF16() ? 1
             : t.getElementType().isBF16() ? 27
             : t.getElementType().isF32() ? 0 : 1;
      os_ << "  " << outsName << "[" << ri << "].dtype = " << dt << ";\n";
      // bytes
      int64_t nelems = 1;
      for (int d = 0; d < t.getRank(); ++d) nelems *= t.getDimSize(d);
      int eb = (dt == 1 || dt == 27) ? 2 : 4;
      os_ << "  " << outsName << "[" << ri << "].data = ::operator new("
          << (nelems * eb) << ");\n";
    }
    std::string insName = fresh();
    os_ << "  TensorInfo " << insName << "[" << callOp.getNumOperands() << "] = {";
    for (auto [i, arg] : llvm::enumerate(callOp.getOperands()))
      os_ << (i ? ", " : "") << nameOf(arg);
    os_ << "};\n";
    os_ << "  if (mlir::runtime::hostLaunchAscendCKernel(\n";
    os_ << "        \"" << callOp.getCallee() << "\",\n";
    os_ << "        /*kernelBinariesDir=*/\"" << cfg.kernelBinariesDir << "\",\n";
    os_ << "        /*tilingsPath=*/\"" << cfg.tilingsPath << "\",\n";
    os_ << "        " << insName << ", " << callOp.getNumOperands() << ",\n";
    os_ << "        " << outsName << ", " << callOp.getResults().size() << ") != 0) {\n";
    os_ << "    fprintf(stderr, \"hostLaunchAscendCKernel(" << callOp.getCallee() << ") failed\\n\");\n";
    os_ << "    return;\n";
    os_ << "  }\n";
    for (auto [ri, res] : llvm::enumerate(callOp.getResults()))
      names_[res] = outsName + "[" + std::to_string(ri) + "]";
```

`cfg` needs to be reachable from `CoordEmitter::emitCall`. Pass it through the emitter constructor:

```cpp
class CoordEmitter {
public:
  CoordEmitter(ModuleOp module, const AclnnBackendConfig &cfg, llvm::raw_ostream &os)
      : module_(module), cfg_(cfg), os_(os) {}
  // ...
private:
  const AclnnBackendConfig &cfg_;
  // ...
};
```
Update the `CoordEmitter emitter(module, os);` site in `buildNetworkHostCpp` to `CoordEmitter emitter(module, cfg, os);`.

Add the helper include to the file-header emit:
```cpp
  os << "#include \"Runtime/Execution/HostLaunchHelper.h\"\n";
  os << "#include <cstdio>\n";
```

- [ ] **Step 7: Build**

```
ninja -C build
```
Expected: clean build. If `NativeExecutionRunner`'s public API is different from the assumption in step 4, fix the helper's call into it now (search-and-replace).

- [ ] **Step 8: Run lit, expect PASS**

```
build/bin/llvm-lit -v test/Conversion/AclnnBackend/ascendc-launch.mlir
```
Expected: PASS — `// TODO: launch` gone; `hostLaunchAscendCKernel("kernel_group0",` present.

- [ ] **Step 9: Re-run all aclnn-backend lits**

```
build/bin/llvm-lit -v test/Conversion/AclnnBackend/
```
Expected: all PASS (host-mode, ascendc-launch, cli-tilings).

- [ ] **Step 10: Commit**

```
git add include/Runtime/Execution/HostLaunchHelper.h \
        lib/Runtime/Execution/HostLaunchHelper.cpp \
        lib/Runtime/Execution/CMakeLists.txt \
        lib/Runtime/AclnnBackend/AclnnBackend.cpp \
        test/Conversion/AclnnBackend/ascendc-launch.mlir
git commit -m "feat(aclnn-backend): emit AscendC kernel launches via HostLaunchHelper

Replace the // TODO: launch ... stub with code that calls into
hostLaunchAscendCKernel — a thin C-callable wrapper around
NativeExecutionRunner in Simulation mode. The generated network_host.cpp now
launches AscendC kernels on the CANN CPU sim (camodel) without aclInit.
Tilings + kernel binary dir come from --tilings / --kernel-binaries
introduced in the previous commit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: `--dump-intermediates DIR` runtime flag in generated host

The generated `network` entry point should accept a runtime config that toggles
intermediate dumping. Simplest: a separate `extern "C" void network_set_dump_dir(const char*)`
that the binary's `main()` (in the example's harness) calls before invoking `network()`.

**Files:**
- Modify: `lib/Runtime/AclnnBackend/AclnnBackend.cpp`
- Test: existing `test/Conversion/AclnnBackend/ascendc-launch.mlir` extended

- [ ] **Step 1: Extend the lit test**

Append to `test/Conversion/AclnnBackend/ascendc-launch.mlir`:

```mlir
// CHECK: extern "C" void network_set_dump_dir(const char *dir)
// CHECK:   mlir::runtime::hostLaunchSetDumpIntermediatesDir(dir)
```

- [ ] **Step 2: Run lit, expect FAIL**

```
build/bin/llvm-lit -v test/Conversion/AclnnBackend/ascendc-launch.mlir
```
Expected: FAIL.

- [ ] **Step 3: Emit the setter**

In `lib/Runtime/AclnnBackend/AclnnBackend.cpp` `buildNetworkHostCpp`, after the `extern "C" void network(...)` block:

```cpp
  os << "\nextern \"C\" void network_set_dump_dir(const char *dir) {\n";
  os << "  mlir::runtime::hostLaunchSetDumpIntermediatesDir(dir);\n";
  os << "}\n";
```

- [ ] **Step 4: Build & test**

```
ninja -C build
build/bin/llvm-lit -v test/Conversion/AclnnBackend/ascendc-launch.mlir
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add lib/Runtime/AclnnBackend/AclnnBackend.cpp \
        test/Conversion/AclnnBackend/ascendc-launch.mlir
git commit -m "feat(aclnn-backend): expose network_set_dump_dir for intermediate dumps

Generated host exports network_set_dump_dir(const char*); calling it before
network() routes per-kernel inputs/outputs to DIR/<kernel>_in_<i>.npy /
_out_<i>.npy via HostLaunchHelper. Used by network_runner.py phase 3 to
collect autotune inputs.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: `network_runner.py` — Phase 1 (outline / emit-network-json)

Phase 1 only: take the input mode and produce `WORK/groups/{network.mlir, network.json,
kernel_group*.mlir}`. Ship a unit-test that exercises both input modes against
fixtures.

**Files:**
- Create: `python/network_runner.py`
- Create: `python/runner_utils/__init__.py`
- Create: `python/runner_utils/run_subprocess.py`
- Create: `python/runner_utils/network_json.py`
- Create: `python/tests/test_network_runner_phase1.py`

- [ ] **Step 1: Skeleton + Phase 1 only**

Write `python/network_runner.py`:

```python
#!/usr/bin/env python3
"""Network runner: orchestrate mixed AscendC+aclnn network compilation & execution.

See docs/superpowers/specs/2026-05-13-network-runner-mixed-cpu-sim-design.md.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AFIR_OPT = os.environ.get("AFIR_OPT", str(REPO / "build/bin/afir-opt"))
AFIR_TRANSLATE = os.environ.get("AFIR_TRANSLATE", str(REPO / "build/bin/afir-translate"))
ACLNN_BACKEND = os.environ.get("ACLNN_BACKEND", str(REPO / "build/bin/aclnn-backend"))
RUNTIME_SESSION = os.environ.get("RUNTIME_SESSION", str(REPO / "build/bin/runtime-session"))
AUTOTUNER = os.environ.get("AUTOTUNER", str(REPO / "build/bin/autotuner"))


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def phase1_outline_or_emit_json(args, work):
    groups = work / "groups"
    groups.mkdir(parents=True, exist_ok=True)

    if args.input_linalg:
        # Pipe: --linalg-fold-unit-extent-dims | --vector-plan-group-analysis +
        # --vector-plan-group-outline=output-dir=...
        intermediate = work / "model_unit_folded.mlir"
        run([AFIR_OPT, "--linalg-fold-unit-extent-dims", args.input_linalg,
             "-o", str(intermediate)])
        run([AFIR_OPT, "--vector-plan-group-analysis",
             f"--vector-plan-group-outline=output-dir={groups}",
             str(intermediate), "-o", str(work / "_outlined_combined.mlir")])
    else:
        # Hand-written network.mlir + kernel_group*.mlir in args.input_network
        src = Path(args.input_network)
        if not src.is_dir():
            sys.exit(f"--input-network must be a directory: {src}")
        if not (src / "network.mlir").exists():
            sys.exit(f"missing {src / 'network.mlir'}")
        for f in src.glob("*.mlir"):
            shutil.copy(f, groups / f.name)
        run([AFIR_OPT, str(groups / "network.mlir"),
             f"--emit-network-json=path={groups / 'network.json'}",
             "-o", "/dev/null"])

    nj = groups / "network.json"
    if not nj.exists():
        sys.exit(f"phase 1 did not produce {nj}")
    print(f"phase 1 OK → {nj}")
    return groups


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-linalg")
    g.add_argument("--input-network")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--expected", nargs="+", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--soc", default="Ascend910B1")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-2)
    args = ap.parse_args()

    work = Path(args.workdir).absolute()
    work.mkdir(parents=True, exist_ok=True)

    groups = phase1_outline_or_emit_json(args, work)
    # Phase 2-5: see follow-up tasks.
    print(f"workdir: {work}")
    print(f"groups:  {groups}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Helper modules**

Write `python/runner_utils/__init__.py` (empty).

Write `python/runner_utils/run_subprocess.py`:

```python
import subprocess, sys

def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)
```

Write `python/runner_utils/network_json.py`:

```python
"""Helpers for the network.json schema."""
import json
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class NetworkJson:
    function: str
    inputs:   List[Dict[str, Any]]
    kernels:  List[Dict[str, Any]]
    outputs:  List[Dict[str, Any]]

    @classmethod
    def load(cls, path: str) -> "NetworkJson":
        with open(path) as f:
            d = json.load(f)
        return cls(d["function"], d["inputs"], d["kernels"], d["outputs"])

    def ascendc_kernels(self):
        return [k for k in self.kernels if k["kind"] == "ascendc"]

    def kernel_by_id(self, kid):
        for k in self.kernels:
            if k["id"] == kid:
                return k
        raise KeyError(kid)
```

- [ ] **Step 3: Unit test**

Write `python/tests/test_network_runner_phase1.py`:

```python
import os, subprocess, tempfile
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "python/network_runner.py"

def test_phase1_input_network_mixed(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "network.mlir").write_text("""
module {
  func.func private @kernel_group0(tensor<8xf16>) -> tensor<8xf16>
  func.func private @__aclnn_softmax(tensor<8xf16>) -> tensor<8xf16>
      attributes {aclnn.op = "Softmax"}
  func.func @model(%x: tensor<8xf16>) -> tensor<8xf16> {
    %a = call @kernel_group0(%x) : (tensor<8xf16>) -> tensor<8xf16>
    %s = call @__aclnn_softmax(%a) : (tensor<8xf16>) -> tensor<8xf16>
    return %s : tensor<8xf16>
  }
}
""")
    (src / "kernel_group0.mlir").write_text("module {}\n")  # placeholder

    work = tmp_path / "work"
    res = subprocess.run([
        "python3", str(RUNNER),
        "--input-network", str(src),
        "--inputs", "/dev/null",
        "--expected", "/dev/null",
        "--workdir", str(work),
    ], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    nj = work / "groups" / "network.json"
    assert nj.exists()
    import json
    d = json.loads(nj.read_text())
    assert d["function"] == "model"
    kinds = [k["kind"] for k in d["kernels"]]
    assert kinds == ["ascendc", "aclnn"]
```

- [ ] **Step 4: Run test, expect PASS (after build is current)**

```
ninja -C build  # ensure afir-opt has the --emit-network-json pass from Task 2
PYTHONPATH=python pytest python/tests/test_network_runner_phase1.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add python/network_runner.py python/runner_utils/ python/tests/test_network_runner_phase1.py
git commit -m "feat(network-runner): phase 1 (outline / emit-network-json)

Two input modes: --input-linalg runs --vector-plan-group-outline; --input-network
copies a hand-written DIR and runs --emit-network-json. Both produce
WORK/groups/{network.mlir, network.json, kernel_group*.mlir}. Phase 1 only;
phases 2-5 in follow-up commits.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: `network_runner.py` — Phase 2 (per-kernel codegen + compile)

For each `kind == "ascendc"` kernel: lower → translate → compile to artifact via
`runtime-session --kernel ... --output ...`.

**Files:**
- Modify: `python/network_runner.py`

- [ ] **Step 1: Add Phase 2 implementation**

Append to `python/network_runner.py` (above `main`):

```python
def phase2_codegen_compile(work, groups, network):
    artifacts = work / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for k in network.ascendc_kernels():
        kid = k["id"]
        src = groups / k["file"]
        lowered = work / f"{kid}_lowered.mlir"
        cpp = work / f"{kid}.cpp"
        space = work / f"{kid}_space.json"
        run([AFIR_OPT, str(src), "--vector-plan-codegen", "-o", str(lowered)])
        run([AFIR_TRANSLATE, "-mlir-to-cann", str(lowered),
             "-o", str(cpp), f"--tiling-space-out={space}"])
        run([RUNTIME_SESSION,
             "--kernel", str(cpp),
             "--kernel-kind", "vec",
             "--output", str(artifacts / kid),
             "--name", kid])
    print(f"phase 2 OK → {artifacts}")
    return artifacts
```

Add at the top of `network_runner.py`:
```python
from runner_utils.network_json import NetworkJson
```

In `main`, after the `phase1_outline_or_emit_json` line:
```python
    network = NetworkJson.load(groups / "network.json")
    artifacts = phase2_codegen_compile(work, groups, network)
```

- [ ] **Step 2: Smoke test on twochain**

```
cat > /tmp/twochain.mlir <<'EOF'
... (same content as Task 1 step 9)
EOF
PYTHONPATH=python python3 python/network_runner.py \
  --input-linalg /tmp/twochain.mlir \
  --inputs /dev/null --expected /dev/null \
  --workdir /tmp/twg-runner
ls /tmp/twg-runner/artifacts/
```
Expected: `kernel_group0/`, `kernel_group1/` directories under `artifacts`, each containing the compiled `.o` and `tiling_space.json` (or wherever `runtime-session --output` puts them — check `examples/relu-e2e/run.sh` for the expected layout).

- [ ] **Step 3: Commit**

```
git add python/network_runner.py
git commit -m "feat(network-runner): phase 2 — per-kernel codegen+translate+compile

For each ascendc kernel: --vector-plan-codegen, -mlir-to-cann (writes .cpp +
tiling_space.json), runtime-session --kernel (compiles to artifact dir).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: `network_runner.py` — Phase 3 (default-tilings build, dump intermediates)

Build a `network_host.cpp` with default tilings, link it, run with `--dump-intermediates`,
and harvest the per-kernel input/output `.npy` files for use in Phase 4.

**Files:**
- Modify: `python/network_runner.py`
- Create: `python/runner_utils/build_host.py` — small g++ link helper.

- [ ] **Step 1: g++ link helper**

Write `python/runner_utils/build_host.py`:

```python
import os, subprocess
from pathlib import Path

def link_host(host_cpp: Path, out_binary: Path, repo_root: Path,
              cann_home: str, cann_arch: str = "x86_64-linux") -> None:
    cann_lib = f"{cann_home}/{cann_arch}/lib64"
    cann_devlib = f"{cann_home}/{cann_arch}/devlib"
    cann_inc = f"{cann_home}/{cann_arch}/include"
    cmd = [
        "g++", "-std=c++17", "-O2",
        "-I", str(repo_root / "include"),
        "-I", cann_inc,
        str(host_cpp),
        str(repo_root / "lib/Runtime/AclnnOps.cpp"),
        # link against the runtime-execution lib that contains HostLaunchHelper
        "-L", str(repo_root / "build/lib"),
        "-lAFIRRuntimeExecution",  # name from CMake target; verify
        "-L", cann_lib, "-L", cann_devlib,
        "-lascendcl", "-lopapi_transformer", "-lnnopbase", "-lascend_hal",
        "-Wl,-rpath," + ":".join([cann_lib, cann_devlib,
                                  f"{cann_home}/{cann_arch}/simulator/Ascend910B1/lib"]),
        "-o", str(out_binary),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
```

(Library name `-lAFIRRuntimeExecution` is a placeholder; check
`lib/Runtime/Execution/CMakeLists.txt` for the actual `add_mlir_library(...)` target
name and substitute.)

- [ ] **Step 2: Phase 3 driver**

Append to `python/network_runner.py`:

```python
def phase3_default_build_and_dump(work, groups, network, artifacts, args):
    # Build a default-tilings JSON: each kernel's params at their schema "default".
    default_tilings = {}
    for k in network.ascendc_kernels():
        kid = k["id"]
        space = json.loads((work / f"{kid}_space.json").read_text())
        params = {p["name"]: p["default"] for p in space["tiling_params"]
                  if p.get("kind") == "tunable"}
        # the helper expects _block_dim too; eval block_dim_expr lazily:
        params["_block_dim"] = eval_block_dim(space, params)
        default_tilings[kid] = params
    tilings_path = work / "tilings_default.json"
    tilings_path.write_text(json.dumps(default_tilings, indent=2))

    # Generate host C++.
    host_cpp = work / "network_host_default.cpp"
    run([ACLNN_BACKEND,
         "--input", str(groups / "network.mlir"),
         "--output", str(host_cpp),
         "--tilings", str(tilings_path),
         "--kernel-binaries", str(artifacts)])

    # g++ link.
    from runner_utils.build_host import link_host
    binary = work / "network_test_default"
    link_host(host_cpp, binary, REPO,
              cann_home=os.environ.get("ASCEND_HOME_PATH",
                                       "/home/gser/Ascend/cann-9.0.0"))

    # Run with --dump-intermediates DIR.
    inter = work / "intermediates_default"
    inter.mkdir(parents=True, exist_ok=True)
    out_npy = work / "output_default.npy"
    cmd = [str(binary)]
    for p in args.inputs:    cmd += ["--input", p]
    cmd += ["--output", str(out_npy)]
    cmd += ["--dump-intermediates", str(inter)]
    run(cmd)
    print(f"phase 3 OK → {inter}")
    return tilings_path, inter


def eval_block_dim(space, params):
    """Evaluate space['block_dim_expr'] under integer params; minimal grammar."""
    import re
    expr = space.get("block_dim_expr") or "1"
    # Replace identifiers with their int values
    def repl(m):
        name = m.group(0)
        if name in params:
            return str(int(params[name]))
        return name
    expr = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", repl, expr)
    # Evaluate using the same grammar autotuner does: + - * /, ceil(a/b)
    expr2 = re.sub(r"ceil\(([^,]+)/([^)]+)\)", r"-(-(\1)//(\2))", expr)
    try:
        return int(eval(expr2, {"__builtins__": {}}, {}))
    except Exception:
        return 1
```

In `main`, after Phase 2:
```python
    tilings_path, inter = phase3_default_build_and_dump(work, groups, network, artifacts, args)
```

The example's harness binary needs to:
1. Parse `--input PATH ...` (multiple), `--output PATH`, `--dump-intermediates DIR` flags.
2. Load each input npy into a `TensorInfo` (host buffer + shape + dtype).
3. Allocate output `TensorInfo`s.
4. Call `network_set_dump_dir(DIR)` if `--dump-intermediates` was given.
5. Call `network(inputs, numIn, outputs, numOut, /*stream=*/nullptr)`.
6. Save `outputs[i]` to the `--output` path (one per output).

The harness lives in the example, not in the runner — see Task 12. For Phase 3 to be
testable, Task 12 has to land first (or the test stops at "host_cpp generated +
linked").

- [ ] **Step 3: Smoke test (gates on Task 12 for end-to-end)**

For now, just check the generated host_cpp compiles standalone:
```
PYTHONPATH=python python3 python/network_runner.py \
  --input-linalg /tmp/twochain.mlir \
  --inputs /dev/null /dev/null /dev/null /dev/null /dev/null /dev/null \
  --expected /dev/null /dev/null \
  --workdir /tmp/twg-runner 2>&1 | tee /tmp/twg-phase3.log || true
ls /tmp/twg-runner/network_host_default.cpp
ls /tmp/twg-runner/network_test_default 2>/dev/null && echo "linked OK" || echo "link failed (expected before Task 12 harness)"
```

- [ ] **Step 4: Commit**

```
git add python/network_runner.py python/runner_utils/build_host.py
git commit -m "feat(network-runner): phase 3 — default-tilings build + dump intermediates

Generate tilings_default.json (each kernel's space defaults + evaluated
block_dim_expr), invoke aclnn-backend, g++ link the host C++ against
HostLaunchHelper + AclnnOps. The binary, when run with --dump-intermediates,
writes per-kernel input/output buffers consumed by phase 4 autotune.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: `network_runner.py` — Phase 4 (per-kernel autotune)

For each ascendc kernel: read its inputs/outputs from `intermediates_default/`, hand
to `autotuner` with the kernel's `_space.json` and `.cpp`, parse `kgN_best.json`, build
up `tilings_best`.

**Files:**
- Modify: `python/network_runner.py`

- [ ] **Step 1: Phase 4 implementation**

Append to `python/network_runner.py`:

```python
def phase4_autotune(work, network, inter):
    tilings_best = {}
    for k in network.ascendc_kernels():
        kid = k["id"]
        # inputs = <kid>_in_*.npy in dump dir; expected = <kid>_out_0.npy.
        ins = sorted(str(p) for p in inter.glob(f"{kid}_in_*.npy"))
        expected = inter / f"{kid}_out_0.npy"
        if not expected.exists():
            sys.exit(f"phase 4: missing {expected}")
        space = work / f"{kid}_space.json"
        cpp = work / f"{kid}.cpp"
        best = work / f"{kid}_best.json"
        # shape arg: read from intermediates side-cars
        shape_args = build_shape_str(inter, kid, space)
        cmd = [AUTOTUNER,
               "--space", str(space), "--kernel", str(cpp),
               "--inputs", ",".join(ins),
               "--expected", str(expected),
               "--shape", shape_args,
               "--output", str(best)]
        run(cmd)
        d = json.loads(best.read_text())
        params = dict(d.get("params", {}))
        # autotuner doesn't write _block_dim; recompute it from space.
        space_d = json.loads(space.read_text())
        params["_block_dim"] = eval_block_dim(space_d, params)
        tilings_best[kid] = params
    tilings_path = work / "tilings_best.json"
    tilings_path.write_text(json.dumps(tilings_best, indent=2))
    print(f"phase 4 OK → {tilings_path}")
    return tilings_path


def build_shape_str(inter, kid, space):
    """Resolve shape_key tokens in space → concrete dim values from dumped npy meta."""
    import json as _json, glob, re
    parts = []
    space_d = _json.loads(open(space).read())
    for p in space_d["tiling_params"]:
        key = p.get("shape_key")
        if not key: continue
        # key looks like "arg0_dim1"
        m = re.match(r"arg(\d+)_dim(\d+)", key)
        if not m: continue
        argIdx, dimIdx = int(m.group(1)), int(m.group(2))
        meta = inter / f"{kid}_in_{argIdx}.npy.meta.json"
        if not meta.exists(): continue
        d = _json.loads(meta.read_text())
        parts.append(f"{p['name']}={d['shape'][dimIdx]}")
    return ",".join(parts)
```

In `main`, after Phase 3:
```python
    tilings_best = phase4_autotune(work, network, inter)
```

- [ ] **Step 2: Verify on existing single-kernel autotune flow**

Check that `autotuner --output FILE` writes a JSON with `params` map. If not, look at
`tools/autotuner/autotuner_main.cpp:40` (`Best-config JSON output path`) and adapt:

```
build/bin/autotuner --help | head -20
```
Expected: confirm the format of `best_config.json`. Adjust the parsing in step 1 if
the schema is `{"XBLOCK": 128, ...}` (flat map) instead of `{"params": {...}}`.

- [ ] **Step 3: Commit**

```
git add python/network_runner.py
git commit -m "feat(network-runner): phase 4 — per-kernel autotune via existing autotuner

For each ascendc kernel: feed its dumped intermediate inputs (from phase 3)
to autotuner with the kernel's tiling_space.json + .cpp; parse best_config.json;
recompute _block_dim from the kernel's space + chosen params; aggregate into
tilings_best.json for phase 5.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 11: `network_runner.py` — Phase 5 (best-tilings build, run, verify)

Re-invoke `aclnn-backend` with the best-tilings JSON, g++ link, run, compare each
network output against `--expected` with atol/rtol.

**Files:**
- Modify: `python/network_runner.py`

- [ ] **Step 1: Phase 5 implementation**

Append to `python/network_runner.py`:

```python
def phase5_final_run_verify(work, groups, artifacts, tilings_best_path, args):
    host_cpp = work / "network_host.cpp"
    run([ACLNN_BACKEND,
         "--input", str(groups / "network.mlir"),
         "--output", str(host_cpp),
         "--tilings", str(tilings_best_path),
         "--kernel-binaries", str(artifacts)])
    from runner_utils.build_host import link_host
    binary = work / "network_test"
    link_host(host_cpp, binary, REPO,
              cann_home=os.environ.get("ASCEND_HOME_PATH",
                                       "/home/gser/Ascend/cann-9.0.0"))
    # Run.  outputs are written to one npy per network output, in order.
    out_dir = work / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary)]
    for p in args.inputs:    cmd += ["--input", p]
    out_paths = [str(out_dir / f"out{i}.npy") for i, _ in enumerate(args.expected)]
    for p in out_paths:      cmd += ["--output", p]
    run(cmd)

    # Verify.
    import numpy as np
    fail = False
    for i, (got, want) in enumerate(zip(out_paths, args.expected)):
        a = np.load(got).astype(np.float32)
        b = np.load(want).astype(np.float32)
        if a.shape != b.shape:
            print(f"network.output[{i}]: SHAPE MISMATCH got={a.shape} want={b.shape}  FAIL")
            fail = True; continue
        diff = np.abs(a - b)
        max_diff = float(diff.max())
        ok = np.allclose(a, b, atol=args.atol, rtol=args.rtol)
        print(f"network.output[{i}]: max_diff={max_diff:.4g}  {'PASS' if ok else 'FAIL'}")
        fail |= not ok
    return 0 if not fail else 1
```

In `main`, after Phase 4:
```python
    rc = phase5_final_run_verify(work, groups, artifacts, tilings_best, args)
    sys.exit(rc)
```

- [ ] **Step 2: End-to-end test gated on Task 12 example**

Phase 5 doesn't have an isolated test — it requires a real example. See Task 12.

- [ ] **Step 3: Commit**

```
git add python/network_runner.py
git commit -m "feat(network-runner): phase 5 — best-tilings build, run, verify

Re-emit network_host.cpp with autotuned tilings, link, run on CPU sim, compare
each network output against --expected with atol/rtol. Returns non-zero on
any FAIL.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 12: `examples/mixed-attn-e2e/` example

Hand-written mixed network: pre-norm `kernel_group0` (linalg add → mul) → flash-attention
(aclnn) → post-proj `kernel_group1` (linalg mul → add). Plus harness, gen_inputs.py,
run.sh.

Tensor shapes use the existing aclnn-attn-e2e example's: q/k/v are
`tensor<1x2x16x8xf16>`, mask is `tensor<1x2x16x16xf16>`, init is the same shape as q.

**Files:**
- Create: `examples/mixed-attn-e2e/network.mlir`
- Create: `examples/mixed-attn-e2e/kernel_group0.mlir`
- Create: `examples/mixed-attn-e2e/kernel_group1.mlir`
- Create: `examples/mixed-attn-e2e/gen_inputs.py`
- Create: `examples/mixed-attn-e2e/harness.cpp`
- Create: `examples/mixed-attn-e2e/run.sh`

- [ ] **Step 1: kernel_group0 — pre-norm (q := q * scale + bias, where scale/bias are inputs)**

Write `examples/mixed-attn-e2e/kernel_group0.mlir`:

```mlir
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func private @kernel_group0(
      %q: tensor<1x2x16x8xf16>, %scale: tensor<1x2x16x8xf16>,
      %bias: tensor<1x2x16x8xf16>, %init: tensor<1x2x16x8xf16>)
      -> tensor<1x2x16x8xf16> {
    %r = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                         iterator_types = ["parallel","parallel","parallel","parallel"]}
         ins(%q, %scale, %bias : tensor<1x2x16x8xf16>, tensor<1x2x16x8xf16>, tensor<1x2x16x8xf16>)
         outs(%init : tensor<1x2x16x8xf16>) {
    ^bb0(%a: f16, %s: f16, %b: f16, %o: f16):
      %m = arith.mulf %a, %s : f16
      %p = arith.addf %m, %b : f16
      linalg.yield %p : f16
    } -> tensor<1x2x16x8xf16>
    return %r : tensor<1x2x16x8xf16>
  }
}
```

- [ ] **Step 2: kernel_group1 — post-proj (out := fa_out * proj_scale)**

Write `examples/mixed-attn-e2e/kernel_group1.mlir`:

```mlir
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func private @kernel_group1(
      %fa: tensor<1x2x16x8xf16>, %proj: tensor<1x2x16x8xf16>,
      %init: tensor<1x2x16x8xf16>) -> tensor<1x2x16x8xf16> {
    %r = linalg.generic {indexing_maps = [#map, #map, #map],
                         iterator_types = ["parallel","parallel","parallel","parallel"]}
         ins(%fa, %proj : tensor<1x2x16x8xf16>, tensor<1x2x16x8xf16>)
         outs(%init : tensor<1x2x16x8xf16>) {
    ^bb0(%a: f16, %p: f16, %o: f16):
      %m = arith.mulf %a, %p : f16
      linalg.yield %m : f16
    } -> tensor<1x2x16x8xf16>
    return %r : tensor<1x2x16x8xf16>
  }
}
```

- [ ] **Step 3: network.mlir — stitch the three calls**

Write `examples/mixed-attn-e2e/network.mlir`:

```mlir
module {
  func.func private @kernel_group0(tensor<1x2x16x8xf16>, tensor<1x2x16x8xf16>,
                                    tensor<1x2x16x8xf16>, tensor<1x2x16x8xf16>)
      -> tensor<1x2x16x8xf16>

  func.func private @__aclnn_flash_attention(
      tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>,
      tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
      attributes {aclnn.op = "FlashAttentionScore", aclnn.layout = "BNSD"}

  func.func private @kernel_group1(tensor<1x2x16x8xf16>, tensor<1x2x16x8xf16>,
                                    tensor<1x2x16x8xf16>) -> tensor<1x2x16x8xf16>

  func.func @model(
      %q: tensor<1x2x16x8xf16>, %scale: tensor<1x2x16x8xf16>, %bias: tensor<1x2x16x8xf16>,
      %k: tensor<1x2x16x8xf16>, %v: tensor<1x2x16x8xf16>,
      %mask: tensor<1x2x16x16xf16>, %init0: tensor<1x2x16x8xf16>,
      %init_fa: tensor<1x2x16x8xf16>, %proj: tensor<1x2x16x8xf16>,
      %init1: tensor<1x2x16x8xf16>) -> tensor<1x2x16x8xf16> {
    %qprime = call @kernel_group0(%q, %scale, %bias, %init0)
        : (tensor<1x2x16x8xf16>, tensor<1x2x16x8xf16>, tensor<1x2x16x8xf16>, tensor<1x2x16x8xf16>)
        -> tensor<1x2x16x8xf16>
    %qc  = tensor.cast %qprime    : tensor<1x2x16x8xf16> to tensor<?x?x?x?xf16>
    %kc  = tensor.cast %k         : tensor<1x2x16x8xf16> to tensor<?x?x?x?xf16>
    %vc  = tensor.cast %v         : tensor<1x2x16x8xf16> to tensor<?x?x?x?xf16>
    %mc  = tensor.cast %mask      : tensor<1x2x16x16xf16> to tensor<?x?x?x?xf16>
    %ic  = tensor.cast %init_fa   : tensor<1x2x16x8xf16> to tensor<?x?x?x?xf16>
    %fa  = call @__aclnn_flash_attention(%qc, %kc, %vc, %mc, %ic)
        : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>,
           tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
    %fac = tensor.cast %fa : tensor<?x?x?x?xf16> to tensor<1x2x16x8xf16>
    %out = call @kernel_group1(%fac, %proj, %init1)
        : (tensor<1x2x16x8xf16>, tensor<1x2x16x8xf16>, tensor<1x2x16x8xf16>)
        -> tensor<1x2x16x8xf16>
    return %out : tensor<1x2x16x8xf16>
  }
}
```

- [ ] **Step 4: gen_inputs.py — numpy reference**

Write `examples/mixed-attn-e2e/gen_inputs.py`:

```python
"""Generate inputs.npy and expected.npy for mixed-attn-e2e.

The reference path mirrors what the host-mode FlashAttention in AclnnOps.cpp computes,
followed by the elementwise pre/post ops.
"""
import argparse, numpy as np, os

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x).astype(np.float32)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    od = args.out_dir
    rng = np.random.default_rng(0)

    shape = (1, 2, 16, 8)
    mask_shape = (1, 2, 16, 16)
    q = rng.standard_normal(shape).astype(np.float16) * 0.1
    k = rng.standard_normal(shape).astype(np.float16) * 0.1
    v = rng.standard_normal(shape).astype(np.float16) * 0.1
    scale = np.full(shape, 1.0, dtype=np.float16)
    bias  = np.zeros(shape, dtype=np.float16)
    mask  = np.zeros(mask_shape, dtype=np.float16)
    init0 = np.zeros(shape, dtype=np.float16)
    init_fa = np.zeros(shape, dtype=np.float16)
    proj  = np.full(shape, 1.0, dtype=np.float16)
    init1 = np.zeros(shape, dtype=np.float16)

    # Reference: q' = q*scale + bias
    q_prime = (q.astype(np.float32) * scale.astype(np.float32) + bias.astype(np.float32)).astype(np.float16)
    # FlashAttentionScore reference (BNSD): out = softmax(qk^T/sqrt(d)+mask) @ v
    d = q_prime.shape[-1]
    qf = q_prime.astype(np.float32); kf = k.astype(np.float32); vf = v.astype(np.float32)
    scores = np.einsum("bnqd,bnkd->bnqk", qf, kf) / np.sqrt(d) + mask.astype(np.float32)
    attn = softmax(scores, axis=-1)
    fa = np.einsum("bnqk,bnkd->bnqd", attn, vf).astype(np.float16)
    # post-proj: out = fa * proj
    out = (fa.astype(np.float32) * proj.astype(np.float32)).astype(np.float16)

    for name, arr in [("q", q), ("scale", scale), ("bias", bias),
                       ("k", k), ("v", v), ("mask", mask),
                       ("init0", init0), ("init_fa", init_fa),
                       ("proj", proj), ("init1", init1),
                       ("expected", out)]:
        np.save(os.path.join(od, f"{name}.npy"), arr)
        print(f"{name}: {arr.shape} {arr.dtype}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: harness.cpp — main()**

Write `examples/mixed-attn-e2e/harness.cpp`:

```cpp
// Driver for the runner-emitted network() entry point.
// Parses --input, --output, --dump-intermediates, loads npy buffers, calls
// network(), saves outputs to npy.

#include "Runtime/AclnnOps.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

extern "C" void network(mlir::runtime::aclnn::TensorInfo *inputs, int numIn,
                        mlir::runtime::aclnn::TensorInfo *outputs, int numOut,
                        void *stream);
extern "C" void network_set_dump_dir(const char *dir);

namespace {

struct NpyMeta {
  std::vector<int64_t> shape;
  int dtype = 1;  // FLOAT16
};

// Minimal .npy v1.0 reader: assumes float16 / float32, little-endian.
NpyMeta readNpyHeader(std::ifstream &is) {
  char magic[6]; is.read(magic, 6);
  uint8_t ver[2]; is.read(reinterpret_cast<char*>(ver), 2);
  uint16_t hlen; is.read(reinterpret_cast<char*>(&hlen), 2);
  std::string header(hlen, ' '); is.read(header.data(), hlen);
  NpyMeta m;
  // descr
  size_t p = header.find("'descr':");
  size_t q = header.find("'", p + 9);
  std::string descr = header.substr(q + 1, header.find("'", q + 1) - q - 1);
  if (descr == "<f2") m.dtype = 1;       // FLOAT16
  else if (descr == "<f4") m.dtype = 0;  // FLOAT32
  // shape
  p = header.find("'shape':");
  q = header.find("(", p);
  size_t qe = header.find(")", q);
  std::string shapeStr = header.substr(q + 1, qe - q - 1);
  size_t pos = 0;
  while (pos < shapeStr.size()) {
    size_t comma = shapeStr.find(',', pos);
    std::string tok = shapeStr.substr(pos, comma - pos);
    if (!tok.empty() && tok.find_first_not_of(" \t") != std::string::npos)
      m.shape.push_back(std::stoll(tok));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return m;
}

mlir::runtime::aclnn::TensorInfo loadNpy(const std::string &path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) { fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
  NpyMeta m = readNpyHeader(is);
  mlir::runtime::aclnn::TensorInfo t;
  t.rank  = static_cast<int>(m.shape.size());
  t.dtype = m.dtype;
  for (int i = 0; i < t.rank; ++i) t.shape[i] = m.shape[i];
  size_t nelems = 1; for (int i = 0; i < t.rank; ++i) nelems *= m.shape[i];
  size_t bytes  = nelems * (m.dtype == 1 ? 2 : 4);
  t.data = ::operator new(bytes);
  is.read(reinterpret_cast<char*>(t.data), bytes);
  return t;
}

void saveNpy(const std::string &path, const mlir::runtime::aclnn::TensorInfo &t) {
  std::ofstream os(path, std::ios::binary);
  os.write("\x93NUMPY", 6);
  uint8_t ver[2] = {1, 0}; os.write(reinterpret_cast<char*>(ver), 2);
  std::string descr = (t.dtype == 1) ? "<f2" : "<f4";
  std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (";
  for (int i = 0; i < t.rank; ++i) {
    header += std::to_string(t.shape[i]);
    if (i + 1 < t.rank) header += ", ";
  }
  if (t.rank == 1) header += ",";
  header += "), }";
  while ((10 + header.size()) % 64 != 63) header += ' ';
  header += '\n';
  uint16_t hlen = static_cast<uint16_t>(header.size());
  os.write(reinterpret_cast<char*>(&hlen), 2);
  os.write(header.data(), header.size());
  size_t nelems = 1; for (int i = 0; i < t.rank; ++i) nelems *= t.shape[i];
  size_t bytes  = nelems * (t.dtype == 1 ? 2 : 4);
  os.write(reinterpret_cast<char*>(t.data), bytes);
}

} // namespace

int main(int argc, char **argv) {
  std::vector<std::string> inputPaths;
  std::vector<std::string> outputPaths;
  std::string dumpDir;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if      (a == "--input"  && i + 1 < argc) inputPaths.push_back(argv[++i]);
    else if (a == "--output" && i + 1 < argc) outputPaths.push_back(argv[++i]);
    else if (a == "--dump-intermediates" && i + 1 < argc) dumpDir = argv[++i];
    else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
  }
  if (!dumpDir.empty()) network_set_dump_dir(dumpDir.c_str());

  std::vector<mlir::runtime::aclnn::TensorInfo> inputs;
  for (auto &p : inputPaths) inputs.push_back(loadNpy(p));

  std::vector<mlir::runtime::aclnn::TensorInfo> outputs(outputPaths.size());
  // For v1 we know there's exactly 1 output of shape 1x2x16x8 f16; rely on the network
  // to fill `outputs[0]` (which it allocates internally inside the kernel-launch path).
  // Simplest: pre-zero TensorInfo; the network calls hostLaunchAscendCKernel which
  // ::operator new's the result buffer and sets shape/dtype.
  network(inputs.data(), static_cast<int>(inputs.size()),
          outputs.data(), static_cast<int>(outputs.size()), nullptr);

  for (size_t i = 0; i < outputPaths.size(); ++i)
    saveNpy(outputPaths[i], outputs[i]);
  return 0;
}
```

- [ ] **Step 6: run.sh**

Write `examples/mixed-attn-e2e/run.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK="${WORK:-$SCRIPT_DIR/build_e2e}"

source /home/gser/Ascend/ascend-toolkit/set_env.sh
export PATH="$REPO/build/bin:$PATH"

mkdir -p "$WORK"
python3 "$SCRIPT_DIR/gen_inputs.py" --out-dir "$WORK"

cd "$REPO"
PYTHONPATH=python python3 python/network_runner.py \
  --input-network "$SCRIPT_DIR" \
  --inputs   "$WORK/q.npy"      "$WORK/scale.npy" "$WORK/bias.npy" \
             "$WORK/k.npy"      "$WORK/v.npy"     "$WORK/mask.npy" \
             "$WORK/init0.npy"  "$WORK/init_fa.npy" \
             "$WORK/proj.npy"   "$WORK/init1.npy" \
  --expected "$WORK/expected.npy" \
  --workdir  "$WORK" \
  --soc Ascend910B1
```

The runner needs to also g++ link `harness.cpp` into the binary alongside the
generated `network_host.cpp`. Update `python/runner_utils/build_host.py` to accept an
extra C++ source list and append it:

```python
def link_host(host_cpp, out_binary, repo_root, cann_home,
              cann_arch="x86_64-linux", extra_sources=()):
    ...
    cmd = [..., str(host_cpp), *map(str, extra_sources),
           str(repo_root / "lib/Runtime/AclnnOps.cpp"), ...]
```

And in `network_runner.py`'s phase 3 / 5 `link_host(...)` calls, pass
`extra_sources=[Path(args.input_network) / "harness.cpp"]` when `--input-network` mode
is in use. (For `--input-linalg` mode v1 doesn't ship a harness — that's a separate
example follow-up.)

- [ ] **Step 7: Run end-to-end**

```
chmod +x examples/mixed-attn-e2e/run.sh
bash examples/mixed-attn-e2e/run.sh
```

Expected:
- Phase 1 → `build_e2e/groups/{network.mlir, network.json, kernel_group0.mlir,
  kernel_group1.mlir}`
- Phase 2 → `build_e2e/artifacts/{kernel_group0,kernel_group1}/` with `.o` files
- Phase 3 → `build_e2e/network_test_default` runs, writes
  `build_e2e/intermediates_default/kernel_group{0,1}_{in,out}_*.npy`
- Phase 4 → `build_e2e/{kernel_group0,kernel_group1}_best.json`,
  `build_e2e/tilings_best.json`
- Phase 5 → `build_e2e/network_test` runs, writes `build_e2e/outputs/out0.npy`
- Final: `network.output[0]: max_diff=<small>  PASS`

If this fails:
- "aclInit failed" stderr line is expected and benign (hostMode kicks in)
- Numerical PASS depends on the host-mode FlashAttention matching the numpy reference
  — see Risks §6 of the spec; if they diverge, align the formula.

- [ ] **Step 8: Commit**

```
git add examples/mixed-attn-e2e/ python/runner_utils/build_host.py python/network_runner.py
git commit -m "feat(examples): mixed-attn-e2e end-to-end through network_runner

Hand-written mixed network.mlir (kernel_group0 → __aclnn_flash_attention →
kernel_group1) plus per-kernel linalg .mlir files, numpy gen_inputs.py
mirroring the host-mode FlashAttention math, and a thin harness.cpp that
loads/saves .npy and calls the runner-emitted network(). run.sh drives the
five-phase Python runner.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-Review Notes (for the executor)

- After every task, run `build/bin/llvm-lit -v test/` (or at least the directories you
  touched) to ensure no regression in the existing lit suite.
- The runner's `eval_block_dim` in Phase 3 mirrors `autotuner_main.cpp::evalBlockExpr`
  — if the autotuner's grammar evolves, keep these in sync.
- The `inlineBytes` / `outBytes` extension to `TensorBinding` (Task 5 step 4 note) is
  **the single most likely place this plan needs adjustment when implementing.** If
  modifying `TaskGraph.h` is more than ~30 LoC of API change, fall back to the
  temp-file shim described in that step.
- The host-mode CPU FlashAttention (`AclnnOps.cpp::run_FlashAttentionScore`'s
  `g_host_mode == true` branch) and `gen_inputs.py`'s reference must compute the same
  thing for Phase 5 to PASS. If they don't, decide whether to align gen_inputs.py to
  the C++ formula (cheap) or fix the C++ (expensive).
- After the example passes, suggest a follow-up plan for `examples/twochain-e2e/`
  (pure-AscendC, no aclnn) to isolate the AscendC-launch path from the host-mode
  fallback path. Out of scope for v1.
