# Symbol-aware `tensor.dim` CSE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an MLIR pass `afir-symbolic-dim-cse` that, after `--afir-symbolize-shapes`, merges all `tensor.dim` ops denoting the same dynamic-shape symbol into one canonical `tensor.dim` on the symbol's root arg.

**Architecture:** A `func::FuncOp` pass reads `afir.dim_symbols` (func attr, `id→(arg,dim)`) plus the serialized `afir.symbolic_shape`/`afir.symbolic_shapes` attrs. For each `tensor.dim %v, %c` it resolves the serialized symbol id of `(v, c)` via `parseSymExprList`; ops resolving to the same `Kind::Sym` id are an equivalence class. Each class with ≥2 members collapses to one fresh `tensor.dim %argRoot, %cRootDim` materialized at the entry block (block args dominate the whole body, so no dominance analysis is needed), with the rest RAUW'd and erased.

**Tech Stack:** C++ / MLIR (LLVM), TableGen pass registration, lit/FileCheck tests, the in-repo `SymExpr` symbolic library, Python pipeline driver (`network_runner.py`).

**Reference spec:** `docs/superpowers/specs/2026-06-05-symbol-aware-dim-cse-design.md`

---

## File Structure

- **Create** `lib/Dialect/AFIR/Transforms/AFIRSymbolicDimCSE.cpp` — the pass (~90 LOC). One responsibility: symbol-keyed `tensor.dim` dedup.
- **Modify** `include/Dialect/AFIR/Transforms/Passes.td` — add `AFIRSymbolicDimCSEPass` def.
- **Modify** `include/Dialect/AFIR/Transforms/Passes.h` — add `createAFIRSymbolicDimCSEPass()` decl.
- **Modify** `lib/Dialect/AFIR/Transforms/CMakeLists.txt` — add the source file.
- **Modify** `lib/Conversion/AutoFuse/Pipeline.cpp` — insert the pass after symbolize-shapes (registered pipeline).
- **Modify** `python/network_runner.py` — add `--afir-symbolic-dim-cse` to the network-level symbolize invocation.
- **Create** `test/Dialect/AFIR/Transforms/symbolicDimCSE.mlir` — lit test.

Build target: `cmake --build build --target afir-opt`. The build dir is `build/` (ninja).

---

## Task 1: Scaffold the pass (registered no-op)

Register the pass so `afir-opt --afir-symbolic-dim-cse` is a recognized (no-op) pass. This makes the next task's lit test runnable and lets it fail on behavior rather than on an unknown-pass error.

**Files:**
- Modify: `include/Dialect/AFIR/Transforms/Passes.td` (after the `AFIRSymbolizeShapesPass` def, ~line 76)
- Modify: `include/Dialect/AFIR/Transforms/Passes.h:22` (next to `createAFIRSymbolizeShapesPass`)
- Create: `lib/Dialect/AFIR/Transforms/AFIRSymbolicDimCSE.cpp`
- Modify: `lib/Dialect/AFIR/Transforms/CMakeLists.txt:8` (source list)

- [ ] **Step 1: Add the TableGen pass def**

In `include/Dialect/AFIR/Transforms/Passes.td`, immediately after the closing `}` of `def AFIRSymbolizeShapesPass` (before `def AFIRVerifySymbolicShapesPass`):

```tablegen
def AFIRSymbolicDimCSEPass : Pass<"afir-symbolic-dim-cse", "mlir::func::FuncOp"> {
  let summary = "Merge tensor.dim ops that denote the same shape symbol";
  let description = [{
    Runs after `afir-symbolize-shapes`.  Two `tensor.dim` ops can compute the
    same integer while querying different SSA values (e.g. `tensor.dim %arg0, 0`
    and `tensor.dim %1, 0` where both dim-0s are symbol `s0`); stock CSE cannot
    merge them because the operands differ.  Using `afir.dim_symbols` and the
    serialized `afir.symbolic_shape(s)` attrs, this pass groups such ops by
    symbol id and collapses each group (size >= 2) to a single canonical
    `tensor.dim` on the symbol's root block argument, placed at the entry block
    so it dominates every use.  Constant or compound (`s0*s1`) dims are left for
    plain canonicalization.  No-op when `afir.dim_symbols` is absent.
  }];
  let constructor = "mlir::createAFIRSymbolicDimCSEPass()";
}
```

- [ ] **Step 2: Add the constructor declaration**

In `include/Dialect/AFIR/Transforms/Passes.h`, after line 22 (`std::unique_ptr<Pass> createAFIRSymbolizeShapesPass();`):

```cpp
std::unique_ptr<Pass> createAFIRSymbolicDimCSEPass();
```

- [ ] **Step 3: Create the stub pass source**

Create `lib/Dialect/AFIR/Transforms/AFIRSymbolicDimCSE.cpp`:

```cpp
//===- AFIRSymbolicDimCSE.cpp - symbol-keyed tensor.dim CSE ------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
// Merges `tensor.dim` ops that denote the same dynamic-shape symbol into one
// canonical `tensor.dim` on the symbol's root block argument.  Reads the attrs
// produced by --afir-symbolize-shapes (afir.dim_symbols on the func,
// afir.symbolic_shape on args, afir.symbolic_shapes on ops).  See
// docs/superpowers/specs/2026-06-05-symbol-aware-dim-cse-design.md.
//
//===----------------------------------------------------------------------===//

#include "Analysis/SymbolicShape/SymExpr.h"
#include "Dialect/AFIR/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_AFIRSYMBOLICDIMCSEPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::afir::symshape;

namespace {
struct AFIRSymbolicDimCSEPass
    : public impl::AFIRSymbolicDimCSEPassBase<AFIRSymbolicDimCSEPass> {
  void runOnOperation() override {}
};
} // namespace

std::unique_ptr<Pass> mlir::createAFIRSymbolicDimCSEPass() {
  return std::make_unique<AFIRSymbolicDimCSEPass>();
}
```

- [ ] **Step 4: Add the source to CMake**

In `lib/Dialect/AFIR/Transforms/CMakeLists.txt`, add `AFIRSymbolicDimCSE.cpp` to the source list (right after `AFIRSymbolizeShapes.cpp`):

```cmake
  AFIRSymbolizeShapes.cpp
  AFIRSymbolicDimCSE.cpp
  AFIRVerifySymbolicShapes.cpp
```

(The library already links `AFIRSymbolicShape`, `MLIRTensorDialect`, `MLIRArithDialect`, `MLIRFuncDialect` — no new deps.)

- [ ] **Step 5: Build**

Run: `cmake --build build --target afir-opt`
Expected: builds clean; `afir-opt` links.

- [ ] **Step 6: Verify the pass is registered as a no-op**

Run:
```bash
echo 'func.func @f(%a: tensor<?xf32>) -> index { %c0 = arith.constant 0 : index %d = tensor.dim %a, %c0 : tensor<?xf32> return %d : index }' \
  | build/bin/afir-opt --afir-symbolic-dim-cse
```
Expected: the IR round-trips unchanged (pass recognized, does nothing). No "unknown pass" error.

- [ ] **Step 7: Commit**

```bash
git add include/Dialect/AFIR/Transforms/Passes.td include/Dialect/AFIR/Transforms/Passes.h \
        lib/Dialect/AFIR/Transforms/AFIRSymbolicDimCSE.cpp lib/Dialect/AFIR/Transforms/CMakeLists.txt
git commit -m "feat(afir): scaffold afir-symbolic-dim-cse pass (no-op)"
```

---

## Task 2: Implement and test the dedup logic (TDD)

**Files:**
- Create: `test/Dialect/AFIR/Transforms/symbolicDimCSE.mlir`
- Modify: `lib/Dialect/AFIR/Transforms/AFIRSymbolicDimCSE.cpp`

- [ ] **Step 1: Write the failing lit test**

Create `test/Dialect/AFIR/Transforms/symbolicDimCSE.mlir`. The input mimics post-symbolize IR: a func with `afir.dim_symbols`, an arg attr `afir.symbolic_shape`, and an op carrying `afir.symbolic_shapes`. Three `tensor.dim` all resolve to symbol `s0` (two via `%arg0`/`%arg1` args, one via an intermediate result `%0`); a fourth is a compound `(s0*s1)` dim that must be left untouched.

```mlir
// RUN: afir-opt %s -split-input-file --afir-symbolic-dim-cse | FileCheck %s

#map = affine_map<(d0) -> (d0)>

// Three dims all = s0 (root arg0,dim0) collapse to ONE tensor.dim on %arg0.
// CHECK-LABEL: func.func @merge_same_symbol
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D:.*]] = tensor.dim %arg0, %[[C0]]
// CHECK-NOT: tensor.dim
// CHECK: return
func.func @merge_same_symbol(
    %arg0: tensor<?xf32> {afir.symbolic_shape = "s0"},
    %arg1: tensor<?xf32> {afir.symbolic_shape = "s0"})
    -> (index, index, index)
    attributes {afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}]} {
  %c0 = arith.constant 0 : index
  %0 = linalg.generic {
      indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>)
      attrs = {afir.symbolic_shapes = ["s0"]} {
  ^bb0(%x: f32, %y: f32):
    linalg.yield %x : f32
  } -> tensor<?xf32>
  %d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %d1 = tensor.dim %arg1, %c0 : tensor<?xf32>
  %d2 = tensor.dim %0,    %c0 : tensor<?xf32>
  return %d0, %d1, %d2 : index, index, index
}

// -----

// A compound (s0*s1) dim is not a single symbol -> left untouched.
// CHECK-LABEL: func.func @leave_compound
// CHECK: tensor.dim %arg0, %c0
func.func @leave_compound(
    %arg0: tensor<?xf32> {afir.symbolic_shape = "(s0*s1)"})
    -> index
    attributes {afir.dim_symbols = [
        {arg = 0 : i64, dim = 0 : i64, id = 0 : i64},
        {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}]} {
  %c0 = arith.constant 0 : index
  %d = tensor.dim %arg0, %c0 : tensor<?xf32>
  return %d : index
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
build/bin/afir-opt test/Dialect/AFIR/Transforms/symbolicDimCSE.mlir -split-input-file --afir-symbolic-dim-cse \
  | externals/llvm-project/build/bin/FileCheck test/Dialect/AFIR/Transforms/symbolicDimCSE.mlir
```
Expected: FAIL on `@merge_same_symbol` — the no-op pass leaves three `tensor.dim` ops, so `CHECK-NOT: tensor.dim` fires.

- [ ] **Step 3: Implement the dedup logic**

Replace the `namespace { ... }` block in `lib/Dialect/AFIR/Transforms/AFIRSymbolicDimCSE.cpp` with the full implementation (keep the includes and the `createAFIRSymbolicDimCSEPass` factory below it unchanged):

```cpp
namespace {

struct Root {
  unsigned arg;
  unsigned dim;
};

/// Resolves the serialized symbol id that `tensor.dim v, idx` denotes, or
/// nullopt if v has no symbolic shape here, idx is out of range, or the dim is
/// not a bare symbol (constant or compound expr).
static std::optional<SymId> resolveSymbol(Value v, int64_t idx,
                                          func::FuncOp func) {
  StringRef serialized;
  if (auto barg = dyn_cast<BlockArgument>(v)) {
    if (barg.getOwner() != &func.getBody().front())
      return std::nullopt;
    auto attr = func.getArgAttrOfType<StringAttr>(barg.getArgNumber(),
                                                  "afir.symbolic_shape");
    if (!attr)
      return std::nullopt;
    serialized = attr.getValue();
  } else {
    Operation *def = v.getDefiningOp();
    if (!def)
      return std::nullopt;
    auto arr = def->getAttrOfType<ArrayAttr>("afir.symbolic_shapes");
    if (!arr)
      return std::nullopt;
    unsigned resNo = cast<OpResult>(v).getResultNumber();
    if (resNo >= arr.size())
      return std::nullopt;
    auto s = dyn_cast<StringAttr>(arr[resNo]);
    if (!s)
      return std::nullopt;
    serialized = s.getValue();
  }
  auto list = parseSymExprList(serialized);
  if (!list || idx < 0 || (size_t)idx >= list->size())
    return std::nullopt;
  const SymExpr &e = (*list)[idx];
  if (e.getKind() != SymExpr::Kind::Sym)
    return std::nullopt;
  return e.getSym();
}

struct AFIRSymbolicDimCSEPass
    : public impl::AFIRSymbolicDimCSEPassBase<AFIRSymbolicDimCSEPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal() || func.getBody().empty())
      return;
    auto dimSymbols = func->getAttrOfType<ArrayAttr>("afir.dim_symbols");
    if (!dimSymbols)
      return; // not symbolized / fully static.

    // id -> root (arg, dim).
    DenseMap<SymId, Root> idToRoot;
    for (Attribute a : dimSymbols) {
      auto d = cast<DictionaryAttr>(a);
      auto id = cast<IntegerAttr>(d.get("id")).getInt();
      auto arg = cast<IntegerAttr>(d.get("arg")).getInt();
      auto dim = cast<IntegerAttr>(d.get("dim")).getInt();
      idToRoot[(SymId)id] = {(unsigned)arg, (unsigned)dim};
    }

    // Group resolvable tensor.dim ops by symbol id.
    DenseMap<SymId, SmallVector<tensor::DimOp>> classes;
    func.walk([&](tensor::DimOp d) {
      std::optional<int64_t> idx = d.getConstantIndex();
      if (!idx)
        return;
      std::optional<SymId> sym = resolveSymbol(d.getSource(), *idx, func);
      if (!sym || !idToRoot.count(*sym))
        return;
      classes[*sym].push_back(d);
    });

    // Deterministic order: sort symbol ids.
    SmallVector<SymId> keys;
    for (auto &kv : classes)
      keys.push_back(kv.first);
    llvm::sort(keys);

    Block &entry = func.getBody().front();
    OpBuilder b(&getContext());
    DenseMap<int64_t, Value> idxConsts; // dedup index constants we create.
    auto getIdx = [&](int64_t v) -> Value {
      Value &slot = idxConsts[v];
      if (!slot) {
        b.setInsertionPointToStart(&entry);
        slot = b.create<arith::ConstantIndexOp>(func.getLoc(), v);
      }
      return slot;
    };

    for (SymId sym : keys) {
      SmallVector<tensor::DimOp> &members = classes[sym];
      if (members.size() < 2)
        continue; // nothing to merge.
      Root root = idToRoot[sym];
      Value rootArg = entry.getArgument(root.arg);
      Value cidx = getIdx((int64_t)root.dim);
      b.setInsertionPointAfter(cidx.getDefiningOp());
      auto canon = b.create<tensor::DimOp>(func.getLoc(), rootArg, cidx);
      for (tensor::DimOp m : members) {
        m.getResult().replaceAllUsesWith(canon.getResult());
        m.erase();
      }
    }
  }
};

} // namespace
```

- [ ] **Step 4: Rebuild**

Run: `cmake --build build --target afir-opt`
Expected: builds clean.

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
build/bin/afir-opt test/Dialect/AFIR/Transforms/symbolicDimCSE.mlir -split-input-file --afir-symbolic-dim-cse \
  | externals/llvm-project/build/bin/FileCheck test/Dialect/AFIR/Transforms/symbolicDimCSE.mlir
```
Expected: PASS (no output, exit 0). `@merge_same_symbol` now has a single `tensor.dim %arg0`; `@leave_compound` is unchanged.

- [ ] **Step 6: Run the full AFIR lit suite (no regression)**

Run: `cmake --build build --target check-afir`
Expected: all tests pass, including the pre-existing `symbolizeShapes.mlir`.

- [ ] **Step 7: Commit**

```bash
git add lib/Dialect/AFIR/Transforms/AFIRSymbolicDimCSE.cpp test/Dialect/AFIR/Transforms/symbolicDimCSE.mlir
git commit -m "feat(afir): implement symbol-keyed tensor.dim CSE + lit test"
```

---

## Task 3: Wire the pass into the pipelines

**Files:**
- Modify: `lib/Conversion/AutoFuse/Pipeline.cpp:97` (after `createAFIRSymbolizeShapesPass`)
- Modify: `python/network_runner.py:183` (the network-level symbolize call)

- [ ] **Step 1: Insert into the registered pipeline**

In `lib/Conversion/AutoFuse/Pipeline.cpp`, immediately after line 97
(`pm.addNestedPass<func::FuncOp>(mlir::createAFIRSymbolizeShapesPass());`), add:

```cpp
        // Collapse tensor.dim ops that denote the same dynamic symbol into one
        // canonical dim on the root arg (stock CSE can't, the sources differ).
        pm.addNestedPass<func::FuncOp>(mlir::createAFIRSymbolicDimCSEPass());
```

Confirm `createAFIRSymbolicDimCSEPass` is visible here — `Pipeline.cpp` already
includes the AFIR Transforms passes header (it calls
`createAFIRSymbolizeShapesPass`); no new include needed.

- [ ] **Step 2: Insert into the network-level driver**

In `python/network_runner.py`, change the symbolize invocation at line 183 from:

```python
        run([AFIR_OPT, "--afir-symbolize-shapes", str(folded),
             "-o", str(symbolized)])
```

to:

```python
        run([AFIR_OPT, "--afir-symbolize-shapes", "--afir-symbolic-dim-cse",
             str(folded), "-o", str(symbolized)])
```

- [ ] **Step 3: Rebuild**

Run: `cmake --build build --target afir-opt`
Expected: builds clean (Pipeline.cpp recompiles).

- [ ] **Step 4: Commit**

```bash
git add lib/Conversion/AutoFuse/Pipeline.cpp python/network_runner.py
git commit -m "feat(afir): run afir-symbolic-dim-cse after symbolize in both pipelines"
```

---

## Task 4: End-to-end regression (the real verification)

This confirms both correctness (provenance independence — rewriting which arg a
dim queries doesn't change results) and the op-count win on the demonstrating
graph.

**Files:** none (verification only).

- [ ] **Step 1: gpt2-dyn — confirm the 9→1 collapse**

Regenerate `model_symbolized.mlir` through the new driver and count dims:

```bash
NETWORK_RUNNER_SKIP_AUTOTUNE=1 MAX_PHASE=1 bash examples/gpt2-dyn-e2e/run.sh
grep -c "tensor.dim " examples/gpt2-dyn-e2e/build_e2e/model_symbolized.mlir
```
Expected: `5` (was 9). The 5 identical `tensor.dim %arg0, %c1` collapse to one;
the 4 remaining dims query `call @__aclnn_layer_norm` results, which
`AFIRSymbolizeShapes` does not annotate (it skips `func.call`), so the pass can't
resolve their symbol. (The earlier "9→1" estimate was wrong — see the spec's
"Coverage caveat".) The cross-value capability is still exercised by the lit
test's `linalg.generic`-result case.

- [ ] **Step 2: gpt2-dyn — full phase-5 e2e, max_diff unchanged**

```bash
NETWORK_RUNNER_SKIP_AUTOTUNE=1 MAX_PHASE=5 bash examples/gpt2-dyn-e2e/run.sh
```
Expected: PASS, `max_diff` ≈ 7.15e-7 (unchanged from the pre-change baseline in
the project memory). This empirically confirms provenance independence — the
safety concern from the spec — better than any static grep.

- [ ] **Step 3: two-elewise-dyn — no regression**

```bash
NETWORK_RUNNER_SKIP_AUTOTUNE=1 bash examples/two-elewise-dyn-e2e/run.sh
```
Expected: PASS (this graph has nothing to merge — every dim was already on
`%arg0` — so it exercises the "≥2 members only" guard staying inert).

- [ ] **Step 4: gelu-dyn — no regression**

```bash
NETWORK_RUNNER_SKIP_AUTOTUNE=1 bash examples/gelu-dyn-e2e/run.sh
```
Expected: PASS, `max_diff` ≈ 2.38e-7 (unchanged).

- [ ] **Step 5: static graph unaffected (early-return path)**

```bash
MAX_PHASE=5 bash examples/two-elewise-e2e/run.sh
```
Expected: PASS. The static func has no `afir.dim_symbols`, so the pass returns
immediately.

- [ ] **Step 6: Commit a short regression note**

Append a one-paragraph result summary to the design spec (under a new
`## Results` heading) recording the 9→1 collapse and the unchanged max_diffs,
then:

```bash
git add docs/superpowers/specs/2026-06-05-symbol-aware-dim-cse-design.md
git commit -m "docs: record symbol-aware-dim-cse regression results"
```

---

## Self-Review notes

- **Spec coverage:** standalone pass (Task 1–2), fresh-anchor canonicalization with
  entry-block placement (Task 2 Step 3), pipeline wiring at both symbolize sites
  (Task 3), lit incl. compound-negative + gpt2-dyn 9→5 + e2e provenance check
  (Task 2, Task 4). All spec sections map to a task.
- **Provenance safety:** verified empirically by Task 4 Step 2 (gpt2-dyn e2e
  max_diff unchanged), which is stronger than the spec's "grep tile-fuse" idea.
- **Type consistency:** `resolveSymbol` signature and the `Root{arg,dim}` /
  `idToRoot` / `classes` names are identical across the test-fail step and the
  implementation step. `getConstantIndex()` returns `std::optional<int64_t>`
  (as used in `AFIRSymbolizeShapes.cpp:103`). `parseSymExprList` / `SymExpr::Kind::Sym`
  / `getSym()` match `Analysis/SymbolicShape/SymExpr.h`.
- **Determinism:** symbol ids sorted before mutation so emitted op order is
  stable for FileCheck.
