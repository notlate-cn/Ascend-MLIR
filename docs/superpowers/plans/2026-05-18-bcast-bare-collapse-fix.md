# Bcast Bare-Collapse `memref.collapse_shape` Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `bcast-trailing-e2e` and `bcast-leading-e2e` PASS by extending `FlattenGMPtrPass` to flatten `ascendc.global_tensor.set_global_buffer` ops whose buffer is a bare `memref.collapse_shape` (no surrounding `memref.subview`).

**Architecture:** `FlattenGMPtrPass` today only flattens SGB ops whose buffer is reached through `memref.cast → memref.subview`. The bcast pattern "load the whole broadcast operand" produces `memref.collapse_shape %arg : memref<HxW>xf32 into memref<H*W>xf32` directly into SGB — no subview. Extend the pass to also handle this case: emit `emitasc.reinterpret_cast` on the underlying block arg with `offset = 0`, then DCE the collapse_shape via the existing dead-view sweep. No new infrastructure; surgical addition to `flattenGMPtr`.

**Tech Stack:** MLIR, AscendC/EmitAsc dialects, C++.

---

## File Structure

- **Modify:** `lib/Conversion/AscendCPrepareForEmit/FlattenGMPtrPass.cpp` — extend the SGB collector + rewriter to accept bare-collapse / bare-blockarg buffers.
- **Modify:** `test/Conversion/AscendCPrepareForEmit/flatten-gm-ptr.mlir` — add lit case covering bare `memref.collapse_shape` and bare block arg.
- **Verify:** `examples/bcast-leading-e2e/run.sh`, `examples/bcast-trailing-e2e/run.sh` — gate via existing run scripts.

---

## Task 1: Reproduce the bug and capture failing state

**Files:**
- Read: `examples/bcast-leading-e2e/bcast_leading_kernel.mlir`
- Read: `lib/Conversion/AscendCPrepareForEmit/FlattenGMPtrPass.cpp:114-118,166-196`

- [ ] **Step 1: Reproduce the failure to confirm starting state**

```bash
cd /home/gser/code/Ascend-MLIR && source examples/env.sh
export PATH=$PWD/build/bin:$PATH
cd examples/bcast-leading-e2e && bash run.sh 2>&1 | tail -10
```

Expected: error `'memref.collapse_shape' op unable to find printer for op` at `bcast_leading_kernel.mlir:151`.

- [ ] **Step 2: Inspect the surviving collapse_shape in the kernel**

```bash
grep -n "collapse_shape" examples/bcast-leading-e2e/bcast_leading_kernel.mlir
```

Expected: 3 hits — line 151 defines `%collapse_shape = memref.collapse_shape %arg1 [[0, 1]] : memref<4x32xf32> into memref<128xf32>`; lines 199 and 238 consume it directly in `ascendc.global_tensor.set_global_buffer`.

- [ ] **Step 3: Confirm `peelCastsToSubview` is the gatekeeper that drops these ops**

Read `FlattenGMPtrPass.cpp:114-118` — `peelCastsToSubview` walks through `memref.cast` only, returns `getDefiningOp<SubViewOp>()`. For a bare `collapse_shape` it returns null, so the SGB op at `:167-170` is never collected.

---

## Task 2: Add lit test for bare collapse_shape

**Files:**
- Modify: `test/Conversion/AscendCPrepareForEmit/flatten-gm-ptr.mlir`

- [ ] **Step 1: Append a failing test case for bare collapse_shape**

Add to the end of `test/Conversion/AscendCPrepareForEmit/flatten-gm-ptr.mlir`:

```mlir
// Bare memref.collapse_shape (no surrounding subview): the bcast pattern
// "load the whole operand" feeds collapse_shape directly into set_global_buffer.
// CHECK-LABEL: func.func @test_bare_collapse
// CHECK: emitasc.reinterpret_cast
// CHECK-NOT: memref.collapse_shape
// CHECK-NOT: memref.subview
func.func @test_bare_collapse(%arg0: memref<4x32xf32>) {
  %gt = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
  %col = memref.collapse_shape %arg0 [[0, 1]]
       : memref<4x32xf32> into memref<128xf32>
  ascendc.global_tensor.set_global_buffer %gt, %col
    : !ascendc.global_tensor<*xf32>, memref<128xf32>
  return
}

// Bare block arg with identity layout, directly into set_global_buffer.
// CHECK-LABEL: func.func @test_bare_blockarg
// CHECK: emitasc.reinterpret_cast
// CHECK-NOT: memref.collapse_shape
func.func @test_bare_blockarg(%arg0: memref<128xf32>) {
  %gt = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
  ascendc.global_tensor.set_global_buffer %gt, %arg0
    : !ascendc.global_tensor<*xf32>, memref<128xf32>
  return
}
```

- [ ] **Step 2: Run the lit test and verify it fails as expected**

```bash
cd build && ninja check-afir 2>&1 | grep -A4 "flatten-gm-ptr"
```

Expected: FAIL on the two new test cases (CHECK directives won't be satisfied — `emitasc.reinterpret_cast` is missing, `memref.collapse_shape` survives).

---

## Task 3: Extend FlattenGMPtrPass to flatten bare-collapse / bare-blockarg SGB ops

**Files:**
- Modify: `lib/Conversion/AscendCPrepareForEmit/FlattenGMPtrPass.cpp:163-196`

- [ ] **Step 1: Replace the SGB collector + rewriter to cover the no-subview case**

In `lib/Conversion/AscendCPrepareForEmit/FlattenGMPtrPass.cpp`, replace lines `163-196` (the entire "── 7b. Flatten subview + set_global_buffer ──" block) with:

```cpp
  // ── 7b. Flatten set_global_buffer ─────────────────────────────────────────
  // Three buffer shapes are accepted (in priority order):
  //   1. cast?-subview chain  → flatten through resolveGMChain (offset != 0)
  //   2. cast?-collapse chain → reinterpret on underlying block arg, offset 0
  //   3. bare block arg       → reinterpret directly, offset 0
  // ascir-translate cannot print memref.collapse_shape; without (2) the bcast
  // pattern (whole-operand load → collapse_shape feeds SGB directly) leaks a
  // collapse_shape into emission and crashes the printer.

  auto peelCasts = [](Value v) {
    while (auto castOp = v.getDefiningOp<memref::CastOp>())
      v = castOp.getSource();
    return v;
  };

  SmallVector<GlobalTensorSetGlobalBufferOp> setGlobalBufferOps;
  func.walk([&](GlobalTensorSetGlobalBufferOp op) {
    setGlobalBufferOps.push_back(op);
  });

  for (GlobalTensorSetGlobalBufferOp sgbOp : setGlobalBufferOps) {
    Value buf = peelCasts(sgbOp.getBuffer());

    OpBuilder b(sgbOp);
    Location loc = sgbOp.getLoc();

    if (auto subview = buf.getDefiningOp<memref::SubViewOp>()) {
      auto [ba, acc] = resolveGMChain(subview.getResult(), b, loc);
      if (!ba)
        continue;
      Value flatOffsetI32 = b.create<arith::IndexCastOp>(loc, i32Ty, acc);
      Type elemTy = cast<MemRefType>(ba.getType()).getElementType();
      Value flatBase =
          b.create<emitasc::ReinterpretCastOp>(loc, mkFlatTy(elemTy), ba);
      b.create<GlobalTensorSetGlobalBufferOp>(loc, sgbOp.getTensor(), flatBase,
                                              flatOffsetI32);
      sgbOp.erase();
      if (subview.use_empty())
        subview.erase();
      continue;
    }

    // No subview: walk through collapse_shape to the underlying block arg.
    Value ptrSrc = buf;
    while (true) {
      if (auto colOp = ptrSrc.getDefiningOp<memref::CollapseShapeOp>()) {
        ptrSrc = colOp.getSrc();
        continue;
      }
      if (auto castOp = ptrSrc.getDefiningOp<memref::CastOp>()) {
        ptrSrc = castOp.getSource();
        continue;
      }
      break;
    }
    auto ba = dyn_cast<BlockArgument>(ptrSrc);
    if (!ba)
      continue;
    Type elemTy = cast<MemRefType>(ba.getType()).getElementType();
    Value flatBase =
        b.create<emitasc::ReinterpretCastOp>(loc, mkFlatTy(elemTy), ba);
    Value zeroI32 = b.create<arith::ConstantOp>(
        loc, i32Ty, b.getIntegerAttr(i32Ty, 0));
    b.create<GlobalTensorSetGlobalBufferOp>(loc, sgbOp.getTensor(), flatBase,
                                            zeroI32);
    sgbOp.erase();
  }
```

The trailing `// Rewrite memref.dim %collapse_shape, %const ...` block at original line `198` onward (the dead-view sweep) remains unchanged — it is now exercised more broadly because more `collapse_shape` ops become dead.

- [ ] **Step 2: Rebuild**

```bash
cd build && ninja afir-opt afir-translate 2>&1 | tail -3
```

Expected: build succeeds.

- [ ] **Step 3: Run the lit test again — should pass now**

```bash
cd build && ninja check-afir 2>&1 | tail -10
```

Expected: all lit tests PASS, including the new `test_bare_collapse` and `test_bare_blockarg`.

---

## Task 4: Verify bcast-leading-e2e and bcast-trailing-e2e pass end-to-end

**Files:**
- Run: `examples/bcast-leading-e2e/run.sh`
- Run: `examples/bcast-trailing-e2e/run.sh`

- [ ] **Step 1: Run bcast-leading-e2e**

```bash
cd /home/gser/code/Ascend-MLIR && source examples/env.sh
export PATH=$PWD/build/bin:$PATH
cd examples/bcast-leading-e2e && bash run.sh 2>&1 | tail -10
```

Expected: ends with `session.validation=pass`. No `memref.collapse_shape` printer error.

- [ ] **Step 2: Run bcast-trailing-e2e**

```bash
cd /home/gser/code/Ascend-MLIR && source examples/env.sh
export PATH=$PWD/build/bin:$PATH
cd examples/bcast-trailing-e2e && bash run.sh 2>&1 | tail -10
```

Expected: ends with `session.validation=pass`.

- [ ] **Step 3: Confirm no regression in adjacent bcast / shape-preserving e2e**

```bash
cd /home/gser/code/Ascend-MLIR && source examples/env.sh
export PATH=$PWD/build/bin:$PATH
for d in examples/add-broadcast-concat examples/two-elewise-e2e examples/dyn-bucketed-e2e; do
  echo "=== $d ==="
  (cd "$d" && bash run.sh 2>&1 | tail -3) || break
done
```

Expected: all three end in `session.validation=pass`.

---

## Task 5: Commit

**Files:**
- Modify: `lib/Conversion/AscendCPrepareForEmit/FlattenGMPtrPass.cpp`
- Modify: `test/Conversion/AscendCPrepareForEmit/flatten-gm-ptr.mlir`

- [ ] **Step 1: Stage and commit**

```bash
cd /home/gser/code/Ascend-MLIR
git add lib/Conversion/AscendCPrepareForEmit/FlattenGMPtrPass.cpp \
        test/Conversion/AscendCPrepareForEmit/flatten-gm-ptr.mlir
git commit -m "$(cat <<'EOF'
fix(flatten-gm-ptr): flatten SGB with bare collapse_shape / block arg

FlattenGMPtrPass previously only rewrote ascendc.global_tensor.set_global_buffer
ops whose buffer reached a memref.subview through optional memref.casts.
The bcast pattern "load the whole broadcast operand" produces a bare
memref.collapse_shape fed directly into SGB (no surrounding subview), so
those SGBs were dropped on the floor and the collapse_shape survived into
emission.  ascir-translate has no printer for memref.collapse_shape, so
bcast-leading-e2e / bcast-trailing-e2e crashed at translate time with
"'memref.collapse_shape' op unable to find printer for op".

Extend the pass to also flatten SGBs whose buffer is cast?-collapse_shape*
or a bare block arg, emitting an emitasc.reinterpret_cast on the underlying
arg with offset 0.  The existing dead-view cleanup at the tail of the pass
then DCEs the dead collapse_shape.

Closes the bcast-trailing-e2e / bcast-leading-e2e gates.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Expected: clean commit, working tree clean (modulo unrelated tracked changes).

- [ ] **Step 2: Verify final state**

```bash
git log -1 --stat
```

Expected: commit shows the two edits above; nothing else.
