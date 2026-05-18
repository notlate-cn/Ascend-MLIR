# Ascend Runtime Graph And Queue Lifetime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the runtime-session mainline by proving larger generated DAGs run numerically and by making AscendC queue tensor lifetime a checked contract.

**Architecture:** Add a deterministic queue-lifetime checker first, then wire it into full-pipeline and example verification. Expand runtime acceptance with one rank-mixed two-kernel DAG, one three-kernel DAG, and one RMSNorm reduction-core demo, all using aligned `N=64` paths so the old DataCopy first-lane issue cannot hide behind a tail fallback.

**Tech Stack:** MLIR LIT/FileCheck, Python numpy data generation, Bash example pipelines, `afir-opt`, `afir-translate`, `runtime-session`, AscendC queue ops, xvm verification through `ssh xvm@orb`.

---

## Scope

This plan covers the agreed priority 1 and priority 2 work:

1. Complex graph runtime acceptance:
   - rank-mixed two-kernel DAG
   - three-kernel DAG
   - RMSNorm reduction-core runtime demo
2. Queue lifecycle audit:
   - static checker for `ascendc.que_bind.deque_tensor` ownership
   - integration into example and LIT gates
   - targeted fixes for any path that dequeues without a matching `free_tensor`

This plan does not implement new math lowerings for `math.exp` or `math.rsqrt`. Full softmax/RMSNorm runtime demos should follow after those ops lower to supported AscendC kernels. This plan still adds a real RMSNorm subgraph slice: `sum(x*x)` per row, which exercises vector multiply plus row reduction.

## File Structure

- Create: `test/tools/check_ascend_queue_lifetime.py`
  - Standalone checker for generated MLIR containing AscendC queue operations.
- Create: `test/tools/ascend-queue-lifetime.mlir`
  - RED/GREEN lit coverage for the checker.
- Modify: `test/tools/examples/example_pipelines.sh`
  - Run the lifetime checker on generated example IR.
  - Add new examples to the suite.
- Modify: `test/tools/examples/example-pipelines.mlir`
  - Update expected example count.
- Create: `examples/two-kernel-rank-mix-dag/step0_input.mlir`
  - `kernel_a: a[N] + b[N] -> mid[N]`
  - `kernel_b: mid[N] broadcast over K, multiply scale[N,K] -> out[N,K]`
- Create: `examples/two-kernel-rank-mix-dag/gen_data.py`
  - Generates `input_a.npy`, `input_b.npy`, `input_scale.npy`, and `output_expected.npy`.
- Create: `examples/two-kernel-rank-mix-dag/run-mainline.sh`
  - Full Phase 0 -> Phase 5 -> runtime-session DAG script.
- Create: `examples/two-kernel-rank-mix-dag/run.sh`
  - Thin wrapper to `run-mainline.sh`.
- Create: `examples/three-kernel-dag/step0_input.mlir`
  - `kernel_a: a + b -> mid0`
  - `kernel_b: mid0 * c -> mid1`
  - `kernel_c: mid1 + d -> out`
- Create: `examples/three-kernel-dag/gen_data.py`
  - Generates four inputs and expected output.
- Create: `examples/three-kernel-dag/run-mainline.sh`
  - Full Phase 0 -> Phase 5 -> runtime-session DAG script.
- Create: `examples/three-kernel-dag/run.sh`
  - Thin wrapper to `run-mainline.sh`.
- Create: `examples/rmsnorm-reduction-core/step0_input.mlir`
  - `squared = x * x`, `sum_sq = reduce_sum(squared, axis=1)`.
- Create: `examples/rmsnorm-reduction-core/gen_data.py`
  - Generates f16 input and f16 row-sum expected output with f32 accumulation.
- Create: `examples/rmsnorm-reduction-core/run-mainline.sh`
  - Full Phase 0 -> Phase 5 -> runtime-session script.
- Create: `examples/rmsnorm-reduction-core/run.sh`
  - Thin wrapper to `run-mainline.sh`.
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
  - Add a short verification record after all gates pass.

## Task 1: Add Queue Lifetime Checker

**Files:**
- Create: `test/tools/check_ascend_queue_lifetime.py`
- Create: `test/tools/ascend-queue-lifetime.mlir`

- [ ] **Step 1: Write the checker lit test before the checker exists**

Create `test/tools/ascend-queue-lifetime.mlir`:

```mlir
// RUN: split-file %s %t
// RUN: not python3 %S/check_ascend_queue_lifetime.py %t/bad.mlir 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: python3 %S/check_ascend_queue_lifetime.py %t/good.mlir | FileCheck %s --check-prefix=GOOD

// BAD: missing free_tensor for deque result %1 from queue %q
// GOOD: queue_lifetime.ok good.mlir deque=1 free=1

//--- bad.mlir
module {
  func.func @bad(%arg0: !ascendc.queue<vecin, 1>) {
    %1 = ascendc.que_bind.deque_tensor %arg0 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
    return
  }
}

//--- good.mlir
module {
  func.func @good(%arg0: !ascendc.queue<vecin, 1>) {
    %1 = ascendc.que_bind.deque_tensor %arg0 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
    ascendc.que_bind.free_tensor %arg0, %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
    return
  }
}
```

- [ ] **Step 2: Verify RED**

Run on xvm:

```bash
rsync -aR ./test/tools/ascend-queue-lifetime.mlir xvm@orb:/home/niu/code/Ascend-MLIR/
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/tools/ascend-queue-lifetime.mlir'
```

Expected: FAIL because `test/tools/check_ascend_queue_lifetime.py` does not exist.

- [ ] **Step 3: Implement the checker**

Create `test/tools/check_ascend_queue_lifetime.py`:

```python
#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path


DEQUE_RE = re.compile(
    r"(?P<tensor>%[A-Za-z0-9_.$-]+)\s*=\s*"
    r"ascendc\.que_bind\.deque_tensor\s+(?P<queue>%[A-Za-z0-9_.$-]+)\b"
)
FREE_RE = re.compile(
    r"ascendc\.que_bind\.free_tensor\s+"
    r"(?P<queue>%[A-Za-z0-9_.$-]+)\s*,\s*(?P<tensor>%[A-Za-z0-9_.$-]+)\b"
)


def check_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    dequeues: dict[tuple[str, str], int] = {}
    frees: dict[tuple[str, str], list[int]] = {}

    for line_no, line in enumerate(text.splitlines(), 1):
        deque_match = DEQUE_RE.search(line)
        if deque_match:
            key = (deque_match.group("queue"), deque_match.group("tensor"))
            dequeues[key] = line_no

        free_match = FREE_RE.search(line)
        if free_match:
            key = (free_match.group("queue"), free_match.group("tensor"))
            frees.setdefault(key, []).append(line_no)

    errors: list[str] = []
    for key, line_no in sorted(dequeues.items(), key=lambda item: item[1]):
        if key not in frees:
            queue, tensor = key
            errors.append(
                f"{path}:{line_no}: missing free_tensor for deque result "
                f"{tensor} from queue {queue}"
            )
            continue
        first_free = frees[key][0]
        if first_free <= line_no:
            queue, tensor = key
            errors.append(
                f"{path}:{line_no}: free_tensor for deque result {tensor} "
                f"from queue {queue} appears before the deque"
            )

    for key, line_numbers in sorted(frees.items(), key=lambda item: item[1][0]):
        if key not in dequeues:
            continue
        if len(line_numbers) > 1:
            queue, tensor = key
            joined = ",".join(str(n) for n in line_numbers)
            errors.append(
                f"{path}:{joined}: duplicate free_tensor for deque result "
                f"{tensor} from queue {queue}"
            )

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(
        f"queue_lifetime.ok {path.name} deque={len(dequeues)} "
        f"free={sum(1 for key in frees if key in dequeues)}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    status = 0
    for path in args.paths:
        status |= check_file(path)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
rsync -aR ./test/tools/check_ascend_queue_lifetime.py ./test/tools/ascend-queue-lifetime.mlir xvm@orb:/home/niu/code/Ascend-MLIR/
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v test/tools/ascend-queue-lifetime.mlir'
```

Expected: PASS 1/1.

- [ ] **Step 5: Commit**

```bash
git add test/tools/check_ascend_queue_lifetime.py test/tools/ascend-queue-lifetime.mlir
git commit -m "test: add AscendC queue lifetime checker"
```

## Task 2: Wire Queue Lifetime Checker Into Full Pipeline Gates

**Files:**
- Modify: `test/tools/examples/example_pipelines.sh`
- Create: `test/Conversion/ascend-queue-lifetime-full-pipeline.mlir`

- [ ] **Step 1: Add a full-pipeline generated-IR gate**

Create `test/Conversion/ascend-queue-lifetime-full-pipeline.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower --ascend-parallelize --ascend-prepare-for-emit --ascend-canonicalize-cann-signature -o %t.mlir
// RUN: python3 %S/../tools/check_ascend_queue_lifetime.py %t.mlir | FileCheck %s

// CHECK: queue_lifetime.ok

#identity = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

module {
  func.func @queue_lifetime_full_pipeline(
      %arg0: tensor<64x64xf16>,
      %arg1: tensor<64x64xf16>,
      %arg2: tensor<64xf16>) -> tensor<64xf16> {
    %mul_init = tensor.empty() : tensor<64x64xf16>
    %mul = linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>)
      outs(%mul_init : tensor<64x64xf16>) {
    ^bb0(%x: f16, %y: f16, %out: f16):
      %v = arith.mulf %x, %y : f16
      linalg.yield %v : f16
    } -> tensor<64x64xf16>

    %zero = arith.constant 0.0 : f16
    %sum_empty = tensor.empty() : tensor<64xf16>
    %sum_init = linalg.fill ins(%zero : f16)
      outs(%sum_empty : tensor<64xf16>) -> tensor<64xf16>
    %sum = linalg.generic {
      indexing_maps = [#identity, #row],
      iterator_types = ["parallel", "reduction"]
    } ins(%mul : tensor<64x64xf16>)
      outs(%sum_init : tensor<64xf16>) {
    ^bb0(%x: f16, %acc: f16):
      %v = arith.addf %x, %acc : f16
      linalg.yield %v : f16
    } -> tensor<64xf16>

    %out_init = tensor.empty() : tensor<64xf16>
    %out = linalg.generic {
      indexing_maps = [#row, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%sum, %arg2 : tensor<64xf16>, tensor<64xf16>)
      outs(%out_init : tensor<64xf16>) {
    ^bb0(%x: f16, %bias: f16, %out_arg: f16):
      %v = arith.addf %x, %bias : f16
      linalg.yield %v : f16
    } -> tensor<64xf16>

    return %out : tensor<64xf16>
  }
}
```

- [ ] **Step 2: Verify focused gate**

Run:

```bash
rsync -aR ./test/tools/check_ascend_queue_lifetime.py ./test/Conversion/ascend-queue-lifetime-full-pipeline.mlir xvm@orb:/home/niu/code/Ascend-MLIR/
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-queue-lifetime-full-pipeline.mlir'
```

Expected: PASS 1/1. If it fails with a missing `free_tensor`, fix the producer path before continuing.

- [ ] **Step 3: Wire checker into example suite**

Modify `test/tools/examples/example_pipelines.sh` inside `run_example()` after the success-marker checks and before `echo "PASS [${name}]"`:

```bash
  local generated_ir=(
    "${REPO_ROOT}/examples/${name}/build_mainline/step5_ascendc.mlir"
    "${REPO_ROOT}/examples/${name}/build_mainline/step6_ascendc.mlir"
    "${REPO_ROOT}/examples/${name}/build_mainline/step7_kernel_ir.mlir"
    "${REPO_ROOT}/examples/${name}/build_mainline/step8_kernel_ir.mlir"
    "${REPO_ROOT}/examples/${name}/build_mainline/step9_cann.mlir"
  )
  local existing_ir=()
  for ir in "${generated_ir[@]}"; do
    if [[ -f "${ir}" ]]; then
      existing_ir+=("${ir}")
    fi
  done
  if ((${#existing_ir[@]} > 0)); then
    if ! python3 "${REPO_ROOT}/test/tools/check_ascend_queue_lifetime.py" "${existing_ir[@]}" >>"${log_file}" 2>&1; then
      echo "FAIL [${name}] queue lifetime check failed"
      tail -n 80 "${log_file}" || true
      failures+=("${name}:queue-lifetime")
      return
    fi
  fi
```

- [ ] **Step 4: Verify examples still pass**

Run:

```bash
rsync -aR ./test/tools/check_ascend_queue_lifetime.py ./test/tools/examples/example_pipelines.sh xvm@orb:/home/niu/code/Ascend-MLIR/
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/tools/examples/example-pipelines.mlir'
```

Expected: existing 7 examples pass and every generated IR file reports `queue_lifetime.ok` in the per-example log.

- [ ] **Step 5: Commit**

```bash
git add test/Conversion/ascend-queue-lifetime-full-pipeline.mlir test/tools/examples/example_pipelines.sh
git commit -m "test: gate examples on AscendC queue lifetime"
```

## Task 3: Add Rank-Mixed Two-Kernel DAG Example

**Files:**
- Create: `examples/two-kernel-rank-mix-dag/step0_input.mlir`
- Create: `examples/two-kernel-rank-mix-dag/gen_data.py`
- Create: `examples/two-kernel-rank-mix-dag/run-mainline.sh`
- Create: `examples/two-kernel-rank-mix-dag/run.sh`
- Modify: `test/tools/examples/example_pipelines.sh`
- Modify: `test/tools/examples/example-pipelines.mlir`

- [ ] **Step 1: Create the high-level IR**

Create `examples/two-kernel-rank-mix-dag/step0_input.mlir`:

```mlir
// Rank-mixed two-kernel DAG.
//
// kernel_a: a[N] + b[N] -> mid[N]
// kernel_b: mid[N] broadcast over K, multiply scale[N,K] -> out[N,K]

module {
  func.func @kernel_a(%a : tensor<?xf16>, %b : tensor<?xf16>) -> tensor<?xf16> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %a, %c0 : tensor<?xf16>
    %init = tensor.empty(%n) : tensor<?xf16>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<?xf16>, tensor<?xf16>)
      outs(%init : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %sum = arith.addf %x, %y : f16
      linalg.yield %sum : f16
    } -> tensor<?xf16>
    return %out : tensor<?xf16>
  }

  func.func @kernel_b(%mid : tensor<?xf16>, %scale : tensor<?x?xf16>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = tensor.dim %scale, %c0 : tensor<?x?xf16>
    %k = tensor.dim %scale, %c1 : tensor<?x?xf16>
    %init = tensor.empty(%n, %k) : tensor<?x?xf16>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%mid, %scale : tensor<?xf16>, tensor<?x?xf16>)
      outs(%init : tensor<?x?xf16>) {
    ^bb0(%x: f16, %s: f16, %o: f16):
      %prod = arith.mulf %x, %s : f16
      linalg.yield %prod : f16
    } -> tensor<?x?xf16>
    return %out : tensor<?x?xf16>
  }
}
```

- [ ] **Step 2: Create data generator**

Create `examples/two-kernel-rank-mix-dag/gen_data.py`:

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a = rng.uniform(-1.0, 1.0, size=(args.n,)).astype(np.float16)
    b = rng.uniform(0.0, 2.0, size=(args.n,)).astype(np.float16)
    scale = rng.uniform(0.5, 1.5, size=(args.n, args.k)).astype(np.float16)
    expected = ((a + b)[:, None] * scale).astype(np.float16)

    np.save(out_dir / "input_a.npy", a)
    np.save(out_dir / "input_b.npy", b)
    np.save(out_dir / "input_scale.npy", scale)
    np.save(out_dir / "output_expected.npy", expected)

    print(f"N={args.n}, K={args.k}, seed={args.seed}")
    print(f"input_a:     {a.shape} {a.dtype}")
    print(f"input_b:     {b.shape} {b.dtype}")
    print(f"input_scale: {scale.shape} {scale.dtype}")
    print(f"output:      {expected.shape} {expected.dtype}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create run scripts**

Create the first version by copying the current two-kernel runner:

```bash
mkdir -p examples/two-kernel-rank-mix-dag
cp examples/two-kernel-dag/run-mainline.sh examples/two-kernel-rank-mix-dag/run-mainline.sh
cp examples/two-kernel-dag/run.sh examples/two-kernel-rank-mix-dag/run.sh
```

Then edit `examples/two-kernel-rank-mix-dag/run-mainline.sh` with these concrete changes:

1. Replace the header computation comment with:

```bash
#   kernel_a: a[N] + b[N] -> mid[N]
#   kernel_b: mid[N] * scale[N,K] -> out[N,K]
```

2. Add `K=33` after `N=64`.
3. Add argument parsing:

```bash
    --k)
      K="$2"
      shift 2
      ;;
```

4. Print `shape.K=$K` after `shape.N=$N`.
5. In the data generation stage, call:

```bash
"$PYTHON" "$DIR/gen_data.py" --n "$N" --k "$K" --seed "$SEED" --out-dir "$BUILD_DIR"
```

6. Replace the tiling shape map with:

```python
shape_values = {
    "arg0_dim0": n,
    "arg1_dim0": n,
    "arg1_dim1": k,
}
```

and pass `"$PHASE5_TILING_SPACE" "$N" "$K"` to that Python snippet.

7. In `kernel_b` manifest inputs, replace input `c` with:

```json
{ "name": "scale", "path": "${BUILD_DIR}/input_scale.npy" }
```

8. In `kernel_b` manifest outputs, replace shape `[${N}]` with:

```json
"shape": [${N}, ${K}]
```

- [ ] **Step 4: Verify focused example**

Run:

```bash
rsync -aR ./examples/two-kernel-rank-mix-dag ./test/tools/check_ascend_queue_lifetime.py xvm@orb:/home/niu/code/Ascend-MLIR/
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/two-kernel-rank-mix-dag/run.sh --log'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && python3 test/tools/check_ascend_queue_lifetime.py examples/two-kernel-rank-mix-dag/build_mainline/step5_ascendc.mlir examples/two-kernel-rank-mix-dag/build_mainline/step7_kernel_ir.mlir examples/two-kernel-rank-mix-dag/build_mainline/step8_cann.mlir'
```

Expected:

```text
manifest.kernel_entries=2
manifest.kernelGraph.nodes=2
session.plan.tasks=2
session.plan[0]=kernel_a
session.plan[1]=kernel_b
session.validation=pass
queue_lifetime.ok ...
```

- [ ] **Step 5: Add to example suite**

Modify `test/tools/examples/example_pipelines.sh`:

```bash
EXAMPLES=(
  "add-broadcast-concat"
  "broadcast-add-reduce"
  "gather-elementwise-fusion"
  "relu-broadcast-transpose"
  "split-relu-brc-add-mul"
  "matmul-add-leakyrelu"
  "two-kernel-dag"
  "two-kernel-rank-mix-dag"
)
```

Modify `test/tools/examples/example-pipelines.mlir`:

```mlir
// CHECK: INFO: executing 8 example pipelines
// CHECK: EXECUTED: 8 example pipelines
```

- [ ] **Step 6: Commit**

```bash
git add examples/two-kernel-rank-mix-dag test/tools/examples/example_pipelines.sh test/tools/examples/example-pipelines.mlir
git commit -m "test: add rank-mixed two-kernel runtime DAG"
```

## Task 4: Add Three-Kernel DAG Example

**Files:**
- Create: `examples/three-kernel-dag/step0_input.mlir`
- Create: `examples/three-kernel-dag/gen_data.py`
- Create: `examples/three-kernel-dag/run-mainline.sh`
- Create: `examples/three-kernel-dag/run.sh`
- Modify: `test/tools/examples/example_pipelines.sh`
- Modify: `test/tools/examples/example-pipelines.mlir`

- [ ] **Step 1: Create high-level IR**

Create `examples/three-kernel-dag/step0_input.mlir`:

```mlir
// Three generated kernels in one runtime-session DAG.
//
// kernel_a: a[N] + b[N] -> mid0[N]
// kernel_b: mid0[N] * c[N] -> mid1[N]
// kernel_c: mid1[N] + d[N] -> out[N]

module {
  func.func @kernel_a(%a : tensor<?xf16>, %b : tensor<?xf16>) -> tensor<?xf16> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %a, %c0 : tensor<?xf16>
    %init = tensor.empty(%n) : tensor<?xf16>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<?xf16>, tensor<?xf16>)
      outs(%init : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %v = arith.addf %x, %y : f16
      linalg.yield %v : f16
    } -> tensor<?xf16>
    return %out : tensor<?xf16>
  }

  func.func @kernel_b(%mid0 : tensor<?xf16>, %c : tensor<?xf16>) -> tensor<?xf16> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %mid0, %c0 : tensor<?xf16>
    %init = tensor.empty(%n) : tensor<?xf16>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%mid0, %c : tensor<?xf16>, tensor<?xf16>)
      outs(%init : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %v = arith.mulf %x, %y : f16
      linalg.yield %v : f16
    } -> tensor<?xf16>
    return %out : tensor<?xf16>
  }

  func.func @kernel_c(%mid1 : tensor<?xf16>, %d : tensor<?xf16>) -> tensor<?xf16> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %mid1, %c0 : tensor<?xf16>
    %init = tensor.empty(%n) : tensor<?xf16>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%mid1, %d : tensor<?xf16>, tensor<?xf16>)
      outs(%init : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %v = arith.addf %x, %y : f16
      linalg.yield %v : f16
    } -> tensor<?xf16>
    return %out : tensor<?xf16>
  }
}
```

- [ ] **Step 2: Create data generator**

Create `examples/three-kernel-dag/gen_data.py`:

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a = rng.uniform(-1.0, 1.0, size=(args.n,)).astype(np.float16)
    b = rng.uniform(0.0, 2.0, size=(args.n,)).astype(np.float16)
    c = rng.uniform(0.5, 1.5, size=(args.n,)).astype(np.float16)
    d = rng.uniform(-0.25, 0.25, size=(args.n,)).astype(np.float16)
    expected = (((a + b) * c) + d).astype(np.float16)

    np.save(out_dir / "input_a.npy", a)
    np.save(out_dir / "input_b.npy", b)
    np.save(out_dir / "input_c.npy", c)
    np.save(out_dir / "input_d.npy", d)
    np.save(out_dir / "output_expected.npy", expected)

    print(f"N={args.n}, seed={args.seed}")
    print(f"input_a: {a.shape} {a.dtype}")
    print(f"input_b: {b.shape} {b.dtype}")
    print(f"input_c: {c.shape} {c.dtype}")
    print(f"input_d: {d.shape} {d.dtype}")
    print(f"output:  {expected.shape} {expected.dtype}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create run scripts**

Start by copying the two-kernel runner:

```bash
mkdir -p examples/three-kernel-dag
cp examples/two-kernel-dag/run-mainline.sh examples/three-kernel-dag/run-mainline.sh
cp examples/two-kernel-dag/run.sh examples/three-kernel-dag/run.sh
```

Edit `examples/three-kernel-dag/run-mainline.sh`:

1. Replace expected names check:

```python
if names != ["kernel_a", "kernel_b", "kernel_c"]:
    raise SystemExit(f"expected kernel_entries for kernel_a/kernel_b/kernel_c, got {names}")
if len(nodes) != 3:
    raise SystemExit(f"expected three kernelGraph nodes, got {len(nodes)}")
print("manifest.kernel_entries=3")
print("manifest.kernelGraph.nodes=3")
```

2. Add `ARTIFACT_C="$BUILD_DIR/artifact_kernel_c"`.
3. Compile kernel C:

```bash
"$RUNTIME_SESSION" \
  --kernel "$BUILD_DIR/step9_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_C" \
  --name kernel_c
```

4. Replace run manifest tasks with three tasks:

```json
{
  "backend": "sim",
  "tasks": [
    {
      "task_id": "kernel_a",
      "artifact_root": "${ARTIFACT_A}",
      "inputs": [
        { "name": "a", "path": "${BUILD_DIR}/input_a.npy" },
        { "name": "b", "path": "${BUILD_DIR}/input_b.npy" }
      ],
      "outputs": [
        { "name": "mid0", "shape": [${N}], "dtype": "f16" }
      ],
      "tiling": { "schema": "${PHASE5_TILING_SPACE}", "params": "${TILING_PARAMS}" },
      "block_dim": ${BLOCK_DIM},
      "workspace_size": 16777216,
      "profiling": true
    },
    {
      "task_id": "kernel_b",
      "dependencies": ["kernel_a"],
      "artifact_root": "${ARTIFACT_B}",
      "inputs": [
        { "name": "mid0", "source": "task_output", "upstream_task": "kernel_a", "upstream_output": "mid0" },
        { "name": "c", "path": "${BUILD_DIR}/input_c.npy" }
      ],
      "outputs": [
        { "name": "mid1", "shape": [${N}], "dtype": "f16" }
      ],
      "tiling": { "schema": "${PHASE5_TILING_SPACE}", "params": "${TILING_PARAMS}" },
      "block_dim": ${BLOCK_DIM},
      "workspace_size": 16777216,
      "profiling": true
    },
    {
      "task_id": "kernel_c",
      "dependencies": ["kernel_b"],
      "artifact_root": "${ARTIFACT_C}",
      "inputs": [
        { "name": "mid1", "source": "task_output", "upstream_task": "kernel_b", "upstream_output": "mid1" },
        { "name": "d", "path": "${BUILD_DIR}/input_d.npy" }
      ],
      "outputs": [
        { "name": "out", "path": "${ACTUAL_OUTPUT}", "shape": [${N}], "dtype": "f16" }
      ],
      "tiling": { "schema": "${PHASE5_TILING_SPACE}", "params": "${TILING_PARAMS}" },
      "block_dim": ${BLOCK_DIM},
      "workspace_size": 16777216,
      "profiling": true
    }
  ]
}
```

5. Update log greps:

```bash
grep -q '^session.plan.tasks=3$' "$VALIDATION_LOG"
grep -q '^session.plan\[0\]=kernel_a$' "$VALIDATION_LOG"
grep -q '^session.plan\[1\]=kernel_b$' "$VALIDATION_LOG"
grep -q '^session.plan\[2\]=kernel_c$' "$VALIDATION_LOG"
grep -q '^session.profile.count=3$' "$VALIDATION_LOG"
grep -q '^session.runtime.counter.planned_task_count=3$' "$VALIDATION_LOG"
grep -q '^session.runtime.counter.serialized_launch_count=3$' "$VALIDATION_LOG"
```

- [ ] **Step 4: Verify focused example**

Run:

```bash
rsync -aR ./examples/three-kernel-dag ./test/tools/check_ascend_queue_lifetime.py xvm@orb:/home/niu/code/Ascend-MLIR/
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/three-kernel-dag/run.sh --log'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && python3 test/tools/check_ascend_queue_lifetime.py examples/three-kernel-dag/build_mainline/step5_ascendc.mlir examples/three-kernel-dag/build_mainline/step7_kernel_ir.mlir examples/three-kernel-dag/build_mainline/step8_cann.mlir'
```

Expected:

```text
manifest.kernel_entries=3
manifest.kernelGraph.nodes=3
session.plan.tasks=3
session.plan[2]=kernel_c
session.validation=pass
queue_lifetime.ok ...
```

- [ ] **Step 5: Add to example suite**

Modify `test/tools/examples/example_pipelines.sh`:

```bash
EXAMPLES=(
  "add-broadcast-concat"
  "broadcast-add-reduce"
  "gather-elementwise-fusion"
  "relu-broadcast-transpose"
  "split-relu-brc-add-mul"
  "matmul-add-leakyrelu"
  "two-kernel-dag"
  "two-kernel-rank-mix-dag"
  "three-kernel-dag"
)
```

Modify `test/tools/examples/example-pipelines.mlir`:

```mlir
// CHECK: INFO: executing 9 example pipelines
// CHECK: EXECUTED: 9 example pipelines
```

- [ ] **Step 6: Commit**

```bash
git add examples/three-kernel-dag test/tools/examples/example_pipelines.sh test/tools/examples/example-pipelines.mlir
git commit -m "test: add three-kernel runtime DAG"
```

## Task 5: Add RMSNorm Reduction-Core Runtime Demo

**Files:**
- Create: `examples/rmsnorm-reduction-core/step0_input.mlir`
- Create: `examples/rmsnorm-reduction-core/gen_data.py`
- Create: `examples/rmsnorm-reduction-core/run-mainline.sh`
- Create: `examples/rmsnorm-reduction-core/run.sh`
- Modify: `test/tools/examples/example_pipelines.sh`
- Modify: `test/tools/examples/example-pipelines.mlir`

- [ ] **Step 1: Create high-level IR**

Create `examples/rmsnorm-reduction-core/step0_input.mlir`:

```mlir
// RMSNorm reduction core: sum(x * x) per row.
// This is the runtime-supported reduction core before math.rsqrt lowering.

#identity = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

module {
  func.func @rmsnorm_reduction_core(%input: tensor<?x?xf16>) -> tensor<?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %input, %c0 : tensor<?x?xf16>
    %n = tensor.dim %input, %c1 : tensor<?x?xf16>

    %sq_init = tensor.empty(%m, %n) : tensor<?x?xf16>
    %squared = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<?x?xf16>)
      outs(%sq_init : tensor<?x?xf16>) {
    ^bb0(%x: f16, %out: f16):
      %sq = arith.mulf %x, %x : f16
      linalg.yield %sq : f16
    } -> tensor<?x?xf16>

    %zero = arith.constant 0.0 : f16
    %sum_empty = tensor.empty(%m) : tensor<?xf16>
    %sum_init = linalg.fill ins(%zero : f16)
      outs(%sum_empty : tensor<?xf16>) -> tensor<?xf16>
    %sum = linalg.generic {
      indexing_maps = [#identity, #row],
      iterator_types = ["parallel", "reduction"]
    } ins(%squared : tensor<?x?xf16>)
      outs(%sum_init : tensor<?xf16>) {
    ^bb0(%x: f16, %acc: f16):
      %v = arith.addf %x, %acc : f16
      linalg.yield %v : f16
    } -> tensor<?xf16>

    return %sum : tensor<?xf16>
  }
}
```

- [ ] **Step 2: Create data generator**

Create `examples/rmsnorm-reduction-core/gen_data.py`:

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = rng.uniform(-1.0, 1.0, size=(args.m, args.n)).astype(np.float16)
    expected = (x.astype(np.float32) * x.astype(np.float32)).sum(axis=1).astype(np.float16)

    np.save(out_dir / "input_x.npy", x)
    np.save(out_dir / "output_expected.npy", expected)

    print(f"M={args.m}, N={args.n}, seed={args.seed}")
    print(f"input_x: {x.shape} {x.dtype}")
    print(f"output:  {expected.shape} {expected.dtype}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create run script**

Use `examples/broadcast-add-reduce/run-mainline.sh` as the base:

```bash
mkdir -p examples/rmsnorm-reduction-core
cp examples/broadcast-add-reduce/run-mainline.sh examples/rmsnorm-reduction-core/run-mainline.sh
cat > examples/rmsnorm-reduction-core/run.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$DIR/run-mainline.sh" "$@"
SH
chmod +x examples/rmsnorm-reduction-core/run.sh
```

Edit `examples/rmsnorm-reduction-core/run-mainline.sh`:

1. Set defaults:

```bash
M=64
N=96
SEED=44
```

2. Use title:

```bash
echo " rmsnorm reduction-core Ascend mainline pipeline"
```

3. Use generator:

```bash
"$PYTHON" "$DIR/gen_data.py" --m "$M" --n "$N" --seed "$SEED" --out-dir "$BUILD_DIR"
```

4. Use one input in run manifest:

```json
"inputs": [
  { "name": "arg0", "path": "${BUILD_DIR}/input_x.npy" }
],
"outputs": [
  { "name": "out", "path": "${ACTUAL_OUTPUT}", "shape": [${M}], "dtype": "f16" }
]
```

5. Use tiling shape values:

```python
shape_values = {
    "arg0_dim0": m,
    "arg0_dim1": n,
}
```

6. Compare against `output_expected.npy`.

- [ ] **Step 4: Verify focused demo**

Run:

```bash
rsync -aR ./examples/rmsnorm-reduction-core ./test/tools/check_ascend_queue_lifetime.py xvm@orb:/home/niu/code/Ascend-MLIR/
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/rmsnorm-reduction-core/run.sh --log'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && python3 test/tools/check_ascend_queue_lifetime.py examples/rmsnorm-reduction-core/build_mainline/step6_ascendc.mlir examples/rmsnorm-reduction-core/build_mainline/step8_kernel_ir.mlir examples/rmsnorm-reduction-core/build_mainline/step9_cann.mlir'
```

Expected:

```text
session.plan.tasks=1
session.backend=sim
session.result=success
session.validation=pass
queue_lifetime.ok ...
```

- [ ] **Step 5: Add to example suite**

Modify `test/tools/examples/example_pipelines.sh`:

```bash
EXAMPLES=(
  "add-broadcast-concat"
  "broadcast-add-reduce"
  "gather-elementwise-fusion"
  "relu-broadcast-transpose"
  "split-relu-brc-add-mul"
  "matmul-add-leakyrelu"
  "two-kernel-dag"
  "two-kernel-rank-mix-dag"
  "three-kernel-dag"
  "rmsnorm-reduction-core"
)
```

Modify `test/tools/examples/example-pipelines.mlir`:

```mlir
// CHECK: INFO: executing 10 example pipelines
// CHECK: EXECUTED: 10 example pipelines
```

- [ ] **Step 6: Commit**

```bash
git add examples/rmsnorm-reduction-core test/tools/examples/example_pipelines.sh test/tools/examples/example-pipelines.mlir
git commit -m "test: add RMSNorm reduction-core runtime demo"
```

## Task 6: Full Queue Lifecycle Audit And Targeted Fixes

**Files:**
- Inspect: `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp`
- Inspect: `lib/Conversion/LinalgToAscendC/DataMoveConversion.cpp`
- Inspect: `lib/Conversion/AscendCBufferPlacement/AscendCBufferPlacementPass.cpp`
- Inspect: `lib/Target/CannKernel/CannTranslation.cpp`
- Modify only the file that owns any failing dequeue/free path.

- [ ] **Step 1: Run checker on generated examples**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && for d in examples/*/build_mainline; do files=$(find "$d" -maxdepth 1 -type f \( -name "*.mlir" \) | tr "\n" " "); if [ -n "$files" ]; then python3 test/tools/check_ascend_queue_lifetime.py $files; fi; done'
```

Expected: every checked file prints `queue_lifetime.ok`.

- [ ] **Step 2: Run checker on Target input fixtures**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && python3 test/tools/check_ascend_queue_lifetime.py test/Target/*-input.mlir'
```

Expected: every checked file prints `queue_lifetime.ok`.

- [ ] **Step 3: Fix any missing-free path at the producer layer**

Use this mapping:

| Failure location | Owning fix |
| --- | --- |
| Missing free after `copyGmToVecin` in generated all-parallel/reduction generic | Extend `OwnedQueueTensor` tracking in `ComputeConversion.cpp` around the specific `readTensor` / `copyGmToVecin` call site. |
| Missing free in explicit GM -> VECIN data movement | Fix `DataMoveConversion.cpp` so `TQueBindFreeTensorOp` is inserted after the final consumer or before block terminator, not before use. |
| Missing free in mix/matmul queues | Fix the corresponding `qA/qB/qC` dequeue owner in `ComputeConversion.cpp`, preserving existing `allocHoisted` behavior. |
| Missing free only in a `test/Target/*-input.mlir` fixture | Fix the fixture to match the real lowering contract, then run the owning `cann-translate-*` test. |

For each fixed path, add or update the smallest LIT test that reaches it and run that test before moving on.

- [ ] **Step 4: Verify no scalar lane patches remain**

Run:

```bash
rg -n "GetValue\\(0\\)|Adds\\(.*\\[0\\]|repair lane|lane 0|first-lane" lib test examples
```

Expected: no result that repairs only lane 0. Existing legitimate comments about scalar tail fallback are acceptable only if they loop over all elements or tail elements.

- [ ] **Step 5: Commit**

If Task 6 required fixes:

```bash
git add lib/Conversion/LinalgToAscendC/ComputeConversion.cpp lib/Conversion/LinalgToAscendC/DataMoveConversion.cpp lib/Conversion/AscendCBufferPlacement/AscendCBufferPlacementPass.cpp lib/Target/CannKernel/CannTranslation.cpp test/Conversion test/Target
git commit -m "fix: close remaining AscendC queue lifetime gaps"
```

If Task 6 required no fixes:

```bash
git status -sb
```

Expected: no new source changes from Task 6.

## Task 7: Final Verification, Tracking, And Push

**Files:**
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Run build gate**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && cmake --build build --target afir-opt afir-translate runtime-session -j6'
```

Expected: exit 0.

- [ ] **Step 2: Run LIT gate**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion build/test/Target build/test/tools/examples/example-pipelines.mlir test/tools/ascend-queue-lifetime.mlir'
```

Expected: all discovered tests pass.

- [ ] **Step 3: Run C++ unit gate**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ctest --test-dir build -R "Ascend" --output-on-failure'
```

Expected: all Ascend unit tests pass.

- [ ] **Step 4: Run focused runtime demos**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/two-kernel-dag/run.sh --log && bash examples/two-kernel-rank-mix-dag/run.sh --log && bash examples/three-kernel-dag/run.sh --log && bash examples/rmsnorm-reduction-core/run.sh --log'
```

Expected: every demo prints `session.validation=pass`.

- [ ] **Step 5: Update tracking doc**

Append this record to `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md` under the latest verification/status area:

```markdown
### Runtime graph and queue-lifetime hardening

- Status: Done
- Scope:
  - Added static AscendC queue lifetime checker for generated MLIR.
  - Added queue-lifetime gate to example pipeline verification.
  - Added rank-mixed two-kernel DAG runtime demo.
  - Added three-kernel DAG runtime demo.
  - Added RMSNorm reduction-core runtime demo.
- Verification:
  - `cmake --build build --target afir-opt afir-translate runtime-session -j6`
  - `llvm-lit -v build/test/Conversion build/test/Target build/test/tools/examples/example-pipelines.mlir test/tools/ascend-queue-lifetime.mlir`
  - `ctest --test-dir build -R "Ascend" --output-on-failure`
  - `bash examples/two-kernel-dag/run.sh --log`
  - `bash examples/two-kernel-rank-mix-dag/run.sh --log`
  - `bash examples/three-kernel-dag/run.sh --log`
  - `bash examples/rmsnorm-reduction-core/run.sh --log`
```

- [ ] **Step 6: Commit tracking update**

```bash
git add docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md
git commit -m "docs: record runtime graph hardening status"
```

- [ ] **Step 7: Push**

```bash
git push origin dev-nyh
git status -sb
```

Expected: branch is clean and not ahead of `origin/dev-nyh`.

## Self-Review

- Spec coverage:
  - Complex graph acceptance is covered by Tasks 3, 4, and 5.
  - Queue lifecycle audit is covered by Tasks 1, 2, and 6.
  - Release-style verification is covered by Task 7.
- Placeholder scan:
  - No deferred-work markers.
  - No undefined future functions in required command paths.
  - Unknown failures are handled by concrete owning-file mapping in Task 6.
- Type consistency:
  - Queue checker consistently keys by `(queue, tensor)`.
  - Runtime DAG manifests consistently use `task_output`, `upstream_task`, and `upstream_output`.
  - New examples use aligned `N=64` by default.
