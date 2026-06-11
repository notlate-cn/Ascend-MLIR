# Network Runner: Mixed AscendC + aclnn, on CPU Sim (Y-cpp)

**Status:** design (2026-05-13)
**Branch:** `llm-net`
**Successor of:** the brainstorming session 2026-05-13 (`network.mlir` end-to-end flow).

## 1. Goal

Take a `network.mlir` whose body is a sequence of `call` ops, each calling either an
**AscendC kernel group** (`@kernel_groupN`, lowered through `--vector-plan-codegen`) or
an **aclnn op** (`@__aclnn_xxx` with `aclnn.op` / `aclnn.layout` attrs), and produce a
**single `network_host.cpp` binary** that runs the whole network on this CPU machine
(no NPU hardware) and verifies its output against `expected.npy`.

Same binary runs on real NPU later: when `aclInit` succeeds, host-mode is off and
aclnn ops dispatch to the real library.

## 2. Why this shape

Three facts forced the design:

1. **AscendC kernels can run on CANN's CPU sim (`camodel`).** `runtime-session
   --backend sim` does it via `librtSetDevice / rtKernelLaunch` from
   `libruntime_camodel.so`, bypassing `aclInit` entirely
   (`lib/Runtime/Execution/NativeExecutionRunner.cpp` is the working reference).
2. **aclnn ops cannot.** `aclInit` requires a real device (`InitSocVersion: cannot get
   soc version, errorCode 507008` against camodel). CANN ships no CPU/host backend
   library for aclnn (`libopapi*` are device-only; `simulator/` has no `libaclnn*`).
   The only path on this machine is a hand-written CPU equivalent per op — exactly
   what `lib/Runtime/AclnnOps.cpp`'s existing `g_host_mode` branch does for
   `FlashAttentionScore`.
3. **The existing `aclnn-backend` tool already parses mixed `network.mlir`** and emits
   `network_host.cpp` with aclnn dispatch — but its AscendC-launch site is a
   `// TODO: launch ...` stub (`lib/Runtime/AclnnBackend/AclnnBackend.cpp:100`), and
   the generated host calls `aclInit` (so today's binary cannot run on CPU sim).

So the Y-cpp design is:

- finish the AscendC-launch stub so `network_host.cpp` launches kernels via `rt*`
  directly to camodel (no `aclInit` dependency);
- when `aclInit` fails, set `g_host_mode = true` and continue (instead of exiting),
  so aclnn nodes fall through to their host-mode C++ implementations;
- keep autotune offline (Python driver), bake best tilings into the generated host;
- start with the single example `mixed-attn-e2e`.

## 3. Pieces

### 3.1 `network.json` emitter — used in two modes

The runner needs a structured view of the call graph (per-kernel shapes, dtype, kind,
upstream/input dataflow) regardless of how the network was produced. Factor JSON
emission into a small reusable utility that walks a `func` body of `call`s and writes
`network.json` (schema below). It runs in two contexts:

1. **As part of `--vector-plan-group-outline=output-dir=DIR`** — after the pass
   produces `DIR/network.mlir` and `DIR/kernel_group*.mlir`, also emit
   `DIR/network.json`. (For linalg-input flows; produces only `kind:"ascendc"`
   nodes today.)
2. **As a standalone `afir-opt` pass `--emit-network-json=path=DIR/network.json`** —
   for hand-written / mixed `network.mlir` (the v1 mixed-attn case), the user invokes
   it explicitly: `afir-opt network.mlir --emit-network-json=path=...`. Detects aclnn
   nodes by the callee's `aclnn.op` attribute.

`network.json` schema:

```json
{
  "function": "model",
  "inputs":  [{"name": "arg0", "shape": [...], "dtype": "f16"}, ...],
  "kernels": [
    {"id": "kernel_group0", "kind": "ascendc", "file": "kernel_group0.mlir",
     "args":    [{"from": "input",  "name": "arg0"}, ...],
     "results": [{"name": "kernel_group0_r0", "shape": [...], "dtype": "f16"}]},
    {"id": "fa0",           "kind": "aclnn",  "op": "FlashAttentionScore",
     "layout": "BNSD",
     "args":    [{"from": "kernel", "kernel": "kernel_group0", "result": 0}, ...],
     "results": [...]}
  ],
  "outputs": [
    {"name": "out0", "from": "kernel", "kernel": "kernel_groupN", "result": 0}
  ]
}
```

- `kernels` is in `call`-appearance order in `network.mlir`. Outline already arranges
  these by SSA dominance, so this is the topological order — runner consumes it
  sequentially.
- `kind` distinguishes AscendC-kernel vs aclnn-op nodes. The pass detects an aclnn
  node by checking the callee for the `aclnn.op` attr.
- v1 supports static shapes only (dynamic-shape network runner is a separate piece).
- Implementation: a shared helper (~80 LOC) that takes a `func::FuncOp` (the network
  function) + the surrounding module, walks `call` ops, classifies callee `kind` by
  `aclnn.op` attr presence, and writes the JSON. Called from both
  `GroupOutlinePass.cpp` (mode 1) and a new tiny pass `EmitNetworkJsonPass.cpp` (mode
  2). New lits: `test/Conversion/Group/group-outline/network-json.mlir` (mode 1),
  `test/Dialect/AFIR/Transforms/emit-network-json-mixed.mlir` (mode 2).

### 3.2 `aclnn-backend`: finish the AscendC-launch site, drop `aclInit` requirement

Two changes to `lib/Runtime/AclnnBackend/AclnnBackend.cpp` and the generated
`network_host.cpp`:

**A. Generated-binary startup (host mode is fine, not fatal):**
```cpp
if (aclInit(nullptr) != 0) {
  fprintf(stderr, "[network_host] aclInit failed — running in host-mode "
                  "(aclnn ops use CPU reference; AscendC kernels use camodel sim)\n");
  mlir::runtime::aclnn::setHostMode(true);
}
// AscendC kernel launches do NOT require aclInit; they go straight to rt*.
```

**B. AscendC kernel launch (replace the `// TODO: launch ...` stub):**

Mirror what `NativeExecutionRunner` does in sim mode:

1. once-per-process: `dlopen("libruntime_camodel.so")`, resolve `rtSetDevice`,
   `rtDevBinaryRegister`, `rtFunctionRegister`, `rtMalloc`, `rtFree`, `rtMemcpy`,
   `rtKernelLaunch`, etc.;
2. once-per-kernel: `rtDevBinaryRegister(<kernel.o bytes>)` →
   `rtFunctionRegister` → cache the function handle;
3. per-call: `rtMalloc` GM I/O buffers → `rtMemcpy` H2D → build args struct
   `{input_ptrs..., output_ptrs..., workspace_ptr, TilingData_blob, block_dim}` →
   `rtKernelLaunch` → `rtMemcpy` D2H.

The TilingData blob and `block_dim` are **baked at codegen time** from the
runner-supplied per-kernel best tiling (passed to `aclnn-backend` as JSON; see 3.4).
The kernel binary path (`.o` produced by `runtime-session --kernel ... --output ...`)
is also passed in.

Fallback: if `aclInit` succeeded (real NPU available), AscendC kernel launch can
optionally use `aclrt*` for symmetry — but for v1 we always go through `rt*` to keep
the path uniform. Real-NPU path is a follow-up.

### 3.3 `AclnnOps.cpp`: extend host-mode for ops we use

`lib/Runtime/AclnnOps.cpp` already has the pattern for FlashAttentionScore: a
`run_FlashAttentionScore(...)` that branches on `g_host_mode` between a hand-written
C++ kernel and the real `aclnnFlashAttentionScore` call. Each new aclnn op the network
uses needs its `run_<Op>` with both branches. v1 uses only FlashAttentionScore (already
done) — no new op to add.

### 3.4 `network_runner.py`: orchestration

Lives at `python/network_runner.py`. Two input modes:

```
network_runner.py
  ( --input-linalg model.mlir             # runs outline → produces network.{mlir,json}
  | --input-network DIR )                 # DIR has network.mlir + kernel_group*.mlir,
                                          # runner runs --emit-network-json on it
  --inputs  in0.npy in1.npy ...           # network inputs, in func-arg order
  --expected o0.npy o1.npy ...            # network outputs, in network-output order
  --workdir WORK
  --soc Ascend910B1
  [--atol 1e-3 --rtol 1e-2]
```

v1 mixed-attn uses `--input-network`. Pure-AscendC linalg flows use `--input-linalg`.

Phase 1 — **obtain `network.{mlir,json}` + per-kernel `.mlir`**:

`--input-linalg`:
```
afir-opt --linalg-fold-unit-extent-dims model.mlir |
afir-opt --vector-plan-group-analysis '--vector-plan-group-outline=output-dir=WORK/groups'
→ WORK/groups/{network.mlir, network.json, kernel_group*.mlir}
```

`--input-network DIR`:
```
copy DIR/{network.mlir, kernel_group*.mlir} → WORK/groups/
afir-opt WORK/groups/network.mlir --emit-network-json=path=WORK/groups/network.json
```

Phase 2 — **lower + compile each AscendC kernel** (skip `kind:"aclnn"` entries —
those need no codegen):
```
For each k in network.json["kernels"] where kind == "ascendc":
  afir-opt kernel_groupK.mlir --vector-plan-codegen → kgK_lowered.mlir
  afir-translate -mlir-to-cann kgK_lowered.mlir → WORK/kgK.cpp + WORK/kgK_space.json
  runtime-session --kernel WORK/kgK.cpp --kernel-kind vec
                  --output WORK/artifacts/kgK --name kgK   # compile only
```

Phase 3 — **default-tilings build, dump intermediates** (the (α) path):
```
For each k where kind == "ascendc":
  pick the default tiling from kgK_space.json (each param's "default")
Write WORK/tilings_default.json = {kgK: {param: value, ...}, ...}
aclnn-backend --input WORK/groups/network.mlir
              --output WORK/network_host.cpp
              --tilings WORK/tilings_default.json
              --kernel-binaries WORK/artifacts
g++ ... → WORK/network_test_default
WORK/network_test_default --inputs ... --output WORK/output_default.npy
                          --dump-intermediates WORK/intermediates_default
  → WORK/intermediates_default/<kernel_id>_in_<n>.npy / _out_<n>.npy for every kernel
```

(`--dump-intermediates` is a runtime flag of the *generated* binary: after each
kernel's d2h copy, write the buffer to a npy. `aclnn-backend` always emits the
plumbing for it; the binary writes nothing if the flag is absent.)

Phase 4 — **per-kernel autotune**:
```
For each k where kind == "ascendc":
  inputs   = WORK/intermediates_default/<k>_in_*.npy
  expected = WORK/intermediates_default/<k>_out_0.npy   # default-tiling output as ref;
                                                         # catches tiling instability
  autotuner --space WORK/kgK_space.json --kernel WORK/kgK.cpp
            --inputs <inputs> --expected <expected>
            --shape <derived-from-network.json>
            --output WORK/kgK_best.json
  read WORK/kgK_best.json → tilings_best[k]
```

Single-output AscendC kernels only in v1 (autotuner today takes one `--expected`).
Multi-output (e.g. horizontal fusion → one kernel returning 2 tensors) is a follow-up.

Phase 5 — **best-tilings build, run, verify**:
```
Write WORK/tilings_best.json
aclnn-backend --input WORK/groups/network.mlir
              --output WORK/network_host.cpp
              --tilings WORK/tilings_best.json
              --kernel-binaries WORK/artifacts
g++ ... → WORK/network_test
WORK/network_test --inputs ... --output WORK/output.npy
For each network output:
  load actual + expected, compare with atol/rtol → print max_diff, PASS/FAIL
```

Exit non-zero on any FAIL.

### 3.5 v1 example: `examples/mixed-attn-e2e/`

Hand-written `network.mlir`:

```mlir
module {
  func.func private @kernel_group0(...) -> tensor<...>     // pre-norm elementwise
  func.func private @__aclnn_flash_attention(tensor<?x?x?x?xf16>, ...)
      -> tensor<?x?x?x?xf16>
      attributes {aclnn.op = "FlashAttentionScore", aclnn.layout = "BNSD"}
  func.func private @kernel_group1(...) -> tensor<...>     // post-proj elementwise
  func.func @model(...) -> tensor<...> {
    %a  = call @kernel_group0(...) : (...) -> ...
    %fa = call @__aclnn_flash_attention(%a, %k, %v, %mask, %init) : (...) -> ...
    %r  = call @kernel_group1(%fa, ...) : (...) -> ...
    return %r : ...
  }
}
```

Plus `kernel_group0.mlir`, `kernel_group1.mlir` (hand-written linalg modules), a
`gen_inputs.py` (numpy: writes `q.npy / k.npy / v.npy / mask.npy / init.npy` and an
`expected.npy` computed from the same numpy reference math the host-mode FlashAttention
implements), and a thin `run.sh` calling `network_runner.py`.

This single example exercises:
- mixed `kind` dispatch in the runner;
- inter-kind buffer plumbing in the generated host (AscendC out → aclnn in →
  AscendC in);
- per-kernel autotune across non-trivial dataflow;
- end-to-end accuracy on CPU sim.

A pure-AscendC `twochain-e2e` example is a useful follow-up to isolate the
AscendC-launch path, but not required for v1.

## 4. Validation

For v1, **only the network's final output is compared to `expected.npy`**. Per-kernel
candidate tilings are validated against the same kernel's default-tiling output (Phase
4) — this catches tiling instability but not codegen bugs that affect every tiling
(those surface at the final compare; bisection is manual on a 2–3-kernel test).

A real per-kernel debugging path (instrumented host that pulls intermediates per-tiling
and compares to a separate golden) is deferred.

## 5. Out of scope (v1)

- `RecognizeAclnnPatterns` (compiler pass that *produces* aclnn nodes from linalg
  patterns like softmax/layernorm) — v1 hand-writes the mixed `network.mlir`.
- Multi-output AscendC kernels in autotune (e.g. horizontal fusion).
- Dynamic shapes in `network.json`.
- Real-NPU execution of the generated binary (works in principle once `aclInit`
  succeeds, but not exercised here).
- `tensor.cast`/`tensor.empty`/`tensor.collapse_shape` glue ops in the network body
  (the existing aclnn-attn-e2e flavor has them; v1 mixed-attn keeps the network body
  to pure `call`s + `return`).
- Per-aclnn-op host-mode beyond what `AclnnOps.cpp` already implements
  (FlashAttentionScore). Adding more ops is mechanical when needed.

## 6. Risks

- **AscendC-launch in generated host** is the biggest unknown. Mitigation: start by
  reading `NativeExecutionRunner.cpp` end-to-end and porting the relevant subset to
  the host emitter, mirroring its sim-mode call sequence verbatim.
- **`--dump-intermediates` mode** is new behavior in `aclnn-backend`. If it ends up
  awkward to wire, fall back to running each AscendC kernel standalone via
  `runtime-session` per-kernel (with per-kernel npy plumbing) for the autotune phase
  only — i.e. the (β) path scoped to autotune. The final binary remains the unified
  one.
- **CPU-reference math mismatch** between the host-mode FlashAttention and what
  `gen_inputs.py` computes for `expected.npy` would silently turn into a numerical
  failure. Mitigation: `gen_inputs.py` uses the same formula (call into the same C++
  via ctypes, or replicate it in numpy and unit-test alignment once).

## 7. Files touched (summary)

| File | Change |
|---|---|
| `lib/Conversion/VectorPlan/GroupOutline/GroupOutlinePass.cpp` | call shared JSON emitter when `output-dir` is set |
| `lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.{h,cpp}` | new: shared `func::FuncOp` → `network.json` helper |
| `lib/Dialect/AFIR/Transforms/EmitNetworkJsonPass.cpp` | new: `--emit-network-json=path=...` standalone pass |
| `lib/Runtime/AclnnBackend/AclnnBackend.cpp` | finish AscendC-launch site; emit `aclInit` host-mode fallback in generated host |
| `tools/aclnn-backend/aclnn-backend.cpp` | add `--tilings`, `--kernel-binaries`, `--dump-intermediates` CLI |
| `python/network_runner.py` | new: orchestrates the 5 phases |
| `examples/mixed-attn-e2e/` | new: `network.mlir`, `kernel_group{0,1}.mlir`, `gen_inputs.py`, `run.sh` |
| `test/Conversion/Group/group-outline/network-json.mlir` | new: lit |
