# matmul-add-leakyrelu

End-to-end AFIR pipeline example: `matmul(A[M,K], B[K,N]) + broadcast(bias[N]) → leaky_relu(0.001)`.

## Computation Graph

```
A[M,K] (f16) × B[K,N] (f16) → C[M,N] (f32)    # linalg.matmul
C[M,N] + broadcast(bias[N]) → D[M,N]            # linalg.generic (bias add)
max(D, D × 0.001) → E[M,N]                      # linalg.generic (leaky_relu)
```

Dimensions: M=128, K=256, N=128 (BLOCK_DIM=1, single core).

## Pipeline Stages

| File | Pass(es) | Description |
|------|----------|-------------|
| `step0_input.mlir` | — | linalg-on-tensor source |
| `step2_transform.mlir` | `--transform-interpreter` | 3-level tiling script (TB/Tb/K) |
| `step2_tiled.mlir` | generated | Tiled linalg loops with ascendc annotations |
| `step3_bufferized.mlir` | `--one-shot-bufferize` | memref-based IR |
| `step4_buffer_placement.mlir` | `--ascendc-buffer-placement` | memory_space annotated |
| `step5_ascendc.mlir` | `--linalg-to-ascendc` | AscendC ops (mmad, add_l2, mul_l2, max_l2) |
| `step6_parallelize.mlir` | `--ascendc-parallelize` | Block index inserted |
| `step7_kernel.mlir` | `--ascendc-prepare-for-emit` | Tiling struct emitted |
| `step7_cann.mlir` | `--canonicalize-cann-signature` | CANN calling convention |
| `step8_kernel.cpp` | `afir-translate -mlir-to-cann` | AscendC C++ kernel |

## Tiling Strategy

3-level tiling matching the `matmul-add-relu-sum` example:

```
scf.for %TB_M {ascendc.parallel}
  scf.for %TB_N {ascendc.parallel,
                 prologue="lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN",
                 epilogue="result:VECOUT->GM"}
    scf.for %Tb_M
      scf.for %Tb_N
        scf.for %K {prologue="lhs:A1->A2,rhs:B1->B2", epilogue="acc:CO1->VECIN"}
          linalg.matmul         (ascendc.unit="AiCore.Cube")
        linalg.generic [add+bias] (ascendc.unit="AiCore.Vector")
        linalg.generic [leaky_relu] (ascendc.unit="AiCore.Vector")
```

leaky_relu is lowered to `duplicate_l2(alpha) + mul_l2(x*alpha) + max_l2(x, x*alpha)`.

## Running

Only supported on `xvm`:

```bash
cd /home/niu/code/Ascend-MLIR
source examples/env.sh
bash examples/matmul-add-leakyrelu/run.sh
```

Expected output: `PASS` at the end.

## Acceptance Criteria

1. All AFIR pipeline stages complete without error
2. `step8_kernel.cpp` contains `__global__ __aicore__ void matmul_add_leakyrelu(`
3. RuntimeMix mix-validator outputs `PASS`
