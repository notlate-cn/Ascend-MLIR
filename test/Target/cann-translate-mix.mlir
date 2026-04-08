// RUN: afir-translate -mlir-to-cann %S/cann-translate-mix-input.mlir | FileCheck %s

// Expected shape: one cube region, one boundary transfer/synchronization layer,
// and one vector region in a single chained mix lowering path.
// The input fixture intentionally feeds the vector region from both:
//   1. the cube->boundary payload selected by generic mix analysis, and
//   2. a separate bias broadcast branch that merges at add_l2.
// Generic single-chain validation must still accept this shape.
// CHECK: #define __AFIR_RUNTIME_MIX_KERNEL_FUN_H__
// CHECK: #define ASCENDC_CUBE_ONLY
// CHECK: #include "kernel_operator.h"
// CHECK: #include "lib/matmul_intf.h"
// CHECK-NOT: struct TilingData {
// CHECK: __aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
// CHECK-LABEL: extern "C" __global__ __aicore__ void matmul_add_leakyrelu(
// CHECK: GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR out, GM_ADDR workspace,
// CHECK: GM_ADDR tilingGm
// CHECK: KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
// CHECK: if ASCEND_IS_AIC {
// CHECK: REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
// CHECK: mm.SetBias(biasGM);
// CHECK: mm.template IterateAll(cGM);
// CHECK: mm.End();
// CHECK: CrossCoreSetFlag<0x2, PIPE_FIX>(3);
// CHECK: if ASCEND_IS_AIV {
// CHECK: uint32_t count = static_cast<uint32_t>(tiling.singleCoreM * tiling.singleCoreN / 2);
// CHECK: CrossCoreWaitFlag(3);
// CHECK: DataCopy(reluInLocal, cGM, count);
// CHECK: LeakyRelu(outLocal, inLocal, static_cast<float>(0.001000f), count);

// This fixture locks the current generic single-chain translator shape for the
// mixed cube/boundary/vector path without depending on the sample kernel name.
