// RUN: ascend-mlir-translate -mlir-to-cann %S/cann-translate-mix-relu-input.mlir | FileCheck %s

// Expected shape: one cube region, one boundary transfer/synchronization layer,
// and one vector region in a single chained mix lowering path.
// CHECK: #define __AFIR_RUNTIME_MIX_KERNEL_FUN_H__
// CHECK: #define ASCENDC_CUBE_ONLY
// CHECK: #include "kernel_operator.h"
// CHECK: #include "lib/matmul_intf.h"
// CHECK-NOT: struct TilingData {
// CHECK: __aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
// CHECK-LABEL: extern "C" __global__ __aicore__ void
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
// CHECK: Relu(outLocal, inLocal, count);
// CHECK-NOT: LeakyRelu(

// This fixture keeps the same generic single-chain shell while pinning the
// epilogue inference that drops the leaky-relu alpha path to plain ReLU.
