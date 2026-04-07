// RUN: afir-translate -mlir-to-cann %S/../../examples/matmul-add-leakyrelu/step7_cann.mlir | FileCheck %s

// CHECK: #define __AFIR_RUNTIME_MIX_KERNEL_FUN_H__
// CHECK: #define ASCENDC_CUBE_ONLY
// CHECK: #include "kernel_operator.h"
// CHECK: #include "lib/matmul_intf.h"
// CHECK-NOT: struct TilingData {
// CHECK: __aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
// CHECK: extern "C" __global__ __aicore__ void matmul_add_leakyrelu(
// CHECK-SAME: GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR out, GM_ADDR workspace,
// CHECK-SAME: GM_ADDR tilingGm
// CHECK: KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
// CHECK: if ASCEND_IS_AIC {
// CHECK: REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
// CHECK: mm.SetBias(biasGM);
// CHECK: if ASCEND_IS_AIV {
// CHECK: LeakyRelu(outLocal, inLocal, static_cast<float>(0.001000f), count);

// Intentionally empty: this test checks translation of the real mix example
// pipeline artifact instead of embedding a large step7_cann module inline.
