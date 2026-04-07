// RUN: afir-translate -mlir-to-cann %S/cann-translate-mix-no-bias-input.mlir | FileCheck %s

// CHECK: #define __AFIR_RUNTIME_MIX_KERNEL_FUN_H__
// CHECK: KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
// CHECK: if ASCEND_IS_AIC {
// CHECK: REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
// CHECK: CrossCoreSetFlag<0x2, PIPE_FIX>(3);
// CHECK: if ASCEND_IS_AIV {
// CHECK: uint32_t count = static_cast<uint32_t>(tiling.singleCoreM * tiling.singleCoreN / 2);
// CHECK: CrossCoreWaitFlag(3);
// CHECK-NOT: biasGM
// CHECK-NOT: SetBias
// CHECK-NOT: mm.SetBias(
// CHECK: LeakyRelu(outLocal, inLocal, static_cast<float>(0.001000f), count);

// Intentionally empty: this test checks translation of a dedicated local fixture
// that translator-side bias inference can drop bias setup and mm.SetBias when
// the vector-region broadcast/add chain is absent.
