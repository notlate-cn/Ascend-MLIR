// RUN: python3 -c "from pathlib import Path; src = Path(r'%S/../../examples/matmul-add-leakyrelu/step7_cann.mlir').read_text(); src = src.replace('1.000000e-03 : f32', '0.000000e+00 : f32', 1); Path(r'%t').write_text(src)"
// RUN: afir-translate -mlir-to-cann %t | FileCheck %s

// CHECK: #define __AFIR_RUNTIME_MIX_KERNEL_FUN_H__
// CHECK: KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
// CHECK: if ASCEND_IS_AIC {
// CHECK: REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
// CHECK: mm.SetBias(biasGM);
// CHECK: CrossCoreSetFlag<0x2, PIPE_FIX>(3);
// CHECK: if ASCEND_IS_AIV {
// CHECK: uint32_t count = static_cast<uint32_t>(tiling.singleCoreM * tiling.singleCoreN / 2);
// CHECK: CrossCoreWaitFlag(3);
// CHECK: Relu(outLocal, inLocal, count);
// CHECK-NOT: LeakyRelu(

// Intentionally empty: this test mutates the checked-in mix example to verify
// that translator-side epilogue inference switches from LeakyRelu(alpha) to
// Relu when the duplicated scalar constant becomes zero.
