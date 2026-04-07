// RUN: python3 -c "from pathlib import Path; src = Path(r'%S/../../examples/matmul-add-leakyrelu/step7_cann.mlir').read_text().splitlines(); out = [line for line in src if 'broadcast_l2' not in line and 'add_l2' not in line]; Path(r'%t').write_text('\n'.join(out) + '\n')"
// RUN: afir-translate -mlir-to-cann %t | FileCheck %s

// CHECK: #define __AFIR_RUNTIME_MIX_KERNEL_FUN_H__
// CHECK: KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
// CHECK: CrossCoreSetFlag<0x2, PIPE_FIX>(3);
// CHECK: uint32_t count = static_cast<uint32_t>(tiling.singleCoreM * tiling.singleCoreN / 2);
// CHECK: CrossCoreWaitFlag(3);
// CHECK-NOT: biasGM
// CHECK-NOT: SetBias
// CHECK: LeakyRelu(outLocal, inLocal, static_cast<float>(0.001000f), count);

// Intentionally empty: this test mutates the checked-in mix example to verify
// that translator-side bias inference can drop bias setup and mm.SetBias when
// the vector-region broadcast/add chain is absent.
