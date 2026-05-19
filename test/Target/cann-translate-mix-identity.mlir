// RUN: sed -e '/ascendc.duplicate_l2/d' -e '/ascendc.mul_l2/d' -e '/ascendc.max_l2/d' -e 's/ascendc.que_bind.enque_tensor %22, %106/ascendc.que_bind.enque_tensor %22, %95/' %S/cann-translate-mix-input.mlir | afir-translate -mlir-to-cann - | FileCheck %s

// Locks the supported mix path for matmul+bias fragments whose vector region
// only forwards the bias-add result to the physical output buffer. This is the
// generic shell needed by logical reshape/view outputs such as QKV head splits.
// CHECK-LABEL: extern "C" __global__ __aicore__ void matmul_add_leakyrelu(
// CHECK: mm.SetBias(biasGM);
// CHECK: const uint32_t vectorRegionOpCount = 2;
// CHECK: for (uint32_t i = 0; i < count; ++i)
// CHECK: outLocal.SetValue(i, inLocal.GetValue(i));
// CHECK: outLocal.SetSize(count);
// CHECK-NOT: LeakyRelu(
// CHECK-NOT: Relu(
