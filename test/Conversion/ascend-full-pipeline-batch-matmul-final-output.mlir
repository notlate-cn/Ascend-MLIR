// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower --ascend-parallelize --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | FileCheck %s --implicit-check-not=linalg.

// CHECK-LABEL: func.func @batch_matmul_final_output
// CHECK-SAME: ascendc.kernel_kind = "cube"
// CHECK: ascendc.mmad
// CHECK: ascendc.data_copy_co12dst
// CHECK: ascendc.global_tensor
// CHECK: ascendc.global_tensor.set_global_buffer
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: memref.store

module {
  func.func @batch_matmul_final_output(
      %lhs: tensor<2x4x8xf16>,
      %rhs: tensor<2x8x16xf16>) -> tensor<2x4x16xf32> {
    %empty = tensor.empty() : tensor<2x4x16xf32>
    %out = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<2x4x8xf16>, tensor<2x8x16xf16>)
        outs(%empty : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
    return %out : tensor<2x4x16xf32>
  }
}
