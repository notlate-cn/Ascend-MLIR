// RUN: afir-opt %s --ascend-normalize --ascend-kernelize | FileCheck %s

func.func @transpose_chain(%arg0: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %empty0 = tensor.empty() : tensor<8x4xf16>
  %t0 = linalg.transpose ins(%arg0 : tensor<4x8xf16>)
      outs(%empty0 : tensor<8x4xf16>) permutation = [1, 0]
  %empty1 = tensor.empty() : tensor<4x8xf16>
  %t1 = linalg.transpose ins(%t0 : tensor<8x4xf16>)
      outs(%empty1 : tensor<4x8xf16>) permutation = [1, 0]
  return %t1 : tensor<4x8xf16>
}

// CHECK: module attributes
// CHECK-SAME: ascend.kernel_graph.edges
// CHECK-SAME: carried_buffers = ["kernel_0_to_kernel_1_operand0"]
// CHECK-SAME: from = "kernel_0"
// CHECK-SAME: to = "kernel_1"
// CHECK: linalg.transpose
// CHECK-SAME: ascend.kernel = "kernel_0"
// CHECK: linalg.transpose
// CHECK-SAME: ascend.kernel = "kernel_1"
