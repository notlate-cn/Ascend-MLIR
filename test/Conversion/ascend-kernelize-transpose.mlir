// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @named_transpose(%arg0: tensor<4x8xf16>) -> tensor<8x4xf16> {
  %empty = tensor.empty() : tensor<8x4xf16>
  %0 = linalg.transpose ins(%arg0 : tensor<4x8xf16>)
      outs(%empty : tensor<8x4xf16>) permutation = [1, 0]
  return %0 : tensor<8x4xf16>
}

func.func @generic_transpose(%arg0: tensor<4x8xf16>) -> tensor<8x4xf16> {
  %empty = tensor.empty() : tensor<8x4xf16>
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<4x8xf16>)
      outs(%empty : tensor<8x4xf16>) {
    ^bb0(%x: f16, %out: f16):
      linalg.yield %x : f16
    } -> tensor<8x4xf16>
  return %0 : tensor<8x4xf16>
}

func.func @named_rank3_transpose(%arg0: tensor<2x4x8xf16>)
    -> tensor<4x2x8xf16> {
  %empty = tensor.empty() : tensor<4x2x8xf16>
  %0 = linalg.transpose ins(%arg0 : tensor<2x4x8xf16>)
      outs(%empty : tensor<4x2x8xf16>) permutation = [1, 0, 2]
  return %0 : tensor<4x2x8xf16>
}

func.func @generic_rank3_transpose(%arg0: tensor<2x4x8xf16>)
    -> tensor<4x2x8xf16> {
  %empty = tensor.empty() : tensor<4x2x8xf16>
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d1, d0, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<2x4x8xf16>)
      outs(%empty : tensor<4x2x8xf16>) {
    ^bb0(%x: f16, %out: f16):
      linalg.yield %x : f16
    } -> tensor<4x2x8xf16>
  return %0 : tensor<4x2x8xf16>
}

// CHECK: DependencyAnalysis
// CHECK: op_id = 0
// CHECK-SAME: op = "linalg.transpose"
// CHECK-SAME: access = "LayoutTransform"
// CHECK-SAME: iterators = [parallel, parallel]
// CHECK: op_id = 1
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: access = "LayoutTransform"
// CHECK: op_id = 2
// CHECK-SAME: op = "linalg.transpose"
// CHECK-SAME: access = "LayoutTransform"
// CHECK-SAME: iterators = [parallel, parallel, parallel]
// CHECK: op_id = 3
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: access = "LayoutTransform"
// CHECK: OpRoleClassification
// CHECK: op_id = 0
// CHECK-SAME: roles = ["Primary", "Vector", "LayoutTransform"]
// CHECK-SAME: op_role = "vector"
// CHECK: op_id = 1
// CHECK-SAME: roles = ["Primary", "Vector", "LayoutTransform"]
// CHECK-SAME: op_role = "vector"
// CHECK: op_id = 2
// CHECK-SAME: roles = ["Primary", "Vector", "LayoutTransform"]
// CHECK-SAME: op_role = "vector"
// CHECK: op_id = 3
// CHECK-SAME: roles = ["Primary", "Vector", "LayoutTransform"]
// CHECK-SAME: op_role = "vector"
// CHECK: KernelPartition
// CHECK: kernel_pattern = "kernel_0"
// CHECK-SAME: internal_ops = [0]
// CHECK-SAME: primary_ops = [0]
// CHECK: kernel_pattern = "kernel_1"
// CHECK-SAME: internal_ops = [1]
// CHECK-SAME: primary_ops = [1]
// CHECK: linalg.transpose
// CHECK-SAME: ascend.kernel = "kernel_0"
// CHECK-SAME: ascend.op_role = "vector"
// CHECK: linalg.generic
// CHECK-SAME: ascend.kernel = "kernel_1"
// CHECK-SAME: ascend.op_role = "vector"
// CHECK: linalg.transpose
// CHECK-SAME: ascend.kernel = "kernel_2"
// CHECK-SAME: ascend.op_role = "vector"
// CHECK: linalg.generic
// CHECK-SAME: ascend.kernel = "kernel_3"
// CHECK-SAME: ascend.op_role = "vector"
