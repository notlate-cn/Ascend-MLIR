// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @generic_matmul_like(%lhs: tensor<4x8xf16>,
                               %rhs: tensor<8x16xf16>,
                               %out: tensor<4x16xf16>) -> tensor<4x16xf16> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x16xf16>)
      outs(%out : tensor<4x16xf16>) {
    ^bb0(%x: f16, %y: f16, %acc: f16):
      %product = arith.mulf %x, %y : f16
      %sum = arith.addf %acc, %product : f16
      linalg.yield %sum : f16
    } -> tensor<4x16xf16>
  return %0 : tensor<4x16xf16>
}

// CHECK: DependencyAnalysis
// CHECK: op_id = 0
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: access = "Contraction"
// CHECK-SAME: iterators = [parallel, parallel, reduction]
// CHECK: OpRoleClassification
// CHECK: op_id = 0
// CHECK-SAME: roles = ["Primary", "Cube"]
// CHECK-SAME: op_role = "cube"
// CHECK: linalg.generic
// CHECK-SAME: ascend.op_role = "cube"
