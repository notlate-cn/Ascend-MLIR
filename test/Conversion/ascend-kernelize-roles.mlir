// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @branch_merge(%a: tensor<16xf32>, %b: tensor<16xf32>,
    %c: tensor<16xf32>) -> tensor<16xf32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%a : tensor<16xf32>)
      outs(%c : tensor<16xf32>) {
    ^bb0(%x: f32, %out: f32):
      linalg.yield %x : f32
    } -> tensor<16xf32>
  %1 = linalg.generic {
      ascend.branch_root = true,
      ascend.branch_group = 99 : i64,
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%0 : tensor<16xf32>)
      outs(%c : tensor<16xf32>) {
    ^bb0(%x: f32, %out: f32):
      %scale = arith.mulf %x, %x : f32
      linalg.yield %scale : f32
    } -> tensor<16xf32>
  %2 = linalg.generic {
      ascend.merge_root = true,
      ascend.merge_group = 88 : i64,
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%0 : tensor<16xf32>)
      outs(%c : tensor<16xf32>) {
    ^bb0(%x: f32, %out: f32):
      %one = "arith.constant"() <{value = 1.000000e+00 : f32}> {
        ascend.kernel = "stale_kernel",
        ascend.op_role = "stale_role",
        ascend.op_roles = ["StaleRole"],
        ascend.primary = true
      } : () -> f32
      %inc = arith.addf %x, %one : f32
      linalg.yield %inc : f32
    } -> tensor<16xf32>
  %3 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%1, %2 : tensor<16xf32>, tensor<16xf32>)
      outs(%b : tensor<16xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<16xf32>
  return %3 : tensor<16xf32>
}

func.func @reduction(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<4x8xf32>)
      outs(%empty : tensor<4xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %sum = arith.addf %acc, %x : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: StructuralMarking
// CHECK: op_id = 0
// CHECK-SAME: branch_root = true
// CHECK-SAME: branch_group = 0
// CHECK: op_id = 1
// CHECK-SAME: branch_group = 0
// CHECK-SAME: merge_group = 0
// CHECK: op_id = 2
// CHECK-SAME: branch_group = 0
// CHECK-SAME: merge_group = 0
// CHECK: op_id = 3
// CHECK-SAME: merge_root = true
// CHECK-SAME: merge_group = 0
// CHECK: OpRoleClassification
// CHECK: op_id = 0
// CHECK-SAME: roles = ["Primary", "Vector", "Injective", "Branch"]
// CHECK-SAME: op_role = "vector"
// CHECK: op_id = 1
// CHECK-SAME: roles = ["Primary", "Vector", "Injective", "Branch", "Merge"]
// CHECK-SAME: op_role = "vector"
// CHECK: op_id = 2
// CHECK-SAME: roles = ["Primary", "Vector", "Injective", "Branch", "Merge"]
// CHECK-SAME: op_role = "vector"
// CHECK: op_id = 3
// CHECK-SAME: roles = ["Primary", "Vector", "Injective", "Merge"]
// CHECK-SAME: op_role = "vector"
// CHECK: roles = ["Reduction"]
// CHECK-SAME: op_role = "reduction"
// CHECK-NOT: ascend.branch_group = 99
// CHECK-NOT: ascend.merge_group = 88
// CHECK-NOT: stale_kernel
// CHECK-NOT: stale_role
// CHECK-NOT: StaleRole
