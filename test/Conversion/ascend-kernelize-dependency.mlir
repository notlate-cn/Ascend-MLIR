// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @dependency_chain(%a: tensor<16xf32>, %b: tensor<16xf32>, %c: tensor<16xf32>)
    -> tensor<16xf32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%a, %b : tensor<16xf32>, tensor<16xf32>)
      outs(%c : tensor<16xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<16xf32>
  %1 = linalg.generic {
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
  return %1 : tensor<16xf32>
}

func.func @broadcast_and_contraction(%a: tensor<16xf32>, %b: tensor<16x16xf32>,
    %c: tensor<16x16xf32>, %lhs: tensor<16x8xf32>, %rhs: tensor<8x32xf32>,
    %out: tensor<16x32xf32>) -> (tensor<16x16xf32>, tensor<16x32xf32>) {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<16xf32>, tensor<16x16xf32>)
      outs(%c : tensor<16x16xf32>) {
    ^bb0(%x: f32, %y: f32, %out_elem: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<16x16xf32>
  %1 = linalg.matmul
      ins(%lhs, %rhs : tensor<16x8xf32>, tensor<8x32xf32>)
      outs(%out : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0, %1 : tensor<16x16xf32>, tensor<16x32xf32>
}

func.func @non_projected_parallel_indexing(%a: tensor<32xf32>,
    %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0 + d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<32xf32>)
      outs(%c : tensor<16x16xf32>) {
    ^bb0(%x: f32, %out: f32):
      linalg.yield %x : f32
    } -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

func.func @transpose_indexing(%a: tensor<4x8xf32>, %c: tensor<8x4xf32>)
    -> tensor<8x4xf32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<4x8xf32>)
      outs(%c : tensor<8x4xf32>) {
    ^bb0(%x: f32, %out: f32):
      linalg.yield %x : f32
    } -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

func.func @mixed_broadcast_transpose_indexing(%a: tensor<4x8xf32>,
    %b: tensor<8xf32>, %c: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<4x8xf32>, tensor<8xf32>)
      outs(%c : tensor<8x4xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK: DependencyAnalysis
// CHECK: op_id = 0
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: access = "Elementwise"
// CHECK-SAME: producers = 0
// CHECK-SAME: consumers = 1
// CHECK: op_id = 1
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: access = "Elementwise"
// CHECK-SAME: producers = 1
// CHECK-SAME: consumers = 0
// CHECK-SAME: result_rank = 1
// CHECK-SAME: iterators = [parallel]
// CHECK-SAME: has_reduction = false
// CHECK-SAME: only_parallel = true
// CHECK: op_id = 2
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: access = "Broadcast"
// CHECK-SAME: result_rank = 2
// CHECK-SAME: iterators = [parallel, parallel]
// CHECK-SAME: has_reduction = false
// CHECK-SAME: only_parallel = true
// CHECK: op_id = 3
// CHECK-SAME: op = "linalg.matmul"
// CHECK-SAME: access = "Contraction"
// CHECK-SAME: result_rank = 2
// CHECK-SAME: iterators = [parallel, parallel, reduction]
// CHECK-SAME: has_reduction = true
// CHECK-SAME: only_parallel = false
// CHECK: op_id = 4
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: access = "Unknown"
// CHECK-SAME: result_rank = 2
// CHECK-SAME: iterators = [parallel, parallel]
// CHECK-SAME: has_reduction = false
// CHECK-SAME: only_parallel = true
// CHECK: op_id = 5
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: access = "LayoutTransform"
// CHECK-SAME: result_rank = 2
// CHECK-SAME: iterators = [parallel, parallel]
// CHECK-SAME: has_reduction = false
// CHECK-SAME: only_parallel = true
// CHECK: op_id = 6
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: access = "LayoutTransform"
// CHECK-SAME: result_rank = 2
// CHECK-SAME: iterators = [parallel, parallel]
// CHECK-SAME: has_reduction = false
// CHECK-SAME: only_parallel = true
// CHECK: Kernelize report
