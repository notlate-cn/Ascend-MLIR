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
// CHECK: Kernelize report
