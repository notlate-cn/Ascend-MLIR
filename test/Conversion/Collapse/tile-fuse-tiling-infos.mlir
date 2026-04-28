// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: vector_plan.tiling_infos = [{fields = [{abi_index = 0 : i32, arg_index = 2 : i32, default_value = 128 : i64, kind = "tunable", name = "XBLOCK"}, {abi_index = 1 : i32, arg_index = 3 : i32, default_value = 16 : i64, kind = "tunable", name = "XBLOCK_SUB"}], kernel_id = "pointwise"}]

func.func @pointwise(%a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%a : tensor<1024xf32>) outs(%b : tensor<1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}
