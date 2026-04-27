// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s
//
// 1-D pointwise (single parallel axis d0=32768, no collapse).
// Phase 2 must add XBLOCK (default=128) and XBLOCK_SUB (default=16) func args.

// CHECK: func.func @pointwise(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {vector_plan.default_tile_size = 128 : i64}
// CHECK-SAME: %[[XBLOCK_SUB:[^ ,)]*]]: index {vector_plan.default_tile_size = 16 : i64}

func.func @pointwise(%a: tensor<32768xf32>, %b: tensor<32768xf32>,
                     %c: tensor<32768xf32>) -> tensor<32768xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%a, %b : tensor<32768xf32>, tensor<32768xf32>)
    outs(%c : tensor<32768xf32>) {
  ^bb0(%a0: f32, %b0: f32, %c0: f32):
    %add = arith.addf %a0, %b0 : f32
    linalg.yield %add : f32
  } -> tensor<32768xf32>
  return %result : tensor<32768xf32>
}
