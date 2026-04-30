// RUN: afir-opt %s "--vector-plan-tile-fuse=max-full-loop-iters=8" 2>&1 | FileCheck %s
//
// BCast escape: d2=32 > max-full-loop-iters=8 → BCAST_TILE_0 (Inner) instead of BCast Full loop.
// Expected: 3 args (XBLOCK, XBLOCK_SUB, BCAST_TILE_0); NO BCast Full scf.for.
// Instead BCAST_TILE_0 appears as an additional Inner loop after XBLOCK_SUB.

// CHECK: func.func @bcast_escape(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {vector_plan.default_tile_size = 128 : i64}
// CHECK-SAME: %[[XBLOCK_SUB:[^ ,)]*]]: index {vector_plan.default_tile_size = 16 : i64}
// CHECK-SAME: %[[BCAST_TILE:[^ ,)]*]]: index {vector_plan.default_tile_size = 16 : i64}

// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[XBLOCK]]
// CHECK: } {ascendc.parallel}

func.func @bcast_escape(%a: tensor<1024x512x32xf32>,
                         %b: tensor<1024x512xf32>,
                         %c: tensor<1024x512x32xf32>) -> tensor<1024x512x32xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%a, %b : tensor<1024x512x32xf32>, tensor<1024x512xf32>)
    outs(%c : tensor<1024x512x32xf32>) {
  ^bb0(%a0: f32, %b0: f32, %c0: f32):
    %mul = arith.mulf %a0, %b0 : f32
    linalg.yield %mul : f32
  } -> tensor<1024x512x32xf32>
  return %result : tensor<1024x512x32xf32>
}
