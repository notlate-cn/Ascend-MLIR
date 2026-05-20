// RUN: afir-opt %s "--auto-fuse-tile-fuse=enable-reduction-split=true" 2>&1 | FileCheck %s
//
// Phase 4: reduction split with RBLOCK. d1 becomes RBLOCK_0 Inner parameter.
// Expected: XBLOCK + XBLOCK_SUB + RBLOCK_0 func args.
// Outer loop {ascendc.parallel}, inner parallel loop, linalg.fill, RBLOCK scf.for, insert_slice.

// With --enable-reduction-split=true both the Common (RBLOCK-split) and
// the FullLoad (R kept whole) drafts are feasible — enumerated in that order.
// v0 is the RBLOCK-split body this test was written for.
// CHECK: func.func @reduce_split__v0(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {auto_fuse.default_tile_size = 128 : i64}
// CHECK-SAME: %[[XBLOCK_SUB:[^ ,)]*]]: index {auto_fuse.default_tile_size = 16 : i64}
// CHECK-SAME: %[[RBLOCK:[^ ,)]*]]: index {auto_fuse.default_tile_size = 64 : i64}
// CHECK-SAME: afir.reduce_template = "Common"

// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[XBLOCK]]
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[XBLOCK]] step %[[XBLOCK_SUB]]
// CHECK: linalg.fill
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[RBLOCK]]
// CHECK: linalg.generic
// CHECK: tensor.insert_slice
// CHECK: } {ascendc.parallel}

// v1 is the FullLoad sibling — no RBLOCK, R kept whole.
// CHECK: func.func @reduce_split__v1(
// CHECK-SAME: afir.reduce_template = "FullLoad"

func.func @reduce_split(%a: tensor<1024x512xf32>,
                         %c: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%a : tensor<1024x512xf32>)
    outs(%c : tensor<1024xf32>) {
  ^bb0(%a0: f32, %acc: f32):
    %add = arith.addf %a0, %acc : f32
    linalg.yield %add : f32
  } -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}
