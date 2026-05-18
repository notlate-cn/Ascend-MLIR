// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s
//
// Broadcast = a parallel axis: `out[d0,d1,d2] = a[d0,d1,d2] * b[d0,d1]` — b is
// constant along d2.  After collapse([0,1],[2]) the iteration space is
// [d0'(524288, parallel), d2(16, broadcast)].  d2 is *not* special to the
// scheduler — it's an ordinary non-ub parallel axis, so it goes whole-dim (no
// loop, no `BCAST_n` / `BCAST_TILE_n` tunable).  Only the block axis d0' gets
// XBLOCK + XBLOCK_SUB.  The tiled generic keeps b at its reduced shape
// (1-D, mapped (d0,d1)->(d0)) — LinalgToAscendC replicates it on-chip.

// CHECK-DAG: #[[ID2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[PROJ:.*]] = affine_map<(d0, d1) -> (d0)>

// Exactly two tunable args (XBLOCK, XBLOCK_SUB) — no BCAST*.
// CHECK: func.func @bcast_op__v0(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {vector_plan.default_tile_size = 128 : i64}
// CHECK-SAME: %[[XBLOCK_SUB:[^ ,)]*]]: index {vector_plan.default_tile_size = 16 : i64})

// 3-D operands collapsed to 2-D ([d0', d2]).
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1], [2]] : tensor<1024x512x16xf32> into tensor<524288x16xf32>

// Outer XBLOCK + inner XBLOCK_SUB — exactly two scf.for, no third loop over d2.
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[XBLOCK]]
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[XBLOCK_SUB]]

// b sliced at its reduced (1-D) shape; the tiled generic reads it (d0,d1)->(d0).
// CHECK: tensor.extract_slice %{{.*}}[%{{.*}}] [%[[XBLOCK_SUB]]] [%{{.*}}] : tensor<524288xf32> to tensor<?xf32>
// CHECK: linalg.generic {indexing_maps = [#[[ID2]], #[[PROJ]], #[[ID2]]], iterator_types = ["parallel", "parallel"]}
// CHECK: arith.mulf
// CHECK: } {ascendc.parallel}

func.func @bcast_op(%a: tensor<1024x512x16xf32>,
                    %b: tensor<1024x512xf32>,
                    %c: tensor<1024x512x16xf32>) -> tensor<1024x512x16xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%a, %b : tensor<1024x512x16xf32>, tensor<1024x512xf32>)
    outs(%c : tensor<1024x512x16xf32>) {
  ^bb0(%a0: f32, %b0: f32, %c0: f32):
    %mul = arith.mulf %a0, %b0 : f32
    linalg.yield %mul : f32
  } -> tensor<1024x512x16xf32>
  return %result : tensor<1024x512x16xf32>
}
