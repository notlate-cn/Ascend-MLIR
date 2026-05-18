// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s
//
// Leading-axis broadcast = a parallel axis: `out[a,b,c] = x[a,b,c] + y[b,c]` — y
// is constant along the *leading* iteration dim a.  After collapse([1,2]) the
// iteration space is [a(8, parallel, broadcast), bc(128, parallel, block axis)].
// a is not special to the scheduler — it's an ordinary non-ub parallel axis, so
// it goes whole-dim (no loop, no `BCAST_n` / `BCAST_TILE_n` tunable).  Only the
// block axis bc gets XBLOCK + XBLOCK_SUB.  The tiled generic keeps y at its
// reduced shape (1-D, mapped (d0,d1)->(d1)) — LinalgToAscendC replicates it
// on-chip.

// CHECK-DAG: #[[ID2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[PROJ:.*]] = affine_map<(d0, d1) -> (d1)>

// Exactly two tunable args (XBLOCK, XBLOCK_SUB) — no BCAST*.
// CHECK: func.func @bcast_leading__v0(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {vector_plan.default_tile_size = 128 : i64}
// CHECK-SAME: %[[XBLOCK_SUB:[^ ,)]*]]: index {vector_plan.default_tile_size = 16 : i64})

// 3-D operands collapsed to 2-D ([a, bc]); y (rank-2) collapsed to 1-D.
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2]] : tensor<8x4x32xf32> into tensor<8x128xf32>
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1]] : tensor<4x32xf32> into tensor<128xf32>

// Outer XBLOCK + inner XBLOCK_SUB — exactly two scf.for, no third loop over a.
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[XBLOCK]]
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[XBLOCK_SUB]]

// y sliced at its reduced (1-D) shape; the tiled generic reads it (d0,d1)->(d1).
// CHECK: tensor.extract_slice %{{.*}}[%{{.*}}] [%[[XBLOCK_SUB]]] [%{{.*}}] : tensor<128xf32> to tensor<?xf32>
// CHECK: linalg.generic {indexing_maps = [#[[ID2]], #[[PROJ]], #[[ID2]]], iterator_types = ["parallel", "parallel"]}
// CHECK: arith.addf
// CHECK: } {ascendc.parallel}

func.func @bcast_leading(%x: tensor<8x4x32xf32>, %y: tensor<4x32xf32>) -> tensor<8x4x32xf32> {
  %o = tensor.empty() : tensor<8x4x32xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(a,b,c) -> (a,b,c)>,
                     affine_map<(a,b,c) -> (b,c)>,
                     affine_map<(a,b,c) -> (a,b,c)>],
    iterator_types = ["parallel","parallel","parallel"]}
    ins(%x, %y : tensor<8x4x32xf32>, tensor<4x32xf32>) outs(%o : tensor<8x4x32xf32>) {
  ^bb0(%xi: f32, %yi: f32, %oi: f32):
    %s = arith.addf %xi, %yi : f32
    linalg.yield %s : f32
  } -> tensor<8x4x32xf32>
  return %r : tensor<8x4x32xf32>
}
