// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s
//
// Middle-axis broadcast = a parallel axis: `out[a,b,c] = x[a,b,c] + y[a,c]` — y
// is constant along the *middle* iteration dim b.  `a` and `c` can't collapse
// (b sits between them), so the iteration space stays 3-D
// [a(8, parallel, block axis), b(16, broadcast), c(32, parallel)].  b is not
// special to the scheduler — it's an ordinary parallel axis, so it goes
// whole-dim (no loop, no `BCAST_n` / `BCAST_TILE_n` tunable).  Only the block
// axis a gets XBLOCK + XBLOCK_SUB.  The tiled generic keeps y at its reduced
// shape (2-D, mapped (a,b,c)->(a,c)) — LinalgToAscendC replicates it on-chip.

// CHECK-DAG: #[[ID3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[PROJ:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// Exactly two tunable args (XBLOCK, XBLOCK_SUB) — no BCAST*.
// CHECK: func.func @bcast_middle(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {vector_plan.default_tile_size = 128 : i64}
// CHECK-SAME: %[[XBLOCK_SUB:[^ ,)]*]]: index {vector_plan.default_tile_size = 16 : i64})

// Block axis is `a` (extent 8) — no collapse_shape (b separates a from c).
// CHECK-NOT: tensor.collapse_shape

// Outer XBLOCK + inner XBLOCK_SUB — exactly two scf.for.
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[XBLOCK]]
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[XBLOCK_SUB]]

// y sliced at its reduced (2-D) shape; the tiled generic reads it (a,b,c)->(a,c).
// CHECK: tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}] [%[[XBLOCK_SUB]], %{{.*}}] [%{{.*}}, %{{.*}}] : tensor<8x32xf32> to tensor<?x?xf32>
// CHECK: linalg.generic {indexing_maps = [#[[ID3]], #[[PROJ]], #[[ID3]]], iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK: arith.addf
// CHECK: } {ascendc.parallel}

func.func @bcast_middle(%x: tensor<8x16x32xf32>, %y: tensor<8x32xf32>) -> tensor<8x16x32xf32> {
  %o = tensor.empty() : tensor<8x16x32xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(a,b,c) -> (a,b,c)>,
                     affine_map<(a,b,c) -> (a,c)>,
                     affine_map<(a,b,c) -> (a,b,c)>],
    iterator_types = ["parallel","parallel","parallel"]}
    ins(%x, %y : tensor<8x16x32xf32>, tensor<8x32xf32>) outs(%o : tensor<8x16x32xf32>) {
  ^bb0(%xi: f32, %yi: f32, %oi: f32):
    %s = arith.addf %xi, %yi : f32
    linalg.yield %s : f32
  } -> tensor<8x16x32xf32>
  return %r : tensor<8x16x32xf32>
}
