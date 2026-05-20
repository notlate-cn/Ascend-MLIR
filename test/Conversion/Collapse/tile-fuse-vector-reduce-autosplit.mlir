// RUN: afir-opt %s --auto-fuse-tile-fuse 2>&1 | FileCheck %s
//
// Tiling-case enumeration + cost model (P5b): a 2-D reduction whose reduce axis
// is way too big to fit on-chip whole.  enumerateTilingCases emits the "R kept
// whole" draft, the "R ub-split" draft (R·4B = 256 KiB > 32 KiB budget makes it
// a split candidate even *without* --enable-reduction-split), and an ∞-scored
// RCore variant; costEstimate rejects the first (and RCore) as infeasible, so
// the picked plan ub-splits d1 → an RBLOCK_0 tunable appears.

// CHECK: func.func @reduce_autosplit__v0(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {auto_fuse.default_tile_size = 128 : i64}
// CHECK-SAME: %[[XBLOCK_SUB:[^ ,)]*]]: index {auto_fuse.default_tile_size = 16 : i64}
// CHECK-SAME: %[[RBLOCK:[^ ,)]*]]: index {auto_fuse.default_tile_size = 64 : i64}
// CHECK-SAME: afir.reduce_template = "Common"

// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[XBLOCK]]
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[XBLOCK_SUB]]
// CHECK: linalg.fill
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[RBLOCK]]
// CHECK: linalg.generic
// CHECK: tensor.insert_slice
// CHECK: } {ascendc.parallel}

func.func @reduce_autosplit(%a: tensor<64x65536xf32>,
                             %c: tensor<64xf32>) -> tensor<64xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%a : tensor<64x65536xf32>)
    outs(%c : tensor<64xf32>) {
  ^bb0(%a0: f32, %acc: f32):
    %add = arith.addf %a0, %acc : f32
    linalg.yield %add : f32
  } -> tensor<64xf32>
  return %result : tensor<64xf32>
}
