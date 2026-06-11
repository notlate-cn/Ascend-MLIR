// RUN: afir-opt %s --auto-fuse-tile-fuse 2>&1 | FileCheck %s --check-prefix=TILE
// RUN: afir-opt %s --auto-fuse-codegen 2>&1 | FileCheck %s --check-prefix=COMBINE
//
// Full-reduce-sum (no parallel axis) — exercises the RCore template:
//   out = sum_{d0}( x[d0] )
//
// Because there's no parallel axis, costEstimate marks every Common/FullLoad
// draft infeasible (g.yAxes.empty() → kInfeasible), and the RCore draft
// (reduceIsBlock=true, ubTilingAxisR on R) is the only feasible pick.
// AscendCRCoreCombinePass then rewrites the per-core write into the
// AF-kRCore-style two-segment pattern: each core writes its partial to
// workspace[block_idx], soft SyncAll, block 0 scalar-sums and writes the
// final scalar.

// Picker: single variant, RCore template, XBLOCK + RBLOCK_0 only.
// TILE: func.func @full_reduce__v0(
// TILE-SAME: %{{.*}}: index {auto_fuse.default_tile_size = 128 : i64}
// TILE-SAME: %{{.*}}: index {auto_fuse.default_tile_size = 64 : i64}
// TILE-SAME: afir.reduce_template = "RCore"
// TILE-NOT: func.func @full_reduce__v1

// Combine pass: soft-sync via SyncAll<false>(gmWs, ubWs, usedCores) +
// block-0-guarded scalar sum + DataCopyPad to output.  The hardware-flag
// SyncAll<false>() (no args) hangs on sim, so we explicitly emit the soft
// variant with an explicit workspace.
//
// COMBINE: func.func @full_reduce__v0(
// COMBINE-SAME: afir.reduce_template = "RCore"
// COMBINE-SAME: cann.num_inputs = 1 : i32
// COMBINE: AscendC::SyncAll<false>(_afir_sync_gm, _afir_sync_ub
// COMBINE: AscendC::GetBlockIdx() == 0
// COMBINE: _afir_total += _afir_ws_lt.GetValue
// COMBINE: AscendC::DataCopyPad(_afir_out_gt, _afir_res_lt

module {
  func.func @full_reduce(%x: tensor<?xf32>, %init: tensor<f32>) -> tensor<f32> {
    %out = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]}
        ins(%x : tensor<?xf32>) outs(%init : tensor<f32>) {
    ^bb0(%in: f32, %acc: f32):
      %v = arith.addf %acc, %in : f32
      linalg.yield %v : f32
    } -> tensor<f32>
    return %out : tensor<f32>
  }
}
