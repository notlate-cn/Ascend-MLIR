// 1D full-reduce-sum — end-to-end example for the RCore template.
//
// Computation:
//   out = sum_{d0}( x[d0] )
//
// Input  shape: x[D0]    f32
// Output shape: out      f32  (scalar)
//
// Iterator types: ["reduction"].  There is **no parallel axis** at all — the
// only way to use multiple cores is to dispatch the R axis itself across
// cores.  TilePlanGen routes this through the **RCore** template:
//   - XBLOCK tiles the R axis across cores (block_dim = ceil(D0/XBLOCK))
//   - RBLOCK_0 tiles R within each core
// Each core computes a partial sum over its R slice.
//
// AscendCRCoreCombinePass then:
//   1. Redirects each core's partial → workspace[block_idx] (skipping a
//      reserved 256-byte soft-sync flag area at the start).
//   2. Inserts AscendC::SyncAll<false>(gmWs, ubWs, usedCores) — soft sync via
//      GM atomic counter (the hardware-flag SyncAll() variant hangs on sim).
//   3. Block 0 reads workspace[64..64+block_dim), scalar-sums the partials,
//      writes the final scalar to out.
//
// %init is the DPS accumulator; the kernel zero-initializes its per-core
// VECCALC acc internally (Duplicate 0), so the runtime-allocated output
// buffer does not need to be pre-zeroed.

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
