// RUN: afir-opt %s --auto-fuse-tile-fuse 2>&1 | FileCheck %s
//
// P6c gate: FullLoad is enumerated for reduce funcs but always ∞-scored
// (oversized OR codegen-missing), so the picked plan is the Common path
// — visible here as RBLOCK_0 / XBLOCK_SUB tile params + a Common-shaped
// constraints list (Divides XBLOCK_SUB|XBLOCK, no FullLoad-specific entry).
//
// The R axis (524288 floats) is large enough to trigger the auto-split
// path; if FullLoad were selectable it would beat the R-split, but it's
// rejected here because its codegen path doesn't exist yet (and even if
// it did, 524288·4 bytes > kReductionTileBudgetBytes = 32 KiB would still
// reject as oversized).
//
// Companion of tile-fuse-tiling-infos.mlir which covers the pointwise case.

// CHECK: auto_fuse.tiling_infos
// constraints: XBLOCK_SUB|XBLOCK divides + LeBytes UB.  (The tail-offset 32B
// reject constraint was dropped: ragged tail GM store goes through DataCopyPad.)
// CHECK-SAME: constraints = [{kind = "divides", lhs = "XBLOCK_SUB", rhs = "XBLOCK"}, {kind = "le_bytes",
// CHECK-SAME: name = "XBLOCK"
// CHECK-SAME: name = "XBLOCK_SUB"
// CHECK-SAME: name = "RBLOCK_0"

#m_in   = affine_map<(d0, d1) -> (d0, d1)>
#m_init = affine_map<(d0, d1) -> (d0)>

func.func @reduce_big_r(%x: tensor<8x524288xf32>,
                        %init: tensor<8xf32>) -> tensor<8xf32> {
  %r = linalg.generic {
      indexing_maps = [#m_in, #m_init],
      iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<8x524288xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %v = arith.addf %acc, %in : f32
    linalg.yield %v : f32
  } -> tensor<8xf32>
  return %r : tensor<8xf32>
}
