// RUN: afir-opt --auto-fuse-tile-fuse %s | FileCheck %s
//
// LayerNorm-style scale+bias:
//   input[4,8,16] — full (d0,d1,d2)→(d0,d1,d2)  Case C
//   scale[16]     — (d0,d1,d2)→(d2)              Case A: d0,d1 (all G-axes) absent
//   bias[16]      — (d0,d1,d2)→(d2)              Case A
//
// d2 is a reduction axis; the only candidate parallel group is G={d0,d1}.
// BCast analysis for candidate G={d0,d1}:
//   input:  all G-axes (d0,d1) present → full coverage → no BCast contribution.
//   scale:  d0 and d1 BOTH absent from map → ALL of G absent → Case A → no BCast contribution.
//   bias:   same → Case A → no BCast contribution.
//   BCast(G) = {} → no pruning → G={d0,d1} unchanged (size 2, keep) → collapse!
//
// Expected: tensor.collapse_shape [[0,1],[2]] on input → tensor<32x16xf16>
//           scale/bias (Case A) passed through unchanged — no collapse on them.
//           tensor.expand_shape to restore return type tensor<4x8x16xf16>
//
func.func @kernel_group0(
    %input: tensor<4x8x16xf16>,
    %scale: tensor<16xf16>,
    %bias:  tensor<16xf16>) -> tensor<4x8x16xf16> {
  %init = tensor.empty() : tensor<4x8x16xf16>
  %r = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d2)>,
      affine_map<(d0, d1, d2) -> (d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%input, %scale, %bias : tensor<4x8x16xf16>, tensor<16xf16>, tensor<16xf16>)
    outs(%init : tensor<4x8x16xf16>) {
    ^bb0(%a: f16, %s: f16, %b: f16, %o: f16):
      %mul = arith.mulf %a, %s : f16
      %add = arith.addf %mul, %b : f16
      linalg.yield %add : f16
  } -> tensor<4x8x16xf16>
  return %r : tensor<4x8x16xf16>
}

// Map aliases emitted for the 2D maps used in the collapsed generic.
// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d1)>
//
// CHECK-LABEL: func.func @kernel_group0
//
// collapse_shape group: Case C input collapses d0*d1 = 4*8 = 32
// CHECK: tensor.collapse_shape
// CHECK-SAME: into tensor<32x16xf16>
//
// scale and bias are Case A — no collapse applied to them
// CHECK-NOT: tensor.collapse_shape{{.*}}tensor<16xf16>
//
// linalg.generic group: uses 2D maps after collapse; iterator_types drops d0 parallel
// CHECK: linalg.generic
// CHECK-SAME: [#[[$MAP]], #[[$MAP1]], #[[$MAP1]], #[[$MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
//
// expand_shape group: restores return type
// CHECK: tensor.expand_shape
// CHECK-SAME: into tensor<4x8x16xf16>
