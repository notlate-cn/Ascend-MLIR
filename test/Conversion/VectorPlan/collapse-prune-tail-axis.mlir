// RUN: afir-opt --vector-plan-tile-fuse %s | FileCheck %s
//
// 3 parallel axes. Input %x has map (d0,d1,d2)->(d0,d1): d2 absent, d0+d1 present
// → d2 is BCast (partial). d2 is at tail of candidate {d0,d1,d2}.
// Pruning: [d0,d1](size 2, keep) → collapse d0*d1.
//
// CHECK-LABEL: func.func @kernel_group0
func.func @kernel_group0(
    %in: tensor<4x8x16xf16>,
    %x:  tensor<4x8xf16>) -> tensor<4x8x16xf16> {
  %init = tensor.empty() : tensor<4x8x16xf16>
  %r = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%in, %x : tensor<4x8x16xf16>, tensor<4x8xf16>)
    outs(%init : tensor<4x8x16xf16>) {
    ^bb0(%a: f16, %b: f16, %o: f16):
      %add = arith.addf %a, %b : f16
      linalg.yield %add : f16
  } -> tensor<4x8x16xf16>
  return %r : tensor<4x8x16xf16>
}
// d2 BCast → prune to [d0,d1] → collapse d0*d1 = 32
// CHECK: tensor.collapse_shape
// CHECK-SAME: into tensor<32x16xf16>
