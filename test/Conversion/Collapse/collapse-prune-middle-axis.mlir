// RUN: afir-opt --vector-plan-tile-fuse %s | FileCheck %s
//
// 3 parallel axes. Input %x has map (d0,d1,d2)->(d0,d2): d1 absent, d0+d2 present
// → d1 is BCast (partial coverage). d1 sits in the middle of candidate {d0,d1,d2}.
// Pruning: [d0](size 1, discard) + [d2](size 1, discard) → no collapse.
//
// CHECK-LABEL: func.func @kernel_group0
func.func @kernel_group0(
    %in: tensor<4x3x8xf16>,
    %x:  tensor<4x8xf16>) -> tensor<4x3x8xf16> {
  %init = tensor.empty() : tensor<4x3x8xf16>
  %r = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%in, %x : tensor<4x3x8xf16>, tensor<4x8xf16>)
    outs(%init : tensor<4x3x8xf16>) {
    ^bb0(%a: f16, %b: f16, %o: f16):
      %add = arith.addf %a, %b : f16
      linalg.yield %add : f16
  } -> tensor<4x3x8xf16>
  return %r : tensor<4x3x8xf16>
}
// CHECK-NOT: tensor.collapse_shape
