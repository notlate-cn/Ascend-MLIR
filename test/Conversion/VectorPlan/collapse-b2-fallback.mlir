// RUN: afir-opt --vector-plan-tile-fuse %s | FileCheck %s
//
// One input has G-axes present but non-consecutively ordered → B2 classification.
// hasAnyB2=true → noCollapse=true → no IR transform.
//
// CHECK-LABEL: func.func @kernel_group0
func.func @kernel_group0(
    %in:  tensor<4x8x16xf32>,
    %b2:  tensor<8x4x16xf32>) -> tensor<4x8x16xf32> {
  %init = tensor.empty() : tensor<4x8x16xf32>
  %r = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d0, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%in, %b2 : tensor<4x8x16xf32>, tensor<8x4x16xf32>)
    outs(%init : tensor<4x8x16xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %add = arith.addf %a, %b : f32
      linalg.yield %add : f32
  } -> tensor<4x8x16xf32>
  return %r : tensor<4x8x16xf32>
}
// B2 input → noCollapse → no tensor.collapse_shape
// CHECK-NOT: tensor.collapse_shape
