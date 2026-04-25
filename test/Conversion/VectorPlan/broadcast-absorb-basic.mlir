// RUN: afir-opt --vector-plan-broadcast-absorb %s | FileCheck %s

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-LABEL: func.func @absorb_into_generic
func.func @absorb_into_generic(
    %x: tensor<4x8xf16>,
    %y: tensor<4x3x8xf16>) -> tensor<4x3x8xf16> {
  %init = tensor.empty() : tensor<4x3x8xf16>
  %b = linalg.broadcast ins(%x : tensor<4x8xf16>)
                        outs(%init : tensor<4x3x8xf16>)
                        dimensions = [1]
  %r = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%b, %y : tensor<4x3x8xf16>, tensor<4x3x8xf16>)
    outs(%init : tensor<4x3x8xf16>) {
    ^bb0(%a: f16, %bv: f16, %o: f16):
      %add = arith.addf %a, %bv : f16
      linalg.yield %add : f16
  } -> tensor<4x3x8xf16>
  return %r : tensor<4x3x8xf16>
}
// CHECK-NOT: linalg.broadcast
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[$MAP]],
