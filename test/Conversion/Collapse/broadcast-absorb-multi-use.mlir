// RUN: afir-opt --vector-plan-broadcast-absorb %s | FileCheck %s
//
// broadcast result has two uses (generic + return) → pattern does not fire → preserved.
//
// CHECK-LABEL: func.func @multi_use_broadcast
func.func @multi_use_broadcast(%x: tensor<4x8xf16>) -> (tensor<4x3x8xf16>, tensor<4x3x8xf16>) {
  %init = tensor.empty() : tensor<4x3x8xf16>
  %b = linalg.broadcast ins(%x : tensor<4x8xf16>)
                        outs(%init : tensor<4x3x8xf16>)
                        dimensions = [1]
  %r = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%b : tensor<4x3x8xf16>)
    outs(%init : tensor<4x3x8xf16>) {
    ^bb0(%a: f16, %o: f16):
      linalg.yield %a : f16
  } -> tensor<4x3x8xf16>
  return %b, %r : tensor<4x3x8xf16>, tensor<4x3x8xf16>
}
// broadcast preserved (two uses)
// CHECK: linalg.broadcast
