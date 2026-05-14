// Dynamic-shape trivial elementwise: out = x + bias.
// Signature matches the aclnn middle segment's tensor<?x?x?x?xf16> output.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func private @kernel_group1(
      %x: tensor<?x?x?x?xf16>, %bias: tensor<?x?x?x?xf16>,
      %init: tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16> {
    %r = linalg.generic {indexing_maps = [#map, #map, #map],
                         iterator_types = ["parallel","parallel","parallel","parallel"]}
         ins(%x, %bias : tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>)
         outs(%init : tensor<?x?x?x?xf16>) {
    ^bb0(%a: f16, %b: f16, %o: f16):
      %p = arith.addf %a, %b : f16
      linalg.yield %p : f16
    } -> tensor<?x?x?x?xf16>
    return %r : tensor<?x?x?x?xf16>
  }
}
