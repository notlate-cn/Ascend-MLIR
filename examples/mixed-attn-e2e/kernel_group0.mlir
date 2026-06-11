#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func private @kernel_group0(
      %q: tensor<1x1x2x8xf16>, %scale: tensor<1x1x2x8xf16>,
      %bias: tensor<1x1x2x8xf16>, %init: tensor<1x1x2x8xf16>)
      -> tensor<1x1x2x8xf16> {
    %r = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                         iterator_types = ["parallel","parallel","parallel","parallel"]}
         ins(%q, %scale, %bias : tensor<1x1x2x8xf16>, tensor<1x1x2x8xf16>, tensor<1x1x2x8xf16>)
         outs(%init : tensor<1x1x2x8xf16>) {
    ^bb0(%a: f16, %s: f16, %b: f16, %o: f16):
      %m = arith.mulf %a, %s : f16
      %p = arith.addf %m, %b : f16
      linalg.yield %p : f16
    } -> tensor<1x1x2x8xf16>
    return %r : tensor<1x1x2x8xf16>
  }
}
