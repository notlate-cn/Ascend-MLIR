// P6 demo for multi-variant tiling. The bcast on dim-0 forces Collapse to
// keep the [M, N] iteration space split (M=256 parallel, N=64 broadcast-y),
// so enumerateTilingCases produces TWO ubY candidates and — with P6's
// relax-non-block-uby cost-model knob — both pass feasibility, producing
// kernel_group0__v0 (ubY=M) and kernel_group0__v1 (ubY=N).
//
// The autotuner --family mode then profiles both and picks the faster one.

#map_full  = affine_map<(d0, d1) -> (d0, d1)>
#map_bcast = affine_map<(d0, d1) -> (d1)>

func.func @model(%a: tensor<256x64xf16>, %b: tensor<64xf16>,
                  %init: tensor<256x64xf16>) -> tensor<256x64xf16> {
  %r = linalg.generic {
    indexing_maps = [#map_full, #map_bcast, #map_full],
    iterator_types = ["parallel", "parallel"]
  } ins(%a, %b : tensor<256x64xf16>, tensor<64xf16>)
    outs(%init : tensor<256x64xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<256x64xf16>
  return %r : tensor<256x64xf16>
}
