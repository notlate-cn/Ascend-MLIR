// Two independent elementwise adds at extent 2048 (32×64 fp16).
// Goes through --input-linalg → group-outline → dynamic kernel_group{0,1}.
// Sized to maximize the surviving autotune trial count after block_dim ≤ 32
// and SUB ≤ XBLOCK pruning, so we can measure ROI of additional pruning.
#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @model(%a: tensor<32x64xf16>, %b: tensor<32x64xf16>,
                  %c: tensor<32x64xf16>, %d: tensor<32x64xf16>,
                  %i0: tensor<32x64xf16>, %i1: tensor<32x64xf16>)
    -> (tensor<32x64xf16>, tensor<32x64xf16>) {
  %x = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel", "parallel"]}
       ins(%a, %b : tensor<32x64xf16>, tensor<32x64xf16>) outs(%i0 : tensor<32x64xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.addf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<32x64xf16>
  %y = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel", "parallel"]}
       ins(%c, %d : tensor<32x64xf16>, tensor<32x64xf16>) outs(%i1 : tensor<32x64xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.mulf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<32x64xf16>
  return %x, %y : tensor<32x64xf16>, tensor<32x64xf16>
}
