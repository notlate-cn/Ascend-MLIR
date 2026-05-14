// Two completely independent elementwise linalg.generic ops.
// Designed to exercise the auto-outline path: --vector-plan-group-analysis
// + --vector-plan-group-outline should split this into kernel_group0 (add)
// and kernel_group1 (mul) with disjoint inputs.
//
// Shape 4x4 f16 keeps total elements (16) ≤ smallest XBLOCK candidate (16),
// so eval_block_dim returns 1 even when block_dim_expr is empty.
#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @model(%a: tensor<4x4xf16>, %b: tensor<4x4xf16>,
                  %c: tensor<4x4xf16>, %d: tensor<4x4xf16>,
                  %i0: tensor<4x4xf16>, %i1: tensor<4x4xf16>)
    -> (tensor<4x4xf16>, tensor<4x4xf16>) {
  %x = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel", "parallel"]}
       ins(%a, %b : tensor<4x4xf16>, tensor<4x4xf16>) outs(%i0 : tensor<4x4xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.addf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<4x4xf16>
  %y = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel", "parallel"]}
       ins(%c, %d : tensor<4x4xf16>, tensor<4x4xf16>) outs(%i1 : tensor<4x4xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.mulf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<4x4xf16>
  return %x, %y : tensor<4x4xf16>, tensor<4x4xf16>
}
