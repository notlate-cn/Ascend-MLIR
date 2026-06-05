// Two independent elementwise CHAINS (2 fused ops each).
//   chain0:  t = a + b ;  out0 = t * e   -> kernel_group0 (add,mul fused)
//   chain1:  u = c * d ;  out1 = u + f   -> kernel_group1 (mul,add fused)
//
// The two chains share no inputs / no SSA, so --auto-fuse-group-analysis
// + --auto-fuse-group-outline still split into kernel_group0 + kernel_group1.
// Within each chain the SSA dep (t / u) makes group-analysis fuse the two ops
// into one Vector kernel. Intermediates t/u use tensor.empty() and are folded
// away by --linalg-fuse-elementwise-ops, so they need no init args.
//
// Shape 4x4 f16 keeps total elements (16) ≤ smallest XBLOCK candidate (16),
// so eval_block_dim returns 1 even when block_dim_expr is empty.
#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @model(%a: tensor<4x4xf16>, %b: tensor<4x4xf16>,
                  %c: tensor<4x4xf16>, %d: tensor<4x4xf16>,
                  %e: tensor<4x4xf16>, %f: tensor<4x4xf16>,
                  %i0: tensor<4x4xf16>, %i1: tensor<4x4xf16>)
    -> (tensor<4x4xf16>, tensor<4x4xf16>) {
  %et = tensor.empty() : tensor<4x4xf16>
  %t = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel", "parallel"]}
       ins(%a, %b : tensor<4x4xf16>, tensor<4x4xf16>) outs(%et : tensor<4x4xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.addf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<4x4xf16>
  %x = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel", "parallel"]}
       ins(%t, %e : tensor<4x4xf16>, tensor<4x4xf16>) outs(%i0 : tensor<4x4xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.mulf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<4x4xf16>
  %eu = tensor.empty() : tensor<4x4xf16>
  %u = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel", "parallel"]}
       ins(%c, %d : tensor<4x4xf16>, tensor<4x4xf16>) outs(%eu : tensor<4x4xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.mulf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<4x4xf16>
  %y = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel", "parallel"]}
       ins(%u, %f : tensor<4x4xf16>, tensor<4x4xf16>) outs(%i1 : tensor<4x4xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.addf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<4x4xf16>
  return %x, %y : tensor<4x4xf16>, tensor<4x4xf16>
}
