// Dynamic-shape twin of two-elewise-e2e: two independent elementwise CHAINS
// (2 fused ops each), but every tensor is fully dynamic 3D `tensor<?x?x?xf16>`.
//   chain0:  t = a + b ;  out0 = t * e   -> kernel_group0 (add,mul fused)
//   chain1:  u = c * d ;  out1 = u + f   -> kernel_group1 (mul,add fused)
//
// All three axes are dynamic, so --afir-symbolize-shapes turns the tile axis
// extent into a symbolic product (s0*s1*s2) resolved from the input npy shape
// at launch — contrast the static 4x4 case whose axis_extent_expr is "16".
// Outputs use tensor.empty(%d0,%d1,%d2): dynamic dims need the runtime sizes,
// so there are no fixed init args (unlike the static case's %i0/%i1).
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @model(%a: tensor<?x?x?xf16>, %b: tensor<?x?x?xf16>,
                  %c: tensor<?x?x?xf16>, %d: tensor<?x?x?xf16>,
                  %e: tensor<?x?x?xf16>, %f: tensor<?x?x?xf16>)
    -> (tensor<?x?x?xf16>, tensor<?x?x?xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %a, %c0 : tensor<?x?x?xf16>
  %d1 = tensor.dim %a, %c1 : tensor<?x?x?xf16>
  %d2 = tensor.dim %a, %c2 : tensor<?x?x?xf16>

  %et = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf16>
  %t = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%a, %b : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%et : tensor<?x?x?xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.addf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<?x?x?xf16>
  %eo0 = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf16>
  %x = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%t, %e : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%eo0 : tensor<?x?x?xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.mulf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<?x?x?xf16>

  %eu = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf16>
  %u = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%c, %d : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%eu : tensor<?x?x?xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.mulf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<?x?x?xf16>
  %eo1 = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf16>
  %y = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%u, %f : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%eo1 : tensor<?x?x?xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.addf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<?x?x?xf16>

  return %x, %y : tensor<?x?x?xf16>, tensor<?x?x?xf16>
}
