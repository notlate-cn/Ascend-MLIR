// RUN: afir-opt --vector-plan-group-analysis %s | FileCheck %s

// Reduce followed by pointwise: vertical fusion → same group_id.

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>

func.func @reduce_pointwise(%in: tensor<4x8xf16>,
                             %rinit: tensor<4xf16>,
                             %pinit: tensor<4xf16>) -> tensor<4xf16> {
  %r = linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "reduction"]
  } ins(%in : tensor<4x8xf16>) outs(%rinit : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16):
    %s = arith.addf %a, %b : f16
    linalg.yield %s : f16
  } -> tensor<4xf16>

  %out = linalg.generic {
    indexing_maps = [#map2, #map2],
    iterator_types = ["parallel"]
  } ins(%r : tensor<4xf16>) outs(%pinit : tensor<4xf16>) {
  ^bb0(%in2: f16, %o: f16):
    %relu = arith.maximumf %in2, %o : f16
    linalg.yield %relu : f16
  } -> tensor<4xf16>
  return %out : tensor<4xf16>
}

// Both ops share the same group_id.
// CHECK: vector_plan.group_id = [[G:[0-9]+]]
// CHECK: vector_plan.group_id = [[G]]
