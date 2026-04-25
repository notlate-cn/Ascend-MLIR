// RUN: afir-opt --vector-plan-group-analysis --vector-plan-group-outline %s | FileCheck %s

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

// CHECK: func.func @reduce_pointwise(%[[IN:.*]]: tensor<4x8xf16>, %[[RINIT:.*]]: tensor<4xf16>, %[[PINIT:.*]]: tensor<4xf16>) -> tensor<4xf16>
// CHECK: %[[CALL:.*]] = func.call @kernel_group[[G:[0-9]+]](%[[IN]]) : (tensor<4x8xf16>) -> tensor<4xf16>
// CHECK: return %[[CALL]] : tensor<4xf16>
// CHECK: func.func private @kernel_group[[G]](%[[ARG0:.*]]: tensor<4x8xf16>) -> tensor<4xf16>
// CHECK: linalg.generic
// CHECK: linalg.generic
// CHECK-NOT: vector_plan.
