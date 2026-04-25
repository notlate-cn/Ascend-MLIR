// RUN: afir-opt --vector-plan-group-analysis --vector-plan-group-outline %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>

func.func @single(%x: tensor<8xf16>, %init: tensor<8xf16>) -> tensor<8xf16> {
  %out = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]
  } ins(%x : tensor<8xf16>) outs(%init : tensor<8xf16>) {
  ^bb0(%in: f16, %o: f16):
    linalg.yield %in : f16
  } -> tensor<8xf16>
  return %out : tensor<8xf16>
}

// CHECK: func.func @single(%[[X:.*]]: tensor<8xf16>, %[[INIT:.*]]: tensor<8xf16>) -> tensor<8xf16>
// CHECK: %[[CALL:.*]] = func.call @kernel_group[[G:[0-9]+]](%[[X]]) : (tensor<8xf16>) -> tensor<8xf16>
// CHECK: return %[[CALL]] : tensor<8xf16>
// CHECK: func.func private @kernel_group[[G]](%[[ARG0:.*]]: tensor<8xf16>) -> tensor<8xf16>
// CHECK: linalg.generic
// CHECK-NOT: vector_plan.group_id
// CHECK-NOT: vector_plan.topo_index
