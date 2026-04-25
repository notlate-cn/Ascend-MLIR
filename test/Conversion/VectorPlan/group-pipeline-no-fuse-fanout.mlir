// RUN: afir-opt --vector-plan-group-analysis --vector-plan-group-outline %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>

func.func @no_fuse_fanout(%x: tensor<8xf16>,
                          %init0: tensor<8xf16>,
                          %init1: tensor<8xf16>,
                          %init2: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>) {
  %mid = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]
  } ins(%x : tensor<8xf16>) outs(%init0 : tensor<8xf16>) {
  ^bb0(%a: f16, %o: f16):
    linalg.yield %a : f16
  } -> tensor<8xf16>

  %use1 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]
  } ins(%mid : tensor<8xf16>) outs(%init1 : tensor<8xf16>) {
  ^bb0(%a: f16, %o: f16):
    %v = arith.negf %a : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  %use2 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]
  } ins(%mid : tensor<8xf16>) outs(%init2 : tensor<8xf16>) {
  ^bb0(%a: f16, %o: f16):
    %v = arith.maximumf %a, %o : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  return %use1, %use2 : tensor<8xf16>, tensor<8xf16>
}

// CHECK: func.func @no_fuse_fanout(%[[X:.*]]: tensor<8xf16>, %[[I0:.*]]: tensor<8xf16>, %[[I1:.*]]: tensor<8xf16>, %[[I2:.*]]: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>)
// CHECK: %[[MID:.*]] = func.call @kernel_group[[G0:[0-9]+]](%[[X]]) : (tensor<8xf16>) -> tensor<8xf16>
// CHECK: %[[R0:.*]], %[[R1:.*]] = func.call @kernel_group[[G1:[0-9]+]](%[[MID]]) : (tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>)
// CHECK: return %[[R0]], %[[R1]] : tensor<8xf16>, tensor<8xf16>
// CHECK: func.func private @kernel_group[[G0]](%[[ARG0:.*]]: tensor<8xf16>) -> tensor<8xf16>
// CHECK: linalg.generic
// CHECK: func.func private @kernel_group[[G1]](%[[ARG1:.*]]: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>)
// CHECK-COUNT-2: linalg.generic
// CHECK-NOT: vector_plan.
