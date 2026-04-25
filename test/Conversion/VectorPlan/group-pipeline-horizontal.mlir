// RUN: afir-opt --vector-plan-group-analysis --vector-plan-group-outline %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>

func.func @horizontal_fuse(%x: tensor<8xf16>,
                           %y1: tensor<8xf16>,
                           %y2: tensor<8xf16>,
                           %init1: tensor<8xf16>,
                           %init2: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>) {
  %s1 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%x, %y1 : tensor<8xf16>, tensor<8xf16>) outs(%init1 : tensor<8xf16>) {
  ^bb0(%a: f16, %b: f16, %o: f16):
    %v = arith.addf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  %s2 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%x, %y2 : tensor<8xf16>, tensor<8xf16>) outs(%init2 : tensor<8xf16>) {
  ^bb0(%a: f16, %b: f16, %o: f16):
    %v = arith.addf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  return %s1, %s2 : tensor<8xf16>, tensor<8xf16>
}

// CHECK: func.func @horizontal_fuse(%[[X:.*]]: tensor<8xf16>, %[[Y1:.*]]: tensor<8xf16>, %[[Y2:.*]]: tensor<8xf16>, %[[I1:.*]]: tensor<8xf16>, %[[I2:.*]]: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>)
// CHECK: %[[R0:.*]], %[[R1:.*]] = func.call @kernel_group[[G:[0-9]+]](%[[X]], %[[Y1]], %[[Y2]]) : (tensor<8xf16>, tensor<8xf16>, tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>)
// CHECK: return %[[R0]], %[[R1]] : tensor<8xf16>, tensor<8xf16>
// CHECK: func.func private @kernel_group[[G]](%[[ARG0:.*]]: tensor<8xf16>, %[[ARG1:.*]]: tensor<8xf16>, %[[ARG2:.*]]: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>)
// CHECK-COUNT-2: linalg.generic
// CHECK-NOT: vector_plan.
