// RUN: afir-opt --vector-plan-group-analysis %s | FileCheck %s

// Single linalg op: assigned group_id=0 and topo_index=0.

func.func @single(%x: tensor<8xf16>, %init: tensor<8xf16>) -> tensor<8xf16> {
  %out = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%x : tensor<8xf16>) outs(%init : tensor<8xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<8xf16>
  return %out : tensor<8xf16>
}

// CHECK: vector_plan.group_id = 0
// CHECK: vector_plan.topo_index = 0
