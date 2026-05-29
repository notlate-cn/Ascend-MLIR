// RUN: rm -rf %t && mkdir -p %t
// RUN: afir-opt --auto-fuse-group-analysis '--auto-fuse-group-outline=output-dir=%t' %s
// RUN: FileCheck %s < %t/network.provenance.json

// Producer-consumer fusion of two single-arith generics:
//   gen0: addf  → role "add"
//   gen1: mulf consuming gen0  → role "mul"
// Vertically fused into a single kernel_groupN.  Expected:
//   - source_ops has TWO entries (one per generic)
//   - fused_ops_summary joins them in topological order → "add+mul"
// Order matters: producer first, consumer second.

#map = affine_map<(d0) -> (d0)>

func.func @add_then_mul(%a: tensor<8xf16>, %b: tensor<8xf16>, %c: tensor<8xf16>,
                         %init1: tensor<8xf16>, %init2: tensor<8xf16>)
    -> tensor<8xf16> {
  %s = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%a, %b : tensor<8xf16>, tensor<8xf16>) outs(%init1 : tensor<8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  %p = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%s, %c : tensor<8xf16>, tensor<8xf16>) outs(%init2 : tensor<8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.mulf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  return %p : tensor<8xf16>
}

// CHECK: "fused_ops_summary": "add+mul"
// CHECK: "op_role": "add"
// CHECK: "op_role": "mul"
