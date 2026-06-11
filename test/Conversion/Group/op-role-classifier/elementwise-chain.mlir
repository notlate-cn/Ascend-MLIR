// RUN: rm -rf %t && mkdir -p %t
// RUN: afir-opt --auto-fuse-group-analysis '--auto-fuse-group-outline=output-dir=%t' %s
// RUN: FileCheck %s < %t/network.provenance.json

// Multi-arith-op body in a single linalg.generic (rule 6 of OpRoleClassifier).
// Body has add + mul + max → classifier should return "elementwise_chain"
// regardless of which arith op appears first.

#map = affine_map<(d0) -> (d0)>

func.func @chain(%a: tensor<8xf16>, %b: tensor<8xf16>,
                  %c: tensor<8xf16>, %init: tensor<8xf16>)
    -> tensor<8xf16> {
  %r = linalg.generic {
    indexing_maps = [#map, #map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%a, %b, %c : tensor<8xf16>, tensor<8xf16>, tensor<8xf16>)
    outs(%init : tensor<8xf16>) {
  ^bb0(%x: f16, %y: f16, %z: f16, %o: f16):
    %s = arith.addf %x, %y : f16
    %p = arith.mulf %s, %z : f16
    %m = arith.maximumf %p, %x : f16
    linalg.yield %m : f16
  } -> tensor<8xf16>

  return %r : tensor<8xf16>
}

// CHECK: "fused_ops_summary": "elementwise_chain"
// CHECK: "op_role": "elementwise_chain"
