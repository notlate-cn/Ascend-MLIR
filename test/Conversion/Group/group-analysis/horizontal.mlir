// RUN: afir-opt --auto-fuse-group-analysis %s | FileCheck %s

// Two elementwise ops sharing input %x: horizontal fusion → same group_id.

#map = affine_map<(d0) -> (d0)>

func.func @horizontal(%x: tensor<8xf16>,
                       %y1: tensor<8xf16>, %y2: tensor<8xf16>,
                       %init1: tensor<8xf16>, %init2: tensor<8xf16>)
    -> (tensor<8xf16>, tensor<8xf16>) {
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

// Both ops share the same group_id.
// CHECK: auto_fuse.group_id = [[G:[0-9]+]]
// CHECK: auto_fuse.group_id = [[G]]
