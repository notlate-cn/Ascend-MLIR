// RUN: rm -rf %t && mkdir -p %t
// RUN: afir-opt --auto-fuse-group-analysis '--auto-fuse-group-outline=output-dir=%t' %s
// RUN: FileCheck %s < %t/network.provenance.json

// Four independent single-arith-op generics (rule 5 of OpRoleClassifier).
// Each gets its own kernel_groupN since they share no inputs/outputs;
// each kernel's source_ops[0].op_role should be the friendly arith op name
// ("add", "mul", "sub", "max" — the trailing 'f' / "imum" stripped).

#map = affine_map<(d0) -> (d0)>

func.func @single_arith(%a0: tensor<8xf16>, %b0: tensor<8xf16>, %i0: tensor<8xf16>,
                         %a1: tensor<8xf16>, %b1: tensor<8xf16>, %i1: tensor<8xf16>,
                         %a2: tensor<8xf16>, %b2: tensor<8xf16>, %i2: tensor<8xf16>,
                         %a3: tensor<8xf16>, %b3: tensor<8xf16>, %i3: tensor<8xf16>)
    -> (tensor<8xf16>, tensor<8xf16>, tensor<8xf16>, tensor<8xf16>) {
  %r0 = linalg.generic {
    indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]
  } ins(%a0, %b0 : tensor<8xf16>, tensor<8xf16>) outs(%i0 : tensor<8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  %r1 = linalg.generic {
    indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]
  } ins(%a1, %b1 : tensor<8xf16>, tensor<8xf16>) outs(%i1 : tensor<8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.mulf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  %r2 = linalg.generic {
    indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]
  } ins(%a2, %b2 : tensor<8xf16>, tensor<8xf16>) outs(%i2 : tensor<8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.subf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  %r3 = linalg.generic {
    indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]
  } ins(%a3, %b3 : tensor<8xf16>, tensor<8xf16>) outs(%i3 : tensor<8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.maximumf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  return %r0, %r1, %r2, %r3 : tensor<8xf16>, tensor<8xf16>, tensor<8xf16>, tensor<8xf16>
}

// CHECK: "op_role": "add"
// CHECK: "op_role": "mul"
// CHECK: "op_role": "sub"
// CHECK: "op_role": "max"
