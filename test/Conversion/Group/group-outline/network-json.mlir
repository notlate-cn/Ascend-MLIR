// RUN: rm -rf %t && mkdir -p %t
// RUN: afir-opt --vector-plan-group-analysis '--vector-plan-group-outline=output-dir=%t' %s
// RUN: FileCheck %s < %t/network.json

#map = affine_map<(d0) -> (d0)>

func.func @two(%a: tensor<8xf16>, %b: tensor<8xf16>,
                %c: tensor<8xf16>, %d: tensor<8xf16>,
                %i0: tensor<8xf16>, %i1: tensor<8xf16>)
    -> (tensor<8xf16>, tensor<8xf16>) {
  %x = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<8xf16>, tensor<8xf16>) outs(%i0 : tensor<8xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.addf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>
  %y = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel"]}
       ins(%c, %d : tensor<8xf16>, tensor<8xf16>) outs(%i1 : tensor<8xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.mulf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>
  return %x, %y : tensor<8xf16>, tensor<8xf16>
}

// CHECK:      "function": "two"
// CHECK:      "inputs": [
// CHECK:        "dtype": "f16"
// CHECK:        "name": "arg0"
// CHECK:        "shape": [
// CHECK:          8
// CHECK:        ]
// CHECK-COUNT-5: "dtype": "f16"
// CHECK:      "kernels":
// CHECK:        "file": "kernel_group0.mlir"
// CHECK:        "id": "kernel_group0"
// CHECK:        "kind": "ascendc"
// CHECK:        "file": "kernel_group1.mlir"
// CHECK:        "id": "kernel_group1"
// CHECK:        "kind": "ascendc"
// CHECK:      "outputs":
// CHECK:        "from": "kernel"
// CHECK:        "kernel": "kernel_group0"
// CHECK:        "from": "kernel"
// CHECK:        "kernel": "kernel_group1"
