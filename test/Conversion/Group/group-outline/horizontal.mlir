// RUN: rm -rf %t && mkdir -p %t
// RUN: afir-opt --vector-plan-group-analysis '--vector-plan-group-outline=output-dir=%t' %s
// RUN: FileCheck %s --check-prefix=NET    < %t/network.mlir
// RUN: FileCheck %s --check-prefix=KERNEL < %t/kernel_group0.mlir

// Two elementwise ops sharing %x: horizontally fused → single kernel_group0.

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

// network.mlir: single call returning two tensors, no linalg ops.
// NET: func.func private @kernel_group0
// NET: func.func @horizontal(
// NET:   call @kernel_group0
// NET-NOT: linalg.generic
// NET-NOT: vector_plan.

// kernel_group0.mlir: both linalg ops present, returns two tensors.
// KERNEL: func.func private @kernel_group0(
// KERNEL-COUNT-2: linalg.generic
// KERNEL-NOT: vector_plan.
