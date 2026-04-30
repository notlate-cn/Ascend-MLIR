// RUN: rm -rf %t && mkdir -p %t
// RUN: afir-opt --vector-plan-group-analysis '--vector-plan-group-outline=output-dir=%t' %s
// RUN: FileCheck %s --check-prefix=NET    < %t/network.mlir
// RUN: FileCheck %s --check-prefix=KERNEL < %t/kernel_group0.mlir

// Single elementwise op → outlined into kernel_group0.

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

// network.mlir: coordinator calls kernel, no linalg op, no vector_plan attrs.
// NET: func.func private @kernel_group0
// NET: func.func @single(
// NET:   call @kernel_group0
// NET-NOT: linalg.generic
// NET-NOT: vector_plan.

// kernel_group0.mlir: kernel contains the linalg op, no vector_plan attrs.
// KERNEL: func.func private @kernel_group0(
// KERNEL:   linalg.generic
// KERNEL-NOT: vector_plan.
