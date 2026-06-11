// RUN: rm -rf %t && mkdir -p %t
// RUN: afir-opt --auto-fuse-group-analysis '--auto-fuse-group-outline=output-dir=%t' %s
// RUN: FileCheck %s --check-prefix=NET    < %t/network.mlir
// RUN: FileCheck %s --check-prefix=KERNEL < %t/kernel_group0.mlir

// Producer fans out to two consumers.  The consumers fuse horizontally first,
// then the producer merges vertically → all three ops end up in kernel_group0.

#map = affine_map<(d0) -> (d0)>

func.func @fanout(%x: tensor<8xf16>,
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

// network.mlir: single call (all three ops fused), no linalg ops.
// NET: func.func private @kernel_group0
// NET: func.func @fanout(
// NET:   call @kernel_group0
// NET-NOT: linalg.generic
// NET-NOT: auto_fuse.

// kernel_group0.mlir: all three linalg ops inside the kernel.
// KERNEL: func.func private @kernel_group0(
// KERNEL-COUNT-3: linalg.generic
// KERNEL-NOT: auto_fuse.
