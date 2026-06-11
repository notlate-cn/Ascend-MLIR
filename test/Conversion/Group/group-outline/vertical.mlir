// RUN: rm -rf %t && mkdir -p %t
// RUN: afir-opt --auto-fuse-group-analysis '--auto-fuse-group-outline=output-dir=%t' %s
// RUN: FileCheck %s --check-prefix=NET    < %t/network.mlir
// RUN: FileCheck %s --check-prefix=KERNEL < %t/kernel_group0.mlir

// Reduce + pointwise vertically fused → single kernel_group0 with both ops.

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>

func.func @reduce_pointwise(%in: tensor<4x8xf16>,
                             %rinit: tensor<4xf16>,
                             %pinit: tensor<4xf16>) -> tensor<4xf16> {
  %r = linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "reduction"]
  } ins(%in : tensor<4x8xf16>) outs(%rinit : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16):
    %s = arith.addf %a, %b : f16
    linalg.yield %s : f16
  } -> tensor<4xf16>

  %out = linalg.generic {
    indexing_maps = [#map2, #map2],
    iterator_types = ["parallel"]
  } ins(%r : tensor<4xf16>) outs(%pinit : tensor<4xf16>) {
  ^bb0(%in2: f16, %o: f16):
    %relu = arith.maximumf %in2, %o : f16
    linalg.yield %relu : f16
  } -> tensor<4xf16>
  return %out : tensor<4xf16>
}

// network.mlir: single call to kernel_group0, no linalg ops.
// NET: func.func private @kernel_group0
// NET: func.func @reduce_pointwise(
// NET:   call @kernel_group0
// NET-NOT: linalg.generic
// NET-NOT: auto_fuse.

// kernel_group0.mlir: both linalg ops present, no attributes.
// KERNEL: func.func private @kernel_group0(
// KERNEL-COUNT-2: linalg.generic
// KERNEL-NOT: auto_fuse.
