// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower --ascend-parallelize --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | tee %t.mlir | FileCheck %s
// RUN: python3 %S/../tools/check_ascend_queue_lifetime.py %t.mlir | FileCheck %s --check-prefix=LIFETIME

// CHECK-LABEL: func.func @queue_lifetime_full_pipeline
// CHECK: ascendc.que_bind.deque_tensor
// CHECK: ascendc.add_l2
// CHECK: ascendc.que_bind.free_tensor
// CHECK: ascendc.que_bind.deque_tensor
// CHECK: ascendc.que_bind.free_tensor
// CHECK-NOT: linalg.generic
// LIFETIME: queue_lifetime.ok {{.*}} deque=3 free=3

#identity = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @queue_lifetime_full_pipeline(
      %arg0: tensor<64x64xf16>,
      %arg1: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %empty = tensor.empty() : tensor<64x64xf16>
    %out = linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>)
      outs(%empty : tensor<64x64xf16>) {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %v = arith.addf %x, %y : f16
      linalg.yield %v : f16
    } -> tensor<64x64xf16>
    return %out : tensor<64x64xf16>
  }
}
