// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower --ascend-parallelize --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | FileCheck %s

// CHECK-LABEL: func.func @rank2_elementwise_add
// CHECK-SAME: %{{.*}}: memref<70x128xf16>
// CHECK-SAME: %{{.*}}: memref<ui8>
// CHECK-SAME: !emitasc.py_struct<"TilingData"
// CHECK-SAME: ascend.schedule.selected_tile_shape = array<i64: 32, 128>
// CHECK-SAME: ascend.schedule.tail_policies = ["masked_tail", "masked_tail"]
// CHECK-SAME: cann.num_inputs = 2 : i32
// CHECK: ascendc.get_block_idx
// CHECK: arith.muli %{{.*}}, %c32
// CHECK: scf.if
// CHECK: ascendc.data_copy_l2
// CHECK: ascendc.add_l2
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: scf.for
// CHECK-NOT: linalg.generic

#identity = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @rank2_elementwise_add(
      %arg0: tensor<70x128xf16>,
      %arg1: tensor<70x128xf16>) -> tensor<70x128xf16> {
    %empty = tensor.empty() : tensor<70x128xf16>
    %out = linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0, %arg1 : tensor<70x128xf16>, tensor<70x128xf16>)
      outs(%empty : tensor<70x128xf16>) {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %v = arith.addf %x, %y : f16
      linalg.yield %v : f16
    } -> tensor<70x128xf16>
    return %out : tensor<70x128xf16>
  }
}
