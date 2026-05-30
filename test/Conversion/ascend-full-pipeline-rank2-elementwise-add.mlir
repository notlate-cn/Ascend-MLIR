// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower --ascend-parallelize --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | FileCheck %s

// CHECK-LABEL: func.func @rank2_elementwise_add
// CHECK-SAME: %{{.*}}: memref<70x128xf16>
// CHECK-SAME: %{{.*}}: memref<ui8>
// CHECK-SAME: !emitasc.py_struct<"TilingData"
// CHECK-SAME: ["TB_M", "TB_N"
// CHECK-SAME: ascend.schedule.tail_policies = ["masked_tail", "masked_tail"]
// CHECK-SAME: ascend.schedule.tile_binding = "symbolic"
// CHECK-SAME: cann.num_inputs = 2 : i32
// CHECK: emitasc.member %{{.*}} "TB_M"
// CHECK: emitasc.member %{{.*}} "TB_N"
// CHECK: arith.index_cast %{{.*}} : i64 to index
// CHECK: ascendc.get_block_idx
// CHECK: arith.muli %{{.*}}, %{{.*}} : index
// CHECK: scf.if
// CHECK: scf.for
// CHECK: emitasc.verbatim
// CHECK: ascendc.add_l2
// CHECK-NEXT: ascendc.pipe_barrier pipe_all
// CHECK-NEXT: ascendc.que_bind.free_tensor
// CHECK-NEXT: ascendc.que_bind.free_tensor
// CHECK: emitasc.verbatim
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
