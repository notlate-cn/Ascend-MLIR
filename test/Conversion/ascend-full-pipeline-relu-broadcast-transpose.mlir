// RUN: afir-opt %s --linalg-generalize-named-ops --linalg-fuse-elementwise-ops --ascend-normalize --ascend-kernelize --ascend-schedule --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower --ascend-parallelize --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | FileCheck %s

// CHECK-LABEL: func.func @relu_transpose_broadcast_add
// CHECK-SAME: %{{.*}}: memref<?x1xf16>
// CHECK-SAME: %{{.*}}: memref<?x?xf16>
// CHECK-SAME: %{{.*}}: memref<?x?xf16
// CHECK-SAME: %{{.*}}: memref<ui8>
// CHECK-SAME: !emitasc.py_struct<"TilingData"
// CHECK-SAME: ["dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg0_1"]
// CHECK-SAME: ascend.schedule.tail_policies = ["masked_tail", "masked_tail"]
// CHECK-SAME: cann.num_inputs = 2 : i32
// CHECK-NOT: linalg.generic
// CHECK: emitasc.member %{{.*}} "dim_arg0_0"
// CHECK-NEXT: emitasc.member %{{.*}} "dim_arg1_0"
// CHECK-NEXT: emitasc.member %{{.*}} "dim_arg1_1"
// CHECK-NEXT: emitasc.member %{{.*}} "dim_arg0_1"
// CHECK-NOT: linalg.generic
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: linalg.generic
// CHECK: ascendc.broadcast_l2
// CHECK-NOT: linalg.generic
// CHECK: ascendc.duplicate_l2
// CHECK-NOT: linalg.generic
// CHECK: ascendc.max_l2
// CHECK-NOT: linalg.generic
// CHECK: ascendc.add_l2
// CHECK-NOT: linalg.generic
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: linalg.generic
// CHECK: return
// CHECK-NOT: linalg.generic

module {
  func.func @relu_transpose_broadcast_add(
      %data0: tensor<?x1xf16>,
      %data1: tensor<?x?xf16>) -> tensor<?x?xf16> {

    %c0 = arith.constant 0 : index
    %m = tensor.dim %data0, %c0 : tensor<?x1xf16>
    %n = tensor.dim %data1, %c0 : tensor<?x?xf16>

    %zero_f16 = arith.constant 0.0 : f16
    %relu_init = tensor.empty(%m) : tensor<?x1xf16>
    %relu = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%data0 : tensor<?x1xf16>)
      outs(%relu_init : tensor<?x1xf16>) {
    ^bb0(%x: f16, %out: f16):
      %r = arith.maximumf %x, %zero_f16 : f16
      linalg.yield %r : f16
    } -> tensor<?x1xf16>

    %out_init = tensor.empty(%n, %m) : tensor<?x?xf16>
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, 0)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%relu, %data1 : tensor<?x1xf16>, tensor<?x?xf16>)
      outs(%out_init : tensor<?x?xf16>) {
    ^bb0(%relu_val: f16, %data_val: f16, %out: f16):
      %sum = arith.addf %relu_val, %data_val : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>

    return %result : tensor<?x?xf16>
  }
}
