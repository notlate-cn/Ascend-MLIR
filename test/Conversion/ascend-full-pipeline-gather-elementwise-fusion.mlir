// RUN: afir-opt %s --ascend-normalize --ascend-kernelize \
// RUN:   --ascend-schedule='target-tile-policy=legacy-default' \
// RUN:   --ascend-realize='materialization-mode=memory-space-annotate' \
// RUN:   --ascend-compute-lower --ascend-parallelize \
// RUN:   --ascend-prepare-for-emit --ascend-canonicalize-cann-signature \
// RUN:   | FileCheck %s --implicit-check-not=linalg.

// CHECK-LABEL: func.func @relu_index_select_add
// CHECK-SAME: ascend.schedule.tail_plan
// CHECK-SAME: selected = "
// CHECK-SAME: buffering = "
// CHECK-SAME: ascend.schedule.tail_policies
// CHECK-SAME: cann.num_inputs = 3 : i32
// CHECK: emitasc.member %arg5 "TB_M"
// CHECK: emitasc.member %arg5 "TB_N"
// CHECK: emitasc.member %arg5 "dim_arg1_0"
// CHECK: emitasc.member %arg5 "dim_arg0_1"
// CHECK: ascendc.get_block_idx
// CHECK: scf.if
// CHECK: scf.for [[K_TILE:%[^ ]+]] =
// CHECK: ascendc.pipe.init_queue
// CHECK: ascendc.tbuf.get_tensor
// CHECK: emitasc.reinterpret_cast %arg2
// CHECK: ascendc.global_tensor.set_global_buffer
// CHECK: ascendc.data_copy_l2
// CHECK: scf.for [[ROW:%[^ ]+]] =
// CHECK-NOT: arith.addi [[ROW]], [[K_TILE]]
// CHECK: arith.addi [[ROW]], {{%[0-9]+}} : index
// CHECK: ascendc.global_tensor.bracket
// CHECK: ascendc.que_bind.deque_tensor
// CHECK: emitasc.verbatim
// CHECK-NEXT: emitasc.verbatim
// CHECK-SAME: static_cast<half>(0)
// CHECK: ascendc.tbuf.get_with_offset
// CHECK: ascendc.gather_l2
// CHECK: ascendc.max_l2
// CHECK: ascendc.add_l2
// CHECK: emitasc.verbatim
// CHECK: ascendc.tbuf.get_tensor
// CHECK: emitasc.verbatim
// CHECK-SAME: _ascend_row_stride
// CHECK-NOT: ascendc.que_bind.deque_tensor
// CHECK: return

#col = affine_map<(d0, d1) -> (d1)>
#id = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @relu_index_select_add(%data: tensor<?x?xf16>,
                                   %indices: tensor<?xi64>,
                                   %bias: tensor<?xf16>)
      -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f16

    %dim_m = tensor.dim %data, %c0 : tensor<?x?xf16>
    %dim_n = tensor.dim %data, %c1 : tensor<?x?xf16>
    %dim_k = tensor.dim %indices, %c0 : tensor<?xi64>

    %empty_relu = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %relu_out = linalg.generic {
      indexing_maps = [#id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%data : tensor<?x?xf16>)
      outs(%empty_relu : tensor<?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      %v = arith.maximumf %in, %zero : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    %empty_gathered = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %gathered = linalg.generic {
      indexing_maps = [#col, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%indices : tensor<?xi64>)
      outs(%empty_gathered : tensor<?x?xf16>) {
    ^bb0(%idx: i64, %out: f16):
      %i = linalg.index 0 : index
      %idx_cast = arith.index_cast %idx : i64 to index
      %val = tensor.extract %relu_out[%i, %idx_cast] : tensor<?x?xf16>
      linalg.yield %val : f16
    } -> tensor<?x?xf16>

    %empty_out = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %out = linalg.generic {
      indexing_maps = [#id, #col, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%gathered, %bias : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_out : tensor<?x?xf16>) {
    ^bb0(%g: f16, %b: f16, %o: f16):
      %v = arith.addf %g, %b : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    return %out : tensor<?x?xf16>
  }
}
