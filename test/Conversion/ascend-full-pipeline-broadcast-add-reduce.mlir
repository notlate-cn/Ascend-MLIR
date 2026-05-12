// RUN: afir-opt %s --linalg-fuse-elementwise-ops --ascend-normalize --ascend-kernelize --ascend-schedule --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower --ascend-parallelize --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | FileCheck %s

// CHECK-LABEL: func.func @broadcast_add_reducesum
// CHECK-SAME: %{{.*}}: memref<ui8>
// CHECK-SAME: %{{.*}}: !emitasc.py_struct<"TilingData"
// CHECK: ascend.schedule.selected_tile_shape = array<i64: 64, 15000>
// CHECK-SAME: ascend.schedule.tail_policies = ["masked_tail", "full_extent"]
// CHECK-SAME: cann.num_inputs = 2 : i32
// CHECK: ascendc.get_block_idx
// CHECK: ascendc.data_copy_l2
// CHECK: ascendc.reduce_sum_2d_l2
// CHECK: emitasc.verbatim
// CHECK-NOT: linalg.

#broadcast_map = affine_map<(d0, d1) -> (d0)>
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @broadcast_add_reducesum(
      %input_a : tensor<640xf16>,
      %input_b : tensor<640x15000xf16>
  ) -> tensor<640xf16> {

    %empty_tensor_c = tensor.empty() : tensor<640x15000xf16>
    %tensor_c = linalg.generic {
      indexing_maps = [#broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%input_a : tensor<640xf16>)
      outs(%empty_tensor_c : tensor<640x15000xf16>) {
    ^bb0(%a_value: f16, %c_output: f16):
      linalg.yield %a_value : f16
    } -> tensor<640x15000xf16>

    %empty_tensor_d = tensor.empty() : tensor<640x15000xf16>
    %tensor_d = linalg.generic {
      indexing_maps = [#full_access_map, #full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%tensor_c, %input_b : tensor<640x15000xf16>, tensor<640x15000xf16>)
      outs(%empty_tensor_d : tensor<640x15000xf16>) {
    ^bb0(%c_value: f16, %b_value: f16, %d_output: f16):
      %sum = arith.addf %c_value, %b_value : f16
      linalg.yield %sum : f16
    } -> tensor<640x15000xf16>

    %zero_value = arith.constant 0.0 : f16
    %empty_tensor_e = tensor.empty() : tensor<640xf16>
    %init_tensor_e = linalg.fill ins(%zero_value : f16)
      outs(%empty_tensor_e : tensor<640xf16>) -> tensor<640xf16>
    %tensor_e = linalg.generic {
      indexing_maps = [#full_access_map, #broadcast_map],
      iterator_types = ["parallel", "reduction"]
    } ins(%tensor_d : tensor<640x15000xf16>)
      outs(%init_tensor_e : tensor<640xf16>) {
    ^bb0(%d_value: f16, %accumulator: f16):
      %new_accumulator = arith.addf %accumulator, %d_value : f16
      linalg.yield %new_accumulator : f16
    } -> tensor<640xf16>

    return %tensor_e : tensor<640xf16>
  }
}
