// RUN: afir-opt %s --linalg-fuse-elementwise-ops --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower --ascend-parallelize --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | FileCheck %s --implicit-check-not=memref.copy --implicit-check-not=linalg.

// CHECK-LABEL: func.func @ewop_broadcast_concat
// CHECK-SAME: cann.num_inputs = 4 : i32
// CHECK-DAG: ascendc.get_block_idx
// CHECK-DAG: ascendc.get_block_idx
// CHECK-DAG: ascendc.broadcast_l2
// CHECK-DAG: ascendc.broadcast_l2
// CHECK-DAG: ascendc.add_l2
// CHECK-DAG: ascendc.mul_l2
// CHECK: return

#broadcast_map = affine_map<(d0, d1) -> (d0)>
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @ewop_broadcast_concat(
      %input_a : tensor<?xf16>,
      %input_b : tensor<?x?xf16>,
      %input_c : tensor<?xf16>,
      %input_d : tensor<?x?xf16>
  ) -> tensor<?x?xf16> {
    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index

    %dim_m = tensor.dim %input_a, %idx_0 : tensor<?xf16>
    %dim_n = tensor.dim %input_b, %idx_1 : tensor<?x?xf16>

    %init_c = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %result_c = linalg.generic {
      indexing_maps = [#broadcast_map, #full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%input_a, %input_b : tensor<?xf16>, tensor<?x?xf16>)
      outs(%init_c : tensor<?x?xf16>) {
    ^bb0(%a_val: f16, %b_val: f16, %c_out: f16):
      %sum = arith.addf %a_val, %b_val : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>

    %init_d = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %result_d = linalg.generic {
      indexing_maps = [#broadcast_map, #full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%input_c, %input_d : tensor<?xf16>, tensor<?x?xf16>)
      outs(%init_d : tensor<?x?xf16>) {
    ^bb0(%c_val: f16, %d_val: f16, %e_out: f16):
      %prod = arith.mulf %c_val, %d_val : f16
      linalg.yield %prod : f16
    } -> tensor<?x?xf16>

    %output = tensor.concat dim(0) %result_c, %result_d
        : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>

    return %output : tensor<?x?xf16>
  }
}
