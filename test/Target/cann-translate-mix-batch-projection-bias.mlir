// RUN: ascend-mlir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: extern "C" __global__ __aicore__ void batch_projection_bias
// CHECK: KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
// CHECK: TCubeTiling tiling;
// CHECK: CopyTiling(&tiling, tilingGm);
// CHECK: const uint32_t batch =
// CHECK: for (uint32_t b = 0; b < batch; ++b)
// CHECK: acc += lhsGM
// CHECK: acc += biasGM[col];
// CHECK: outGM
// CHECK-LABEL: extern "C" __global__ __aicore__ void matmul_projection_bias
// CHECK: KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
// CHECK: TCubeTiling tiling;
// CHECK: for (uint32_t row = 0; row < m; ++row)
// CHECK: acc += lhsGM
// CHECK: acc += biasGM[col];

module {
  func.func @batch_projection_bias(
      %lhs: memref<?x?x128xf32>,
      %rhs: memref<?x128x384xf32>,
      %dead0: memref<?x?x384xf32>,
      %dead1: memref<?x?x128xf32>,
      %bias: memref<384xf32>,
      %out: memref<?x?x384xf32>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData_batch_projection_bias", [i64],
                                   ["dim_arg0_0"]>
  ) attributes {
      abi_matmul_epilogue_kind = "BiasAdd",
      abi_matmul_has_bias = true,
      abi_matmul_layout_a = "ND",
      abi_matmul_layout_b = "ND",
      abi_matmul_layout_c = "ND",
      abi_matmul_op_kind = "batch_matmul",
      abi_matmul_trans_a = false,
      abi_matmul_trans_b = false,
      ascendc.aicore,
      ascendc.global,
      ascendc.kernel_kind = "mix",
      cann.num_inputs = 5 : i32} {
    func.return
  }

  func.func @matmul_projection_bias(
      %lhs: memref<?x1x4x32xf32>,
      %rhs: memref<128x128xf32>,
      %dead0: memref<?x128xf32>,
      %dead1: memref<?x?x128xf32>,
      %dead2: memref<?x?x32xf32>,
      %bias: memref<128xf32>,
      %out: memref<?x128xf32>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData_matmul_projection_bias", [i64],
                                   ["dim_arg0_0"]>
  ) attributes {
      abi_matmul_epilogue_kind = "BiasAdd",
      abi_matmul_has_bias = true,
      abi_matmul_layout_a = "ND",
      abi_matmul_layout_b = "ND",
      abi_matmul_layout_c = "ND",
      abi_matmul_op_kind = "matmul",
      abi_matmul_trans_a = false,
      abi_matmul_trans_b = false,
      ascendc.aicore,
      ascendc.global,
      ascendc.kernel_kind = "mix",
      cann.num_inputs = 6 : i32} {
    func.return
  }
}
