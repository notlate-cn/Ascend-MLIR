// RUN: afir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: extern "C" __global__ __aicore__ void kernel_a
// CHECK-LABEL: extern "C" __global__ __aicore__ void kernel_b
// CHECK: KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
// CHECK: for (uint32_t b = 0; b < batch; ++b)
// CHECK: CrossCoreSetFlag<0x2, PIPE_FIX>(3);
// CHECK: CrossCoreWaitFlag(3);

module {
  func.func @kernel_a(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData_kernel_a", [i64], ["dim_arg0_0"]>
  ) attributes {
      ascendc.aicore,
      ascendc.global,
      ascendc.kernel_kind = "vec",
      cann.num_inputs = 1 : i32} {
    func.return
  }

  func.func @kernel_b(
      %q: memref<?x?x?xf16>,
      %key: memref<?x?x?xf16>,
      %bias: memref<?x?x?xf32>,
      %out: memref<?x?x?xf32>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData_kernel_b", [i64], ["dim_arg0_0"]>
  ) attributes {
      abi_matmul_batch_shape = [2],
      abi_matmul_epilogue_kind = "None",
      abi_matmul_has_bias = false,
      abi_matmul_layout_a = "ND",
      abi_matmul_layout_b = "ND",
      abi_matmul_layout_c = "ND",
      abi_matmul_op_kind = "batch_matmul",
      abi_matmul_trans_a = false,
      abi_matmul_trans_b = false,
      ascendc.aicore,
      ascendc.global,
      ascendc.kernel_kind = "mix",
      cann.num_inputs = 3 : i32} {
    func.return
  }
}
