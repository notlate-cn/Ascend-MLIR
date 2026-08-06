// RUN: ascend-mlir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: extern "C" __global__ __aicore__ void global_bracket_dst_copy(
// CHECK: AscendC::GlobalTensor<half> _afir_gt;
// CHECK: _afir_gt.SetGlobalBuffer(
// CHECK-SAME: GetPhyAddr(
// CHECK: AscendC::DataCopy(_afir_gt,

module {
  func.func @global_bracket_dst_copy(
      %input: memref<?xf16>,
      %output: memref<?xf16>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData_global_bracket_dst_copy", [i64], ["N"]>
  ) attributes {
      ascendc.aicore,
      ascendc.global,
      ascendc.kernel_kind = "vec",
      cann.num_inputs = 1 : i32} {
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %pipe = ascendc.pipe
    %src_buf = ascendc.tbuf : <vecout>
    ascendc.pipe.init_buffer %pipe, %src_buf, %c32 : !ascendc.tbuf<vecout>, index
    %src = ascendc.tbuf.get_tensor %src_buf : !ascendc.tbuf<vecout>, !ascendc.local_tensor<*xf16>
    %dst = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
    ascendc.global_tensor.set_global_buffer %dst, %output : !ascendc.global_tensor<*xf16>, memref<?xf16>
    %dst_row = ascendc.global_tensor.bracket %dst(%c16) : !ascendc.global_tensor<*xf16>, index, !ascendc.global_tensor<*xf16>
    ascendc.data_copy_l2 %dst_row, %src, %c16 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
    return
  }
}
