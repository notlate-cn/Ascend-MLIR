// RUN: ascend-mlir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: void gather_count_uses_result_count
// CHECK: uint32_t _afir_idx32_count = static_cast<uint32_t>(c16_i32);
// CHECK: uint32_t _afir_idx32_padded_count = _afir_idx32_count == 0u ? 0u : ((_afir_idx32_count + 127u) / 128u) * 128u;
// CHECK: uint32_t _afir_idx32_bytes = _afir_idx32_padded_count * sizeof(uint32_t);
// CHECK: SetSize(_afir_idx32_count);
// CHECK: for (uint32_t _afir_i = 0; _afir_i < _afir_idx32_count; _afir_i++)
// CHECK: for (uint32_t _afir_i = _afir_idx32_count; _afir_i < _afir_idx32_padded_count; _afir_i++)
// CHECK: _afir_idx32_0.SetValue(_afir_i, 0u);
// CHECK: _afir_idx32_0.SetSize(_afir_idx32_padded_count);
// CHECK: uint32_t _afir_gather_count = static_cast<uint32_t>(c16_i32);
// CHECK: uint32_t _afir_gather_padded_count = _afir_gather_count == 0u ? 0u : ((_afir_gather_count + 127u) / 128u) * 128u;
// CHECK: SetSize(_afir_gather_padded_count);
// CHECK: SetSize((uint32_t)c128_idx);
// CHECK: AscendC::Gather(
module {
  func.func @gather_count_uses_result_count(
      %arg0: memref<?xf16>,
      %arg1: memref<?xi64>,
      %arg2: memref<?xf16>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 2 : i32} {
    %pipe = ascendc.pipe
    %src_gt = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
    %idx_gt = ascendc.global_tensor : !ascendc.global_tensor<*xi64>
    %c0 = arith.constant 0 : i32
    %src_cast = emitasc.reinterpret_cast %arg0 : memref<?xf16> to memref<?xf16, 22 : i32>
    %idx_cast = emitasc.reinterpret_cast %arg1 : memref<?xi64> to memref<?xi64, 22 : i32>
    ascendc.global_tensor.set_global_buffer %src_gt, %src_cast, %c0 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
    ascendc.global_tensor.set_global_buffer %idx_gt, %idx_cast, %c0 : !ascendc.global_tensor<*xi64>, memref<?xi64, 22 : i32>, i32
    %src_tbuf = ascendc.tbuf : <veccalc>
    %idx_tbuf = ascendc.tbuf : <veccalc>
    %dst_tbuf = ascendc.tbuf : <veccalc>
    %src_bytes = arith.constant 128 : index
    ascendc.pipe.init_buffer %pipe, %src_tbuf, %src_bytes : !ascendc.tbuf<veccalc>, index
    %src = ascendc.tbuf.get_tensor %src_tbuf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
    %idx = ascendc.tbuf.get_tensor %idx_tbuf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xi64>
    %dst = ascendc.tbuf.get_tensor %dst_tbuf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
    %n = arith.constant 64 : i32
    %k = arith.constant 16 : i32
    ascendc.data_copy_l2 %src, %src_gt, %n : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, i32
    ascendc.data_copy_l2 %idx, %idx_gt, %k : !ascendc.local_tensor<*xi64>, !ascendc.global_tensor<*xi64>, i32
    ascendc.gather_l2 %dst, %src, %idx, %c0, %k
        : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xi64>, i32, i32
    return
  }
}
