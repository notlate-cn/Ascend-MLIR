// RUN: afir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: void mul_l2_inplace_chunked
// CHECK: uint32_t _afir_mul_count = static_cast<uint32_t>(
// CHECK: for (uint32_t _afir_off = 0; _afir_off < _afir_mul_count; _afir_off += 1024u) {
// CHECK: uint32_t _afir_chunk = ((_afir_mul_count - _afir_off) < 1024u) ? (_afir_mul_count - _afir_off) : 1024u;
// CHECK: AscendC::Mul({{.*}}[_afir_off], {{.*}}[_afir_off], {{.*}}[_afir_off], _afir_chunk);
// CHECK: SetSize(_afir_mul_count);
module {
  func.func @mul_l2_inplace_chunked(
      %arg0: memref<?xf16>,
      %arg1: memref<?xf16>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %dst_tbuf = ascendc.tbuf : <veccalc>
    %lhs_tbuf = ascendc.tbuf : <veccalc>
    %rhs_tbuf = ascendc.tbuf : <veccalc>
    %dst = ascendc.tbuf.get_tensor %dst_tbuf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
    %rhs = ascendc.tbuf.get_tensor %rhs_tbuf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
    %count = arith.constant 480000 : i32
    ascendc.mul_l2 %dst, %dst, %rhs, %count {ascendc.unit = "AiCore.Vector"}
        : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
    return
  }

  // CHECK-LABEL: void mul_l2_out_of_place_direct
  // CHECK-NOT: _afir_mul_count
  // CHECK: AscendC::Mul({{[^[]*}}, {{[^[]*}}, {{[^[]*}}, {{.*}});
  func.func @mul_l2_out_of_place_direct(
      %arg0: memref<?xf16>,
      %arg1: memref<?xf16>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %dst_tbuf = ascendc.tbuf : <veccalc>
    %lhs_tbuf = ascendc.tbuf : <veccalc>
    %rhs_tbuf = ascendc.tbuf : <veccalc>
    %dst = ascendc.tbuf.get_tensor %dst_tbuf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
    %lhs = ascendc.tbuf.get_tensor %lhs_tbuf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
    %rhs = ascendc.tbuf.get_tensor %rhs_tbuf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
    %count = arith.constant 480000 : i32
    ascendc.mul_l2 %dst, %lhs, %rhs, %count {ascendc.unit = "AiCore.Vector"}
        : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
    return
  }
}
