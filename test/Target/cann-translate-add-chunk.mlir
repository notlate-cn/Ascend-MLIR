// RUN: ascend-mlir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: void add_l2_chunked
// CHECK: uint32_t _ascend_add_count = static_cast<uint32_t>(
// CHECK: for (uint32_t _ascend_off = 0; _ascend_off < _ascend_add_count; _ascend_off += 1024u) {
// CHECK: uint32_t _ascend_chunk = ((_ascend_add_count - _ascend_off) < 1024u) ? (_ascend_add_count - _ascend_off) : 1024u;
// CHECK: AscendC::Add({{.*}}[_ascend_off], {{.*}}[_ascend_off], {{.*}}[_ascend_off], _ascend_chunk);
// CHECK: SetSize(_ascend_add_count);
module {
  func.func @add_l2_chunked(
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
    ascendc.add_l2 %dst, %lhs, %rhs, %count {ascendc.unit = "AiCore.Vector"}
        : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
    return
  }
}
