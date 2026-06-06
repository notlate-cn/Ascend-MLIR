// RUN: ascend-mlir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: void align_runtime_buffers
// CHECK: uint32_t _ascend_bytes = static_cast<uint32_t>(c12_idx);
// CHECK: uint32_t _ascend_aligned_bytes = _ascend_bytes == 0 ? 0 : ((_ascend_bytes + 31u) / 32u) * 32u;
// CHECK: if (_ascend_aligned_bytes < 32u)
// CHECK-NEXT: _ascend_aligned_bytes = 32u;
// CHECK: InitBuffer({{.*}}, _ascend_aligned_bytes);
// CHECK: InitBuffer({{.*}}, c1_i32, _ascend_aligned_bytes);

module {
  func.func @align_runtime_buffers(
      %arg0: memref<?xf16>,
      %arg1: memref<?xf16>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %pipe = ascendc.pipe
    %queue = ascendc.queue : <vecin, 1>
    %tbuf = ascendc.tbuf : <vecin>
    %c1 = arith.constant 1 : i32
    %c12 = arith.constant 12 : index
    ascendc.pipe.init_buffer %pipe, %tbuf, %c12 : !ascendc.tbuf<vecin>, index
    ascendc.pipe.init_queue %pipe, %queue, %c1, %c12 : !ascendc.queue<vecin, 1>, i32, index
    return
  }
}
