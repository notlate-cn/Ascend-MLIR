// RUN: afir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: void broadcast_tail_fallback
// CHECK: AscendC::PipeBarrier<PIPE_ALL>();
// CHECK: if (_afir_ss[1] == 1u) {
// CHECK: for (uint32_t _afir_r = 0; _afir_r < _afir_ds[0]; ++_afir_r) {
// CHECK-NOT: afir_gm_load<half>
// CHECK: auto _afir_v = {{.*}}.GetValue(_afir_r);
// CHECK: uint32_t _afir_row_offset = _afir_r * _afir_ds[1];
// CHECK: if (((_afir_row_offset * sizeof(half)) % 32u) == 0u && ((_afir_ds[1] * sizeof(half)) % 32u) == 0u) {
// CHECK: for (uint32_t _afir_c = 0; _afir_c < _afir_ds[1]; _afir_c += 1024u) {
// CHECK: uint32_t _afir_chunk = ((_afir_ds[1] - _afir_c) < 1024u) ? (_afir_ds[1] - _afir_c) : 1024u;
// CHECK: AscendC::Duplicate({{.*}}[_afir_row_offset + _afir_c], _afir_v, _afir_chunk);
// CHECK: uint32_t _afir_vec_elems = 32u / sizeof(half);
// CHECK: uint32_t _afir_tail_base = (_afir_chunk / _afir_vec_elems) * _afir_vec_elems;
// CHECK: for (uint32_t _afir_t = _afir_tail_base; _afir_t < _afir_chunk; ++_afir_t)
// CHECK: {{.*}}.SetValue(_afir_row_offset + _afir_c + _afir_t, _afir_v);
// CHECK: } else {
// CHECK: {{.*}}.SetValue(_afir_row_offset + _afir_c, _afir_v);
// CHECK: } else {
// CHECK: AscendC::Broadcast<half, 2, 1>
// CHECK: AscendC::PipeBarrier<PIPE_ALL>();

module {
  func.func @broadcast_tail_fallback(
      %arg0: memref<?xf16>,
      %arg1: memref<?xf16>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %pipe = ascendc.pipe
    %src_q = ascendc.queue : <vecin, 1>
    %gt = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
    %c0 = arith.constant 0 : i32
    %cast = emitasc.reinterpret_cast %arg0 : memref<?xf16> to memref<?xf16, 22 : i32>
    ascendc.global_tensor.set_global_buffer %gt, %cast, %c0 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
    %alloc = ascendc.que_bind.alloc_tensor %src_q : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
    %rows = arith.constant 6 : i32
    ascendc.data_copy_l2 %alloc, %gt, %rows : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, i32
    ascendc.que_bind.enque_tensor %src_q, %alloc : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
    %src = ascendc.que_bind.deque_tensor %src_q : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
    %dst_tbuf = ascendc.tbuf : <veccalc>
    %dst = ascendc.tbuf.get_tensor %dst_tbuf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
    %cols = arith.constant 128 : i32
    %one = arith.constant 1 : i32
    ascendc.broadcast_l2 %dst, %src, %rows, %cols, %rows, %one
        {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>}
        : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
    return
  }

// CHECK-LABEL: void broadcast_full_tile_gm_fallback
// CHECK: AscendC::PipeBarrier<PIPE_ALL>();
// CHECK: if (_afir_ss[1] == 1u) {
// CHECK-NOT: if (_afir_ds[0] < 16u
// CHECK-NOT: afir_gm_load<half>
// CHECK: auto _afir_v = {{.*}}.GetValue(_afir_r);
// CHECK: uint32_t _afir_row_offset = _afir_r * _afir_ds[1];
// CHECK: if (((_afir_row_offset * sizeof(half)) % 32u) == 0u && ((_afir_ds[1] * sizeof(half)) % 32u) == 0u) {
// CHECK: for (uint32_t _afir_c = 0; _afir_c < _afir_ds[1]; _afir_c += 1024u) {
// CHECK: AscendC::Duplicate({{.*}}[_afir_row_offset + _afir_c], _afir_v, _afir_chunk);
// CHECK: uint32_t _afir_vec_elems = 32u / sizeof(half);
// CHECK: uint32_t _afir_tail_base = (_afir_chunk / _afir_vec_elems) * _afir_vec_elems;
// CHECK: for (uint32_t _afir_t = _afir_tail_base; _afir_t < _afir_chunk; ++_afir_t)
// CHECK: {{.*}}.SetValue(_afir_row_offset + _afir_c + _afir_t, _afir_v);
// CHECK: } else {
// CHECK: {{.*}}.SetValue(_afir_row_offset + _afir_c, _afir_v);
// CHECK: } else {
// CHECK: AscendC::Broadcast<half, 2, 1>
// CHECK: AscendC::PipeBarrier<PIPE_ALL>();
func.func @broadcast_full_tile_gm_fallback(
    %arg0: memref<?xf16>,
    %arg1: memref<?xf16>,
    %workspace: memref<ui8>,
    %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim"]>
) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
  %pipe = ascendc.pipe
  %src_q = ascendc.queue : <vecin, 1>
  %gt = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
  %c0 = arith.constant 0 : i32
  %cast = emitasc.reinterpret_cast %arg0 : memref<?xf16> to memref<?xf16, 22 : i32>
  ascendc.global_tensor.set_global_buffer %gt, %cast, %c0 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
  %alloc = ascendc.que_bind.alloc_tensor %src_q : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
  %rows = arith.constant 32 : i32
  ascendc.data_copy_l2 %alloc, %gt, %rows : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, i32
  ascendc.que_bind.enque_tensor %src_q, %alloc : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
  %src = ascendc.que_bind.deque_tensor %src_q : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
  %dst_tbuf = ascendc.tbuf : <veccalc>
  %dst = ascendc.tbuf.get_tensor %dst_tbuf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
  %cols = arith.constant 500 : i32
  %one = arith.constant 1 : i32
  ascendc.broadcast_l2 %dst, %src, %rows, %cols, %rows, %one
      {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>}
      : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
  return
}
}
