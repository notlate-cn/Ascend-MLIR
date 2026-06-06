// RUN: ascend-mlir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: void broadcast_tail_fallback
// CHECK: AscendC::TBuf<AscendC::TPosition::VECCALC> _ascend_bcast_src_tbuf_
// CHECK: AscendC::LocalTensor<half> [[SRC0:_ascend_bcast_src_[0-9]+]] =
// CHECK: [[SRC0]].SetValue(_ascend_i, static_cast<half>(
// CHECK: AscendC::PipeBarrier<PIPE_ALL>();
// CHECK: if (_ascend_ss[1] == 1u) {
// CHECK: for (uint32_t _ascend_r = 0; _ascend_r < _ascend_ds[0]; ++_ascend_r) {
// CHECK-NOT: ascend_gm_load<half>
// CHECK: auto _ascend_v = [[SRC0]].GetValue(_ascend_r);
// CHECK: uint32_t _ascend_row_offset = _ascend_r * _ascend_ds[1];
// CHECK: if (((_ascend_row_offset * sizeof(half)) % 32u) == 0u && ((_ascend_ds[1] * sizeof(half)) % 32u) == 0u) {
// CHECK: for (uint32_t _ascend_c = 0; _ascend_c < _ascend_ds[1]; _ascend_c += 1024u) {
// CHECK: uint32_t _ascend_chunk = ((_ascend_ds[1] - _ascend_c) < 1024u) ? (_ascend_ds[1] - _ascend_c) : 1024u;
// CHECK: AscendC::Duplicate({{.*}}[_ascend_row_offset + _ascend_c], _ascend_v, _ascend_chunk);
// CHECK: uint32_t _ascend_vec_elems = 32u / sizeof(half);
// CHECK: uint32_t _ascend_tail_base = (_ascend_chunk / _ascend_vec_elems) * _ascend_vec_elems;
// CHECK: for (uint32_t _ascend_t = _ascend_tail_base; _ascend_t < _ascend_chunk; ++_ascend_t)
// CHECK: {{.*}}.SetValue(_ascend_row_offset + _ascend_c + _ascend_t, _ascend_v);
// CHECK: } else {
// CHECK: {{.*}}.SetValue(_ascend_row_offset + _ascend_c, _ascend_v);
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
// CHECK: AscendC::TBuf<AscendC::TPosition::VECCALC> _ascend_bcast_src_tbuf_
// CHECK: AscendC::LocalTensor<half> [[SRC1:_ascend_bcast_src_[0-9]+]] =
// CHECK: [[SRC1]].SetValue(_ascend_i, static_cast<half>(
// CHECK: AscendC::PipeBarrier<PIPE_ALL>();
// CHECK: if (_ascend_ss[1] == 1u) {
// CHECK-NOT: if (_ascend_ds[0] < 16u
// CHECK-NOT: ascend_gm_load<half>
// CHECK: auto _ascend_v = [[SRC1]].GetValue(_ascend_r);
// CHECK: uint32_t _ascend_row_offset = _ascend_r * _ascend_ds[1];
// CHECK: if (((_ascend_row_offset * sizeof(half)) % 32u) == 0u && ((_ascend_ds[1] * sizeof(half)) % 32u) == 0u) {
// CHECK: for (uint32_t _ascend_c = 0; _ascend_c < _ascend_ds[1]; _ascend_c += 1024u) {
// CHECK: AscendC::Duplicate({{.*}}[_ascend_row_offset + _ascend_c], _ascend_v, _ascend_chunk);
// CHECK: uint32_t _ascend_vec_elems = 32u / sizeof(half);
// CHECK: uint32_t _ascend_tail_base = (_ascend_chunk / _ascend_vec_elems) * _ascend_vec_elems;
// CHECK: for (uint32_t _ascend_t = _ascend_tail_base; _ascend_t < _ascend_chunk; ++_ascend_t)
// CHECK: {{.*}}.SetValue(_ascend_row_offset + _ascend_c + _ascend_t, _ascend_v);
// CHECK: } else {
// CHECK: {{.*}}.SetValue(_ascend_row_offset + _ascend_c, _ascend_v);
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
