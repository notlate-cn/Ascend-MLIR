// RUN: afir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: void reduce_scalar_barrier
// CHECK: AscendC::PipeBarrier<PIPE_ALL>();
// CHECK-NEXT: {
// CHECK: float _afir_acc = 0.0f;
// CHECK: _afir_acc += static_cast<float>({{.*}}.GetValue(_afir_offset));
// CHECK: {{.*}}.SetValue(_afir_r, static_cast<half>(_afir_acc));

module {
  func.func @reduce_scalar_barrier(
      %arg0: memref<?xf16>,
      %arg1: memref<?xf16>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %pipe = ascendc.pipe
    %dst_q = ascendc.queue : <vecout, 1>
    %src_tbuf = ascendc.tbuf : <veccalc>
    %one = arith.constant 1 : i32
    %rows = arith.constant 32 : index
    %cols = arith.constant 128 : index
    %elem_size = arith.constant 2 : index
    %dst_bytes = arith.muli %rows, %elem_size : index
    %src_elems = arith.muli %rows, %cols : index
    %src_bytes = arith.muli %src_elems, %elem_size : index
    ascendc.pipe.init_queue %pipe, %dst_q, %one, %dst_bytes
        : !ascendc.queue<vecout, 1>, i32, index
    ascendc.pipe.init_buffer %pipe, %src_tbuf, %src_bytes
        : !ascendc.tbuf<veccalc>, index
    %src = ascendc.tbuf.get_tensor %src_tbuf
        : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
    %dst = ascendc.que_bind.alloc_tensor %dst_q
        : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
    ascendc.reduce_sum_2d_l2 %dst, %src {layout = 0 : i32}
        : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>
    return
  }
}
