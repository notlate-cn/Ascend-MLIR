// RUN: ascend-mlir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: void tbuf_queue_roundtrip
// CHECK-NOT: EnQue
// CHECK-NOT: DeQue
// CHECK-NOT: FreeTensor
// CHECK: AscendC::Exp(
// CHECK-NOT: EnQue
// CHECK-NOT: DeQue
// CHECK-NOT: FreeTensor
// CHECK: AscendC::DataCopy(
// CHECK-NOT: EnQue
// CHECK-NOT: DeQue
// CHECK-NOT: FreeTensor

module {
  func.func @tbuf_queue_roundtrip(
      %input: memref<?xf32>,
      %output: memref<?xf32>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData_tbuf_queue_roundtrip", [i64], ["n"]>
  ) attributes {
      ascendc.aicore,
      ascendc.global,
      ascendc.kernel_kind = "vec",
      cann.num_inputs = 1 : i32} {
    %n64 = emitasc.member %tiling "n" : !emitasc.py_struct<"TilingData_tbuf_queue_roundtrip", [i64], ["n"]>, i64
    %n = arith.index_cast %n64 : i64 to index
    %c1_i32 = arith.constant 1 : i32
    %c4 = arith.constant 4 : index
    %bytes = arith.muli %n, %c4 : index
    %pipe = ascendc.pipe
    %src_buf = ascendc.tbuf : <veccalc>
    %dst_buf = ascendc.tbuf : <veccalc>
    %out_q = ascendc.queue : <vecout, 1>
    ascendc.pipe.init_buffer %pipe, %src_buf, %bytes : !ascendc.tbuf<veccalc>, index
    ascendc.pipe.init_buffer %pipe, %dst_buf, %bytes : !ascendc.tbuf<veccalc>, index
    ascendc.pipe.init_queue %pipe, %out_q, %c1_i32, %bytes : !ascendc.queue<vecout, 1>, i32, index
    %src = ascendc.tbuf.get_tensor %src_buf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
    %dst = ascendc.tbuf.get_tensor %dst_buf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
    ascendc.exp_l2 %dst, %src, %n {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
    ascendc.pipe_barrier pipe_all
    ascendc.que_bind.enque_tensor %out_q, %dst : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
    %live = ascendc.que_bind.deque_tensor %out_q : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
    %gt = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
    ascendc.global_tensor.set_global_buffer %gt, %output : !ascendc.global_tensor<*xf32>, memref<?xf32>
    ascendc.data_copy_l2 %gt, %live, %n : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
    ascendc.que_bind.free_tensor %out_q, %live : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
    return
  }
}
