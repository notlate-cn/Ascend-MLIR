// RUN: afir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: void co12dst_f32_scalar
// CHECK: uint32_t _afir_count = (uint32_t)(
// CHECK-SAME: / sizeof(float)
// CHECK: for (uint32_t _afir_i = 0; _afir_i < _afir_count; ++_afir_i)
// CHECK: SetValue(_afir_i,
// CHECK-SAME: GetValue(_afir_i)
// CHECK: SetSize(_afir_count)
// CHECK-NOT: AscendC::DataCopy(

module {
  func.func @co12dst_f32_scalar(
      %input: memref<?xf32>,
      %output: memref<?xf32>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData_co12dst_f32_scalar", [i64], ["n"]>
  ) attributes {
      ascendc.aicore,
      ascendc.global,
      ascendc.kernel_kind = "cube",
      cann.num_inputs = 1 : i32} {
    %n64 = emitasc.member %tiling "n" : !emitasc.py_struct<"TilingData_co12dst_f32_scalar", [i64], ["n"]>, i64
    %n = arith.index_cast %n64 : i64 to index
    %c1_i32 = arith.constant 1 : i32
    %c4 = arith.constant 4 : index
    %bytes = arith.muli %n, %c4 : index
    %pipe = ascendc.pipe
    %co_q = ascendc.queue : <co1, 1>
    %vec_q = ascendc.queue : <vecin, 1>
    %co_tbuf = ascendc.tbuf : <co1>
    %vec_tbuf = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %pipe, %co_tbuf, %bytes : !ascendc.tbuf<co1>, index
    ascendc.pipe.init_queue %pipe, %co_q, %c1_i32, %bytes : !ascendc.queue<co1, 1>, i32, index
    %co_init = ascendc.que_bind.alloc_tensor %co_q : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
    ascendc.que_bind.enque_tensor %co_q, %co_init : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
    %co = ascendc.que_bind.deque_tensor %co_q : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
    ascendc.pipe.init_buffer %pipe, %vec_tbuf, %bytes : !ascendc.tbuf<vecin>, index
    ascendc.pipe.init_queue %pipe, %vec_q, %c1_i32, %bytes : !ascendc.queue<vecin, 1>, i32, index
    %vec = ascendc.que_bind.alloc_tensor %vec_q : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
    %params = ascendc.construct !ascendc.data_copy_co12dst_params()
    ascendc.data_copy_co12dst %vec, %co, %params : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.data_copy_co12dst_params
    ascendc.que_bind.enque_tensor %vec_q, %vec : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
    %live = ascendc.que_bind.deque_tensor %vec_q : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
    ascendc.que_bind.free_tensor %vec_q, %live : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
    ascendc.que_bind.free_tensor %co_q, %co : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
    return
  }
}
