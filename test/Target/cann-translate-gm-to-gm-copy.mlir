// RUN: ascend-mlir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: extern "C" __global__ __aicore__ void gm_to_gm_copy(
// CHECK: uint32_t _ascend_count = (uint32_t)
// CHECK-NOT: AscendC::DataCopy(
// CHECK: for (uint32_t _ascend_i = 0; _ascend_i < _ascend_count; ++_ascend_i)
// CHECK: {{v[0-9]+}}.SetValue(_ascend_i, {{v[0-9]+}}.GetValue(_ascend_i));
// CHECK-LABEL: extern "C" __global__ __aicore__ void gm_to_gm_subview_copy(
// CHECK: GM_ADDR [[IN:v[0-9]+]],
// CHECK-NOT: float* {{v[0-9]+}} =
// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ float*>([[IN]]) + {{.*}});
// CHECK: for (uint32_t _ascend_i = 0; _ascend_i < _ascend_count; ++_ascend_i)

module {
  func.func @gm_to_gm_copy(
      %input: memref<?xf32>,
      %output: memref<?xf32>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["N"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %c0_i32 = arith.constant 0 : i32
    %c16 = arith.constant 16 : index
    %src = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
    %dst = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
    ascendc.global_tensor.set_global_buffer %src, %input, %c0_i32 : !ascendc.global_tensor<*xf32>, memref<?xf32>, i32
    ascendc.global_tensor.set_global_buffer %dst, %output, %c0_i32 : !ascendc.global_tensor<*xf32>, memref<?xf32>, i32
    ascendc.data_copy_l2 %dst, %src, %c16 : !ascendc.global_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
    func.return
  }

  func.func @gm_to_gm_subview_copy(
      %input: memref<?xf32>,
      %output: memref<?xf32>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["N"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %c0_i32 = arith.constant 0 : i32
    %c16 = arith.constant 16 : index
    %subview = memref.subview %input[4] [16] [1]
        : memref<?xf32> to memref<16xf32, strided<[1], offset: 4>>
    %src = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
    %dst = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
    ascendc.global_tensor.set_global_buffer %src, %subview, %c0_i32 : !ascendc.global_tensor<*xf32>, memref<16xf32, strided<[1], offset: 4>>, i32
    ascendc.global_tensor.set_global_buffer %dst, %output, %c0_i32 : !ascendc.global_tensor<*xf32>, memref<?xf32>, i32
    ascendc.data_copy_l2 %dst, %src, %c16 : !ascendc.global_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
    func.return
  }
}
