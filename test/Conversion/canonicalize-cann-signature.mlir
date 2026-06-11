// RUN: afir-opt --canonicalize-cann-signature %s | FileCheck %s

// CHECK-LABEL: func.func @broadcast_add_reducesum
// CHECK-SAME: %[[A:[a-z0-9]+]]: memref<?xf16>
// CHECK-SAME: %[[B:[a-z0-9]+]]: memref<?x?xf16>
// CHECK-SAME: %[[OUT:[a-z0-9]+]]: memref<?xf16
// CHECK-SAME: %[[WS:[a-z0-9]+]]: memref<ui8>
// CHECK-SAME: %[[TILING:[a-z0-9]+]]: !emitasc.py_struct<"TilingData"
// CHECK-SAME: ascendc.kernel_kind = "mix"
// CHECK-SAME: cann.num_inputs = 2
// CHECK-NOT: emitasc.copy_struct
// CHECK: emitasc.member %[[TILING]] "TB_M"

module {
  func.func @broadcast_add_reducesum(
      %input_a: memref<?xf16>,
      %input_b: memref<?x?xf16>,
      %tiling_data: memref<?x!emitasc.py_struct<"TilingData",
          [i64, i64, i64, i64],
          ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, 22 : i32>,
      %output: memref<?xf16>
  ) attributes {ascendc.aicore, ascendc.global, ascendc.kernel_kind = "mix"} {
    %local_tiling = emitasc.copy_struct %tiling_data
        : memref<?x!emitasc.py_struct<"TilingData",
              [i64, i64, i64, i64],
              ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, 22 : i32>,
          !emitasc.py_struct<"TilingData",
              [i64, i64, i64, i64],
              ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>
    %tb_m = emitasc.member %local_tiling "TB_M"
        : !emitasc.py_struct<"TilingData",
              [i64, i64, i64, i64],
              ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>,
          i64
    func.return
  }

  // Test: promoted intermediate buffers appear before the tiling arg (bufferized
  // layout). Real (non-strided) args before tiling: %input + %output = 2; %inter
  // is strided with dynamic offset, so it is not a real I/O arg. num_inputs is
  // derived from data_copy write direction: %input feeds a data_copy_l2 as src,
  // %output is the dst of a data_copy_l2 -> 1 real output -> num_inputs = 1.
  // CHECK-LABEL: func.func @promoted_intermediate_layout
  // CHECK-SAME: cann.num_inputs = 1
  // CHECK-NOT: emitasc.copy_struct
  func.func @promoted_intermediate_layout(
      %input: memref<?xf32>,
      %output: memref<?xf32>,
      %inter: memref<?xf32, strided<[1], offset: ?>>,
      %tiling: memref<?x!emitasc.py_struct<"TD", [i64], ["X"]>, 22 : i32>
  ) attributes {ascendc.aicore, ascendc.global} {
    %c0_i32 = arith.constant 0 : i32
    %local = emitasc.copy_struct %tiling
        : memref<?x!emitasc.py_struct<"TD", [i64], ["X"]>, 22 : i32>,
          !emitasc.py_struct<"TD", [i64], ["X"]>
    %x = emitasc.member %local "X"
        : !emitasc.py_struct<"TD", [i64], ["X"]>, i64
    %lt = ascendc.local_tensor : !ascendc.local_tensor<*xf32>
    // %input: read (data_copy_l2 src)
    %gt_in = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
    %cast_in = emitasc.reinterpret_cast %input : memref<?xf32> to memref<?xf32, 22 : i32>
    ascendc.global_tensor.set_global_buffer %gt_in, %cast_in, %c0_i32
        : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
    ascendc.data_copy_l2 %lt, %gt_in, %c0_i32
        : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, i32
    // %output: written (data_copy_l2 dst)
    %gt_out = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
    %cast_out = emitasc.reinterpret_cast %output : memref<?xf32> to memref<?xf32, 22 : i32>
    ascendc.global_tensor.set_global_buffer %gt_out, %cast_out, %c0_i32
        : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
    ascendc.data_copy_l2 %gt_out, %lt, %c0_i32
        : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    func.return
  }

  // Test: all-broadcast kernel. Every output is promoted to a strided buffer
  // (dynamic offset) and appears before the tiling arg. Real (non-strided) args
  // before tiling: %in0 + %in1 = 2; the strided %out is not a real I/O arg, so
  // num_inputs = 2 - 0 (no non-strided written arg) = 2.
  // CHECK-LABEL: func.func @all_broadcast_layout
  // CHECK-SAME: cann.num_inputs = 2
  // CHECK-NOT: emitasc.copy_struct
  func.func @all_broadcast_layout(
      %in0: memref<?xf32>,
      %in1: memref<?xf32>,
      %out: memref<?xf32, strided<[1], offset: ?>>,
      %tiling: memref<?x!emitasc.py_struct<"TD", [i64], ["X"]>, 22 : i32>
  ) attributes {ascendc.aicore, ascendc.global} {
    %c0_i32 = arith.constant 0 : i32
    %local = emitasc.copy_struct %tiling
        : memref<?x!emitasc.py_struct<"TD", [i64], ["X"]>, 22 : i32>,
          !emitasc.py_struct<"TD", [i64], ["X"]>
    %x = emitasc.member %local "X"
        : !emitasc.py_struct<"TD", [i64], ["X"]>, i64
    %lt = ascendc.local_tensor : !ascendc.local_tensor<*xf32>
    %gt_out = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
    %cast_out = emitasc.reinterpret_cast %out
        : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, 22 : i32>
    ascendc.global_tensor.set_global_buffer %gt_out, %cast_out, %c0_i32
        : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
    ascendc.data_copy_l2 %gt_out, %lt, %c0_i32
        : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    func.return
  }
}
