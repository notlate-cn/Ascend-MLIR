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

// Test: promoted intermediate buffers appear before the tiling arg (bufferized
// layout). Real args before tiling: %input (real) + %output (real) + %inter
// (strided dynamic offset, not real) = 2 real. No real args after tiling.
// numInputs = 2 - 1 = 1 (subtract 1 for the single output before tiling).
// CHECK-LABEL: func.func @promoted_intermediate_layout
// CHECK-SAME: cann.num_inputs = 1
// CHECK-NOT: emitasc.copy_struct

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

  func.func @promoted_intermediate_layout(
      %input: memref<?xf32>,
      %output: memref<?xf32>,
      %inter: memref<?xf32, strided<[1], offset: ?>>,
      %tiling: memref<?x!emitasc.py_struct<"TD", [i64], ["X"]>, 22 : i32>
  ) attributes {ascendc.aicore, ascendc.global} {
    %local = emitasc.copy_struct %tiling
        : memref<?x!emitasc.py_struct<"TD", [i64], ["X"]>, 22 : i32>,
          !emitasc.py_struct<"TD", [i64], ["X"]>
    %x = emitasc.member %local "X"
        : !emitasc.py_struct<"TD", [i64], ["X"]>, i64
    func.return
  }
}
