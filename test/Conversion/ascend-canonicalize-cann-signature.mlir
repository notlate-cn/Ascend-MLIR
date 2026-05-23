// RUN: afir-opt --ascend-canonicalize-cann-signature %s | FileCheck %s

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
// CHECK-LABEL: func.func @uses_global_const
// CHECK-SAME: %[[INPUT:[[:alnum:]_]+]]: memref<4xf32>
// CHECK-SAME: %[[WEIGHTS:[[:alnum:]_]+]]: memref<4xf32>
// CHECK-SAME: %[[OUTPUT:[[:alnum:]_]+]]: memref<4xf32>
// CHECK-SAME: %[[WS:[[:alnum:]_]+]]: memref<ui8>
// CHECK-SAME: %[[TILING:[[:alnum:]_]+]]: !emitasc.py_struct<"TilingData"
// CHECK-SAME: cann.num_inputs = 2
// CHECK-NOT: memref.get_global
// CHECK: memref.load %[[WEIGHTS]]

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

  memref.global "private" @weights : memref<4xf32>
  func.func @uses_global_const(
      %input: memref<4xf32>,
      %tiling_data: memref<?x!emitasc.py_struct<"TilingData",
          [i64],
          ["TB_M"]>, 22 : i32>,
      %output: memref<4xf32>
  ) attributes {ascendc.aicore, ascendc.global, ascendc.kernel_kind = "vec"} {
    %local_tiling = emitasc.copy_struct %tiling_data
        : memref<?x!emitasc.py_struct<"TilingData",
              [i64],
              ["TB_M"]>, 22 : i32>,
          !emitasc.py_struct<"TilingData",
              [i64],
              ["TB_M"]>
    %tb_m = emitasc.member %local_tiling "TB_M"
        : !emitasc.py_struct<"TilingData",
              [i64],
              ["TB_M"]>,
          i64
    %c0 = arith.constant 0 : index
    %weights = memref.get_global @weights : memref<4xf32>
    %v = memref.load %weights[%c0] : memref<4xf32>
    memref.store %v, %output[%c0] : memref<4xf32>
    func.return
  }
}
