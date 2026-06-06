// RUN: ascend-mlir-opt --ascend-canonicalize-cann-signature %s | ascend-mlir-translate -mlir-to-cann - | FileCheck %s

// CHECK-LABEL: extern "C" __global__ __aicore__ void uses_global_const(
// CHECK: GM_ADDR
// CHECK: GM_ADDR
// CHECK: GM_ADDR
// CHECK: GM_ADDR
// CHECK: TilingData
// CHECK: float {{v[0-9]+}} = ascend_gm_load<float>(
// CHECK: ascend_gm_store<float>(

module {
  memref.global "private" constant @weights : memref<4xf32> = dense<1.0> {alignment = 64 : i64}

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
