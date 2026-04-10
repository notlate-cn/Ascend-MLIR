// RUN: afir-translate -mlir-to-cann %s | FileCheck %s

// CHECK: struct TilingData {
// CHECK-NEXT: int64_t TB_M;
// CHECK-NEXT: int64_t TB_N;
// CHECK-NEXT: int64_t dim_arg0_0;
// CHECK-NEXT: int64_t dim_arg1_1;
// CHECK: extern "C" __global__ __aicore__ void broadcast_add_reducesum(
// CHECK-NEXT: GM_ADDR
// CHECK-NEXT: GM_ADDR
// CHECK-NEXT: GM_ADDR
// CHECK-NEXT: GM_ADDR
// CHECK-NEXT: TilingData
// CHECK: ) {
// CHECK-NOT: copy_struct
// CHECK: .TB_M

module {
  func.func @broadcast_add_reducesum(
      %a: memref<?xf16>,
      %b: memref<?x?xf16>,
      %out: memref<?xf16>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData",
          [i64, i64, i64, i64],
          ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 2 : i32} {
    %tb_m = emitasc.member %tiling "TB_M"
        : !emitasc.py_struct<"TilingData",
              [i64, i64, i64, i64],
              ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>,
          i64
    func.return
  }
}
