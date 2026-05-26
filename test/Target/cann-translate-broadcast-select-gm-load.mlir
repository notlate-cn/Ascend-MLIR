// RUN: ascend-mlir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: void broadcast_select_gm_load
// CHECK: afir_gm_load<float>
// CHECK-NOT: v2[v

module {
  func.func @broadcast_select_gm_load(
      %input: memref<?x?x1xf32>,
      %output: memref<?x?x128xf32>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64, i64], ["dim_arg0_0", "dim_arg0_1"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %m64 = emitasc.member %tiling "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64], ["dim_arg0_0", "dim_arg0_1"]>, i64
    %n64 = emitasc.member %tiling "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64], ["dim_arg0_0", "dim_arg0_1"]>, i64
    %m = arith.index_cast %m64 : i64 to index
    %n = arith.index_cast %n64 : i64 to index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    scf.for %i = %c0 to %m step %c1 {
      scf.for %j = %c0 to %n step %c1 {
        scf.for %k = %c0 to %c128 step %c1 {
          %is_m_one = arith.cmpi eq, %m, %c1 : index
          %bi = arith.select %is_m_one, %c0, %i : index
          %is_n_one = arith.cmpi eq, %n, %c1 : index
          %bj = arith.select %is_n_one, %c0, %j : index
          %v = memref.load %input[%bi, %bj, %c0] : memref<?x?x1xf32>
          memref.store %v, %output[%i, %j, %k] : memref<?x?x128xf32>
        }
      }
    }
    return
  }
}
