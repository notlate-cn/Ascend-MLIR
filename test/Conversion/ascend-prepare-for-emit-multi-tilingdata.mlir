// RUN: afir-opt %s --ascend-prepare-for-emit | FileCheck %s
// RUN: afir-opt %s --ascendc-prepare-for-emit | FileCheck %s

// CHECK-DAG: emitasc.declare_py_struct !emitasc.py_struct<"TilingData_kernel_a"
// CHECK-DAG: emitasc.declare_py_struct !emitasc.py_struct<"TilingData_kernel_b"
// CHECK-LABEL: func.func @kernel_a
// CHECK-SAME: !emitasc.py_struct<"TilingData_kernel_a"
// CHECK-LABEL: func.func @kernel_b
// CHECK-SAME: !emitasc.py_struct<"TilingData_kernel_b"

module {
  func.func @kernel_a(%arg0: memref<?xf16>, %arg1: memref<?xf16>, %out: memref<?xf16>) {
    %c0 = arith.constant 0 : index
    %n = memref.dim %arg0, %c0 : memref<?xf16>
    %i64 = arith.index_cast %n : index to i64
    "emitasc.verbatim"(%i64) {value = "/* $0 */"} : (i64) -> ()
    return
  }

  func.func @kernel_b(%arg0: memref<?xf16>, %arg1: memref<?x?xf16>, %out: memref<?x?xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = memref.dim %arg1, %c0 : memref<?x?xf16>
    %n = memref.dim %arg1, %c1 : memref<?x?xf16>
    %mi64 = arith.index_cast %m : index to i64
    %ni64 = arith.index_cast %n : index to i64
    "emitasc.verbatim"(%mi64, %ni64) {value = "/* $0 $1 */"} : (i64, i64) -> ()
    return
  }
}
