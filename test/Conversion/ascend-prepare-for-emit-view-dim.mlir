// RUN: ascend-mlir-opt %s --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | FileCheck %s

module {
  func.func @collapse_dim_kernel(%arg0: memref<?x4x?x?xf32>, %out: memref<?xf32>) {
    %c2 = arith.constant 2 : index
    %collapsed = memref.collapse_shape %arg0 [[0, 1], [2], [3]]
        : memref<?x4x?x?xf32> into memref<?x?x?xf32>
    %k = memref.dim %collapsed, %c2 : memref<?x?x?xf32>
    %k64 = arith.index_cast %k : index to i64
    "emitasc.verbatim"(%k64) {value = "/* $0 */"} : (i64) -> ()
    return
  }
}

// CHECK: emitasc.declare_py_struct !emitasc.py_struct<"TilingData", [i64], ["dim_arg0_3"]>
// CHECK-LABEL: func.func @collapse_dim_kernel(
// CHECK-SAME: !emitasc.py_struct<"TilingData", [i64], ["dim_arg0_3"]>
// CHECK: emitasc.member %{{.*}} "dim_arg0_3"
// CHECK-NOT: memref.dim
// CHECK: emitasc.verbatim
