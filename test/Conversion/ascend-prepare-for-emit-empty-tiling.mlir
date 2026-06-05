// RUN: ascend-mlir-opt %s --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | FileCheck %s

func.func @static_kernel() -> memref<4xf32> {
  %out = memref.alloc() : memref<4xf32>
  return %out : memref<4xf32>
}

// CHECK: emitasc.declare_py_struct !emitasc.py_struct<"TilingData", [], []>
// CHECK-LABEL: func.func @static_kernel(
// CHECK-SAME: %{{.*}}: memref<4xf32
// CHECK-SAME: %{{.*}}: memref<ui8>
// CHECK-SAME: %{{.*}}: !emitasc.py_struct<"TilingData", [], []>
// CHECK-SAME: cann.num_inputs = 0 : i32
