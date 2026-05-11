// RUN: not afir-translate -mlir-to-cann %s --runtime-manifest-out=%t.manifest.json 2>&1 | FileCheck %s
// RUN: rm -f %t.tiling.json
// RUN: afir-translate -mlir-to-cann %s --tiling-space-out=%t.tiling.json > %t.cpp
// RUN: FileCheck %s --input-file=%t.tiling.json --check-prefix=TILING

// CHECK: runtime manifest MVP supports exactly one global kernel
// TILING: "kernel": "kernel_a"
// TILING: "schema_version": "2.0"

module {
  func.func @kernel_a(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    func.return
  }

  func.func @kernel_b(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    func.return
  }
}
