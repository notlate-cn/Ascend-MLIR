// RUN: sed -n '/\/\/ MULTI-BEGIN/,/\/\/ MULTI-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.manifest.json 2>&1 | FileCheck %s
// RUN: rm -f %t.tiling.json
// RUN: sed -n '/\/\/ MULTI-BEGIN/,/\/\/ MULTI-END/p' %s | afir-translate -mlir-to-cann --tiling-space-out=%t.tiling.json > %t.cpp
// RUN: FileCheck %s --input-file=%t.tiling.json --check-prefix=TILING
// RUN: sed -n '/\/\/ BAD-TAIL-BEGIN/,/\/\/ BAD-TAIL-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.bad-tail.manifest.json 2>&1 | FileCheck %s --check-prefix=BAD-TAIL
// RUN: sed -n '/\/\/ NO-ATTR-BEGIN/,/\/\/ NO-ATTR-END/p' %s | afir-translate -mlir-to-cann --runtime-manifest-out=%t.no-attr.manifest.json > %t.no-attr.cpp
// RUN: FileCheck %s --input-file=%t.no-attr.manifest.json --check-prefix=NO-ATTR

// CHECK: runtime manifest MVP supports exactly one global kernel
// TILING: "kernel": "kernel_a"
// TILING: "schema_version": "2.0"
// BAD-TAIL: ascend.schedule.tail_policies element 1 must be a string attribute
// NO-ATTR: "tilingParams": {}

// MULTI-BEGIN
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
// MULTI-END

// BAD-TAIL-BEGIN
module {
  func.func @bad_tail_policy(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {
      ascend.schedule.tail_policies = ["masked_tail", 1 : i64],
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32} {
    func.return
  }
}
// BAD-TAIL-END

// NO-ATTR-BEGIN
module {
  func.func @no_schedule_metadata(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    func.return
  }
}
// NO-ATTR-END
