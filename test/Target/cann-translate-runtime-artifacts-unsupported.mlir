// RUN: sed -n '/\/\/ MULTI-BEGIN/,/\/\/ MULTI-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.manifest.json 2>&1 | FileCheck %s
// RUN: rm -f %t.tiling.json
// RUN: sed -n '/\/\/ MULTI-BEGIN/,/\/\/ MULTI-END/p' %s | afir-translate -mlir-to-cann --tiling-space-out=%t.tiling.json > %t.cpp
// RUN: FileCheck %s --input-file=%t.tiling.json --check-prefix=TILING
// RUN: sed -n '/\/\/ BAD-TAIL-BEGIN/,/\/\/ BAD-TAIL-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.bad-tail.manifest.json 2>&1 | FileCheck %s --check-prefix=BAD-TAIL
// RUN: sed -n '/\/\/ BAD-SELECTED-TILE-SHAPE-TOP-BEGIN/,/\/\/ BAD-SELECTED-TILE-SHAPE-TOP-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.bad-selected-tile-shape-top.manifest.json 2>&1 | FileCheck %s --check-prefix=BAD-SELECTED-TILE-SHAPE-TOP
// RUN: sed -n '/\/\/ BAD-TAIL-POLICIES-TOP-BEGIN/,/\/\/ BAD-TAIL-POLICIES-TOP-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.bad-tail-policies-top.manifest.json 2>&1 | FileCheck %s --check-prefix=BAD-TAIL-POLICIES-TOP
// RUN: sed -n '/\/\/ BAD-TAIL-POLICIES-VALUE-BEGIN/,/\/\/ BAD-TAIL-POLICIES-VALUE-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.bad-tail-policies-value.manifest.json 2>&1 | FileCheck %s --check-prefix=BAD-TAIL-POLICIES-VALUE
// RUN: sed -n '/\/\/ PARTIAL-NO-TAIL-PLAN-BEGIN/,/\/\/ PARTIAL-NO-TAIL-PLAN-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.partial-no-tail-plan.manifest.json 2>&1 | FileCheck %s --check-prefix=PARTIAL-NO-TAIL-PLAN
// RUN: sed -n '/\/\/ PARTIAL-NO-TAIL-POLICIES-BEGIN/,/\/\/ PARTIAL-NO-TAIL-POLICIES-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.partial-no-tail-policies.manifest.json 2>&1 | FileCheck %s --check-prefix=PARTIAL-NO-TAIL-POLICIES
// RUN: sed -n '/\/\/ BAD-TAIL-PLAN-BEGIN/,/\/\/ BAD-TAIL-PLAN-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.bad-tail-plan.manifest.json 2>&1 | FileCheck %s --check-prefix=BAD-TAIL-PLAN
// RUN: sed -n '/\/\/ BAD-TAIL-PLAN-TOP-BEGIN/,/\/\/ BAD-TAIL-PLAN-TOP-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.bad-tail-plan-top.manifest.json 2>&1 | FileCheck %s --check-prefix=BAD-TAIL-PLAN-TOP
// RUN: sed -n '/\/\/ BAD-TAIL-PLAN-I64-BEGIN/,/\/\/ BAD-TAIL-PLAN-I64-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.bad-tail-plan-i64.manifest.json 2>&1 | FileCheck %s --check-prefix=BAD-TAIL-PLAN-I64
// RUN: sed -n '/\/\/ BAD-TAIL-PLAN-SELECTED-BEGIN/,/\/\/ BAD-TAIL-PLAN-SELECTED-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.bad-tail-plan-selected.manifest.json 2>&1 | FileCheck %s --check-prefix=BAD-TAIL-PLAN-SELECTED
// RUN: sed -n '/\/\/ BAD-TAIL-PLAN-AFFECTED-BEGIN/,/\/\/ BAD-TAIL-PLAN-AFFECTED-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.bad-tail-plan-affected.manifest.json 2>&1 | FileCheck %s --check-prefix=BAD-TAIL-PLAN-AFFECTED
// RUN: sed -n '/\/\/ BAD-TAIL-PLAN-ALIGN-BEGIN/,/\/\/ BAD-TAIL-PLAN-ALIGN-END/p' %s | not afir-translate -mlir-to-cann --runtime-manifest-out=%t.bad-tail-plan-align.manifest.json 2>&1 | FileCheck %s --check-prefix=BAD-TAIL-PLAN-ALIGN
// RUN: sed -n '/\/\/ NO-ATTR-BEGIN/,/\/\/ NO-ATTR-END/p' %s | afir-translate -mlir-to-cann --runtime-manifest-out=%t.no-attr.manifest.json > %t.no-attr.cpp
// RUN: FileCheck %s --input-file=%t.no-attr.manifest.json --check-prefix=NO-ATTR

// CHECK: runtime manifest MVP supports exactly one global kernel
// TILING: "kernel": "kernel_a"
// TILING: "schema_version": "2.0"
// BAD-TAIL: ascend.schedule.tail_policies element 1 must be a string attribute
// BAD-SELECTED-TILE-SHAPE-TOP: ascend.schedule.selected_tile_shape must be a dense i64 array attribute
// BAD-TAIL-POLICIES-TOP: ascend.schedule.tail_policies must be an array attribute
// BAD-TAIL-POLICIES-VALUE: ascend.schedule.tail_policies element 0 has unsupported value 'unknown_tail_policy'
// PARTIAL-NO-TAIL-PLAN: schedule metadata requires ascend.schedule.selected_tile_shape, ascend.schedule.tail_policies, and ascend.schedule.tail_plan together
// PARTIAL-NO-TAIL-POLICIES: schedule metadata requires ascend.schedule.selected_tile_shape, ascend.schedule.tail_policies, and ascend.schedule.tail_plan together
// BAD-TAIL-PLAN: ascend.schedule.tail_plan element 0 must be a dictionary attribute
// BAD-TAIL-PLAN-TOP: ascend.schedule.tail_plan must be an array attribute
// BAD-TAIL-PLAN-I64: ascend.schedule.tail_plan element 0 field 'axis' must be an i64 integer attribute
// BAD-TAIL-PLAN-SELECTED: ascend.schedule.tail_plan element 0 field 'selected' has unsupported value 'unknown_tail_policy'
// BAD-TAIL-PLAN-AFFECTED: ascend.schedule.tail_plan element 0 field 'affected' element 0 has unsupported value 'unknown_use'
// BAD-TAIL-PLAN-ALIGN: ascend.schedule.tail_plan element 0 field 'align' must be an i64 integer attribute
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
      ascend.schedule.selected_tile_shape = array<i64: 64>,
      ascend.schedule.tail_policies = ["masked_tail", 1 : i64],
      ascend.schedule.tail_plan = [
        {
          affected = ["data_copy"],
          align = 16 : i64,
          axis = 0 : i64,
          buffering = "separate_tail_buffer",
          selected = "masked_tail"
        }
      ],
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32} {
    func.return
  }
}
// BAD-TAIL-END

// BAD-SELECTED-TILE-SHAPE-TOP-BEGIN
module {
  func.func @bad_selected_tile_shape_top(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {
      ascend.schedule.selected_tile_shape = 64 : i64,
      ascend.schedule.tail_policies = ["masked_tail"],
      ascend.schedule.tail_plan = [
        {
          affected = ["data_copy"],
          align = 16 : i64,
          axis = 0 : i64,
          buffering = "separate_tail_buffer",
          selected = "masked_tail"
        }
      ],
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32} {
    func.return
  }
}
// BAD-SELECTED-TILE-SHAPE-TOP-END

// BAD-TAIL-POLICIES-TOP-BEGIN
module {
  func.func @bad_tail_policies_top(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {
      ascend.schedule.selected_tile_shape = array<i64: 64>,
      ascend.schedule.tail_policies = "masked_tail",
      ascend.schedule.tail_plan = [
        {
          affected = ["data_copy"],
          align = 16 : i64,
          axis = 0 : i64,
          buffering = "separate_tail_buffer",
          selected = "masked_tail"
        }
      ],
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32} {
    func.return
  }
}
// BAD-TAIL-POLICIES-TOP-END

// BAD-TAIL-POLICIES-VALUE-BEGIN
module {
  func.func @bad_tail_policies_value(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {
      ascend.schedule.selected_tile_shape = array<i64: 64>,
      ascend.schedule.tail_policies = ["unknown_tail_policy"],
      ascend.schedule.tail_plan = [
        {
          affected = ["data_copy"],
          align = 16 : i64,
          axis = 0 : i64,
          buffering = "separate_tail_buffer",
          selected = "masked_tail"
        }
      ],
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32} {
    func.return
  }
}
// BAD-TAIL-POLICIES-VALUE-END

// PARTIAL-NO-TAIL-PLAN-BEGIN
module {
  func.func @partial_no_tail_plan(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {
      ascend.schedule.selected_tile_shape = array<i64: 64>,
      ascend.schedule.tail_policies = ["masked_tail"],
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32} {
    func.return
  }
}
// PARTIAL-NO-TAIL-PLAN-END

// PARTIAL-NO-TAIL-POLICIES-BEGIN
module {
  func.func @partial_no_tail_policies(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {
      ascend.schedule.selected_tile_shape = array<i64: 64>,
      ascend.schedule.tail_plan = [
        {
          affected = ["data_copy"],
          align = 16 : i64,
          axis = 0 : i64,
          buffering = "separate_tail_buffer",
          selected = "masked_tail"
        }
      ],
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32} {
    func.return
  }
}
// PARTIAL-NO-TAIL-POLICIES-END

// BAD-TAIL-PLAN-BEGIN
module {
  func.func @bad_tail_plan(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {
      ascend.schedule.selected_tile_shape = array<i64: 64>,
      ascend.schedule.tail_policies = ["masked_tail"],
      ascend.schedule.tail_plan = ["masked_tail"],
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32} {
    func.return
  }
}
// BAD-TAIL-PLAN-END

// BAD-TAIL-PLAN-TOP-BEGIN
module {
  func.func @bad_tail_plan_top(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {
      ascend.schedule.selected_tile_shape = array<i64: 64>,
      ascend.schedule.tail_policies = ["masked_tail"],
      ascend.schedule.tail_plan = "masked_tail",
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32} {
    func.return
  }
}
// BAD-TAIL-PLAN-TOP-END

// BAD-TAIL-PLAN-I64-BEGIN
module {
  func.func @bad_tail_plan_i64(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {
      ascend.schedule.selected_tile_shape = array<i64: 64>,
      ascend.schedule.tail_policies = ["masked_tail"],
      ascend.schedule.tail_plan = [
        {
          affected = ["data_copy"],
          align = 16 : i64,
          axis = 0 : i32,
          buffering = "separate_tail_buffer",
          selected = "masked_tail"
        }
      ],
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32} {
    func.return
  }
}
// BAD-TAIL-PLAN-I64-END

// BAD-TAIL-PLAN-SELECTED-BEGIN
module {
  func.func @bad_tail_plan_selected(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {
      ascend.schedule.selected_tile_shape = array<i64: 64>,
      ascend.schedule.tail_policies = ["masked_tail"],
      ascend.schedule.tail_plan = [
        {
          affected = ["data_copy"],
          align = 16 : i64,
          axis = 0 : i64,
          buffering = "separate_tail_buffer",
          selected = "unknown_tail_policy"
        }
      ],
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32} {
    func.return
  }
}
// BAD-TAIL-PLAN-SELECTED-END

// BAD-TAIL-PLAN-AFFECTED-BEGIN
module {
  func.func @bad_tail_plan_affected(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {
      ascend.schedule.selected_tile_shape = array<i64: 64>,
      ascend.schedule.tail_policies = ["masked_tail"],
      ascend.schedule.tail_plan = [
        {
          affected = ["unknown_use"],
          align = 16 : i64,
          axis = 0 : i64,
          buffering = "separate_tail_buffer",
          selected = "masked_tail"
        }
      ],
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32} {
    func.return
  }
}
// BAD-TAIL-PLAN-AFFECTED-END

// BAD-TAIL-PLAN-ALIGN-BEGIN
module {
  func.func @bad_tail_plan_align(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {
      ascend.schedule.selected_tile_shape = array<i64: 64>,
      ascend.schedule.tail_policies = ["masked_tail"],
      ascend.schedule.tail_plan = [
        {
          affected = ["data_copy"],
          align = 16 : i32,
          axis = 0 : i64,
          buffering = "separate_tail_buffer",
          selected = "masked_tail"
        }
      ],
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32} {
    func.return
  }
}
// BAD-TAIL-PLAN-ALIGN-END

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
