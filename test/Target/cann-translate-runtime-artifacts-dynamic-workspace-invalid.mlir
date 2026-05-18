// RUN: not afir-translate -mlir-to-cann %s --host-tiling-out=%t.host.cpp --cann-soc=Ascend910B2 2>&1 | FileCheck %s

// CHECK: cann.workspace_size_expr references unknown tiling shape field "dim_arg9_0"

module {
  func.func @dynamic_workspace_invalid(
      %a: memref<?x128xf16>,
      %out: memref<?x128xf16>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData",
          [i64],
          ["dim_arg0_0"]>
  ) attributes {
      ascend.schedule.selected_tile_shape = array<i64: 64, 128>,
      ascend.schedule.tail_policies = ["masked_tail", "full_extent"],
      ascend.schedule.tail_plan = [
        {
          affected = ["data_copy", "vector_compute"],
          align = 16 : i64,
          axis = 0 : i64,
          buffering = "separate_tail_buffer",
          selected = "masked_tail"
        },
        {
          affected = ["write_back"],
          align = 0 : i64,
          axis = 1 : i64,
          buffering = "reuse_main_buffer_after_drain",
          selected = "full_extent"
        }
      ],
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32,
      cann.workspace_size_expr = "dim_arg9_0 * 2"} {
    func.return
  }
}
