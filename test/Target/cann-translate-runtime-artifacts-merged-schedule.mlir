// RUN: ascend-mlir-translate -mlir-to-cann %s --artifact-manifest-out=%t.manifest.json --host-tiling-out=%t.host.cpp --cann-soc=Ascend910B2 > %t.cpp
// RUN: FileCheck %s --input-file=%t.manifest.json --check-prefix=MANIFEST
// RUN: FileCheck %s --input-file=%t.host.cpp --check-prefix=HOST

// MANIFEST: "decisionId": "kernel_1.decision.0"
// MANIFEST: "tile_binding": "symbolic"
// MANIFEST: "tile_params": [
// MANIFEST: "default": 32
// MANIFEST: "name": "TB_M"
// MANIFEST-NOT: "decisionId": "kernel_0.decision.0"

// HOST: data.TB_M = 32;
// HOST-NOT: data.TB_M = 0;

module attributes {
    ascend.kernel_graph.edges = [
      {from = "kernel_0", to = "kernel_1", carried_buffers = ["tmp0"]}
    ]} {
  func.func @merged_kernel(
      %a: memref<?xf16>,
      %out: memref<?xf16>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64, i64], ["TB_M", "dim_arg0_0"]>
  ) attributes {
      ascend.schedule.kernel_metadata = [
        {
          decision_id = "kernel_0.decision.0",
          kernel = "kernel_0",
          tile_binding = "symbolic",
          tile_params = [{
            axis = 0 : i64,
            axis_kind = "parallel",
            binding = "runtime",
            default = 24576 : i64,
            extent = -9223372036854775808 : i64,
            name = "TB_M",
            primitive_uses = ["data_copy", "vector_compute"],
            roles = ["kernel_loop"],
            upper_bound = 24576 : i64
          }],
          tail_policies = ["masked_tail"],
          tail_plan = [{
            affected = ["data_copy", "vector_compute"],
            align = 0 : i64,
            axis = 0 : i64,
            buffering = "separate_tail_buffer",
            selected = "masked_tail"
          }]
        },
        {
          decision_id = "kernel_1.decision.0",
          kernel = "kernel_1",
          tile_binding = "symbolic",
          tile_params = [{
            axis = 0 : i64,
            axis_kind = "parallel",
            binding = "runtime",
            default = 32 : i64,
            extent = -9223372036854775808 : i64,
            name = "TB_M",
            primitive_uses = ["data_copy", "vector_compute", "write_back"],
            roles = ["bind_core", "kernel_loop", "vectorize"],
            upper_bound = 32 : i64
          }],
          tail_policies = ["masked_tail"],
          tail_plan = [{
            affected = ["data_copy", "vector_compute", "write_back"],
            align = 0 : i64,
            axis = 0 : i64,
            buffering = "separate_tail_buffer",
            selected = "masked_tail"
          }]
        }
      ],
      ascendc.aicore,
      ascendc.global,
      ascendc.kernel_kind = "vec",
      cann.num_inputs = 1 : i32,
      cann.workspace_size_bytes = 0 : i64} {
    func.return
  }
}
