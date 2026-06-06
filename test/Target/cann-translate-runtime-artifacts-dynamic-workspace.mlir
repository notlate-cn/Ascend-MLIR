// RUN: rm -f %t.cpp %t.tiling.json %t.manifest.json %t.host.cpp
// RUN: ascend-mlir-translate -mlir-to-cann %s --tiling-space-out=%t.tiling.json --artifact-manifest-out=%t.manifest.json --host-tiling-out=%t.host.cpp --cann-soc=Ascend910B2 > %t.cpp
// RUN: FileCheck %s --input-file=%t.tiling.json --check-prefix=TILING
// RUN: FileCheck %s --input-file=%t.manifest.json --check-prefix=MANIFEST
// RUN: FileCheck %s --input-file=%t.host.cpp --check-prefix=HOST

// TILING: "workspace_size_expr": "dim_arg0_0 * 128 * 2"

// MANIFEST: "workspaceSizeExpr": "dim_arg0_0 * 128 * 2"
// MANIFEST: "workspace"
// MANIFEST: "sizeExpr": "dim_arg0_0 * 128 * 2"
// MANIFEST: "workspaceSizeBytes": 0

// HOST: int64_t dynamic_workspace_GetWorkspaceSize(const int64_t* shape_args, int32_t shape_count)
// HOST: if (shape_count != 1 || shape_args == nullptr)
// HOST-NEXT: return -1;
// HOST: return shape_args[0] * 128 * 2;

module {
  func.func @dynamic_workspace(
      %a: memref<?x128xf16>,
      %out: memref<?x128xf16>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData",
          [i64, i64],
          ["TB_M", "dim_arg0_0"]>
  ) attributes {
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
      cann.workspace_size_expr = "dim_arg0_0 * 128 * 2"} {
    %tb_m = emitasc.member %tiling "TB_M"
        : !emitasc.py_struct<"TilingData",
              [i64, i64],
              ["TB_M", "dim_arg0_0"]>,
          i64
    func.return
  }
}
