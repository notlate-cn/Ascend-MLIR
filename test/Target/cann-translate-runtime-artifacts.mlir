// RUN: rm -f %t.cpp %t.legacy.cpp %t.tiling.json %t.manifest.json %t.legacy-manifest.json %t.conflict-manifest.json %t.host.cpp
// RUN: ascend-mlir-translate -mlir-to-cann %s --tiling-space-out=%t.tiling.json --artifact-manifest-out=%t.manifest.json --host-tiling-out=%t.host.cpp --cann-soc=Ascend910B2 > %t.cpp
// RUN: FileCheck %s --input-file=%t.tiling.json --check-prefix=TILING
// RUN: FileCheck %s --input-file=%t.manifest.json --check-prefix=MANIFEST
// RUN: FileCheck %s --input-file=%t.host.cpp --check-prefix=HOST
// RUN: ascend-mlir-translate -mlir-to-cann %s --runtime-manifest-out=%t.legacy-manifest.json > %t.legacy.cpp
// RUN: FileCheck %s --input-file=%t.legacy-manifest.json --check-prefix=MANIFEST
// RUN: not ascend-mlir-translate -mlir-to-cann %s --artifact-manifest-out=%t.manifest.json --runtime-manifest-out=%t.conflict-manifest.json 2>&1 | FileCheck %s --check-prefix=MANIFEST-CONFLICT

// TILING-DAG: "kernel": "broadcast_add_reducesum"
// TILING-DAG: "schema_version": "2.0"
// TILING-DAG: "soc": "Ascend910B2"
// TILING: "fixed": false
// TILING: "name": "TB_M"
// TILING: "fixed": true
// TILING: "name": "dim_arg0_0"
// TILING: "shape_key": "arg0_dim0"
// TILING: "workspace_size_expr": "4096"

// MANIFEST: "guardSet": []
// MANIFEST: "kernelGraph"
// MANIFEST: "kernelName": "broadcast_add_reducesum"
// MANIFEST: "kernel_entries": [
// MANIFEST-NEXT: {
// MANIFEST-DAG: "entry_index": 0,
// MANIFEST-DAG: "kernel_id": "broadcast_add_reducesum",
// MANIFEST: "tilingParams": {
// MANIFEST-NEXT: "selected_tile_shape": [
// MANIFEST-NEXT: 64,
// MANIFEST-NEXT: 15000
// MANIFEST: "tail_plan": [
// MANIFEST-NEXT: {
// MANIFEST-DAG: "affectedPrimitiveUses": [
// MANIFEST-DAG: "data_copy",
// MANIFEST-DAG: "vector_compute"
// MANIFEST-DAG: "alignmentGranularity": 16,
// MANIFEST-DAG: "axis": 0,
// MANIFEST-DAG: "selectedPolicy": "masked_tail",
// MANIFEST-DAG: "tailBufferingMode": "separate_tail_buffer"
// MANIFEST-NEXT: },
// MANIFEST-NEXT: {
// MANIFEST-DAG: "affectedPrimitiveUses": [
// MANIFEST-DAG: "write_back"
// MANIFEST-DAG: "alignmentGranularity": 0,
// MANIFEST-DAG: "axis": 1,
// MANIFEST-DAG: "selectedPolicy": "full_extent",
// MANIFEST-DAG: "tailBufferingMode": "reuse_main_buffer_after_drain"
// MANIFEST: "tail_policies": [
// MANIFEST-NEXT: "masked_tail",
// MANIFEST-NEXT: "full_extent"
// MANIFEST: "shapeArgOrder"
// MANIFEST: "abiPosition": 0
// MANIFEST: "name": "dim_arg0_0"
// MANIFEST: "abiPosition": 1
// MANIFEST: "name": "dim_arg1_1"
// MANIFEST: "workspaceSizeBytes": 4096
// MANIFEST-CONFLICT: cannot pass both --artifact-manifest-out and --runtime-manifest-out with different paths

// HOST: struct TilingData
// HOST: extern "C"
// HOST: int32_t broadcast_add_reducesum_GetTilingSize(void)
// HOST: int32_t broadcast_add_reducesum_GetTiling(const int64_t* shape_args, int32_t shape_count, void* tiling_out)
// HOST: int64_t broadcast_add_reducesum_GetBlockDim(const int64_t* shape_args, int32_t shape_count)
// HOST: int64_t broadcast_add_reducesum_GetWorkspaceSize(const int64_t* shape_args, int32_t shape_count)
// HOST: ? 4096 : -1;

module {
  func.func @broadcast_add_reducesum(
      %a: memref<?xf16>,
      %b: memref<?x?xf16>,
      %out: memref<?xf16>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData",
          [i64, i64, i64, i64],
          ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>
  ) attributes {
      ascend.schedule.selected_tile_shape = array<i64: 64, 15000>,
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
      cann.num_inputs = 2 : i32,
      cann.workspace_size_bytes = 4096 : i64} {
    %tb_m = emitasc.member %tiling "TB_M"
        : !emitasc.py_struct<"TilingData",
              [i64, i64, i64, i64],
              ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>,
          i64
    func.return
  }
}
