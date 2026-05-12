// RUN: rm -f %t.cpp %t.tiling.json %t.manifest.json %t.host.cpp
// RUN: afir-translate -mlir-to-cann %s --tiling-space-out=%t.tiling.json --runtime-manifest-out=%t.manifest.json --host-tiling-out=%t.host.cpp --cann-soc=Ascend910B2 > %t.cpp
// RUN: FileCheck %s --input-file=%t.tiling.json --check-prefix=TILING
// RUN: FileCheck %s --input-file=%t.manifest.json --check-prefix=MANIFEST
// RUN: FileCheck %s --input-file=%t.host.cpp --check-prefix=HOST

// TILING-DAG: "kernel": "broadcast_add_reducesum"
// TILING-DAG: "schema_version": "2.0"
// TILING-DAG: "soc": "Ascend910B2"
// TILING: "fixed": false
// TILING: "name": "TB_M"
// TILING: "fixed": true
// TILING: "name": "dim_arg0_0"
// TILING: "shape_key": "arg0_dim0"
// TILING: "workspace_size_expr": "0"

// MANIFEST: "guardSet": []
// MANIFEST: "kernelGraph"
// MANIFEST: "kernelName": "broadcast_add_reducesum"
// MANIFEST: "tilingParams": {
// MANIFEST: "selected_tile_shape": [
// MANIFEST-NEXT: 64,
// MANIFEST-NEXT: 15000
// MANIFEST: "tail_policies": [
// MANIFEST-NEXT: "masked_tail",
// MANIFEST-NEXT: "full_extent"
// MANIFEST: "shapeArgOrder"
// MANIFEST: "abiPosition": 0
// MANIFEST: "name": "dim_arg0_0"
// MANIFEST: "abiPosition": 1
// MANIFEST: "name": "dim_arg1_1"
// MANIFEST: "workspaceSizeBytes": 0

// HOST: struct TilingData
// HOST: extern "C"
// HOST: int32_t broadcast_add_reducesum_GetTilingSize(void)
// HOST: int32_t broadcast_add_reducesum_GetTiling(const int64_t* shape_args, int32_t shape_count, void* tiling_out)
// HOST: int64_t broadcast_add_reducesum_GetBlockDim(const int64_t* shape_args, int32_t shape_count)
// HOST: int64_t broadcast_add_reducesum_GetWorkspaceSize(const int64_t* shape_args, int32_t shape_count)

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
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 2 : i32} {
    %tb_m = emitasc.member %tiling "TB_M"
        : !emitasc.py_struct<"TilingData",
              [i64, i64, i64, i64],
              ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>,
          i64
    func.return
  }
}
