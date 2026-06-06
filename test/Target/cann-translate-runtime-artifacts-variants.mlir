// RUN: rm -f %t.cpp %t.tiling.json %t.manifest.json %t.host.cpp
// RUN: ascend-mlir-translate -mlir-to-cann %s --tiling-space-out=%t.tiling.json --artifact-manifest-out=%t.manifest.json --host-tiling-out=%t.host.cpp --cann-soc=Ascend910B2 > %t.cpp
// RUN: FileCheck %s --input-file=%t.manifest.json --check-prefix=MANIFEST
// RUN: FileCheck %s --input-file=%t.tiling.json --check-prefix=TILING
// RUN: FileCheck %s --input-file=%t.host.cpp --check-prefix=HOST

// MANIFEST: "guardSet": [
// MANIFEST-NEXT: "arg0_dim0 <= 128",
// MANIFEST-NEXT: "arg0_dim0 > 0"
// MANIFEST: "shapeBucketKey": "M.small_or_fallback"
// MANIFEST: "scheduleEntries": [
// MANIFEST: "blockDim": 4
// MANIFEST: "decisionId": "kernel_variant.small"
// MANIFEST: "fallback": false
// MANIFEST: "guard": "arg0_dim0 <= 128"
// MANIFEST: "priority": 0
// MANIFEST: "shapeBucketKey": "M.small"
// MANIFEST: "structured_lowering": {
// MANIFEST-DAG: "contract": "generic_tiled_loop"
// MANIFEST-DAG: "loop_axes": [
// MANIFEST-DAG: "axis": 0
// MANIFEST-DAG: "axis_kind": "parallel"
// MANIFEST-DAG: "tile_param": "TB_M"
// MANIFEST-DAG: "representation": "symbolic_marker_contract"
// MANIFEST: "workspaceSizeBytes": 1024
// MANIFEST: "blockDim": 1
// MANIFEST: "decisionId": "kernel_variant.fallback"
// MANIFEST: "fallback": true
// MANIFEST: "guard": "arg0_dim0 > 0"
// MANIFEST: "priority": 99
// MANIFEST: "shapeBucketKey": "M.fallback"
// MANIFEST: "workspaceSizeBytes": 2048

// TILING: "guardSet": [
// TILING-NEXT: "arg0_dim0 <= 128",
// TILING-NEXT: "arg0_dim0 > 0"
// TILING: "kernels": [
// TILING: "guardSet": [
// TILING-NEXT: "arg0_dim0 <= 128",
// TILING-NEXT: "arg0_dim0 > 0"
// TILING: "kernel": "kernel_variant"
// TILING: "scheduleEntries": [
// TILING: "decisionId": "kernel_variant.small"
// TILING: "structured_lowering": {
// TILING-DAG: "contract": "generic_tiled_loop"
// TILING: "shapeBucketKey": "M.small"
// TILING: "decisionId": "kernel_variant.fallback"
// TILING: "shapeBucketKey": "M.fallback"
// TILING: "shapeBucketKey": "M.small_or_fallback"

// HOST: static int32_t kernel_variant_SelectScheduleEntry(const int64_t* shape_args)
// HOST: if (shape_args[0] <= 128)
// HOST-NEXT: return 0;
// HOST: if (shape_args[0] > 0)
// HOST-NEXT: return 1;
// HOST: case 0:
// HOST-NEXT: data.TB_M = 64;
// HOST: case 1:
// HOST-NEXT: data.TB_M = 16;
// HOST: int64_t kernel_variant_GetBlockDim(const int64_t* shape_args, int32_t shape_count)
// HOST: case 0:
// HOST-NEXT: return 4;
// HOST: case 1:
// HOST-NEXT: return 1;
// HOST: int64_t kernel_variant_GetWorkspaceSize(const int64_t* shape_args, int32_t shape_count)
// HOST: case 0:
// HOST-NEXT: return 1024;
// HOST: case 1:
// HOST-NEXT: return 2048;

module {
  func.func @kernel_variant(
      %a: memref<?xf16>,
      %out: memref<?xf16>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData",
          [i64, i64],
          ["TB_M", "dim_arg0_0"]>
  ) attributes {
      ascend.schedule.kernel_metadata = [
        {
          block_dim = 4 : i64,
          decision_id = "kernel_variant.small",
          fallback = false,
          guard = "arg0_dim0 <= 128",
          kernel = "kernel_variant",
          priority = 0 : i64,
          shape_bucket_key = "M.small",
          structured_lowering = {
            cache_read_marker = "metadata_deferred",
            cache_write_marker = "metadata_deferred",
            contract = "generic_tiled_loop",
            double_buffer_marker = "none",
            guard_marker_count = 1 : i64,
            loop_axes = [{axis = 0 : i64, axis_kind = "parallel", binding = "runtime", primitive_uses = ["data_copy", "vector_compute"], roles = ["kernel_loop"], tile_param = "TB_M"}],
            pipeline_marker = "none",
            representation = "symbolic_marker_contract",
            tail_marker_count = 1 : i64
          },
          tile_binding = "symbolic",
          tile_params = [{
            axis = 0 : i64,
            axis_kind = "parallel",
            binding = "runtime",
            default = 64 : i64,
            extent = -9223372036854775808 : i64,
            name = "TB_M",
            primitive_uses = ["data_copy", "vector_compute"],
            roles = ["kernel_loop"],
            upper_bound = 128 : i64
          }],
          tail_policies = ["masked_tail"],
          tail_plan = [{
            affected = ["data_copy", "vector_compute"],
            align = 16 : i64,
            axis = 0 : i64,
            buffering = "separate_tail_buffer",
            selected = "masked_tail"
          }],
          workspace_size_bytes = 1024 : i64
        },
        {
          block_dim = 1 : i64,
          decision_id = "kernel_variant.fallback",
          fallback = true,
          guard = "arg0_dim0 > 0",
          kernel = "kernel_variant",
          priority = 99 : i64,
          shape_bucket_key = "M.fallback",
          structured_lowering = {
            cache_read_marker = "metadata_deferred",
            cache_write_marker = "metadata_deferred",
            contract = "generic_tiled_loop",
            double_buffer_marker = "none",
            guard_marker_count = 0 : i64,
            loop_axes = ["arg0_dim0"],
            pipeline_marker = "none",
            representation = "symbolic_marker_contract",
            tail_marker_count = 0 : i64
          },
          tile_binding = "symbolic",
          tile_params = [{
            axis = 0 : i64,
            axis_kind = "parallel",
            binding = "runtime",
            default = 16 : i64,
            extent = -9223372036854775808 : i64,
            name = "TB_M",
            primitive_uses = ["data_copy", "vector_compute"],
            roles = ["kernel_loop"],
            upper_bound = 128 : i64
          }],
          tail_policies = ["scalar_epilogue"],
          tail_plan = [{
            affected = ["data_copy", "vector_compute"],
            align = 1 : i64,
            axis = 0 : i64,
            buffering = "separate_tail_buffer",
            selected = "scalar_epilogue"
          }],
          workspace_size_bytes = 2048 : i64
        }
      ],
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32,
      cann.workspace_size_bytes = 1024 : i64} {
    func.return
  }
}
