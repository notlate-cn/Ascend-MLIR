// RUN: ascend-mlir-translate -mlir-to-cann %s --artifact-manifest-out=%t.manifest.json --host-tiling-out=%t.host.cpp --cann-soc=Ascend910B2 > %t.cpp
// RUN: FileCheck %s --input-file=%t.manifest.json --check-prefix=MANIFEST
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP
// RUN: FileCheck %s --input-file=%t.host.cpp --check-prefix=HOST

// CPP: struct TilingData
// CPP-NOT: struct TilingData
// CPP: extern "C" __global__ __aicore__ void kernel_a
// CPP-NOT: struct TilingData
// CPP: extern "C" __global__ __aicore__ void kernel_b

// HOST: struct TilingData_kernel_a
// HOST: struct TilingData_kernel_b
// HOST: int32_t kernel_a_GetTilingSize(void)
// HOST: int32_t kernel_a_GetTiling(const int64_t* shape_args, int32_t shape_count, void* tiling_out)
// HOST: int64_t kernel_a_GetWorkspaceSize(const int64_t* shape_args, int32_t shape_count)
// HOST: int32_t kernel_b_GetTilingSize(void)
// HOST: int32_t kernel_b_GetTiling(const int64_t* shape_args, int32_t shape_count, void* tiling_out)
// HOST: int64_t kernel_b_GetWorkspaceSize(const int64_t* shape_args, int32_t shape_count)

// MANIFEST: "kernelGraph": {
// MANIFEST: "edges": [
// MANIFEST: "carriedBuffers": [
// MANIFEST-NEXT: "tmp0"
// MANIFEST: "from": "kernel_a"
// MANIFEST: "to": "kernel_b"
// MANIFEST: "nodes": [
// MANIFEST: "entry_index": 0,
// MANIFEST: "name": "kernel_a"
// MANIFEST: "entry_index": 1,
// MANIFEST: "name": "kernel_b"
// MANIFEST: "kernelKind": "vec"
// MANIFEST: "kernelName": "kernel_a"
// MANIFEST: "kernel_entries": [
// MANIFEST: "abi": {
// MANIFEST: "numInputs": 1
// MANIFEST: "numOutputs": 1
// MANIFEST: "workspaceArgIndex": 2
// MANIFEST: "writesToInputArgs": []
// MANIFEST: "entry_index": 0,
// MANIFEST: "kernelKind": "vec"
// MANIFEST: "kernel_id": "kernel_a"
// MANIFEST: "resources": {
// MANIFEST: "executionUnit": "aicore"
// MANIFEST: "kernelKind": "vec"
// MANIFEST: "decisionId": "kernel_a.decision.0"
// MANIFEST: "tile_binding": "symbolic"
// MANIFEST: "tile_params": [
// MANIFEST-NEXT: {
// MANIFEST-DAG: "axis": 0,
// MANIFEST-DAG: "axisKind": "parallel",
// MANIFEST-DAG: "binding": "runtime",
// MANIFEST-DAG: "default": 32,
// MANIFEST-DAG: "extent": 128,
// MANIFEST-DAG: "name": "TB_M",
// MANIFEST-DAG: "primitiveUses": [
// MANIFEST-DAG: "data_copy"
// MANIFEST-DAG: "vector_compute"
// MANIFEST-DAG: "roles": [
// MANIFEST-DAG: "kernel_loop"
// MANIFEST-DAG: "upperBound": 64
// MANIFEST: "shape": {
// MANIFEST: "rank": 1
// MANIFEST: "shapeArgOrder": [
// MANIFEST: "shapeKey": "arg0_dim0"
// MANIFEST: "workspace": {
// MANIFEST: "argIndex": 2
// MANIFEST: "sizeBytes": 1024
// MANIFEST: "abi": {
// MANIFEST: "numInputs": 3
// MANIFEST: "numOutputs": 1
// MANIFEST: "workspaceArgIndex": 4
// MANIFEST: "writesToInputArgs": []
// MANIFEST: "entry_index": 1,
// MANIFEST: "kernelKind": "mix"
// MANIFEST: "kernel_id": "kernel_b"
// MANIFEST: "resources": {
// MANIFEST: "executionUnit": "aicore"
// MANIFEST: "kernelKind": "mix"
// MANIFEST: "decisionId": "kernel_b.decision.0"
// MANIFEST: "shape": {
// MANIFEST: "rank": 1
// MANIFEST: "shapeArgOrder": [
// MANIFEST: "shapeKey": "arg0_dim0"
// MANIFEST: "workspace": {
// MANIFEST: "argIndex": 4
// MANIFEST: "sizeBytes": 2048

module attributes {
    ascend.kernel_graph.edges = [
      {from = "internal_kernel_a", to = "internal_kernel_b", carried_buffers = ["tmp0"]}
    ]} {
  func.func @kernel_a(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim_arg0_0"]>
  ) attributes {
      ascend.schedule.kernel_metadata = [{
        decision_id = "kernel_a.decision.0",
        kernel = "internal_kernel_a",
        tile_binding = "symbolic",
        tile_params = [{
          axis = 0 : i64,
          axis_kind = "parallel",
          binding = "runtime",
          default = 32 : i64,
          extent = 128 : i64,
          name = "TB_M",
          primitive_uses = ["data_copy", "vector_compute"],
          roles = ["kernel_loop"],
          upper_bound = 64 : i64
        }],
        tail_policies = ["masked_tail"],
        tail_plan = [{
          affected = ["data_copy", "vector_compute"],
          align = 16 : i64,
          axis = 0 : i64,
          buffering = "separate_tail_buffer",
          selected = "masked_tail"
        }]
      }],
      ascendc.aicore,
      ascendc.global,
      ascendc.kernel_kind = "vec",
      cann.num_inputs = 1 : i32,
      cann.workspace_size_bytes = 1024 : i64} {
    func.return
  }

  func.func @kernel_b(
      %q: memref<?x?x?xf16>,
      %key: memref<?x?x?xf16>,
      %bias: memref<?x?x?xf32>,
      %out: memref<?x?x?xf32>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim_arg0_0"]>
  ) attributes {
      abi_matmul_batch_shape = [2],
      abi_matmul_epilogue_kind = "None",
      abi_matmul_has_bias = false,
      abi_matmul_layout_a = "ND",
      abi_matmul_layout_b = "ND",
      abi_matmul_layout_c = "ND",
      abi_matmul_op_kind = "batch_matmul",
      abi_matmul_trans_a = false,
      abi_matmul_trans_b = false,
      ascend.schedule.kernel_metadata = [{
        decision_id = "kernel_b.decision.0",
        kernel = "internal_kernel_b",
        tail_policies = ["masked_tail"],
        tail_plan = [{
          affected = ["data_copy", "vector_compute"],
          align = 16 : i64,
          axis = 0 : i64,
          buffering = "separate_tail_buffer",
          selected = "masked_tail"
        }]
      }],
      ascendc.aicore,
      ascendc.global,
      ascendc.kernel_kind = "mix",
      cann.num_inputs = 3 : i32,
      cann.workspace_size_bytes = 2048 : i64} {
    func.return
  }
}
