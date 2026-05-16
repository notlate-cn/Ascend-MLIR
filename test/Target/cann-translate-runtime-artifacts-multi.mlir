// RUN: afir-translate -mlir-to-cann %s --runtime-manifest-out=%t.manifest.json --cann-soc=Ascend910B2 > %t.cpp
// RUN: FileCheck %s --input-file=%t.manifest.json --check-prefix=MANIFEST
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// CPP: struct TilingData
// CPP-NOT: struct TilingData
// CPP: extern "C" __global__ __aicore__ void kernel_a
// CPP-NOT: struct TilingData
// CPP: extern "C" __global__ __aicore__ void kernel_b

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
// MANIFEST: "kernelName": "kernel_a"
// MANIFEST: "kernel_entries": [
// MANIFEST: "entry_index": 0,
// MANIFEST: "kernel_id": "kernel_a"
// MANIFEST: "resources": {
// MANIFEST: "executionUnit": "aicore"
// MANIFEST: "decisionId": "kernel_a.decision.0"
// MANIFEST: "selected_tile_shape": [
// MANIFEST-NEXT: 32
// MANIFEST: "shape": {
// MANIFEST: "rank": 1
// MANIFEST: "shapeArgOrder": [
// MANIFEST: "shapeKey": "arg0_dim0"
// MANIFEST: "workspace": {
// MANIFEST: "argIndex": 2
// MANIFEST: "sizeBytes": 0
// MANIFEST: "entry_index": 1,
// MANIFEST: "kernel_id": "kernel_b"
// MANIFEST: "decisionId": "kernel_b.decision.0"
// MANIFEST: "selected_tile_shape": [
// MANIFEST-NEXT: 64
// MANIFEST: "shape": {
// MANIFEST: "rank": 1
// MANIFEST: "shapeArgOrder": [
// MANIFEST: "shapeKey": "arg0_dim0"
// MANIFEST: "workspace": {
// MANIFEST: "argIndex": 2
// MANIFEST: "sizeBytes": 0

module attributes {
    ascend.kernel_graph.edges = [
      {from = "kernel_a", to = "kernel_b", carried_buffers = ["tmp0"]}
    ]} {
  func.func @kernel_a(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim_arg0_0"]>
  ) attributes {
      ascend.schedule.kernel_metadata = [{
        decision_id = "kernel_a.decision.0",
        kernel = "kernel_a",
        selected_tile_shape = array<i64: 32>,
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
      cann.num_inputs = 1 : i32} {
    func.return
  }

  func.func @kernel_b(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim_arg0_0"]>
  ) attributes {
      ascend.schedule.kernel_metadata = [{
        decision_id = "kernel_b.decision.0",
        kernel = "kernel_b",
        selected_tile_shape = array<i64: 64>,
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
      cann.num_inputs = 1 : i32} {
    func.return
  }
}
