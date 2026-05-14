// RUN: afir-translate -mlir-to-cann %s --runtime-manifest-out=%t.manifest.json --cann-soc=Ascend910B2 > %t.cpp
// RUN: FileCheck %s --input-file=%t.manifest.json --check-prefix=MANIFEST

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
// MANIFEST: "shapeArgOrder": [
// MANIFEST: "shapeKey": "arg0_dim0"
// MANIFEST: "entry_index": 1,
// MANIFEST: "kernel_id": "kernel_b"
// MANIFEST: "shapeArgOrder": [
// MANIFEST: "shapeKey": "arg0_dim0"

module attributes {
    ascend.kernel_graph.edges = [
      {from = "kernel_a", to = "kernel_b", carried_buffers = ["tmp0"]}
    ]} {
  func.func @kernel_a(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim_arg0_0"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    func.return
  }

  func.func @kernel_b(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim_arg0_0"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    func.return
  }
}
