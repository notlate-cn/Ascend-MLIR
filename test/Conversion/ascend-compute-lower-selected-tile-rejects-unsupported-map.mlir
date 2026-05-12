// RUN: not afir-opt %s --ascend-compute-lower 2>&1 | FileCheck %s

#transpose = affine_map<(d0, d1) -> (d1, d0)>
#project_parallel = affine_map<(d0, d1) -> (d0)>

// CHECK: unsupported selected-tile indexing map
func.func @reject_transpose_selected_tile() {
  %in = memref.alloc() : memref<16x64xf16, 9 : i32>
  %out = memref.alloc() : memref<64xf16, 10 : i32>
  %gm = memref.alloc() : memref<64xf16>
  linalg.generic {
      indexing_maps = [#transpose, #project_parallel],
      iterator_types = ["parallel", "reduction"],
      ascend.schedule.selected_tile_shape = array<i64: 64, 16>}
      ins(%in : memref<16x64xf16, 9 : i32>)
      outs(%out : memref<64xf16, 10 : i32>) {
    ^bb0(%x: f16, %acc: f16):
      %next = arith.addf %acc, %x : f16
      linalg.yield %next : f16
  }
  memref.copy %out, %gm : memref<64xf16, 10 : i32> to memref<64xf16>
  return
}
