// RUN: not ascend-mlir-opt %s --ascend-compute-lower 2>&1 | FileCheck %s

#map_parallel = affine_map<(d0, d1) -> (d0)>
#map_reduction = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: unsupported compute kind unknown
func.func @unsupported_reduction_select_body() {
  %lhs = memref.alloc() : memref<8xf32, 9 : i32>
  %rhs = memref.alloc() : memref<8x4xf32, 9 : i32>
  %out = memref.alloc() : memref<8xf32, 10 : i32>
  linalg.generic {
      indexing_maps = [#map_parallel, #map_reduction, #map_parallel],
      iterator_types = ["parallel", "reduction"],
      ascendc.unit = "AiCore.Vector"}
      ins(%lhs, %rhs : memref<8xf32, 9 : i32>, memref<8x4xf32, 9 : i32>)
      outs(%out : memref<8xf32, 10 : i32>) {
    ^bb0(%in: f32, %in_0: f32, %acc: f32):
      %cmp = arith.cmpf olt, %in, %in_0 : f32
      %selected = arith.select %cmp, %in, %in_0 : f32
      %next = arith.addf %acc, %selected : f32
      linalg.yield %next : f32
  }
  return
}
