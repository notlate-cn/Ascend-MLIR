// RUN: afir-opt %s --auto-fuse-fold-shadow-alloc | FileCheck %s
//
// Verify that the post-bufferize shadow-alloc sandwich produced by
// one-shot-bufferize on tail blocks with dynamic-size DPS init operands
// gets folded back to an in-place write.

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @fold_simple
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: linalg.generic
// CHECK-SAME: outs(%{{.*}} : memref<?xf32, strided<[1], offset: ?>>)
func.func @fold_simple(%arg0: memref<64xf32>, %arg1: memref<64xf32>,
                        %off: index, %sz: index) {
  %src = memref.subview %arg1[%off] [%sz] [1]
       : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>>
  %dst = memref.subview %arg1[%off] [%sz] [1]
       : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>>
  %alloc = memref.alloc(%sz) {alignment = 64 : i64} : memref<?xf32>
  memref.copy %src, %alloc
      : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32>
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]
  } ins(%arg0 : memref<64xf32>) outs(%alloc : memref<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.addf %in, %out : f32
    linalg.yield %v : f32
  }
  memref.copy %alloc, %dst
      : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
  return
}

// Counter-example: pre.src and post.dst point to *different* regions of the
// same memref (different offsets).  The fold must bail out to preserve
// semantics — the alloc, both copies, and the generic must all remain.
//
// CHECK-LABEL: func.func @bail_differing_region
// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: linalg.generic
// CHECK: memref.copy
func.func @bail_differing_region(%arg0: memref<64xf32>, %arg1: memref<64xf32>,
                                  %off_a: index, %off_b: index, %sz: index) {
  %src = memref.subview %arg1[%off_a] [%sz] [1]
       : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>>
  %dst = memref.subview %arg1[%off_b] [%sz] [1]
       : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>>
  %alloc = memref.alloc(%sz) {alignment = 64 : i64} : memref<?xf32>
  memref.copy %src, %alloc
      : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32>
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]
  } ins(%arg0 : memref<64xf32>) outs(%alloc : memref<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.addf %in, %out : f32
    linalg.yield %v : f32
  }
  memref.copy %alloc, %dst
      : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
  return
}
