// RUN: afir-opt --annotate-ascendc-kernel-kind %s | FileCheck %s

// CHECK-LABEL: func.func @cube_only
// CHECK-SAME: ascendc.kernel_kind = "cube"
func.func @cube_only(%lhs: memref<4x4xf16>, %rhs: memref<4x4xf16>,
                     %out: memref<4x4xf32>) {
  linalg.matmul {ascendc.unit = "AiCore.Cube"}
      ins(%lhs, %rhs : memref<4x4xf16>, memref<4x4xf16>)
      outs(%out : memref<4x4xf32>)
  return
}

// CHECK-LABEL: func.func @vec_only
// CHECK-SAME: ascendc.kernel_kind = "vec"
func.func @vec_only(%in: memref<4x4xf32>, %out: memref<4x4xf32>) {
  linalg.generic {
    ascendc.unit = "AiCore.Vector",
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%in : memref<4x4xf32>)
    outs(%out : memref<4x4xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  }
  return
}

// CHECK-LABEL: func.func @mix_kind
// CHECK-SAME: ascendc.kernel_kind = "mix"
func.func @mix_kind(%lhs: memref<4x4xf16>, %rhs: memref<4x4xf16>,
                    %acc: memref<4x4xf32>, %in: memref<4x4xf32>,
                    %out: memref<4x4xf32>) {
  linalg.matmul {ascendc.unit = "AiCore.Cube"}
      ins(%lhs, %rhs : memref<4x4xf16>, memref<4x4xf16>)
      outs(%acc : memref<4x4xf32>)
  linalg.generic {
    ascendc.unit = "AiCore.Vector",
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%in : memref<4x4xf32>)
    outs(%out : memref<4x4xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  }
  return
}
