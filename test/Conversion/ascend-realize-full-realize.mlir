// RUN: afir-opt %s --ascend-realize='materialization-mode=full-realize' | FileCheck %s

// Test that full-realize mode tiles linalg.matmul into scf.for loops AND
// bufferizes the tensor IR to memref IR. Function arguments become memrefs,
// tensor.extract_slice becomes memref.subview, and the tiled linalg.matmul
// operates on memref subviews.

func.func @matmul_full_realize(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>,
                                %arg2: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %out = linalg.matmul {
    ascend.kernel = "kernel_0",
    ascend.schedule.decision_id = "kernel_0.decision.0",
    ascend.schedule.structured_lowering = "loop_skeleton_v0",
    ascend.schedule.selected_tile_shape = array<i64: 4, 4, 4>
  } ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>)
    outs(%arg2 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %out : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @matmul_full_realize
// CHECK-SAME:    %arg0: memref<8x8xf32>
// CHECK-SAME:    %arg1: memref<8x8xf32>
// CHECK-SAME:    %arg2: memref<8x8xf32>
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               memref.subview
// CHECK:               linalg.matmul
// CHECK-NOT:     tensor.empty
