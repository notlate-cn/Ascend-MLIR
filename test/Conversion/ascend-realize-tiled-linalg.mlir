// RUN: afir-opt %s --ascend-realize='materialization-mode=tiled-linalg' | FileCheck %s

// Test that tiled-linalg mode tiles linalg.matmul into scf.for loops when
// ascend.schedule.selected_tile_shape is present. The tensor IR is preserved
// (no bufferization); only tiling is applied.

func.func @matmul_tiled(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>,
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

// CHECK-LABEL: func.func @matmul_tiled
// CHECK-SAME:    %arg0: tensor<8x8xf32>
// CHECK-SAME:    %arg1: tensor<8x8xf32>
// CHECK-SAME:    %arg2: tensor<8x8xf32>
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               tensor.extract_slice
// CHECK:               linalg.matmul
// CHECK:               tensor.insert_slice
// CHECK-NOT:     memref
