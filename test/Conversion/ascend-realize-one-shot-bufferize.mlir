// RUN: afir-opt %s --ascend-realize='materialization-mode=one-shot-bufferize' | FileCheck %s
// RUN: not afir-opt %s --ascend-realize='materialization-mode=bad' 2>&1 | FileCheck %s --check-prefix=BAD

func.func @realize_one_shot(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {ascend.normalized = true} {
  %empty = tensor.empty() : tensor<4xf32>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
    outs(%empty : tensor<4xf32>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %v = arith.addf %x, %y : f32
    linalg.yield %v : f32
  } -> tensor<4xf32>
  return %out : tensor<4xf32>
}

// CHECK-LABEL: func.func @realize_one_shot(%arg0: memref<4xf32
// CHECK-SAME: %arg1: memref<4xf32
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: linalg.generic
// CHECK-NOT: tensor.empty
// CHECK: return %[[ALLOC]] : memref<4xf32>

// BAD: unsupported ascend-realize materialization-mode "bad"
