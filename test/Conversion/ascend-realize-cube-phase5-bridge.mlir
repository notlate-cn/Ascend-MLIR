// RUN: sed -n '/\/\/ POSITIVE-BEGIN/,/\/\/ POSITIVE-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' | FileCheck %s --check-prefix=POSITIVE
// RUN: sed -n '/\/\/ FANOUT-BEGIN/,/\/\/ FANOUT-END/p' %s | afir-opt --ascend-realize='materialization-mode=memory-space-annotate' | FileCheck %s --check-prefix=FANOUT

// POSITIVE-LABEL: func.func @matmul_add_leakyrelu
// POSITIVE: memref.alloc{{.*}} : memref<?x?xf16, 1 : i32>
// POSITIVE: memref.copy {{.*}} : memref<?x?xf16> to memref<?x?xf16, 1 : i32>
// POSITIVE: memref.alloc{{.*}} : memref<?x?xf16, 2 : i32>
// POSITIVE: memref.copy {{.*}} : memref<?x?xf16, 1 : i32> to memref<?x?xf16, 2 : i32>
// POSITIVE: memref.alloc{{.*}} : memref<?x?xf16, 3 : i32>
// POSITIVE: memref.copy {{.*}} : memref<?x?xf16> to memref<?x?xf16, 3 : i32>
// POSITIVE: memref.alloc{{.*}} : memref<?x?xf16, 4 : i32>
// POSITIVE: memref.copy {{.*}} : memref<?x?xf16, 3 : i32> to memref<?x?xf16, 4 : i32>
// POSITIVE: memref.alloc{{.*}} : memref<?x?xf32, 7 : i32>
// POSITIVE: linalg.matmul
// POSITIVE-SAME: ascendc.unit = "AiCore.Cube"
// POSITIVE-SAME: ins({{.*}} : memref<?x?xf16, 2 : i32>, memref<?x?xf16, 4 : i32>)
// POSITIVE-SAME: outs({{.*}} : memref<?x?xf32, 7 : i32>)
// POSITIVE: memref.alloc{{.*}} : memref<?x?xf32, 9 : i32>
// POSITIVE: memref.copy {{.*}} : memref<?x?xf32, 7 : i32> to memref<?x?xf32, 9 : i32>
// POSITIVE: linalg.generic
// POSITIVE-SAME: ascendc.unit = "AiCore.Vector"
// POSITIVE: memref.alloc{{.*}} : memref<?x?xf32, 10 : i32>
// POSITIVE: memref.copy {{.*}} : memref<?x?xf32, 10 : i32> to memref<?x?xf32>

// FANOUT-LABEL: func.func @matmul_fanout_rejects_cube_bridge
// FANOUT: linalg.matmul
// FANOUT-SAME: ins({{.*}} : memref<?x?xf16>, memref<?x?xf16>)
// FANOUT-SAME: outs({{.*}} : memref<?x?xf32>)
// FANOUT: linalg.generic
// FANOUT-SAME: ins({{.*}} : memref<?x?xf32>, memref<?xf32>)
// FANOUT: return

// POSITIVE-BEGIN
#identity = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (d1)>

module {
  func.func @matmul_add_leakyrelu(
      %lhs: tensor<?x?xf16>,
      %rhs: tensor<?x?xf16>,
      %bias_arg: tensor<?xf32>,
      %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %matmul = linalg.matmul
        ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%out : tensor<?x?xf32>) -> tensor<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %out, %c0 : tensor<?x?xf32>
    %n = tensor.dim %out, %c1 : tensor<?x?xf32>
    %bias_out = tensor.empty(%m, %n) : tensor<?x?xf32>
    %biased = linalg.generic {
        indexing_maps = [#identity, #bias, #identity],
        iterator_types = ["parallel", "parallel"]
      } ins(%matmul, %bias_arg : tensor<?x?xf32>, tensor<?xf32>)
        outs(%bias_out : tensor<?x?xf32>) {
      ^bb0(%x: f32, %bias: f32, %acc: f32):
        %sum = arith.addf %x, %bias : f32
        linalg.yield %sum : f32
    } -> tensor<?x?xf32>
    %alpha = arith.constant 1.000000e-03 : f32
    %relu_out = tensor.empty(%m, %n) : tensor<?x?xf32>
    %result = linalg.generic {
        indexing_maps = [#identity, #identity],
        iterator_types = ["parallel", "parallel"]
      } ins(%biased : tensor<?x?xf32>)
        outs(%relu_out : tensor<?x?xf32>) {
      ^bb0(%x: f32, %acc: f32):
        %scaled = arith.mulf %x, %alpha : f32
        %max = arith.maximumf %x, %scaled : f32
        linalg.yield %max : f32
    } -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
}
// POSITIVE-END

// FANOUT-BEGIN
#identity = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (d1)>

module {
  func.func @matmul_fanout_rejects_cube_bridge(
      %lhs: tensor<?x?xf16>,
      %rhs: tensor<?x?xf16>,
      %bias_arg: tensor<?xf32>,
      %out: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
      attributes {ascend.normalized = true} {
    %matmul = linalg.matmul {
        ascend.kernel = "kernel_0",
        ascend.op_role = "cube",
        ascend.schedule.decision_id = "kernel_0.decision.0",
        ascend.schedule.schedule_contract = "generic_tiled_loop"
      }
        ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%out : tensor<?x?xf32>) -> tensor<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %out, %c0 : tensor<?x?xf32>
    %n = tensor.dim %out, %c1 : tensor<?x?xf32>
    %bias_out = tensor.empty(%m, %n) : tensor<?x?xf32>
    %biased = linalg.generic {
        indexing_maps = [#identity, #bias, #identity],
        iterator_types = ["parallel", "parallel"]
      } ins(%matmul, %bias_arg : tensor<?x?xf32>, tensor<?xf32>)
        outs(%bias_out : tensor<?x?xf32>)
        attrs = {
          ascend.kernel = "kernel_0",
          ascend.op_role = "vector",
          ascend.schedule.decision_id = "kernel_0.decision.0",
          ascend.schedule.schedule_contract = "generic_tiled_loop"
        } {
      ^bb0(%x: f32, %bias: f32, %acc: f32):
        %sum = arith.addf %x, %bias : f32
        linalg.yield %sum : f32
    } -> tensor<?x?xf32>
    return %matmul, %biased : tensor<?x?xf32>, tensor<?x?xf32>
  }
}
// FANOUT-END
