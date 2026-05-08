// RUN: sed -n '/\/\/ SINGLE-BEGIN/,/\/\/ SINGLE-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s
// RUN: sed -n '/\/\/ MULTI-PRIMARY-BEGIN/,/\/\/ MULTI-PRIMARY-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=MULTI
// RUN: sed -n '/\/\/ MULTI-CUBE-BEGIN/,/\/\/ MULTI-CUBE-END/p' %s | afir-opt --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=CUBE

// SINGLE-BEGIN
func.func @rank2_elementwise(%arg0: tensor<4x8xf32>,
                             %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %empty = tensor.empty() : tensor<4x8xf32>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %out_elem: f32):
    %sum = arith.addf %x, %y : f32
    linalg.yield %sum : f32
  } -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}

func.func @parallel_reduction(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<4x8xf32>)
    outs(%empty : tensor<4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %sum = arith.addf %acc, %x : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %out : tensor<4xf32>
}

func.func @broadcast_elementwise(%arg0: tensor<4x8xf32>,
                                 %arg1: tensor<8xf32>) -> tensor<4x8xf32> {
  %empty = tensor.empty() : tensor<4x8xf32>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8xf32>)
    outs(%empty : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %out_elem: f32):
    %sum = arith.addf %x, %y : f32
    linalg.yield %sum : f32
  } -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}

func.func @matmul(%lhs: tensor<4x8xf32>,
                  %rhs: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %empty = tensor.empty() : tensor<4x16xf32>
  %out = linalg.matmul ins(%lhs, %rhs : tensor<4x8xf32>, tensor<8x16xf32>)
                       outs(%empty : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %out : tensor<4x16xf32>
}
// SINGLE-END

// MULTI-PRIMARY-BEGIN
func.func @vector_chain_into_reduction(%arg0: tensor<4x8xf32>,
                                       %arg1: tensor<4x8xf32>,
                                       %arg2: tensor<4x8xf32>)
    -> tensor<4xf32> {
  %empty0 = tensor.empty() : tensor<4x8xf32>
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty0 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %v = arith.addf %x, %y : f32
    linalg.yield %v : f32
  } -> tensor<4x8xf32>

  %empty1 = tensor.empty() : tensor<4x8xf32>
  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%0, %arg2 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty1 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %v = arith.mulf %x, %y : f32
    linalg.yield %v : f32
  } -> tensor<4x8xf32>

  %empty2 = tensor.empty() : tensor<4xf32>
  %2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%1 : tensor<4x8xf32>)
    outs(%empty2 : tensor<4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %sum = arith.addf %acc, %x : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>

  return %2 : tensor<4xf32>
}
// MULTI-PRIMARY-END

// MULTI-CUBE-BEGIN
func.func @manual_cube_dominant_multi_primary(
    %arg0: tensor<4x16x8xf32>,
    %arg1: tensor<4x16x8xf32>,
    %lhs: tensor<4x8xf32>,
    %rhs: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %vec_empty = tensor.empty() : tensor<4x16x8xf32>
  %vec = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x16x8xf32>, tensor<4x16x8xf32>)
    outs(%vec_empty : tensor<4x16x8xf32>)
    attrs = {
      ascend.v2.kernel = "kernel_0",
      ascend.v2.op_role = "vector",
      ascend.v2.primary = true
    } {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %sum = arith.addf %x, %y : f32
    linalg.yield %sum : f32
  } -> tensor<4x16x8xf32>

  %mat_empty = tensor.empty() : tensor<4x16xf32>
  %mat = linalg.matmul {
      ascend.v2.kernel = "kernel_0",
      ascend.v2.op_role = "cube",
      ascend.v2.primary = true
    } ins(%lhs, %rhs : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%mat_empty : tensor<4x16xf32>) -> tensor<4x16xf32>

  return %mat : tensor<4x16xf32>
}
// MULTI-CUBE-END

// CHECK: AxisCoalescing:
// CHECK-NEXT: kernel = kernel_0
// CHECK-NEXT: logical_axes = 2
// CHECK-NEXT: parallel_axes = [0, 1]
// CHECK-NEXT: reduction_axes = []
// CHECK-NEXT: broadcast_axes = []
// CHECK-NEXT: barriers = 0
// CHECK: AxisCoalescing:
// CHECK-NEXT: kernel = kernel_1
// CHECK-NEXT: logical_axes = 2
// CHECK-NEXT: parallel_axes = [0]
// CHECK-NEXT: reduction_axes = [1]
// CHECK-NEXT: broadcast_axes = []
// CHECK-NEXT: barriers = 0
// CHECK: AxisCoalescing:
// CHECK-NEXT: kernel = kernel_2
// CHECK-NEXT: logical_axes = 2
// CHECK-NEXT: parallel_axes = [0, 1]
// CHECK-NEXT: reduction_axes = []
// CHECK-NEXT: broadcast_axes = [0]
// CHECK-NEXT: barriers = 0
// CHECK: AxisCoalescing:
// CHECK-NEXT: kernel = kernel_3
// CHECK-NEXT: logical_axes = 3
// CHECK-NEXT: parallel_axes = [0, 1]
// CHECK-NEXT: reduction_axes = [2]
// CHECK-NEXT: broadcast_axes = []
// CHECK-NEXT: barriers = 0

// MULTI: SchedulePatternView:
// MULTI-NEXT: kernel = kernel_0
// MULTI-NEXT: ops = 3
// MULTI-NEXT: primary_ops = 2
// MULTI-NEXT: dominant_role = reduction
// MULTI: AxisCoalescing:
// MULTI-NEXT: kernel = kernel_0
// MULTI-NEXT: logical_axes = 2
// MULTI-NEXT: parallel_axes = [0]
// MULTI-NEXT: reduction_axes = [1]
// MULTI-NEXT: broadcast_axes = []
// MULTI-NEXT: barriers = 0
// MULTI: Schedule report
// MULTI: op_role = "reduction"
// MULTI: schedule_family = "reduction_static"
// MULTI: linalg.generic
// MULTI-SAME: ascend.v2.schedule.family = "reduction_static"

// CUBE: SchedulePatternView:
// CUBE-NEXT: kernel = kernel_0
// CUBE-NEXT: ops = 2
// CUBE-NEXT: primary_ops = 2
// CUBE-NEXT: dominant_role = cube
// CUBE: AxisCoalescing:
// CUBE-NEXT: kernel = kernel_0
// CUBE-NEXT: logical_axes = 3
// CUBE-NEXT: parallel_axes = [0, 1]
// CUBE-NEXT: reduction_axes = [2]
// CUBE-NEXT: broadcast_axes = []
// CUBE-NEXT: barriers = 0
// CUBE: Schedule report
// CUBE: op_role = "cube"
// CUBE: schedule_family = "cube_static_matmul"
// CUBE: linalg.generic
// CUBE-SAME: ascend.v2.schedule.family = "cube_static_matmul"
// CUBE: linalg.matmul
// CUBE-SAME: ascend.v2.schedule.family = "cube_static_matmul"
