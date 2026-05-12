// RUN: sed -n '/\/\/ SINGLE-BEGIN/,/\/\/ SINGLE-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s
// RUN: sed -n '/\/\/ MULTI-PRIMARY-BEGIN/,/\/\/ MULTI-PRIMARY-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=MULTI
// RUN: sed -n '/\/\/ MULTI-CUBE-BEGIN/,/\/\/ MULTI-CUBE-END/p' %s | afir-opt --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=CUBE
// RUN: sed -n '/\/\/ CONSTANT-PROJECTION-BEGIN/,/\/\/ CONSTANT-PROJECTION-END/p' %s | afir-opt --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=CONST

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
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.primary = true
    } {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %sum = arith.addf %x, %y : f32
    linalg.yield %sum : f32
  } -> tensor<4x16x8xf32>

  %mat_empty = tensor.empty() : tensor<4x16xf32>
  %mat = linalg.matmul {
      ascend.kernel = "kernel_0",
      ascend.op_role = "cube",
      ascend.primary = true
    } ins(%lhs, %rhs : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%mat_empty : tensor<4x16xf32>) -> tensor<4x16xf32>

  return %mat : tensor<4x16xf32>
}
// MULTI-CUBE-END

// CONSTANT-PROJECTION-BEGIN
func.func @manual_vector_constant_projection_broadcast_transpose(
    %arg0: tensor<8x1xf32>) -> tensor<4x8xf32> {
  %empty = tensor.empty() : tensor<4x8xf32>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d1, 0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<8x1xf32>)
    outs(%empty : tensor<4x8xf32>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.primary = true
    } {
  ^bb0(%x: f32, %out_elem: f32):
    linalg.yield %x : f32
  } -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}
// CONSTANT-PROJECTION-END

// CHECK: AxisCoalescing:
// CHECK-NEXT: kernel = kernel_0
// CHECK-NEXT: logical_axes = 2
// CHECK-NEXT: parallel_axes = [0, 1]
// CHECK-NEXT: reduction_axes = []
// CHECK-NEXT: broadcast_axes = []
// CHECK-NEXT: barriers = 0
// CHECK-NEXT: axis_constraints = [
// CHECK-NEXT: axis=0 kind=parallel roles=[bind_core,kernel_loop,vectorize] tail=masked_tail group=1
// CHECK-NEXT: axis=1 kind=parallel roles=[bind_core,kernel_loop,vectorize] tail=masked_tail group=1
// CHECK-NEXT: ]
// CHECK-NEXT: coalescing_hints = [
// CHECK-NEXT: group=1 kind=vectorizable members=[0,1]
// CHECK-NEXT: ]
// CHECK: AxisCoalescing:
// CHECK-NEXT: kernel = kernel_1
// CHECK-NEXT: logical_axes = 2
// CHECK-NEXT: parallel_axes = [0]
// CHECK-NEXT: reduction_axes = [1]
// CHECK-NEXT: broadcast_axes = []
// CHECK-NEXT: barriers = 0
// CHECK-NEXT: axis_constraints = [
// CHECK-NEXT: axis=0 kind=parallel roles=[bind_core,kernel_loop,vectorize] tail=masked_tail
// CHECK-NEXT: axis=1 kind=reduction roles=[full_reduction] tail=full_extent
// CHECK-NEXT: ]
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
// CHECK-NEXT: axis_constraints = [
// CHECK-NEXT: axis=0 kind=parallel roles=[bind_core,kernel_loop,vectorize] tail=masked_tail group=1
// CHECK-NEXT: axis=1 kind=parallel roles=[bind_core,kernel_loop,vectorize] tail=masked_tail group=1
// CHECK-NEXT: axis=2 kind=reduction roles=[full_reduction] tail=full_extent
// CHECK-NEXT: ]
// CHECK-NEXT: coalescing_hints = [
// CHECK-NEXT: group=1 kind=vectorizable members=[0,1]
// CHECK-NEXT: ]

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
// MULTI-SAME: ascend.schedule.family = "reduction_static"

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
// CUBE-SAME: ascend.schedule.family = "cube_static_matmul"
// CUBE: linalg.matmul
// CUBE-SAME: ascend.schedule.family = "cube_static_matmul"

// CONST: AxisCoalescing:
// CONST-NEXT: kernel = kernel_0
// CONST-NEXT: logical_axes = 2
// CONST-NEXT: parallel_axes = [0, 1]
// CONST-NEXT: reduction_axes = []
// CONST-NEXT: broadcast_axes = [0]
// CONST-NEXT: barriers = 0
// CONST-NEXT: axis_constraints = [
// CONST-NEXT: axis=0 kind=parallel roles=[bind_core,kernel_loop,vectorize,broadcast_projection] tail=masked_tail group=1
// CONST-NEXT: axis=1 kind=parallel roles=[bind_core,kernel_loop,vectorize] tail=masked_tail group=1
// CONST-NEXT: ]
