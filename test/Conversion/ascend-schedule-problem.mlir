// RUN: sed -n '/\/\/ MAIN-BEGIN/,/\/\/ MAIN-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s
// RUN: sed -n '/\/\/ HORIZONTAL-BEGIN/,/\/\/ HORIZONTAL-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=HORIZONTAL

// MAIN-BEGIN
func.func @rank2_vector_chain(%arg0: tensor<4x8xf32>,
                              %arg1: tensor<4x8xf32>,
                              %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %empty0 = tensor.empty() : tensor<4x8xf32>
  %add = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty0 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %out_elem: f32):
    %sum = arith.addf %x, %y : f32
    linalg.yield %sum : f32
  } -> tensor<4x8xf32>

  %empty1 = tensor.empty() : tensor<4x8xf32>
  %mul = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%add, %arg2 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty1 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %out_elem: f32):
    %product = arith.mulf %x, %y : f32
    linalg.yield %product : f32
  } -> tensor<4x8xf32>

  return %mul : tensor<4x8xf32>
}

func.func @dynamic_vector(%arg0: tensor<?x8xf32>,
                          %arg1: tensor<?x8xf32>) -> tensor<?x8xf32> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x8xf32>
  %empty = tensor.empty(%d0) : tensor<?x8xf32>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x8xf32>, tensor<?x8xf32>)
    outs(%empty : tensor<?x8xf32>) {
  ^bb0(%x: f32, %y: f32, %out_elem: f32):
    %sum = arith.addf %x, %y : f32
    linalg.yield %sum : f32
  } -> tensor<?x8xf32>
  return %out : tensor<?x8xf32>
}

func.func @matmul_problem(%lhs: tensor<4x8xf16>,
                          %rhs: tensor<8x16xf16>) -> tensor<4x16xf16> {
  %empty = tensor.empty() : tensor<4x16xf16>
  %out = linalg.matmul ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x16xf16>)
                       outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
  return %out : tensor<4x16xf16>
}

func.func @reduction_problem(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
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
// MAIN-END

// HORIZONTAL-BEGIN
func.func @horizontal_siblings(%arg0: tensor<4x8xf32>,
                               %arg1: tensor<4x8xf32>,
                               %arg2: tensor<4x8xf32>)
    -> (tensor<4x8xf32>, tensor<4x8xf32>) {
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
  } ins(%arg0, %arg2 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty1 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %v = arith.mulf %x, %y : f32
    linalg.yield %v : f32
  } -> tensor<4x8xf32>

  return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}
// HORIZONTAL-END

// CHECK: AxisCoalescing:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   logical_axes = 2
// CHECK-NEXT:   parallel_axes = [0, 1]
// CHECK-NEXT:   reduction_axes = []
// CHECK-NEXT:   broadcast_axes = []
// CHECK-NEXT:   barriers = 0
// CHECK: ScheduleProblem:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   role = vector
// CHECK-NEXT:   result_rank = 2
// CHECK-NEXT:   result_shape = [4, 8]
// CHECK-NEXT:   guard_budget = 8
// CHECK-NEXT:   template_tags = [vector]
// CHECK-NEXT:   shape_constraints = [d0 == 4, d1 == 8]
// CHECK-NEXT:   structure_constraints = [elementwise_chain]
// CHECK-NEXT:   tileable_axes = [axis0, axis1]
// CHECK-NEXT:   required_reduction_axes = []
// CHECK-NEXT:   axis_constraints = [
// CHECK-NEXT:     axis=0 roles=[bind_core,kernel_loop,vectorize] tail=masked_tail
// CHECK-NEXT:     axis=1 roles=[bind_core,kernel_loop,vectorize] tail=masked_tail
// CHECK-NEXT:   ]
// CHECK-NEXT:   coalescing_hints = [
// CHECK-NEXT:     group=1 kind=vectorizable members=[0,1]
// CHECK-NEXT:   ]

// CHECK: AxisCoalescing:
// CHECK-NEXT:   kernel = kernel_1
// CHECK-NEXT:   logical_axes = 2
// CHECK-NEXT:   parallel_axes = [0, 1]
// CHECK-NEXT:   reduction_axes = []
// CHECK-NEXT:   broadcast_axes = []
// CHECK-NEXT:   barriers = 0
// CHECK: ScheduleProblem:
// CHECK-NEXT:   kernel = kernel_1
// CHECK-NEXT:   role = vector
// CHECK-NEXT:   result_rank = 2
// CHECK-NEXT:   result_shape = [?, 8]
// CHECK-NEXT:   guard_budget = 8
// CHECK-NEXT:   template_tags = [vector]
// CHECK-NEXT:   shape_constraints = [d0 dynamic, d1 == 8]
// CHECK-NEXT:   structure_constraints = []
// CHECK-NEXT:   tileable_axes = [arg0_dim0, axis1]
// CHECK-NEXT:   required_reduction_axes = []

// CHECK: AxisCoalescing:
// CHECK-NEXT:   kernel = kernel_2
// CHECK-NEXT:   logical_axes = 3
// CHECK-NEXT:   parallel_axes = [0, 1]
// CHECK-NEXT:   reduction_axes = [2]
// CHECK-NEXT:   broadcast_axes = []
// CHECK-NEXT:   barriers = 0
// CHECK: ScheduleProblem:
// CHECK-NEXT:   kernel = kernel_2
// CHECK-NEXT:   role = cube
// CHECK-NEXT:   result_rank = 2
// CHECK-NEXT:   result_shape = [4, 16]
// CHECK-NEXT:   guard_budget = 8
// CHECK-NEXT:   template_tags = [cube]
// CHECK-NEXT:   shape_constraints = [d0 == 4, d1 == 16]
// CHECK-NEXT:   structure_constraints = [matmul_contract]
// CHECK-NEXT:   tileable_axes = [axis0, axis1]
// CHECK-NEXT:   required_reduction_axes = [axis2]

// CHECK: AxisCoalescing:
// CHECK-NEXT:   kernel = kernel_3
// CHECK-NEXT:   logical_axes = 2
// CHECK-NEXT:   parallel_axes = [0]
// CHECK-NEXT:   reduction_axes = [1]
// CHECK-NEXT:   broadcast_axes = []
// CHECK-NEXT:   barriers = 0
// CHECK: ScheduleProblem:
// CHECK-NEXT:   kernel = kernel_3
// CHECK-NEXT:   role = reduction
// CHECK-NEXT:   result_rank = 1
// CHECK-NEXT:   result_shape = [4]
// CHECK-NEXT:   guard_budget = 8
// CHECK-NEXT:   template_tags = [reduction]
// CHECK-NEXT:   shape_constraints = [d0 == 4]
// CHECK-NEXT:   structure_constraints = [single_reduction_region]
// CHECK-NEXT:   tileable_axes = [axis0]
// CHECK-NEXT:   required_reduction_axes = [axis1]
// CHECK-NEXT:   axis_constraints = [
// CHECK-NEXT:     axis=0 roles=[bind_core,kernel_loop,vectorize] tail=masked_tail
// CHECK-NEXT:     axis=1 roles=[full_reduction] tail=full_extent
// CHECK-NEXT:   ]

// CHECK: Schedule report

// HORIZONTAL: SchedulePatternView:
// HORIZONTAL-NEXT:   kernel = kernel_0
// HORIZONTAL-NEXT:   ops = 2
// HORIZONTAL-NEXT:   primary_ops = 2
// HORIZONTAL-NEXT:   dominant_role = vector
// HORIZONTAL: ScheduleProblem:
// HORIZONTAL-NEXT:   kernel = kernel_0
// HORIZONTAL-NEXT:   role = vector
// HORIZONTAL:   structure_constraints = []
