// RUN: sed -n '/\/\/ SINGLE-BEGIN/,/\/\/ SINGLE-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s
// RUN: sed -n '/\/\/ MULTI-PRIMARY-BEGIN/,/\/\/ MULTI-PRIMARY-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=MULTI
// RUN: sed -n '/\/\/ MULTI-CUBE-BEGIN/,/\/\/ MULTI-CUBE-END/p' %s | afir-opt --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=CUBE
// RUN: sed -n '/\/\/ CONSTANT-PROJECTION-BEGIN/,/\/\/ CONSTANT-PROJECTION-END/p' %s | afir-opt --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=CONST
// RUN: sed -n '/\/\/ GATHER-TAIL-BEGIN/,/\/\/ GATHER-TAIL-END/p' %s | afir-opt --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=TAIL
// RUN: sed -n '/\/\/ EMBEDDING-TAIL-BEGIN/,/\/\/ EMBEDDING-TAIL-END/p' %s | afir-opt --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=EMBED
// RUN: sed -n '/\/\/ PERMUTED-GATHER-TAIL-BEGIN/,/\/\/ PERMUTED-GATHER-TAIL-END/p' %s | afir-opt --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=PERMUTE
// RUN: sed -n '/\/\/ POST-REDUCE-BEGIN/,/\/\/ POST-REDUCE-END/p' %s | afir-opt --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=POSTREDUCE
// RUN: sed -n '/\/\/ CONFLICT-BEGIN/,/\/\/ CONFLICT-END/p' %s | not afir-opt --ascend-schedule 2>&1 | FileCheck %s --check-prefix=CONFLICT

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

// GATHER-TAIL-BEGIN
#tail_col = affine_map<(d0, d1) -> (d1)>
#tail_id = affine_map<(d0, d1) -> (d0, d1)>
func.func @manual_gather_tail_contract(%data: tensor<?x?xf16>,
                                       %indices: tensor<?xi64>,
                                       %m: index,
                                       %k: index) -> tensor<?x?xf16> {
  %empty = tensor.empty(%m, %k) : tensor<?x?xf16>
  %out = linalg.generic {
      indexing_maps = [#tail_col, #tail_id],
      iterator_types = ["parallel", "parallel"]
    } ins(%indices : tensor<?xi64>)
      outs(%empty : tensor<?x?xf16>)
      attrs = {
        ascend.kernel = "kernel_0",
        ascend.op_role = "vector",
        ascend.primary = true,
        gather_dim = 1 : i64
      } {
  ^bb0(%idx: i64, %out_elem: f16):
    %i = linalg.index 0 : index
    %idx_cast = arith.index_cast %idx : i64 to index
    %val = tensor.extract %data[%i, %idx_cast] : tensor<?x?xf16>
    linalg.yield %val : f16
  } -> tensor<?x?xf16>
  return %out : tensor<?x?xf16>
}
// GATHER-TAIL-END

// EMBEDDING-TAIL-BEGIN
#embed_row = affine_map<(d0, d1) -> (d0)>
#embed_id = affine_map<(d0, d1) -> (d0, d1)>
func.func @manual_embedding_tail_contract(%weight: tensor<?x?xf16>,
                                          %indices: tensor<?xi32>,
                                          %m: index,
                                          %k: index) -> tensor<?x?xf16> {
  %empty = tensor.empty(%m, %k) : tensor<?x?xf16>
  %out = linalg.generic {
      indexing_maps = [#embed_row, #embed_id],
      iterator_types = ["parallel", "parallel"]
    } ins(%indices : tensor<?xi32>)
      outs(%empty : tensor<?x?xf16>)
      attrs = {
        ascend.kernel = "kernel_0",
        ascend.op_role = "vector",
        ascend.primary = true,
        embedding_dim = 0 : i64
      } {
  ^bb0(%idx: i32, %out_elem: f16):
    %j = linalg.index 1 : index
    %idx64 = arith.extsi %idx : i32 to i64
    %idx_cast = arith.index_cast %idx64 : i64 to index
    %val = tensor.extract %weight[%idx_cast, %j] : tensor<?x?xf16>
    linalg.yield %val : f16
  } -> tensor<?x?xf16>
  return %out : tensor<?x?xf16>
}
// EMBEDDING-TAIL-END

// PERMUTED-GATHER-TAIL-BEGIN
#permute_row = affine_map<(d0, d1) -> (d0)>
#permute_id = affine_map<(d0, d1) -> (d0, d1)>
func.func @manual_permuted_gather_tail_contract(%data: tensor<?x?xf16>,
                                                %indices: tensor<?xi64>,
                                                %m: index,
                                                %k: index,
                                                %bias: tensor<?xf16>)
    -> tensor<?x?xf16> {
  %empty = tensor.empty(%m, %k) : tensor<?x?xf16>
  %out = linalg.generic {
      indexing_maps = [#permute_row, #permute_row, #permute_id],
      iterator_types = ["parallel", "parallel"]
    } ins(%indices, %bias : tensor<?xi64>, tensor<?xf16>)
      outs(%empty : tensor<?x?xf16>)
      attrs = {
        ascend.kernel = "kernel_0",
        ascend.op_role = "vector",
        ascend.primary = true,
        gather_dim = 1 : i64
      } {
  ^bb0(%idx: i64, %bias_elem: f16, %out_elem: f16):
    %j = linalg.index 1 : index
    %idx_cast = arith.index_cast %idx : i64 to index
    %val = tensor.extract %data[%j, %idx_cast] : tensor<?x?xf16>
    %sum = arith.addf %val, %bias_elem : f16
    linalg.yield %sum : f16
  } -> tensor<?x?xf16>
  return %out : tensor<?x?xf16>
}
// PERMUTED-GATHER-TAIL-END

// POST-REDUCE-BEGIN
#post_reduce_in = affine_map<(d0, d1) -> (d0, d1)>
#post_reduce_out = affine_map<(d0, d1) -> (d0, 0)>
#post_reduce_singleton = affine_map<(d0, d1) -> (d0, d1)>
func.func @manual_reduction_with_singleton_vector_epilogue(
    %arg0: tensor<4x8xf32>, %init: tensor<4x1xf32>) -> tensor<4x1xf32> {
  %scale = arith.constant 8.000000e+00 : f32
  %red = linalg.generic {
    indexing_maps = [#post_reduce_in, #post_reduce_out],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<4x8xf32>)
    outs(%init : tensor<4x1xf32>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "reduction",
      ascend.primary = true
    } {
  ^bb0(%x: f32, %acc: f32):
    %sum = arith.addf %acc, %x : f32
    linalg.yield %sum : f32
  } -> tensor<4x1xf32>

  %out = linalg.generic {
    indexing_maps = [#post_reduce_singleton, #post_reduce_singleton],
    iterator_types = ["parallel", "parallel"]
  } ins(%red : tensor<4x1xf32>)
    outs(%init : tensor<4x1xf32>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector"
    } {
  ^bb0(%x: f32, %o: f32):
    %v = arith.divf %x, %scale : f32
    linalg.yield %v : f32
  } -> tensor<4x1xf32>

  return %out : tensor<4x1xf32>
}
// POST-REDUCE-END

// CONFLICT-BEGIN
func.func @manual_axis_static_extent_conflict(
    %a4: tensor<4xf32>,
    %b4: tensor<4xf32>,
    %a8: tensor<8xf32>,
    %b8: tensor<8xf32>) -> (tensor<4xf32>, tensor<8xf32>) {
  %empty0 = tensor.empty() : tensor<4xf32>
  %out0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%a4, %b4 : tensor<4xf32>, tensor<4xf32>)
    outs(%empty0 : tensor<4xf32>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.primary = true
    } {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %sum = arith.addf %x, %y : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>

  %empty1 = tensor.empty() : tensor<8xf32>
  %out1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%a8, %b8 : tensor<8xf32>, tensor<8xf32>)
    outs(%empty1 : tensor<8xf32>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.primary = true
    } {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %product = arith.mulf %x, %y : f32
    linalg.yield %product : f32
  } -> tensor<8xf32>

  return %out0, %out1 : tensor<4xf32>, tensor<8xf32>
}
// CONFLICT-END

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

// TAIL: AxisCoalescing:
// TAIL-NEXT: kernel = kernel_0
// TAIL-NEXT: logical_axes = 2
// TAIL-NEXT: parallel_axes = [0, 1]
// TAIL-NEXT: reduction_axes = []
// TAIL-NEXT: broadcast_axes = [0]
// TAIL-NEXT: barriers = 0
// TAIL-NEXT: axis_constraints = [
// TAIL-NEXT: axis=0 kind=parallel roles=[bind_core,kernel_loop,vectorize,broadcast_projection] tail=masked_tail group=1 allowed_tail=[masked_tail,scalar_epilogue] primitive_uses=[data_copy,vector_compute,write_back] semantic_align=0
// TAIL-NEXT: axis=1 kind=parallel roles=[bind_core,kernel_loop,vectorize] tail=masked_tail group=1 allowed_tail=[masked_tail,scalar_epilogue,pad_and_mask] primitive_uses=[data_copy,vector_compute,write_back,gather_index] semantic_align=16
// TAIL-NEXT: ]

// EMBED: AxisCoalescing:
// EMBED-NEXT: kernel = kernel_0
// EMBED-NEXT: logical_axes = 2
// EMBED-NEXT: parallel_axes = [0, 1]
// EMBED-NEXT: reduction_axes = []
// EMBED-NEXT: broadcast_axes = [1]
// EMBED-NEXT: barriers = 0
// EMBED-NEXT: axis_constraints = [
// EMBED-NEXT: axis=0 kind=parallel roles=[bind_core,kernel_loop,vectorize] tail=masked_tail group=1 allowed_tail=[masked_tail,scalar_epilogue,pad_and_mask] primitive_uses=[data_copy,vector_compute,write_back,gather_index] semantic_align=16
// EMBED-NEXT: axis=1 kind=parallel roles=[bind_core,kernel_loop,vectorize,broadcast_projection] tail=masked_tail group=1 allowed_tail=[masked_tail,scalar_epilogue] primitive_uses=[data_copy,vector_compute,write_back] semantic_align=0
// EMBED-NEXT: ]

// PERMUTE: AxisCoalescing:
// PERMUTE-NEXT: kernel = kernel_0
// PERMUTE-NEXT: logical_axes = 2
// PERMUTE-NEXT: parallel_axes = [0, 1]
// PERMUTE-NEXT: reduction_axes = []
// PERMUTE-NEXT: broadcast_axes = [1]
// PERMUTE-NEXT: barriers = 0
// PERMUTE-NEXT: axis_constraints = [
// PERMUTE-NEXT: axis=0 kind=parallel roles=[bind_core,kernel_loop,vectorize] tail=masked_tail group=1 allowed_tail=[masked_tail,scalar_epilogue,pad_and_mask] primitive_uses=[data_copy,vector_compute,write_back,gather_index] semantic_align=16
// PERMUTE-NEXT: axis=1 kind=parallel roles=[bind_core,kernel_loop,vectorize,broadcast_projection] tail=masked_tail group=1 allowed_tail=[masked_tail,scalar_epilogue] primitive_uses=[data_copy,vector_compute,write_back] semantic_align=0
// PERMUTE-NEXT: ]

// POSTREDUCE: AxisCoalescing:
// POSTREDUCE-NEXT: kernel = kernel_0
// POSTREDUCE-NEXT: logical_axes = 2
// POSTREDUCE-NEXT: parallel_axes = [0]
// POSTREDUCE-NEXT: reduction_axes = [1]
// POSTREDUCE-NEXT: broadcast_axes = []
// POSTREDUCE-NEXT: barriers = 0
// POSTREDUCE-NEXT: axis_constraints = [
// POSTREDUCE-NEXT: axis=0 kind=parallel roles=[bind_core,kernel_loop,vectorize] tail=masked_tail
// POSTREDUCE-NEXT: axis=1 kind=reduction roles=[full_reduction] tail=full_extent
// POSTREDUCE-NEXT: ]

// CONFLICT: conflicting static extent for logical axis 0: 4 vs 8
