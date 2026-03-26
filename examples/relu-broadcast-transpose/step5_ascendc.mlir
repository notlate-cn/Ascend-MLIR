#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_transpose(%arg0: memref<?x?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: i64, %arg4: i64) -> memref<?x?xf16> {
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = ascendc.pipe
    %1 = ascendc.queue : <vecin, 1>
    %2 = ascendc.queue : <vecout, 1>
    %3 = ascendc.queue : <vecin, 1>
    %4 = ascendc.queue : <vecout, 1>
    %5 = ascendc.queue : <vecin, 1>
    %6 = ascendc.queue : <vecout, 1>
    %7 = arith.index_cast %arg4 : i64 to index
    %8 = arith.index_cast %arg3 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf16>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf16>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf16>
    %9 = ascendc.tbuf : <veccalc>
    %10 = ascendc.tbuf : <veccalc>
    %11 = ascendc.queue : <vecin, 1>
    %12 = ascendc.tbuf : <vecin>
    %13 = ascendc.tbuf : <veccalc>
    %14 = ascendc.tbuf : <vecout>
    %15 = ascendc.tbuf : <vecin>
    scf.for %arg5 = %c0 to %dim step %8 {
      %24 = affine.min #map(%arg5)[%dim, %8]
      %subview = memref.subview %arg0[%arg5, 0] [%24, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_3 = memref.subview %arg1[0] [%dim_0] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_4 = memref.subview %alloc[%arg5, 0] [%24, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg6 = %c0 to %24 step %7 {
        %25 = affine.min #map(%arg6)[%24, %7]
        %subview_5 = memref.subview %subview[%arg6, 0] [%25, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %26 = arith.muli %25, %dim_0 : index
        %27 = arith.muli %26, %c2 : index
        ascendc.pipe.init_buffer %0, %15, %27 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %1, %c1_i32, %27 : !ascendc.queue<vecin, 1>, i32, index
        %28 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %29 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %29, %subview_5 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %28, %29, %26 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %1, %28 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %30 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_6 = memref.subview %subview_4[%arg6, 0] [%25, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %14, %27 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %2, %c1_i32, %27 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %0, %13, %27 : !ascendc.tbuf<veccalc>, index
        %31 = ascendc.tbuf.get_tensor %13 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %32 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %32, %subview_3 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1]>>
        %33 = arith.muli %dim_0, %c2 : index
        ascendc.pipe.init_buffer %0, %12, %33 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %11, %c1_i32, %33 : !ascendc.queue<vecin, 1>, i32, index
        %34 = ascendc.que_bind.alloc_tensor %11 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %34, %32, %dim_0 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %11, %34 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %35 = ascendc.que_bind.deque_tensor %11 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %36 = arith.index_cast %25 : index to i32
        %37 = arith.index_cast %dim_0 : index to i32
        ascendc.pipe.init_buffer %0, %10, %27 : !ascendc.tbuf<veccalc>, index
        %38 = ascendc.tbuf.get_tensor %10 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %38, %35, %36, %37, %c1_i32, %37 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %0, %9, %27 : !ascendc.tbuf<veccalc>, index
        %39 = ascendc.tbuf.get_tensor %9 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %39, %cst, %26 : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %31, %30, %39, %26 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %31, %31, %38, %26 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %2, %31 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %40 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %41 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %41, %subview_6 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %41, %40, %26 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %40 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %30 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %alloc_1 = memref.alloc(%dim_0, %dim) {alignment = 64 : i64} : memref<?x?xf16>
    %alloc_2 = memref.alloc(%dim_0, %dim) {alignment = 64 : i64} : memref<?x?xf16>
    %16 = ascendc.tbuf : <vecout>
    %17 = ascendc.tbuf : <vecin>
    scf.for %arg5 = %c0 to %dim_0 step %8 {
      %24 = affine.min #map(%arg5)[%dim_0, %8]
      %subview = memref.subview %alloc[0, %arg5] [%dim, %24] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_3 = memref.subview %alloc_2[%arg5, 0] [%24, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg6 = %c0 to %24 step %7 {
        %25 = affine.min #map(%arg6)[%24, %7]
        %subview_4 = memref.subview %subview[0, %arg6] [%dim, %25] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %26 = arith.muli %dim, %25 : index
        %27 = arith.muli %26, %c2 : index
        ascendc.pipe.init_buffer %0, %17, %27 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %3, %c1_i32, %27 : !ascendc.queue<vecin, 1>, i32, index
        %28 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %29 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %29, %subview_4 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %28, %29, %26 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %3, %28 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %30 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_5 = memref.subview %subview_3[%arg6, 0] [%25, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %16, %27 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %4, %c1_i32, %27 : !ascendc.queue<vecout, 1>, i32, index
        %31 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.transpose %31, %30 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.enque_tensor %4, %31 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %32 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %33 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %33, %subview_5 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %33, %32, %26 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %4, %32 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %3, %30 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %18 = ascendc.tbuf : <veccalc>
    %19 = ascendc.queue : <vecin, 1>
    %20 = ascendc.tbuf : <vecin>
    %21 = ascendc.tbuf : <veccalc>
    %22 = ascendc.tbuf : <vecout>
    %23 = ascendc.tbuf : <vecin>
    scf.for %arg5 = %c0 to %dim_0 step %8 {
      %24 = affine.min #map(%arg5)[%dim_0, %8]
      %subview = memref.subview %alloc_2[%arg5, 0] [%24, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_3 = memref.subview %arg2[0] [%dim] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_4 = memref.subview %alloc_1[%arg5, 0] [%24, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg6 = %c0 to %24 step %7 {
        %25 = affine.min #map(%arg6)[%24, %7]
        %subview_5 = memref.subview %subview[%arg6, 0] [%25, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %26 = arith.muli %25, %dim : index
        %27 = arith.muli %26, %c2 : index
        ascendc.pipe.init_buffer %0, %23, %27 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %5, %c1_i32, %27 : !ascendc.queue<vecin, 1>, i32, index
        %28 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %29 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %29, %subview_5 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %28, %29, %26 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %5, %28 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %30 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_6 = memref.subview %subview_4[%arg6, 0] [%25, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %22, %27 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %6, %c1_i32, %27 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %0, %21, %27 : !ascendc.tbuf<veccalc>, index
        %31 = ascendc.tbuf.get_tensor %21 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %32 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %32, %subview_3 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1]>>
        %33 = arith.muli %dim, %c2 : index
        ascendc.pipe.init_buffer %0, %20, %33 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %19, %c1_i32, %33 : !ascendc.queue<vecin, 1>, i32, index
        %34 = ascendc.que_bind.alloc_tensor %19 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %34, %32, %dim : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %19, %34 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %35 = ascendc.que_bind.deque_tensor %19 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %36 = arith.index_cast %25 : index to i32
        %37 = arith.index_cast %dim : index to i32
        ascendc.pipe.init_buffer %0, %18, %27 : !ascendc.tbuf<veccalc>, index
        %38 = ascendc.tbuf.get_tensor %18 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %38, %35, %36, %37, %c1_i32, %37 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.mul_l2 %31, %30, %38, %26 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %6, %31 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %39 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %40 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %40, %subview_6 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %40, %39, %26 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %6, %39 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %5, %30 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return %alloc_1 : memref<?x?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "relu_bias_add"} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "transpose"} in %transformed : (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "scale_mul"} in %transformed : (!transform.any_op) -> !transform.any_op
    %4 = transform.param.constant true -> !transform.any_param
    %5 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %6 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %7 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_1 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %2 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_3 "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_5 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_5 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_4 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_5 : !transform.any_op
    %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %3 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_7 "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_9 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_9 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_8 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_9 : !transform.any_op
    transform.yield 
  }
}

