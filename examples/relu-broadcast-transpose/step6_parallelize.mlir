#map = affine_map<()[s0, s1, s2] -> (s1, s0 - s2)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
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
    %11 = ascendc.tbuf : <veccalc>
    %12 = ascendc.tbuf : <veccalc>
    %13 = ascendc.tbuf : <vecout>
    %14 = ascendc.tbuf : <vecin>
    %15 = ascendc.get_block_idx : index
    %16 = arith.muli %15, %8 : index
    %17 = arith.cmpi ult, %16, %dim : index
    scf.if %17 {
      %27 = affine.min #map()[%dim, %8, %16]
      %subview = memref.subview %arg0[%16, 0] [%27, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_3 = memref.subview %arg1[0] [%dim_0] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_4 = memref.subview %alloc[%16, 0] [%27, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg5 = %c0 to %27 step %7 {
        %28 = affine.min #map1(%arg5)[%27, %7]
        %subview_5 = memref.subview %subview[%arg5, 0] [%28, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %29 = arith.muli %28, %dim_0 : index
        %30 = arith.muli %29, %c2 : index
        ascendc.pipe.init_buffer %0, %14, %30 : !ascendc.tbuf<vecin>, index
        %31 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %32 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %32, %subview_5 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %31, %32, %29 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %1, %31 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %33 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_6 = memref.subview %subview_4[%arg5, 0] [%28, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %13, %30 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %0, %12, %30 : !ascendc.tbuf<veccalc>, index
        %34 = ascendc.tbuf.get_tensor %12 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %35 = arith.muli %dim_0, %c2 : index
        ascendc.pipe.init_buffer %0, %11, %35 : !ascendc.tbuf<veccalc>, index
        %36 = ascendc.tbuf.get_tensor %11 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %37 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %37, %subview_3 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1]>>
        ascendc.data_copy_l2 %36, %37, %dim_0 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %38 = arith.index_cast %28 : index to i32
        %39 = arith.index_cast %dim_0 : index to i32
        ascendc.pipe.init_buffer %0, %10, %30 : !ascendc.tbuf<veccalc>, index
        %40 = ascendc.tbuf.get_tensor %10 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %40, %36, %38, %39, %c1_i32, %39 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %0, %9, %30 : !ascendc.tbuf<veccalc>, index
        %41 = ascendc.tbuf.get_tensor %9 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %41, %cst, %29 : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %34, %33, %41, %29 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %34, %34, %40, %29 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %2, %34 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %42 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %43 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %43, %subview_6 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %43, %42, %29 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %42 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %33 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %alloc_1 = memref.alloc(%dim_0, %dim) {alignment = 64 : i64} : memref<?x?xf16>
    %alloc_2 = memref.alloc(%dim_0, %dim) {alignment = 64 : i64} : memref<?x?xf16>
    %18 = ascendc.tbuf : <veccalc>
    %19 = ascendc.tbuf : <vecout>
    %20 = ascendc.tbuf : <vecin>
    %21 = arith.cmpi ult, %16, %dim_0 : index
    scf.if %21 {
      %27 = affine.min #map()[%dim_0, %8, %16]
      %subview = memref.subview %alloc[0, %16] [%dim, %27] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_3 = memref.subview %alloc_2[%16, 0] [%27, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg5 = %c0 to %27 step %7 {
        %28 = affine.min #map1(%arg5)[%27, %7]
        %subview_4 = memref.subview %subview[0, %arg5] [%dim, %28] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %29 = arith.muli %dim, %28 : index
        %30 = arith.muli %29, %c2 : index
        ascendc.pipe.init_buffer %0, %20, %30 : !ascendc.tbuf<vecin>, index
        %31 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %32 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %32, %subview_4 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %31, %32, %29 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %3, %31 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %33 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_5 = memref.subview %subview_3[%arg5, 0] [%28, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %19, %30 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %0, %18, %30 : !ascendc.tbuf<veccalc>, index
        %34 = ascendc.tbuf.get_tensor %18 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.enque_tensor %4, %34 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %35 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %36 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %36, %subview_5 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %36, %35, %29 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %4, %35 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %3, %33 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %22 = ascendc.tbuf : <veccalc>
    %23 = ascendc.tbuf : <veccalc>
    %24 = ascendc.tbuf : <veccalc>
    %25 = ascendc.tbuf : <vecout>
    %26 = ascendc.tbuf : <vecin>
    scf.if %21 {
      %27 = affine.min #map()[%dim_0, %8, %16]
      %subview = memref.subview %alloc_2[%16, 0] [%27, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_3 = memref.subview %arg2[0] [%dim] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_4 = memref.subview %alloc_1[%16, 0] [%27, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg5 = %c0 to %27 step %7 {
        %28 = affine.min #map1(%arg5)[%27, %7]
        %subview_5 = memref.subview %subview[%arg5, 0] [%28, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %29 = arith.muli %28, %dim : index
        %30 = arith.muli %29, %c2 : index
        ascendc.pipe.init_buffer %0, %26, %30 : !ascendc.tbuf<vecin>, index
        %31 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %32 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %32, %subview_5 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %31, %32, %29 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %5, %31 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %33 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_6 = memref.subview %subview_4[%arg5, 0] [%28, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %25, %30 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %0, %24, %30 : !ascendc.tbuf<veccalc>, index
        %34 = ascendc.tbuf.get_tensor %24 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %35 = arith.muli %dim, %c2 : index
        ascendc.pipe.init_buffer %0, %23, %35 : !ascendc.tbuf<veccalc>, index
        %36 = ascendc.tbuf.get_tensor %23 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %37 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %37, %subview_3 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1]>>
        ascendc.data_copy_l2 %36, %37, %dim : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %38 = arith.index_cast %28 : index to i32
        %39 = arith.index_cast %dim : index to i32
        ascendc.pipe.init_buffer %0, %22, %30 : !ascendc.tbuf<veccalc>, index
        %40 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %40, %36, %38, %39, %c1_i32, %39 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.mul_l2 %34, %33, %40, %29 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %6, %34 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %41 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %42 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %42, %subview_6 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %42, %41, %29 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %6, %41 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %5, %33 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
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

