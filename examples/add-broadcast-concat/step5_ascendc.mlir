#map = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_concat(%arg0: memref<?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?xf16>, %arg3: memref<?x?xf16>, %arg4: i64, %arg5: i64) -> memref<?x?xf16> {
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = ascendc.pipe
    %1 = ascendc.queue : <vecin, 1>
    %2 = ascendc.queue : <vecout, 1>
    %3 = ascendc.queue : <vecin, 1>
    %4 = ascendc.queue : <vecout, 1>
    %5 = arith.index_cast %arg5 : i64 to index
    %6 = arith.index_cast %arg4 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?xf16>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf16>
    %7 = affine.apply #map()[%dim]
    %alloc = memref.alloc(%7, %dim_0) {alignment = 64 : i64} : memref<?x?xf16>
    %subview = memref.subview %alloc[%dim, 0] [%dim, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    %subview_1 = memref.subview %alloc[0, 0] [%dim, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1]>>
    %8 = ascendc.queue : <vecin, 1>
    %9 = ascendc.tbuf : <vecin>
    %10 = ascendc.tbuf : <veccalc>
    %11 = ascendc.tbuf : <veccalc>
    %12 = ascendc.tbuf : <vecout>
    %13 = ascendc.tbuf : <vecin>
    scf.for %arg6 = %c0 to %dim step %6 {
      %20 = affine.min #map1(%arg6)[%dim, %6]
      %subview_4 = memref.subview %arg0[%arg6] [%20] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_5 = memref.subview %arg1[%arg6, 0] [%20, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %subview_1[%arg6, 0] [%20, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1]>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg7 = %c0 to %20 step %5 {
        %21 = affine.min #map1(%arg7)[%20, %5]
        %subview_7 = memref.subview %subview_4[%arg7] [%21] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %22 = arith.muli %21, %c2 : index
        ascendc.pipe.init_buffer %0, %13, %22 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %1, %c1_i32, %22 : !ascendc.queue<vecin, 1>, i32, index
        %23 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %24 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %24, %subview_7 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %23, %24, %21 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %1, %23 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %25 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_8 = memref.subview %subview_5[%arg7, 0] [%21, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_9 = memref.subview %subview_6[%arg7, 0] [%21, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %26 = arith.muli %21, %dim_0 : index
        %27 = arith.muli %26, %c2 : index
        ascendc.pipe.init_buffer %0, %12, %27 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %2, %c1_i32, %27 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %0, %11, %27 : !ascendc.tbuf<veccalc>, index
        %28 = ascendc.tbuf.get_tensor %11 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %29 = arith.index_cast %21 : index to i32
        %30 = arith.index_cast %dim_0 : index to i32
        ascendc.pipe.init_buffer %0, %10, %27 : !ascendc.tbuf<veccalc>, index
        %31 = ascendc.tbuf.get_tensor %10 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %31, %25, %29, %30, %29, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %32 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %32, %subview_8 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %9, %27 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %8, %c1_i32, %27 : !ascendc.queue<vecin, 1>, i32, index
        %33 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %33, %32, %26 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %8, %33 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %34 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.add_l2 %28, %31, %34, %26 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %2, %28 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %35 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %36 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %36, %subview_9 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %36, %35, %26 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %35 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %25 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %dim_2 = memref.dim %arg2, %c0 : memref<?xf16>
    %dim_3 = memref.dim %arg3, %c1 : memref<?x?xf16>
    %14 = ascendc.queue : <vecin, 1>
    %15 = ascendc.tbuf : <vecin>
    %16 = ascendc.tbuf : <veccalc>
    %17 = ascendc.tbuf : <veccalc>
    %18 = ascendc.tbuf : <vecout>
    %19 = ascendc.tbuf : <vecin>
    scf.for %arg6 = %c0 to %dim_2 step %6 {
      %20 = affine.min #map1(%arg6)[%dim_2, %6]
      %subview_4 = memref.subview %arg2[%arg6] [%20] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_5 = memref.subview %arg3[%arg6, 0] [%20, %dim_3] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %subview[%arg6, 0] [%20, %dim_3] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg7 = %c0 to %20 step %5 {
        %21 = affine.min #map1(%arg7)[%20, %5]
        %subview_7 = memref.subview %subview_4[%arg7] [%21] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %22 = arith.muli %21, %c2 : index
        ascendc.pipe.init_buffer %0, %19, %22 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %3, %c1_i32, %22 : !ascendc.queue<vecin, 1>, i32, index
        %23 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %24 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %24, %subview_7 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %23, %24, %21 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %3, %23 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %25 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_8 = memref.subview %subview_5[%arg7, 0] [%21, %dim_3] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_9 = memref.subview %subview_6[%arg7, 0] [%21, %dim_3] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %26 = arith.muli %21, %dim_3 : index
        %27 = arith.muli %26, %c2 : index
        ascendc.pipe.init_buffer %0, %18, %27 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %4, %c1_i32, %27 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %0, %17, %27 : !ascendc.tbuf<veccalc>, index
        %28 = ascendc.tbuf.get_tensor %17 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %29 = arith.index_cast %21 : index to i32
        %30 = arith.index_cast %dim_3 : index to i32
        ascendc.pipe.init_buffer %0, %16, %27 : !ascendc.tbuf<veccalc>, index
        %31 = ascendc.tbuf.get_tensor %16 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %31, %25, %29, %30, %29, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %32 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %32, %subview_8 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %15, %27 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %14, %c1_i32, %27 : !ascendc.queue<vecin, 1>, i32, index
        %33 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %33, %32, %26 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %14, %33 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %34 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.mul_l2 %28, %31, %34, %26 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %4, %28 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %35 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %36 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %36, %subview_9 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %36, %35, %26 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %4, %35 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %3, %25 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return %alloc : memref<?x?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.tensor.decompose_concat
    } : !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.param.constant true -> !transform.any_param
    %3 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %4 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %5 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.foreach %1 : !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      %tiled_linalg_op, %loops = transform.structured.tile_using_for %arg1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.annotate %loops "ascendc.parallel" = %2 : !transform.any_op, !transform.any_param
      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.annotate %loops_1 "ascendc.prologue" = %3 : !transform.any_op, !transform.any_param
      transform.annotate %loops_1 "ascendc.epilogue" = %4 : !transform.any_op, !transform.any_param
      transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %5 : !transform.any_op, !transform.any_param
      transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    }
    transform.yield 
  }
}

