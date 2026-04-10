#map = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<()[s0, s1, s2] -> (s1, s0 - s2)>
#map2 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
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
    %14 = ascendc.get_block_idx : index
    %15 = arith.muli %14, %6 : index
    %16 = arith.cmpi ult, %15, %dim : index
    scf.if %16 {
      %24 = affine.min #map1()[%dim, %6, %15]
      %subview_4 = memref.subview %arg0[%15] [%24] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_5 = memref.subview %arg1[%15, 0] [%24, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %subview_1[%15, 0] [%24, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1]>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg6 = %c0 to %24 step %5 {
        %25 = affine.min #map2(%arg6)[%24, %5]
        %subview_7 = memref.subview %subview_4[%arg6] [%25] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %26 = arith.muli %25, %c2 : index
        ascendc.pipe.init_buffer %0, %13, %26 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %1, %c1_i32, %26 : !ascendc.queue<vecin, 1>, i32, index
        %27 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %28 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %28, %subview_7 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %27, %28, %25 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %1, %27 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %29 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_8 = memref.subview %subview_5[%arg6, 0] [%25, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_9 = memref.subview %subview_6[%arg6, 0] [%25, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %30 = arith.muli %25, %dim_0 : index
        %31 = arith.muli %30, %c2 : index
        ascendc.pipe.init_buffer %0, %12, %31 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %2, %c1_i32, %31 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %0, %11, %31 : !ascendc.tbuf<veccalc>, index
        %32 = ascendc.tbuf.get_tensor %11 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %33 = arith.index_cast %25 : index to i32
        %34 = arith.index_cast %dim_0 : index to i32
        ascendc.pipe.init_buffer %0, %10, %31 : !ascendc.tbuf<veccalc>, index
        %35 = ascendc.tbuf.get_tensor %10 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %35, %29, %33, %34, %33, %c1_i32 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %36 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %36, %subview_8 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %9, %31 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %8, %c1_i32, %31 : !ascendc.queue<vecin, 1>, i32, index
        %37 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %37, %36, %30 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %8, %37 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %38 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.add_l2 %32, %35, %38, %30 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %2, %32 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %39 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %40 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %40, %subview_9 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %40, %39, %30 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %39 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %29 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %dim_2 = memref.dim %arg2, %c0 : memref<?xf16>
    %dim_3 = memref.dim %arg3, %c1 : memref<?x?xf16>
    %17 = ascendc.queue : <vecin, 1>
    %18 = ascendc.tbuf : <vecin>
    %19 = ascendc.tbuf : <veccalc>
    %20 = ascendc.tbuf : <veccalc>
    %21 = ascendc.tbuf : <vecout>
    %22 = ascendc.tbuf : <vecin>
    %23 = arith.cmpi ult, %15, %dim_2 : index
    scf.if %23 {
      %24 = affine.min #map1()[%dim_2, %6, %15]
      %subview_4 = memref.subview %arg2[%15] [%24] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_5 = memref.subview %arg3[%15, 0] [%24, %dim_3] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %subview[%15, 0] [%24, %dim_3] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg6 = %c0 to %24 step %5 {
        %25 = affine.min #map2(%arg6)[%24, %5]
        %subview_7 = memref.subview %subview_4[%arg6] [%25] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %26 = arith.muli %25, %c2 : index
        ascendc.pipe.init_buffer %0, %22, %26 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %3, %c1_i32, %26 : !ascendc.queue<vecin, 1>, i32, index
        %27 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %28 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %28, %subview_7 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %27, %28, %25 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %3, %27 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %29 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_8 = memref.subview %subview_5[%arg6, 0] [%25, %dim_3] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_9 = memref.subview %subview_6[%arg6, 0] [%25, %dim_3] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %30 = arith.muli %25, %dim_3 : index
        %31 = arith.muli %30, %c2 : index
        ascendc.pipe.init_buffer %0, %21, %31 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %4, %c1_i32, %31 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %0, %20, %31 : !ascendc.tbuf<veccalc>, index
        %32 = ascendc.tbuf.get_tensor %20 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %33 = arith.index_cast %25 : index to i32
        %34 = arith.index_cast %dim_3 : index to i32
        ascendc.pipe.init_buffer %0, %19, %31 : !ascendc.tbuf<veccalc>, index
        %35 = ascendc.tbuf.get_tensor %19 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %35, %29, %33, %34, %33, %c1_i32 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %36 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %36, %subview_8 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %18, %31 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %17, %c1_i32, %31 : !ascendc.queue<vecin, 1>, i32, index
        %37 = ascendc.que_bind.alloc_tensor %17 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %37, %36, %30 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %17, %37 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %38 = ascendc.que_bind.deque_tensor %17 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.mul_l2 %32, %35, %38, %30 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %4, %32 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %39 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %40 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %40, %subview_9 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %40, %39, %30 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %4, %39 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %3, %29 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
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

