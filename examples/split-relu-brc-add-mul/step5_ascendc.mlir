#map = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_split(%arg0: memref<?x?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: i64, %arg6: i64) -> memref<?x?xf16> {
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
    %5 = arith.index_cast %arg6 : i64 to index
    %6 = arith.index_cast %arg5 : i64 to index
    %dim = memref.dim %arg0, %c1 : memref<?x?xf16>
    %dim_0 = memref.dim %arg1, %c0 : memref<?xf16>
    %subview = memref.subview %arg0[0, 0] [%dim_0, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1]>>
    %subview_1 = memref.subview %arg0[%dim_0, 0] [%dim_0, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    %7 = affine.apply #map()[%dim_0]
    %alloc = memref.alloc(%7, %dim) {alignment = 64 : i64} : memref<?x?xf16>
    %subview_2 = memref.subview %alloc[%dim_0, 0] [%dim_0, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    %subview_3 = memref.subview %alloc[0, 0] [%dim_0, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1]>>
    %8 = ascendc.tbuf : <veccalc>
    %9 = ascendc.tbuf : <veccalc>
    %10 = ascendc.queue : <vecin, 1>
    %11 = ascendc.tbuf : <vecin>
    %12 = ascendc.tbuf : <veccalc>
    %13 = ascendc.queue : <vecin, 1>
    %14 = ascendc.tbuf : <vecin>
    %15 = ascendc.tbuf : <veccalc>
    %16 = ascendc.tbuf : <vecout>
    %17 = ascendc.tbuf : <vecin>
    scf.for %arg7 = %c0 to %dim_0 step %6 {
      %28 = affine.min #map1(%arg7)[%dim_0, %6]
      %subview_4 = memref.subview %subview[%arg7, 0] [%28, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1]>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_5 = memref.subview %arg1[%arg7] [%28] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_6 = memref.subview %arg3[0] [%dim] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_7 = memref.subview %subview_3[%arg7, 0] [%28, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1]>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg8 = %c0 to %28 step %5 {
        %29 = affine.min #map1(%arg8)[%28, %5]
        %subview_8 = memref.subview %subview_4[%arg8, 0] [%29, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %30 = arith.muli %29, %dim : index
        %31 = arith.muli %30, %c2 : index
        ascendc.pipe.init_buffer %0, %17, %31 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %1, %c1_i32, %31 : !ascendc.queue<vecin, 1>, i32, index
        %32 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %33 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %33, %subview_8 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %32, %33, %30 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %1, %32 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %34 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_9 = memref.subview %subview_5[%arg8] [%29] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_10 = memref.subview %subview_7[%arg8, 0] [%29, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %16, %31 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %2, %c1_i32, %31 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %0, %15, %31 : !ascendc.tbuf<veccalc>, index
        %35 = ascendc.tbuf.get_tensor %15 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %36 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %36, %subview_9 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        %37 = arith.muli %29, %c2 : index
        ascendc.pipe.init_buffer %0, %14, %37 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %13, %c1_i32, %37 : !ascendc.queue<vecin, 1>, i32, index
        %38 = ascendc.que_bind.alloc_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %38, %36, %29 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %13, %38 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %39 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %40 = arith.index_cast %29 : index to i32
        %41 = arith.index_cast %dim : index to i32
        ascendc.pipe.init_buffer %0, %12, %31 : !ascendc.tbuf<veccalc>, index
        %42 = ascendc.tbuf.get_tensor %12 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %42, %39, %40, %41, %40, %c1_i32 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %43 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %43, %subview_6 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1]>>
        %44 = arith.muli %dim, %c2 : index
        ascendc.pipe.init_buffer %0, %11, %44 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %10, %c1_i32, %44 : !ascendc.queue<vecin, 1>, i32, index
        %45 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %45, %43, %dim : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %10, %45 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %46 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %0, %9, %31 : !ascendc.tbuf<veccalc>, index
        %47 = ascendc.tbuf.get_tensor %9 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %47, %46, %40, %41, %c1_i32, %41 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %0, %8, %31 : !ascendc.tbuf<veccalc>, index
        %48 = ascendc.tbuf.get_tensor %8 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %48, %cst, %30 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %35, %34, %48, %30 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %35, %35, %42, %30 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.mul_l2 %35, %35, %47, %30 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %2, %35 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %49 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %50 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %50, %subview_10 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %50, %49, %30 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %49 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %34 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %18 = ascendc.tbuf : <veccalc>
    %19 = ascendc.tbuf : <veccalc>
    %20 = ascendc.queue : <vecin, 1>
    %21 = ascendc.tbuf : <vecin>
    %22 = ascendc.tbuf : <veccalc>
    %23 = ascendc.queue : <vecin, 1>
    %24 = ascendc.tbuf : <vecin>
    %25 = ascendc.tbuf : <veccalc>
    %26 = ascendc.tbuf : <vecout>
    %27 = ascendc.tbuf : <vecin>
    scf.for %arg7 = %c0 to %dim_0 step %6 {
      %28 = affine.min #map1(%arg7)[%dim_0, %6]
      %subview_4 = memref.subview %subview_1[%arg7, 0] [%28, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_5 = memref.subview %arg2[%arg7] [%28] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_6 = memref.subview %arg4[0] [%dim] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_7 = memref.subview %subview_2[%arg7, 0] [%28, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg8 = %c0 to %28 step %5 {
        %29 = affine.min #map1(%arg8)[%28, %5]
        %subview_8 = memref.subview %subview_4[%arg8, 0] [%29, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %30 = arith.muli %29, %dim : index
        %31 = arith.muli %30, %c2 : index
        ascendc.pipe.init_buffer %0, %27, %31 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %3, %c1_i32, %31 : !ascendc.queue<vecin, 1>, i32, index
        %32 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %33 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %33, %subview_8 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %32, %33, %30 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %3, %32 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %34 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_9 = memref.subview %subview_5[%arg8] [%29] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_10 = memref.subview %subview_7[%arg8, 0] [%29, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %26, %31 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %4, %c1_i32, %31 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %0, %25, %31 : !ascendc.tbuf<veccalc>, index
        %35 = ascendc.tbuf.get_tensor %25 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %36 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %36, %subview_9 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        %37 = arith.muli %29, %c2 : index
        ascendc.pipe.init_buffer %0, %24, %37 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %23, %c1_i32, %37 : !ascendc.queue<vecin, 1>, i32, index
        %38 = ascendc.que_bind.alloc_tensor %23 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %38, %36, %29 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %23, %38 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %39 = ascendc.que_bind.deque_tensor %23 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %40 = arith.index_cast %29 : index to i32
        %41 = arith.index_cast %dim : index to i32
        ascendc.pipe.init_buffer %0, %22, %31 : !ascendc.tbuf<veccalc>, index
        %42 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %42, %39, %40, %41, %40, %c1_i32 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %43 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %43, %subview_6 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1]>>
        %44 = arith.muli %dim, %c2 : index
        ascendc.pipe.init_buffer %0, %21, %44 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %20, %c1_i32, %44 : !ascendc.queue<vecin, 1>, i32, index
        %45 = ascendc.que_bind.alloc_tensor %20 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %45, %43, %dim : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %20, %45 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %46 = ascendc.que_bind.deque_tensor %20 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %0, %19, %31 : !ascendc.tbuf<veccalc>, index
        %47 = ascendc.tbuf.get_tensor %19 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %47, %46, %40, %41, %c1_i32, %41 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %0, %18, %31 : !ascendc.tbuf<veccalc>, index
        %48 = ascendc.tbuf.get_tensor %18 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %48, %cst, %30 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %35, %34, %48, %30 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %35, %35, %42, %30 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.mul_l2 %35, %35, %47, %30 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %4, %35 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %49 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %50 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %50, %subview_10 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %50, %49, %30 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %4, %49 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %3, %34 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
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

