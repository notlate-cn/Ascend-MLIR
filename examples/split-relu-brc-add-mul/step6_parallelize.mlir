#map = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<()[s0, s1, s2] -> (s1, s0 - s2)>
#map2 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
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
    %18 = ascendc.get_block_idx : index
    %19 = arith.muli %18, %6 : index
    %20 = arith.cmpi ult, %19, %dim_0 : index
    scf.if %20 {
      %31 = affine.min #map1()[%dim_0, %6, %19]
      %subview_4 = memref.subview %subview[%19, 0] [%31, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1]>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_5 = memref.subview %arg1[%19] [%31] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_6 = memref.subview %arg3[0] [%dim] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_7 = memref.subview %subview_3[%19, 0] [%31, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1]>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg7 = %c0 to %31 step %5 {
        %32 = affine.min #map2(%arg7)[%31, %5]
        %subview_8 = memref.subview %subview_4[%arg7, 0] [%32, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %33 = arith.muli %32, %dim : index
        %34 = arith.muli %33, %c2 : index
        ascendc.pipe.init_buffer %0, %17, %34 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %1, %c1_i32, %34 : !ascendc.queue<vecin, 1>, i32, index
        %35 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %36 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %36, %subview_8 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %35, %36, %33 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %1, %35 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %37 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_9 = memref.subview %subview_5[%arg7] [%32] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_10 = memref.subview %subview_7[%arg7, 0] [%32, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %16, %34 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %2, %c1_i32, %34 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %0, %15, %34 : !ascendc.tbuf<veccalc>, index
        %38 = ascendc.tbuf.get_tensor %15 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %39 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %39, %subview_9 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        %40 = arith.muli %32, %c2 : index
        ascendc.pipe.init_buffer %0, %14, %40 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %13, %c1_i32, %40 : !ascendc.queue<vecin, 1>, i32, index
        %41 = ascendc.que_bind.alloc_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %41, %39, %32 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %13, %41 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %42 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %43 = arith.index_cast %32 : index to i32
        %44 = arith.index_cast %dim : index to i32
        ascendc.pipe.init_buffer %0, %12, %34 : !ascendc.tbuf<veccalc>, index
        %45 = ascendc.tbuf.get_tensor %12 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %45, %42, %43, %44, %43, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %46 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %46, %subview_6 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1]>>
        %47 = arith.muli %dim, %c2 : index
        ascendc.pipe.init_buffer %0, %11, %47 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %10, %c1_i32, %47 : !ascendc.queue<vecin, 1>, i32, index
        %48 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %48, %46, %dim : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %10, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %49 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %0, %9, %34 : !ascendc.tbuf<veccalc>, index
        %50 = ascendc.tbuf.get_tensor %9 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %50, %49, %43, %44, %c1_i32, %44 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %0, %8, %34 : !ascendc.tbuf<veccalc>, index
        %51 = ascendc.tbuf.get_tensor %8 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %51, %cst, %33 : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %38, %37, %51, %33 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %38, %38, %45, %33 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.mul_l2 %38, %38, %50, %33 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %2, %38 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %52 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %53 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %53, %subview_10 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %53, %52, %33 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %52 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %37 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %21 = ascendc.tbuf : <veccalc>
    %22 = ascendc.tbuf : <veccalc>
    %23 = ascendc.queue : <vecin, 1>
    %24 = ascendc.tbuf : <vecin>
    %25 = ascendc.tbuf : <veccalc>
    %26 = ascendc.queue : <vecin, 1>
    %27 = ascendc.tbuf : <vecin>
    %28 = ascendc.tbuf : <veccalc>
    %29 = ascendc.tbuf : <vecout>
    %30 = ascendc.tbuf : <vecin>
    scf.if %20 {
      %31 = affine.min #map1()[%dim_0, %6, %19]
      %subview_4 = memref.subview %subview_1[%19, 0] [%31, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_5 = memref.subview %arg2[%19] [%31] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_6 = memref.subview %arg4[0] [%dim] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_7 = memref.subview %subview_2[%19, 0] [%31, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg7 = %c0 to %31 step %5 {
        %32 = affine.min #map2(%arg7)[%31, %5]
        %subview_8 = memref.subview %subview_4[%arg7, 0] [%32, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %33 = arith.muli %32, %dim : index
        %34 = arith.muli %33, %c2 : index
        ascendc.pipe.init_buffer %0, %30, %34 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %3, %c1_i32, %34 : !ascendc.queue<vecin, 1>, i32, index
        %35 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %36 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %36, %subview_8 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %35, %36, %33 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %3, %35 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %37 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_9 = memref.subview %subview_5[%arg7] [%32] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_10 = memref.subview %subview_7[%arg7, 0] [%32, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %29, %34 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %4, %c1_i32, %34 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %0, %28, %34 : !ascendc.tbuf<veccalc>, index
        %38 = ascendc.tbuf.get_tensor %28 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %39 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %39, %subview_9 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        %40 = arith.muli %32, %c2 : index
        ascendc.pipe.init_buffer %0, %27, %40 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %26, %c1_i32, %40 : !ascendc.queue<vecin, 1>, i32, index
        %41 = ascendc.que_bind.alloc_tensor %26 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %41, %39, %32 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %26, %41 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %42 = ascendc.que_bind.deque_tensor %26 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %43 = arith.index_cast %32 : index to i32
        %44 = arith.index_cast %dim : index to i32
        ascendc.pipe.init_buffer %0, %25, %34 : !ascendc.tbuf<veccalc>, index
        %45 = ascendc.tbuf.get_tensor %25 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %45, %42, %43, %44, %43, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %46 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %46, %subview_6 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1]>>
        %47 = arith.muli %dim, %c2 : index
        ascendc.pipe.init_buffer %0, %24, %47 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %23, %c1_i32, %47 : !ascendc.queue<vecin, 1>, i32, index
        %48 = ascendc.que_bind.alloc_tensor %23 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %48, %46, %dim : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %23, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %49 = ascendc.que_bind.deque_tensor %23 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %0, %22, %34 : !ascendc.tbuf<veccalc>, index
        %50 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %50, %49, %43, %44, %c1_i32, %44 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %0, %21, %34 : !ascendc.tbuf<veccalc>, index
        %51 = ascendc.tbuf.get_tensor %21 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %51, %cst, %33 : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %38, %37, %51, %33 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %38, %38, %45, %33 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.mul_l2 %38, %38, %50, %33 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %4, %38 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %52 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %53 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %53, %subview_10 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %53, %52, %33 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %4, %52 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %3, %37 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
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

