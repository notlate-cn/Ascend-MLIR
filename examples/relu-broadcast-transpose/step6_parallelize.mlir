#map = affine_map<()[s0, s1, s2] -> (s1, s0 - s2)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module attributes {transform.with_named_sequence} {
  func.func @relu_transpose_broadcast_add(%arg0: memref<?x1xf16>, %arg1: memref<?x?xf16>, %arg2: i64, %arg3: i64) -> memref<?x?xf16> {
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = ascendc.pipe
    %1 = ascendc.queue : <vecin, 1>
    %2 = ascendc.queue : <vecout, 1>
    %3 = arith.index_cast %arg3 : i64 to index
    %4 = arith.index_cast %arg2 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?x1xf16>
    %dim_0 = memref.dim %arg1, %c0 : memref<?x?xf16>
    %alloc = memref.alloc(%dim_0, %dim) {alignment = 64 : i64} : memref<?x?xf16>
    %5 = ascendc.tbuf : <veccalc>
    %6 = ascendc.queue : <vecin, 1>
    %7 = ascendc.tbuf : <vecin>
    %8 = ascendc.tbuf : <veccalc>
    %9 = ascendc.tbuf : <veccalc>
    %10 = ascendc.tbuf : <vecout>
    %11 = ascendc.tbuf : <vecin>
    %12 = ascendc.get_block_idx : index
    %13 = arith.muli %12, %4 : index
    %14 = arith.cmpi ult, %13, %dim_0 : index
    scf.if %14 {
      %15 = affine.min #map()[%dim_0, %4, %13]
      %subview = memref.subview %arg0[0, 0] [%dim, 1] [1, 1] : memref<?x1xf16> to memref<?x1xf16, strided<[1, 1]>>
      %subview_1 = memref.subview %arg1[%13, 0] [%15, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_2 = memref.subview %alloc[%13, 0] [%15, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg4 = %c0 to %15 step %3 {
        %16 = affine.min #map1(%arg4)[%15, %3]
        %17 = arith.muli %dim, %c2 : index
        ascendc.pipe.init_buffer %0, %11, %17 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %1, %c1_i32, %17 : !ascendc.queue<vecin, 1>, i32, index
        %18 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %19 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %19, %subview : !ascendc.global_tensor<*xf16>, memref<?x1xf16, strided<[1, 1]>>
        ascendc.data_copy_l2 %18, %19, %dim : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %1, %18 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %20 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_3 = memref.subview %subview_1[%arg4, 0] [%16, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_4 = memref.subview %subview_2[%arg4, 0] [%16, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %21 = arith.muli %16, %dim : index
        %22 = arith.muli %21, %c2 : index
        ascendc.pipe.init_buffer %0, %10, %22 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %2, %c1_i32, %22 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %0, %9, %22 : !ascendc.tbuf<veccalc>, index
        %23 = ascendc.tbuf.get_tensor %9 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %24 = arith.index_cast %16 : index to i32
        %25 = arith.index_cast %dim : index to i32
        ascendc.pipe.init_buffer %0, %8, %22 : !ascendc.tbuf<veccalc>, index
        %26 = ascendc.tbuf.get_tensor %8 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %26, %20, %24, %25, %c1_i32, %25 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %27 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %27, %subview_3 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %7, %22 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %6, %c1_i32, %22 : !ascendc.queue<vecin, 1>, i32, index
        %28 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %28, %27, %21 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %6, %28 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %29 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %0, %5, %22 : !ascendc.tbuf<veccalc>, index
        %30 = ascendc.tbuf.get_tensor %5 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %30, %cst, %21 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %23, %26, %30, %21 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %23, %23, %29, %21 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %2, %23 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %31 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %32 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %32, %subview_4 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %32, %31, %21 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %31 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %20 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return %alloc : memref<?x?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.param.constant true -> !transform.any_param
    %3 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %4 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %5 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops "ascendc.parallel" = %2 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_1 "ascendc.prologue" = %3 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %4 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %5 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    transform.yield 
  }
}

