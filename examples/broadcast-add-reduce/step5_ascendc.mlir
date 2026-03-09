#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module attributes {transform.with_named_sequence} {
  func.func @broadcast_add_reducesum(%arg0: memref<?xf16>, %arg1: memref<?x?xf16>, %arg2: i64, %arg3: i64) -> memref<?xf16> {
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = ascendc.pipe
    %1 = ascendc.queue : <vecin, 1>
    %2 = ascendc.queue : <vecout, 1>
    %3 = arith.index_cast %arg3 : i64 to index
    %4 = arith.index_cast %arg2 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?xf16>
    %alloc = memref.alloc(%dim) {alignment = 64 : i64} : memref<?xf16>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf16>
    %5 = ascendc.tbuf : <veccalc>
    %6 = ascendc.tbuf : <veccalc>
    %7 = ascendc.tbuf : <veccalc>
    %8 = ascendc.tbuf : <vecout>
    %9 = ascendc.tbuf : <vecin>
    scf.for %arg4 = %c0 to %dim step %4 {
      %10 = affine.min #map(%arg4)[%dim, %4]
      %subview = memref.subview %arg0[%arg4] [%10] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_1 = memref.subview %arg1[%arg4, 0] [%10, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_2 = memref.subview %alloc[%arg4] [%10] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      scf.for %arg5 = %c0 to %10 step %3 {
        %11 = affine.min #map(%arg5)[%10, %3]
        %subview_3 = memref.subview %subview[%arg5] [%11] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %12 = arith.muli %11, %c2 : index
        ascendc.pipe.init_buffer %0, %9, %12 : !ascendc.tbuf<vecin>, index
        %13 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %14 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %14, %subview_3 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %13, %14, %11 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %1, %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %15 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_4 = memref.subview %subview_1[%arg5, 0] [%11, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_5 = memref.subview %subview_2[%arg5] [%11] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        ascendc.pipe.init_buffer %0, %8, %12 : !ascendc.tbuf<vecout>, index
        %16 = arith.muli %11, %dim_0 : index
        %17 = arith.muli %16, %c2 : index
        ascendc.pipe.init_buffer %0, %7, %17 : !ascendc.tbuf<veccalc>, index
        %18 = ascendc.tbuf.get_tensor %7 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %19 = arith.index_cast %11 : index to i32
        %20 = arith.index_cast %dim_0 : index to i32
        ascendc.pipe.init_buffer %0, %6, %17 : !ascendc.tbuf<veccalc>, index
        %21 = ascendc.tbuf.get_tensor %6 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %21, %15, %19, %20, %19, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %0, %5, %17 : !ascendc.tbuf<veccalc>, index
        %22 = ascendc.tbuf.get_tensor %5 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %23 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %23, %subview_4 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %22, %23, %16 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.add_l2 %18, %21, %22, %16 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %18, %18, %18, %16 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        %24 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.reduce_sum_2d_l2 %24, %18 {layout = 0 : i32} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.enque_tensor %2, %24 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %25 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %26 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %26, %subview_5 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %26, %25, %11 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %25 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return %alloc : memref<?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} in %transformed : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %2 = transform.param.constant true -> !transform.any_param
    transform.annotate %loops "ascendc.parallel" = %2 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %3 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %4 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    transform.annotate %loops_1 "ascendc.prologue" = %3 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %4 : !transform.any_op, !transform.any_param
    %5 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %5 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    transform.yield 
  }
}

