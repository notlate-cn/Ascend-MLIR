#map = affine_map<()[s0, s1, s2] -> (s1, s0 - s2)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_gather(%arg0: memref<?x?xf16>, %arg1: memref<?xi32>, %arg2: memref<?xf16>, %arg3: i64, %arg4: i64) -> memref<?x?xf16> {
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = ascendc.pipe
    %1 = ascendc.queue : <vecin, 1>
    %2 = ascendc.queue : <vecout, 1>
    %3 = ascendc.queue : <vecin, 1>
    %4 = ascendc.queue : <vecout, 1>
    %5 = arith.index_cast %arg4 : i64 to index
    %6 = arith.index_cast %arg3 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf16>
    %dim_0 = memref.dim %arg1, %c0 : memref<?xi32>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf16>
    %alloc_1 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf16>
    %7 = ascendc.tbuf : <veccalc>
    %8 = ascendc.tbuf : <veccalc>
    %9 = ascendc.tbuf : <veccalc>
    %10 = ascendc.tbuf : <vecout>
    %11 = ascendc.tbuf : <vecin>
    %12 = ascendc.get_block_idx : index
    %13 = arith.muli %12, %6 : index
    %14 = arith.cmpi ult, %13, %dim : index
    scf.if %14 {
      %20 = affine.min #map()[%dim, %6, %13]
      %subview = memref.subview %arg1[0] [%dim_0] [1] : memref<?xi32> to memref<?xi32, strided<[1]>>
      %subview_2 = memref.subview %arg0[%13, 0] [%20, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_3 = memref.subview %alloc_1[%13, 0] [%20, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg5 = %c0 to %20 step %5 {
        %21 = affine.min #map1(%arg5)[%20, %5]
        %22 = arith.muli %dim_0, %c4 : index
        ascendc.pipe.init_buffer %0, %11, %22 : !ascendc.tbuf<vecin>, index
        %23 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi32>
        %24 = ascendc.global_tensor : !ascendc.global_tensor<*xi32>
        ascendc.global_tensor.set_global_buffer %24, %subview : !ascendc.global_tensor<*xi32>, memref<?xi32, strided<[1]>>
        ascendc.data_copy_l2 %23, %24, %dim_0 : !ascendc.local_tensor<*xi32>, !ascendc.global_tensor<*xi32>, index
        ascendc.que_bind.enque_tensor %1, %23 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi32>
        %25 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi32>
        %subview_4 = memref.subview %subview_2[%arg5, 0] [%21, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_5 = memref.subview %subview_3[%arg5, 0] [%21, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %26 = arith.muli %21, %dim_0 : index
        %27 = arith.muli %26, %c2 : index
        ascendc.pipe.init_buffer %0, %10, %27 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %0, %9, %27 : !ascendc.tbuf<veccalc>, index
        %28 = ascendc.tbuf.get_tensor %9 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %29 = arith.index_cast %21 : index to i32
        %30 = arith.index_cast %dim_0 : index to i32
        ascendc.pipe.init_buffer %0, %8, %27 : !ascendc.tbuf<veccalc>, index
        %31 = ascendc.tbuf.get_tensor %8 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %31, %25, %29, %30, %c1_i32, %30 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xi32>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %0, %7, %27 : !ascendc.tbuf<veccalc>, index
        %32 = ascendc.tbuf.get_tensor %7 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %33 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %33, %subview_4 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %32, %33, %26 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %2, %28 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %34 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %35 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %35, %subview_5 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %35, %34, %26 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %34 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %25 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi32>
      }
    }
    %15 = ascendc.tbuf : <veccalc>
    %16 = ascendc.tbuf : <veccalc>
    %17 = ascendc.tbuf : <veccalc>
    %18 = ascendc.tbuf : <vecout>
    %19 = ascendc.tbuf : <vecin>
    scf.if %14 {
      %20 = affine.min #map()[%dim, %6, %13]
      %subview = memref.subview %alloc_1[%13, 0] [%20, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_2 = memref.subview %arg2[%13] [%20] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_3 = memref.subview %alloc[%13, 0] [%20, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg5 = %c0 to %20 step %5 {
        %21 = affine.min #map1(%arg5)[%20, %5]
        %subview_4 = memref.subview %subview[%arg5, 0] [%21, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %22 = arith.muli %21, %dim_0 : index
        %23 = arith.muli %22, %c2 : index
        ascendc.pipe.init_buffer %0, %19, %23 : !ascendc.tbuf<vecin>, index
        %24 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %25 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %25, %subview_4 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %24, %25, %22 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %3, %24 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %26 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_5 = memref.subview %subview_2[%arg5] [%21] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_6 = memref.subview %subview_3[%arg5, 0] [%21, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %18, %23 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %0, %17, %23 : !ascendc.tbuf<veccalc>, index
        %27 = ascendc.tbuf.get_tensor %17 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %28 = arith.muli %21, %c2 : index
        ascendc.pipe.init_buffer %0, %16, %28 : !ascendc.tbuf<veccalc>, index
        %29 = ascendc.tbuf.get_tensor %16 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %30 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %30, %subview_5 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %29, %30, %21 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %31 = arith.index_cast %21 : index to i32
        %32 = arith.index_cast %dim_0 : index to i32
        ascendc.pipe.init_buffer %0, %15, %23 : !ascendc.tbuf<veccalc>, index
        %33 = ascendc.tbuf.get_tensor %15 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %33, %29, %31, %32, %31, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.add_l2 %27, %26, %33, %22 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %4, %27 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %34 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %35 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %35, %subview_6 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %35, %34, %22 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %4, %34 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %3, %26 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return %alloc : memref<?x?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "gather_by_index"} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "broadcast_add_gathered"} in %transformed : (!transform.any_op) -> !transform.any_op
    %3 = transform.param.constant true -> !transform.any_param
    %4 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %5 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %6 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops "ascendc.parallel" = %3 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_1 "ascendc.prologue" = %4 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %6 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %2 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_3 "ascendc.parallel" = %3 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_5 "ascendc.prologue" = %4 : !transform.any_op, !transform.any_param
    transform.annotate %loops_5 "ascendc.epilogue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_4 "ascendc.unit" = %6 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_5 : !transform.any_op
    transform.yield 
  }
}

