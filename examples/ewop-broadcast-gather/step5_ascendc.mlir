#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
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
    scf.for %arg5 = %c0 to %dim step %6 {
      %17 = affine.min #map(%arg5)[%dim, %6]
      %subview = memref.subview %arg1[0] [%dim_0] [1] : memref<?xi32> to memref<?xi32, strided<[1]>>
      %subview_2 = memref.subview %arg0[%arg5, 0] [%17, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_3 = memref.subview %alloc_1[%arg5, 0] [%17, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg6 = %c0 to %17 step %5 {
        %18 = affine.min #map(%arg6)[%17, %5]
        %19 = arith.muli %dim_0, %c4 : index
        ascendc.pipe.init_buffer %0, %11, %19 : !ascendc.tbuf<vecin>, index
        %20 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi32>
        %21 = ascendc.global_tensor : !ascendc.global_tensor<*xi32>
        ascendc.global_tensor.set_global_buffer %21, %subview : !ascendc.global_tensor<*xi32>, memref<?xi32, strided<[1]>>
        ascendc.data_copy_l2 %20, %21, %dim_0 : !ascendc.local_tensor<*xi32>, !ascendc.global_tensor<*xi32>, index
        ascendc.que_bind.enque_tensor %1, %20 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi32>
        %22 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi32>
        %subview_4 = memref.subview %subview_2[%arg6, 0] [%18, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_5 = memref.subview %subview_3[%arg6, 0] [%18, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %23 = arith.muli %18, %dim_0 : index
        %24 = arith.muli %23, %c2 : index
        ascendc.pipe.init_buffer %0, %10, %24 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %0, %9, %24 : !ascendc.tbuf<veccalc>, index
        %25 = ascendc.tbuf.get_tensor %9 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %26 = arith.index_cast %18 : index to i32
        %27 = arith.index_cast %dim_0 : index to i32
        ascendc.pipe.init_buffer %0, %8, %24 : !ascendc.tbuf<veccalc>, index
        %28 = ascendc.tbuf.get_tensor %8 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %28, %22, %26, %27, %c1_i32, %27 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xi32>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %0, %7, %24 : !ascendc.tbuf<veccalc>, index
        %29 = ascendc.tbuf.get_tensor %7 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %30 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %30, %subview_4 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %29, %30, %23 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %2, %25 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %31 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %32 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %32, %subview_5 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %32, %31, %23 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %31 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %22 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi32>
      }
    }
    %12 = ascendc.tbuf : <veccalc>
    %13 = ascendc.tbuf : <veccalc>
    %14 = ascendc.tbuf : <veccalc>
    %15 = ascendc.tbuf : <vecout>
    %16 = ascendc.tbuf : <vecin>
    scf.for %arg5 = %c0 to %dim step %6 {
      %17 = affine.min #map(%arg5)[%dim, %6]
      %subview = memref.subview %alloc_1[%arg5, 0] [%17, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_2 = memref.subview %arg2[%arg5] [%17] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_3 = memref.subview %alloc[%arg5, 0] [%17, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg6 = %c0 to %17 step %5 {
        %18 = affine.min #map(%arg6)[%17, %5]
        %subview_4 = memref.subview %subview[%arg6, 0] [%18, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %19 = arith.muli %18, %dim_0 : index
        %20 = arith.muli %19, %c2 : index
        ascendc.pipe.init_buffer %0, %16, %20 : !ascendc.tbuf<vecin>, index
        %21 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %22 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %22, %subview_4 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %21, %22, %19 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %3, %21 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %23 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_5 = memref.subview %subview_2[%arg6] [%18] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_6 = memref.subview %subview_3[%arg6, 0] [%18, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %15, %20 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %0, %14, %20 : !ascendc.tbuf<veccalc>, index
        %24 = ascendc.tbuf.get_tensor %14 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %25 = arith.muli %18, %c2 : index
        ascendc.pipe.init_buffer %0, %13, %25 : !ascendc.tbuf<veccalc>, index
        %26 = ascendc.tbuf.get_tensor %13 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %27 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %27, %subview_5 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %26, %27, %18 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %28 = arith.index_cast %18 : index to i32
        %29 = arith.index_cast %dim_0 : index to i32
        ascendc.pipe.init_buffer %0, %12, %20 : !ascendc.tbuf<veccalc>, index
        %30 = ascendc.tbuf.get_tensor %12 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %30, %26, %28, %29, %28, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.add_l2 %24, %23, %30, %19 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %4, %24 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %31 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %32 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %32, %subview_6 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %32, %31, %19 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %4, %31 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %3, %23 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
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

