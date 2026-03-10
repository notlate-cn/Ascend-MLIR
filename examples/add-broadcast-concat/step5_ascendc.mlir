#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_concat(%arg0: memref<?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?xf16>, %arg3: memref<?x?xf16>, %arg4: i64, %arg5: i64) -> memref<?x?xf16> {
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = ascendc.pipe
    %1 = ascendc.queue : <vecin, 1>
    %2 = ascendc.queue : <vecout, 1>
    %3 = ascendc.queue : <vecin, 1>
    %4 = ascendc.queue : <vecout, 1>
    %5 = arith.index_cast %arg5 : i64 to index
    %6 = arith.index_cast %arg4 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?xf16>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf16>
    %7 = arith.muli %dim, %c2 : index
    %alloc = memref.alloc(%7, %dim_0) {alignment = 64 : i64} : memref<?x?xf16>
    %subview = memref.subview %alloc[0, 0] [%dim, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1]>>
    %8 = ascendc.tbuf : <veccalc>
    %9 = ascendc.tbuf : <veccalc>
    %10 = ascendc.tbuf : <veccalc>
    %11 = ascendc.tbuf : <vecout>
    %12 = ascendc.tbuf : <vecin>
    scf.for %arg6 = %c0 to %dim step %6 {
      %18 = affine.min #map(%arg6)[%dim, %6]
      %subview_4 = memref.subview %arg0[%arg6] [%18] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_5 = memref.subview %arg1[%arg6, 0] [%18, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %subview[%arg6, 0] [%18, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1]>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg7 = %c0 to %18 step %5 {
        %19 = affine.min #map(%arg7)[%18, %5]
        %subview_7 = memref.subview %subview_4[%arg7] [%19] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %20 = arith.muli %19, %c2 : index
        ascendc.pipe.init_buffer %0, %12, %20 : !ascendc.tbuf<vecin>, index
        %21 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %22 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %22, %subview_7 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %21, %22, %19 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %1, %21 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %23 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_8 = memref.subview %subview_5[%arg7, 0] [%19, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_9 = memref.subview %subview_6[%arg7, 0] [%19, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %24 = arith.muli %19, %dim_0 : index
        %25 = arith.muli %24, %c2 : index
        ascendc.pipe.init_buffer %0, %11, %25 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %0, %10, %25 : !ascendc.tbuf<veccalc>, index
        %26 = ascendc.tbuf.get_tensor %10 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %27 = arith.index_cast %19 : index to i32
        %28 = arith.index_cast %dim_0 : index to i32
        ascendc.pipe.init_buffer %0, %9, %25 : !ascendc.tbuf<veccalc>, index
        %29 = ascendc.tbuf.get_tensor %9 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %29, %23, %27, %28, %27, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %0, %8, %25 : !ascendc.tbuf<veccalc>, index
        %30 = ascendc.tbuf.get_tensor %8 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %31 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %31, %subview_8 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %30, %31, %24 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.add_l2 %26, %29, %30, %24 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %2, %26 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %32 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %33 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %33, %subview_9 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %33, %32, %24 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %32 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %23 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %subview_1 = memref.subview %alloc[%dim, 0] [%dim, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    %dim_2 = memref.dim %arg2, %c0 : memref<?xf16>
    %dim_3 = memref.dim %arg3, %c1 : memref<?x?xf16>
    %13 = ascendc.tbuf : <veccalc>
    %14 = ascendc.tbuf : <veccalc>
    %15 = ascendc.tbuf : <veccalc>
    %16 = ascendc.tbuf : <vecout>
    %17 = ascendc.tbuf : <vecin>
    scf.for %arg6 = %c0 to %dim_2 step %6 {
      %18 = affine.min #map(%arg6)[%dim_2, %6]
      %subview_4 = memref.subview %arg2[%arg6] [%18] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_5 = memref.subview %arg3[%arg6, 0] [%18, %dim_3] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %subview_1[%arg6, 0] [%18, %dim_3] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg7 = %c0 to %18 step %5 {
        %19 = affine.min #map(%arg7)[%18, %5]
        %subview_7 = memref.subview %subview_4[%arg7] [%19] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %20 = arith.muli %19, %c2 : index
        ascendc.pipe.init_buffer %0, %17, %20 : !ascendc.tbuf<vecin>, index
        %21 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %22 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %22, %subview_7 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %21, %22, %19 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %3, %21 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %23 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_8 = memref.subview %subview_5[%arg7, 0] [%19, %dim_3] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_9 = memref.subview %subview_6[%arg7, 0] [%19, %dim_3] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %24 = arith.muli %19, %dim_3 : index
        %25 = arith.muli %24, %c2 : index
        ascendc.pipe.init_buffer %0, %16, %25 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %0, %15, %25 : !ascendc.tbuf<veccalc>, index
        %26 = ascendc.tbuf.get_tensor %15 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %27 = arith.index_cast %19 : index to i32
        %28 = arith.index_cast %dim_3 : index to i32
        ascendc.pipe.init_buffer %0, %14, %25 : !ascendc.tbuf<veccalc>, index
        %29 = ascendc.tbuf.get_tensor %14 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %29, %23, %27, %28, %27, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %0, %13, %25 : !ascendc.tbuf<veccalc>, index
        %30 = ascendc.tbuf.get_tensor %13 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %31 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %31, %subview_8 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %30, %31, %24 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.mul_l2 %26, %29, %30, %24 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %4, %26 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %32 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %33 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %33, %subview_9 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %33, %32, %24 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %4, %32 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %3, %23 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return %alloc : memref<?x?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "broadcast_add"} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "broadcast_mul"} in %transformed : (!transform.any_op) -> !transform.any_op
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

