#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_split(%arg0: memref<?x?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: i64, %arg5: i64) -> (memref<?x?xf16>, memref<?x?xf16>) {
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
    %7 = arith.index_cast %arg5 : i64 to index
    %8 = arith.index_cast %arg4 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf16>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf16>
    %dim_1 = memref.dim %arg2, %c0 : memref<?xf16>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf16>
    %9 = ascendc.tbuf : <veccalc>
    %10 = ascendc.tbuf : <veccalc>
    %11 = ascendc.tbuf : <veccalc>
    %12 = ascendc.tbuf : <veccalc>
    %13 = ascendc.tbuf : <vecout>
    %14 = ascendc.tbuf : <vecin>
    scf.for %arg6 = %c0 to %dim step %8 {
      %25 = affine.min #map(%arg6)[%dim, %8]
      %subview_5 = memref.subview %arg0[%arg6, 0] [%25, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %arg1[%arg6] [%25] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_7 = memref.subview %alloc[%arg6, 0] [%25, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg7 = %c0 to %25 step %7 {
        %26 = affine.min #map(%arg7)[%25, %7]
        %subview_8 = memref.subview %subview_5[%arg7, 0] [%26, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %27 = arith.muli %26, %dim_0 : index
        %28 = arith.muli %27, %c2 : index
        ascendc.pipe.init_buffer %0, %14, %28 : !ascendc.tbuf<vecin>, index
        %29 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %30 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %30, %subview_8 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %29, %30, %27 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %1, %29 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %31 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_9 = memref.subview %subview_6[%arg7] [%26] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_10 = memref.subview %subview_7[%arg7, 0] [%26, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %13, %28 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %0, %12, %28 : !ascendc.tbuf<veccalc>, index
        %32 = ascendc.tbuf.get_tensor %12 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %33 = arith.muli %26, %c2 : index
        ascendc.pipe.init_buffer %0, %11, %33 : !ascendc.tbuf<veccalc>, index
        %34 = ascendc.tbuf.get_tensor %11 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %35 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %35, %subview_9 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %34, %35, %26 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %36 = arith.index_cast %26 : index to i32
        %37 = arith.index_cast %dim_0 : index to i32
        ascendc.pipe.init_buffer %0, %10, %28 : !ascendc.tbuf<veccalc>, index
        %38 = ascendc.tbuf.get_tensor %10 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %38, %34, %36, %37, %36, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %0, %9, %28 : !ascendc.tbuf<veccalc>, index
        %39 = ascendc.tbuf.get_tensor %9 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %39, %cst, %27 : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %32, %31, %39, %27 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %32, %32, %38, %27 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %2, %32 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %40 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %41 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %41, %subview_10 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %41, %40, %27 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %40 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %31 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %subview = memref.subview %alloc[0, 0] [%dim, %dim_1] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1]>>
    %alloc_2 = memref.alloc(%dim, %dim_1) {alignment = 64 : i64} : memref<?x?xf16>
    %alloc_3 = memref.alloc(%dim, %dim_1) {alignment = 64 : i64} : memref<?x?xf16>
    %15 = ascendc.tbuf : <veccalc>
    %16 = ascendc.tbuf : <veccalc>
    %17 = ascendc.tbuf : <veccalc>
    %18 = ascendc.tbuf : <vecout>
    %19 = ascendc.tbuf : <vecin>
    scf.for %arg6 = %c0 to %dim step %8 {
      %25 = affine.min #map(%arg6)[%dim, %8]
      %subview_5 = memref.subview %subview[%arg6, 0] [%25, %dim_1] [1, 1] : memref<?x?xf16, strided<[?, 1]>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %arg2[0] [%dim_1] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_7 = memref.subview %alloc_3[%arg6, 0] [%25, %dim_1] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg7 = %c0 to %25 step %7 {
        %26 = affine.min #map(%arg7)[%25, %7]
        %subview_8 = memref.subview %subview_5[%arg7, 0] [%26, %dim_1] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %27 = arith.muli %26, %dim_1 : index
        %28 = arith.muli %27, %c2 : index
        ascendc.pipe.init_buffer %0, %19, %28 : !ascendc.tbuf<vecin>, index
        %29 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %30 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %30, %subview_8 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %29, %30, %27 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %3, %29 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %31 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_9 = memref.subview %subview_7[%arg7, 0] [%26, %dim_1] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %18, %28 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %0, %17, %28 : !ascendc.tbuf<veccalc>, index
        %32 = ascendc.tbuf.get_tensor %17 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %33 = arith.muli %dim_1, %c2 : index
        ascendc.pipe.init_buffer %0, %16, %33 : !ascendc.tbuf<veccalc>, index
        %34 = ascendc.tbuf.get_tensor %16 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %35 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %35, %subview_6 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1]>>
        ascendc.data_copy_l2 %34, %35, %dim_1 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %36 = arith.index_cast %26 : index to i32
        %37 = arith.index_cast %dim_1 : index to i32
        ascendc.pipe.init_buffer %0, %15, %28 : !ascendc.tbuf<veccalc>, index
        %38 = ascendc.tbuf.get_tensor %15 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %38, %34, %36, %37, %c1_i32, %37 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.mul_l2 %32, %31, %38, %27 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %4, %32 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %39 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %40 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %40, %subview_9 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %40, %39, %27 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %4, %39 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %3, %31 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %subview_4 = memref.subview %alloc[0, %dim_1] [%dim, %dim_1] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    %20 = ascendc.tbuf : <veccalc>
    %21 = ascendc.tbuf : <veccalc>
    %22 = ascendc.tbuf : <veccalc>
    %23 = ascendc.tbuf : <vecout>
    %24 = ascendc.tbuf : <vecin>
    scf.for %arg6 = %c0 to %dim step %8 {
      %25 = affine.min #map(%arg6)[%dim, %8]
      %subview_5 = memref.subview %subview_4[%arg6, 0] [%25, %dim_1] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %arg3[0] [%dim_1] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_7 = memref.subview %alloc_2[%arg6, 0] [%25, %dim_1] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg7 = %c0 to %25 step %7 {
        %26 = affine.min #map(%arg7)[%25, %7]
        %subview_8 = memref.subview %subview_5[%arg7, 0] [%26, %dim_1] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %27 = arith.muli %26, %dim_1 : index
        %28 = arith.muli %27, %c2 : index
        ascendc.pipe.init_buffer %0, %24, %28 : !ascendc.tbuf<vecin>, index
        %29 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %30 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %30, %subview_8 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %29, %30, %27 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %5, %29 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %31 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_9 = memref.subview %subview_7[%arg7, 0] [%26, %dim_1] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %23, %28 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %0, %22, %28 : !ascendc.tbuf<veccalc>, index
        %32 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %33 = arith.muli %dim_1, %c2 : index
        ascendc.pipe.init_buffer %0, %21, %33 : !ascendc.tbuf<veccalc>, index
        %34 = ascendc.tbuf.get_tensor %21 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %35 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %35, %subview_6 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1]>>
        ascendc.data_copy_l2 %34, %35, %dim_1 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %36 = arith.index_cast %26 : index to i32
        %37 = arith.index_cast %dim_1 : index to i32
        ascendc.pipe.init_buffer %0, %20, %28 : !ascendc.tbuf<veccalc>, index
        %38 = ascendc.tbuf.get_tensor %20 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %38, %34, %36, %37, %c1_i32, %37 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.mul_l2 %32, %31, %38, %27 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %6, %32 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %39 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %40 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %40, %subview_9 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %40, %39, %27 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %6, %39 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %5, %31 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return %alloc_3, %alloc_2 : memref<?x?xf16>, memref<?x?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "relu_broadcast_add"} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "split_scale0"} in %transformed : (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "split_scale1"} in %transformed : (!transform.any_op) -> !transform.any_op
    %4 = transform.param.constant true -> !transform.any_param
    %5 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %6 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %7 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %2 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_1 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %3 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_3 "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_5 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_5 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_4 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_5 : !transform.any_op
    %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_7 "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_9 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_9 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_8 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_9 : !transform.any_op
    transform.yield 
  }
}

