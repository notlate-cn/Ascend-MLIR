// -----// IR Dump After CSE (cse) //----- //
module attributes {vector_plan.tiling_infos = [{block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", fields = [{abi_index = 0 : i32, arg_index = 4 : i32, axis_size = -1 : i64, default_value = 128 : i64, kind = "tunable", name = "XBLOCK"}, {abi_index = 1 : i32, arg_index = 5 : i32, axis_size = -1 : i64, default_value = 16 : i64, kind = "tunable", name = "XBLOCK_SUB"}], kernel_id = "kernel_group1__v0"}]} {
  func.func private @kernel_group1__v0(%arg0: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg1: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg2: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg3: memref<?x?xf32> {afir.symbolic_shape = "s0,s1"}, %arg4: index {vector_plan.default_tile_size = 128 : i64}, %arg5: index {vector_plan.default_tile_size = 16 : i64}) -> memref<?x?xf32> attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}]} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = ascendc.pipe
    %1 = ascendc.queue : <vecin, 1>
    %2 = ascendc.queue : <vecin, 1>
    %3 = ascendc.queue : <vecin, 1>
    %4 = ascendc.queue : <vecout, 1>
    %5 = ascendc.queue : <vecin, 1>
    %6 = ascendc.queue : <vecin, 1>
    %7 = ascendc.queue : <vecin, 1>
    %8 = ascendc.queue : <vecout, 1>
    %dim = memref.dim %arg3, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf32>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
    %collapse_shape_1 = memref.collapse_shape %arg1 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
    %collapse_shape_2 = memref.collapse_shape %arg2 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
    %collapse_shape_3 = memref.collapse_shape %alloc [[0, 1]] : memref<?x?xf32> into memref<?xf32>
    %dim_4 = memref.dim %arg0, %c0 : memref<?x?x?xf32>
    %dim_5 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
    %9 = arith.muli %dim_4, %dim_5 : index
    %dim_6 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
    %10 = arith.muli %arg5, %dim_6 : index
    %11 = arith.muli %10, %c4 : index
    %dim_7 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
    %12 = arith.muli %arg5, %dim_7 : index
    %13 = arith.muli %12, %c4 : index
    %dim_8 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
    %14 = arith.muli %arg5, %dim_8 : index
    %15 = arith.muli %14, %c4 : index
    %16 = arith.muli %arg5, %c4 : index
    %17 = ascendc.tbuf : <veccalc>
    %18 = ascendc.tbuf : <veccalc>
    %19 = ascendc.tbuf : <veccalc>
    %20 = ascendc.tbuf : <vecout>
    %21 = ascendc.tbuf : <vecin>
    %22 = ascendc.tbuf : <vecin>
    %23 = ascendc.tbuf : <vecin>
    %24 = ascendc.tbuf : <veccalc>
    %25 = ascendc.tbuf : <veccalc>
    %26 = ascendc.tbuf : <veccalc>
    %27 = ascendc.tbuf : <vecout>
    ascendc.pipe.init_buffer %0, %27, %16 : !ascendc.tbuf<vecout>, index
    %28 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %0, %28, %15 : !ascendc.tbuf<vecin>, index
    %29 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %0, %29, %13 : !ascendc.tbuf<vecin>, index
    ascendc.pipe.init_queue %0, %1, %c1_i32, %11 : !ascendc.queue<vecin, 1>, i32, index
    %30 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %0, %30, %11 : !ascendc.tbuf<vecin>, index
    ascendc.pipe.init_queue %0, %4, %c1_i32, %16 : !ascendc.queue<vecout, 1>, i32, index
    ascendc.pipe.init_queue %0, %3, %c1_i32, %15 : !ascendc.queue<vecin, 1>, i32, index
    ascendc.pipe.init_queue %0, %2, %c1_i32, %13 : !ascendc.queue<vecin, 1>, i32, index
    %31 = ascendc.get_block_idx : index
    %32 = arith.muli %31, %arg4 : index
    %33 = arith.cmpi ult, %32, %9 : index
    scf.if %33 {
      %34 = arith.subi %9, %32 : index
      %35 = arith.minsi %arg4, %34 : index
      %36 = arith.divsi %35, %arg5 : index
      %37 = arith.muli %36, %arg5 : index
      ascendc.pipe.init_buffer %0, %24, %11 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %0, %25, %11 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %0, %26, %11 : !ascendc.tbuf<veccalc>, index
      scf.for %arg6 = %c0 to %37 step %arg5 {
        %39 = arith.addi %32, %arg6 : index
        %subview = memref.subview %collapse_shape[%39, 0] [%arg5, %dim_6] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_9 = memref.subview %collapse_shape_1[%39, 0] [%arg5, %dim_7] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_10 = memref.subview %collapse_shape_2[%39, 0] [%arg5, %dim_8] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_11 = memref.subview %collapse_shape_3[%39] [%arg5] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
        %40 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %41 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %41, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %40, %41, %10 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %1, %40 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %42 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %43 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %44 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %44, %subview_9 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %43, %44, %12 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %2, %43 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %45 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %46 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %47 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %47, %subview_10 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %46, %47, %14 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %3, %46 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %48 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %49 = ascendc.tbuf.get_tensor %26 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.duplicate_l2 %49, %cst, %10 : !ascendc.local_tensor<*xf32>, f32, index
        %50 = ascendc.tbuf.get_tensor %25 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %50, %42, %45, %10 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %51 = ascendc.tbuf.get_tensor %24 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.mul_l2 %51, %50, %48, %10 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.add_l2 %49, %49, %51, %10 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %52 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.reduce_sum_2d_l2 %52, %49 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.enque_tensor %4, %52 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %53 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %54, %subview_11 : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %54, %53, %arg5 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %4, %53 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %1, %42 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %2, %45 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %3, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
      %38 = arith.cmpi slt, %37, %35 : index
      scf.if %38 {
        %39 = arith.subi %9, %arg5 : index
        %subview = memref.subview %collapse_shape[%39, 0] [%arg5, %dim_6] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_9 = memref.subview %collapse_shape_1[%39, 0] [%arg5, %dim_7] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_10 = memref.subview %collapse_shape_2[%39, 0] [%arg5, %dim_8] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_11 = memref.subview %collapse_shape_3[%39] [%arg5] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
        ascendc.pipe.init_buffer %0, %23, %11 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %5, %c1_i32, %11 : !ascendc.queue<vecin, 1>, i32, index
        %40 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %41 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %41, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %40, %41, %10 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %5, %40 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %42 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %0, %22, %13 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %6, %c1_i32, %13 : !ascendc.queue<vecin, 1>, i32, index
        %43 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %44 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %44, %subview_9 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %43, %44, %12 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %6, %43 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %45 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %0, %21, %15 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %7, %c1_i32, %15 : !ascendc.queue<vecin, 1>, i32, index
        %46 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %47 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %47, %subview_10 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %46, %47, %14 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %7, %46 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %48 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %0, %20, %16 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %8, %c1_i32, %16 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %0, %19, %11 : !ascendc.tbuf<veccalc>, index
        %49 = ascendc.tbuf.get_tensor %19 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.duplicate_l2 %49, %cst, %10 : !ascendc.local_tensor<*xf32>, f32, index
        ascendc.pipe.init_buffer %0, %18, %11 : !ascendc.tbuf<veccalc>, index
        %50 = ascendc.tbuf.get_tensor %18 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %50, %42, %45, %10 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.pipe.init_buffer %0, %17, %11 : !ascendc.tbuf<veccalc>, index
        %51 = ascendc.tbuf.get_tensor %17 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.mul_l2 %51, %50, %48, %10 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.add_l2 %49, %49, %51, %10 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %52 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.reduce_sum_2d_l2 %52, %49 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.enque_tensor %8, %52 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %53 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %54, %subview_11 : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %54, %53, %arg5 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %8, %53 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %5, %42 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %6, %45 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %7, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    %expand_shape = memref.expand_shape %collapse_shape_3 [[0, 1]] output_shape [%dim, %dim_0] : memref<?xf32> into memref<?x?xf32>
    return %expand_shape : memref<?x?xf32>
  }
}


