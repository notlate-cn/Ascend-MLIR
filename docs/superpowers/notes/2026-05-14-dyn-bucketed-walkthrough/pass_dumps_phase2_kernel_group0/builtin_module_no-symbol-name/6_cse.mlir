// -----// IR Dump After CSE (cse) //----- //
module attributes {vector_plan.tiling_infos = [{block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", fields = [{abi_index = 0 : i32, arg_index = 5 : i32, axis_size = -1 : i64, default_value = 128 : i64, kind = "tunable", name = "XBLOCK"}, {abi_index = 1 : i32, arg_index = 6 : i32, axis_size = -1 : i64, default_value = 16 : i64, kind = "tunable", name = "XBLOCK_SUB"}], kernel_id = "kernel_group0__v0"}]} {
  func.func private @kernel_group0__v0(%arg0: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg1: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg2: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg3: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg4: memref<?x?xf32> {afir.symbolic_shape = "s0,s1"}, %arg5: index {vector_plan.default_tile_size = 128 : i64}, %arg6: index {vector_plan.default_tile_size = 16 : i64}) -> memref<?x?xf32> attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}]} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = ascendc.pipe
    %1 = ascendc.queue : <vecin, 1>
    %2 = ascendc.queue : <vecin, 1>
    %3 = ascendc.queue : <vecin, 1>
    %4 = ascendc.queue : <vecin, 1>
    %5 = ascendc.queue : <vecout, 1>
    %6 = ascendc.queue : <vecin, 1>
    %7 = ascendc.queue : <vecin, 1>
    %8 = ascendc.queue : <vecin, 1>
    %9 = ascendc.queue : <vecin, 1>
    %10 = ascendc.queue : <vecout, 1>
    %dim = memref.dim %arg4, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg4, %c1 : memref<?x?xf32>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
    %collapse_shape_1 = memref.collapse_shape %arg1 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
    %collapse_shape_2 = memref.collapse_shape %arg2 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
    %collapse_shape_3 = memref.collapse_shape %arg3 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
    %collapse_shape_4 = memref.collapse_shape %alloc [[0, 1]] : memref<?x?xf32> into memref<?xf32>
    %dim_5 = memref.dim %arg0, %c0 : memref<?x?x?xf32>
    %dim_6 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
    %11 = arith.muli %dim_5, %dim_6 : index
    %dim_7 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
    %12 = arith.muli %arg6, %dim_7 : index
    %13 = arith.muli %12, %c4 : index
    %dim_8 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
    %14 = arith.muli %arg6, %dim_8 : index
    %15 = arith.muli %14, %c4 : index
    %dim_9 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
    %16 = arith.muli %arg6, %dim_9 : index
    %17 = arith.muli %16, %c4 : index
    %dim_10 = memref.dim %collapse_shape_3, %c1 : memref<?x?xf32>
    %18 = arith.muli %arg6, %dim_10 : index
    %19 = arith.muli %18, %c4 : index
    %20 = arith.muli %arg6, %c4 : index
    %21 = ascendc.tbuf : <veccalc>
    %22 = ascendc.tbuf : <veccalc>
    %23 = ascendc.tbuf : <veccalc>
    %24 = ascendc.tbuf : <veccalc>
    %25 = ascendc.tbuf : <vecout>
    %26 = ascendc.tbuf : <vecin>
    %27 = ascendc.tbuf : <vecin>
    %28 = ascendc.tbuf : <vecin>
    %29 = ascendc.tbuf : <vecin>
    %30 = ascendc.tbuf : <veccalc>
    %31 = ascendc.tbuf : <veccalc>
    %32 = ascendc.tbuf : <veccalc>
    %33 = ascendc.tbuf : <veccalc>
    %34 = ascendc.tbuf : <vecout>
    ascendc.pipe.init_buffer %0, %34, %20 : !ascendc.tbuf<vecout>, index
    %35 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %0, %35, %19 : !ascendc.tbuf<vecin>, index
    %36 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %0, %36, %17 : !ascendc.tbuf<vecin>, index
    %37 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %0, %37, %15 : !ascendc.tbuf<vecin>, index
    ascendc.pipe.init_queue %0, %1, %c1_i32, %13 : !ascendc.queue<vecin, 1>, i32, index
    %38 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %0, %38, %13 : !ascendc.tbuf<vecin>, index
    ascendc.pipe.init_queue %0, %5, %c1_i32, %20 : !ascendc.queue<vecout, 1>, i32, index
    ascendc.pipe.init_queue %0, %4, %c1_i32, %19 : !ascendc.queue<vecin, 1>, i32, index
    ascendc.pipe.init_queue %0, %3, %c1_i32, %17 : !ascendc.queue<vecin, 1>, i32, index
    ascendc.pipe.init_queue %0, %2, %c1_i32, %15 : !ascendc.queue<vecin, 1>, i32, index
    %39 = ascendc.get_block_idx : index
    %40 = arith.muli %39, %arg5 : index
    %41 = arith.cmpi ult, %40, %11 : index
    scf.if %41 {
      %42 = arith.subi %11, %40 : index
      %43 = arith.minsi %arg5, %42 : index
      %44 = arith.divsi %43, %arg6 : index
      %45 = arith.muli %44, %arg6 : index
      ascendc.pipe.init_buffer %0, %30, %13 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %0, %31, %13 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %0, %32, %13 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %0, %33, %13 : !ascendc.tbuf<veccalc>, index
      scf.for %arg7 = %c0 to %45 step %arg6 {
        %47 = arith.addi %40, %arg7 : index
        %subview = memref.subview %collapse_shape[%47, 0] [%arg6, %dim_7] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_11 = memref.subview %collapse_shape_1[%47, 0] [%arg6, %dim_8] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_12 = memref.subview %collapse_shape_2[%47, 0] [%arg6, %dim_9] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_13 = memref.subview %collapse_shape_3[%47, 0] [%arg6, %dim_10] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_14 = memref.subview %collapse_shape_4[%47] [%arg6] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
        %48 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %49 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %49, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %48, %49, %12 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %1, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %50 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %51 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %52 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %52, %subview_11 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %51, %52, %14 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %2, %51 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %53 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %54 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %55 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %55, %subview_12 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %54, %55, %16 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %3, %54 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %56 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %57 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %58 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %58, %subview_13 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %57, %58, %18 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %4, %57 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %59 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %60 = ascendc.tbuf.get_tensor %33 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.duplicate_l2 %60, %cst, %12 : !ascendc.local_tensor<*xf32>, f32, index
        %61 = ascendc.tbuf.get_tensor %32 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %61, %50, %53, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %62 = ascendc.tbuf.get_tensor %31 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.mul_l2 %62, %61, %56, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %63 = ascendc.tbuf.get_tensor %30 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %63, %62, %59, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.add_l2 %60, %60, %63, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %64 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.reduce_sum_2d_l2 %64, %60 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.enque_tensor %5, %64 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %65 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %66 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %66, %subview_14 : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %66, %65, %arg6 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %5, %65 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %1, %50 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %2, %53 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %3, %56 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %4, %59 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
      %46 = arith.cmpi slt, %45, %43 : index
      scf.if %46 {
        %47 = arith.subi %11, %arg6 : index
        %subview = memref.subview %collapse_shape[%47, 0] [%arg6, %dim_7] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_11 = memref.subview %collapse_shape_1[%47, 0] [%arg6, %dim_8] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_12 = memref.subview %collapse_shape_2[%47, 0] [%arg6, %dim_9] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_13 = memref.subview %collapse_shape_3[%47, 0] [%arg6, %dim_10] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_14 = memref.subview %collapse_shape_4[%47] [%arg6] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
        ascendc.pipe.init_buffer %0, %29, %13 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %6, %c1_i32, %13 : !ascendc.queue<vecin, 1>, i32, index
        %48 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %49 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %49, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %48, %49, %12 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %6, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %50 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %0, %28, %15 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %7, %c1_i32, %15 : !ascendc.queue<vecin, 1>, i32, index
        %51 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %52 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %52, %subview_11 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %51, %52, %14 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %7, %51 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %53 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %0, %27, %17 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %8, %c1_i32, %17 : !ascendc.queue<vecin, 1>, i32, index
        %54 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %55 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %55, %subview_12 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %54, %55, %16 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %8, %54 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %56 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %0, %26, %19 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %9, %c1_i32, %19 : !ascendc.queue<vecin, 1>, i32, index
        %57 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %58 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %58, %subview_13 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %57, %58, %18 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %9, %57 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %59 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %0, %25, %20 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %10, %c1_i32, %20 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %0, %24, %13 : !ascendc.tbuf<veccalc>, index
        %60 = ascendc.tbuf.get_tensor %24 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.duplicate_l2 %60, %cst, %12 : !ascendc.local_tensor<*xf32>, f32, index
        ascendc.pipe.init_buffer %0, %23, %13 : !ascendc.tbuf<veccalc>, index
        %61 = ascendc.tbuf.get_tensor %23 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %61, %50, %53, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.pipe.init_buffer %0, %22, %13 : !ascendc.tbuf<veccalc>, index
        %62 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.mul_l2 %62, %61, %56, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.pipe.init_buffer %0, %21, %13 : !ascendc.tbuf<veccalc>, index
        %63 = ascendc.tbuf.get_tensor %21 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %63, %62, %59, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.add_l2 %60, %60, %63, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %64 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.reduce_sum_2d_l2 %64, %60 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.enque_tensor %10, %64 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %65 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %66 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %66, %subview_14 : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %66, %65, %arg6 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %10, %65 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %6, %50 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %7, %53 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %8, %56 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %9, %59 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    %expand_shape = memref.expand_shape %collapse_shape_4 [[0, 1]] output_shape [%dim, %dim_0] : memref<?xf32> into memref<?x?xf32>
    return %expand_shape : memref<?x?xf32>
  }
}


