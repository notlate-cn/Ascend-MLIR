// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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
    %dim_11 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
    %dim_12 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
    %dim_13 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
    %dim_14 = memref.dim %collapse_shape_3, %c1 : memref<?x?xf32>
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
      %46 = arith.muli %arg6, %dim_7 : index
      %47 = arith.muli %arg6, %dim_8 : index
      %48 = arith.muli %arg6, %dim_9 : index
      %49 = arith.muli %arg6, %dim_10 : index
      %50 = arith.muli %arg6, %dim_7 : index
      %51 = arith.muli %arg6, %dim_7 : index
      %52 = arith.muli %51, %c4 : index
      %53 = arith.muli %arg6, %dim_7 : index
      %54 = arith.muli %53, %c4 : index
      %55 = arith.muli %arg6, %dim_7 : index
      %56 = arith.muli %55, %c4 : index
      %57 = arith.muli %arg6, %dim_7 : index
      %58 = arith.muli %57, %c4 : index
      ascendc.pipe.init_buffer %0, %30, %58 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %0, %31, %56 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %0, %32, %54 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %0, %33, %52 : !ascendc.tbuf<veccalc>, index
      scf.for %arg7 = %c0 to %45 step %arg6 {
        %60 = arith.addi %40, %arg7 : index
        %subview = memref.subview %collapse_shape[%60, 0] [%arg6, %dim_11] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_15 = memref.subview %collapse_shape_1[%60, 0] [%arg6, %dim_12] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_16 = memref.subview %collapse_shape_2[%60, 0] [%arg6, %dim_13] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_17 = memref.subview %collapse_shape_3[%60, 0] [%arg6, %dim_14] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_18 = memref.subview %collapse_shape_4[%60] [%arg6] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
        %61 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %62 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %62, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %61, %62, %46 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %1, %61 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %63 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %64 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %65 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %65, %subview_15 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %64, %65, %47 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %2, %64 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %66 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %67 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %68 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %68, %subview_16 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %67, %68, %48 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %3, %67 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %69 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %70 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %71 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %71, %subview_17 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %70, %71, %49 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %4, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %72 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %73 = ascendc.tbuf.get_tensor %33 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.duplicate_l2 %73, %cst, %50 : !ascendc.local_tensor<*xf32>, f32, index
        %74 = ascendc.tbuf.get_tensor %32 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %74, %63, %66, %50 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %75 = ascendc.tbuf.get_tensor %31 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.mul_l2 %75, %74, %69, %50 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %76 = ascendc.tbuf.get_tensor %30 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %76, %75, %72, %50 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.add_l2 %73, %73, %76, %50 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %77 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.reduce_sum_2d_l2 %77, %73 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.enque_tensor %5, %77 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %78 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %79 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %79, %subview_18 : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %79, %78, %arg6 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %5, %78 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %1, %63 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %2, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %3, %69 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %4, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
      %59 = arith.cmpi slt, %45, %43 : index
      scf.if %59 {
        %60 = arith.subi %11, %arg6 : index
        %dim_15 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
        %subview = memref.subview %collapse_shape[%60, 0] [%arg6, %dim_15] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %dim_16 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
        %subview_17 = memref.subview %collapse_shape_1[%60, 0] [%arg6, %dim_16] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %dim_18 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
        %subview_19 = memref.subview %collapse_shape_2[%60, 0] [%arg6, %dim_18] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %dim_20 = memref.dim %collapse_shape_3, %c1 : memref<?x?xf32>
        %subview_21 = memref.subview %collapse_shape_3[%60, 0] [%arg6, %dim_20] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_22 = memref.subview %collapse_shape_4[%60] [%arg6] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
        %61 = arith.muli %arg6, %dim_15 : index
        %62 = arith.muli %61, %c4 : index
        ascendc.pipe.init_buffer %0, %29, %62 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %6, %c1_i32, %62 : !ascendc.queue<vecin, 1>, i32, index
        %63 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %64 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %64, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        %65 = arith.muli %arg6, %dim_15 : index
        ascendc.data_copy_l2 %63, %64, %65 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %6, %63 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %66 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %67 = arith.muli %arg6, %dim_16 : index
        %68 = arith.muli %67, %c4 : index
        ascendc.pipe.init_buffer %0, %28, %68 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %7, %c1_i32, %68 : !ascendc.queue<vecin, 1>, i32, index
        %69 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %70 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %70, %subview_17 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        %71 = arith.muli %arg6, %dim_16 : index
        ascendc.data_copy_l2 %69, %70, %71 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %7, %69 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %72 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %73 = arith.muli %arg6, %dim_18 : index
        %74 = arith.muli %73, %c4 : index
        ascendc.pipe.init_buffer %0, %27, %74 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %8, %c1_i32, %74 : !ascendc.queue<vecin, 1>, i32, index
        %75 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %76 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %76, %subview_19 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        %77 = arith.muli %arg6, %dim_18 : index
        ascendc.data_copy_l2 %75, %76, %77 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %8, %75 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %78 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %79 = arith.muli %arg6, %dim_20 : index
        %80 = arith.muli %79, %c4 : index
        ascendc.pipe.init_buffer %0, %26, %80 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %9, %c1_i32, %80 : !ascendc.queue<vecin, 1>, i32, index
        %81 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %82 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %82, %subview_21 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        %83 = arith.muli %arg6, %dim_20 : index
        ascendc.data_copy_l2 %81, %82, %83 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %9, %81 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %84 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %85 = arith.muli %arg6, %c4 : index
        ascendc.pipe.init_buffer %0, %25, %85 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %10, %c1_i32, %85 : !ascendc.queue<vecout, 1>, i32, index
        %86 = arith.muli %arg6, %dim_15 : index
        %87 = arith.muli %arg6, %dim_15 : index
        %88 = arith.muli %87, %c4 : index
        ascendc.pipe.init_buffer %0, %24, %88 : !ascendc.tbuf<veccalc>, index
        %89 = ascendc.tbuf.get_tensor %24 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.duplicate_l2 %89, %cst, %86 : !ascendc.local_tensor<*xf32>, f32, index
        %90 = arith.muli %arg6, %dim_15 : index
        %91 = arith.muli %90, %c4 : index
        ascendc.pipe.init_buffer %0, %23, %91 : !ascendc.tbuf<veccalc>, index
        %92 = ascendc.tbuf.get_tensor %23 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %92, %66, %72, %86 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %93 = arith.muli %arg6, %dim_15 : index
        %94 = arith.muli %93, %c4 : index
        ascendc.pipe.init_buffer %0, %22, %94 : !ascendc.tbuf<veccalc>, index
        %95 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.mul_l2 %95, %92, %78, %86 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %96 = arith.muli %arg6, %dim_15 : index
        %97 = arith.muli %96, %c4 : index
        ascendc.pipe.init_buffer %0, %21, %97 : !ascendc.tbuf<veccalc>, index
        %98 = ascendc.tbuf.get_tensor %21 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %98, %95, %84, %86 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.add_l2 %89, %89, %98, %86 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %99 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.reduce_sum_2d_l2 %99, %89 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.enque_tensor %10, %99 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %100 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %101 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %101, %subview_22 : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %101, %100, %arg6 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %10, %100 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %6, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %7, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %8, %78 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %9, %84 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    %expand_shape = memref.expand_shape %collapse_shape_4 [[0, 1]] output_shape [%dim, %dim_0] : memref<?xf32> into memref<?x?xf32>
    return %expand_shape : memref<?x?xf32>
  }
}


