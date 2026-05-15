// -----// IR Dump After LinalgToAscendCPass (linalg-to-ascendc) //----- //
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
  %dim_9 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
  %dim_10 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
  %dim_11 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
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
  %31 = scf.for %arg6 = %c0 to %9 step %arg4 iter_args(%arg7 = %collapse_shape_3) -> (memref<?xf32>) {
    %32 = arith.subi %9, %arg6 : index
    %33 = arith.minsi %arg4, %32 : index
    %34 = arith.divsi %33, %arg5 : index
    %35 = arith.muli %34, %arg5 : index
    %36 = arith.muli %arg5, %dim_6 : index
    %37 = arith.muli %arg5, %dim_7 : index
    %38 = arith.muli %arg5, %dim_8 : index
    %39 = arith.muli %arg5, %dim_6 : index
    %40 = arith.muli %arg5, %dim_6 : index
    %41 = arith.muli %40, %c4 : index
    %42 = arith.muli %arg5, %dim_6 : index
    %43 = arith.muli %42, %c4 : index
    %44 = arith.muli %arg5, %dim_6 : index
    %45 = arith.muli %44, %c4 : index
    ascendc.pipe.init_buffer %0, %24, %45 : !ascendc.tbuf<veccalc>, index
    ascendc.pipe.init_buffer %0, %25, %43 : !ascendc.tbuf<veccalc>, index
    ascendc.pipe.init_buffer %0, %26, %41 : !ascendc.tbuf<veccalc>, index
    %46 = scf.for %arg8 = %c0 to %35 step %arg5 iter_args(%arg9 = %arg7) -> (memref<?xf32>) {
      %49 = arith.addi %arg6, %arg8 : index
      %subview = memref.subview %collapse_shape[%49, 0] [%arg5, %dim_9] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_12 = memref.subview %collapse_shape_1[%49, 0] [%arg5, %dim_10] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_13 = memref.subview %collapse_shape_2[%49, 0] [%arg5, %dim_11] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_14 = memref.subview %arg9[%49] [%arg5] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      %50 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %51 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %51, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
      ascendc.data_copy_l2 %50, %51, %36 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %1, %50 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %52 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %53 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %54, %subview_12 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
      ascendc.data_copy_l2 %53, %54, %37 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %2, %53 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %55 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %56 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %57 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %57, %subview_13 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
      ascendc.data_copy_l2 %56, %57, %38 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %3, %56 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %58 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %59 = ascendc.tbuf.get_tensor %26 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.duplicate_l2 %59, %cst, %39 : !ascendc.local_tensor<*xf32>, f32, index
      %60 = ascendc.tbuf.get_tensor %25 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %60, %52, %55, %39 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %61 = ascendc.tbuf.get_tensor %24 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.mul_l2 %61, %60, %58, %39 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.add_l2 %59, %59, %61, %39 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %62 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.reduce_sum_2d_l2 %62, %59 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.enque_tensor %4, %62 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %63 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %64 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %64, %subview_14 : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
      ascendc.data_copy_l2 %64, %63, %arg5 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.que_bind.free_tensor %4, %63 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %1, %52 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %2, %55 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %3, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      scf.yield %arg9 : memref<?xf32>
    }
    %47 = arith.cmpi slt, %35, %33 : index
    %48 = scf.if %47 -> (memref<?xf32>) {
      %49 = arith.subi %9, %arg5 : index
      %dim_12 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
      %subview = memref.subview %collapse_shape[%49, 0] [%arg5, %dim_12] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_13 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
      %subview_14 = memref.subview %collapse_shape_1[%49, 0] [%arg5, %dim_13] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_15 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
      %subview_16 = memref.subview %collapse_shape_2[%49, 0] [%arg5, %dim_15] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_17 = memref.subview %46[%49] [%arg5] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      %50 = arith.muli %arg5, %dim_12 : index
      %51 = arith.muli %50, %c4 : index
      ascendc.pipe.init_buffer %0, %23, %51 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %0, %5, %c1_i32, %51 : !ascendc.queue<vecin, 1>, i32, index
      %52 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %53 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %53, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
      %54 = arith.muli %arg5, %dim_12 : index
      ascendc.data_copy_l2 %52, %53, %54 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %5, %52 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %55 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %56 = arith.muli %arg5, %dim_13 : index
      %57 = arith.muli %56, %c4 : index
      ascendc.pipe.init_buffer %0, %22, %57 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %0, %6, %c1_i32, %57 : !ascendc.queue<vecin, 1>, i32, index
      %58 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %59 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %59, %subview_14 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
      %60 = arith.muli %arg5, %dim_13 : index
      ascendc.data_copy_l2 %58, %59, %60 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %6, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %61 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %62 = arith.muli %arg5, %dim_15 : index
      %63 = arith.muli %62, %c4 : index
      ascendc.pipe.init_buffer %0, %21, %63 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %0, %7, %c1_i32, %63 : !ascendc.queue<vecin, 1>, i32, index
      %64 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %65 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %65, %subview_16 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
      %66 = arith.muli %arg5, %dim_15 : index
      ascendc.data_copy_l2 %64, %65, %66 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %7, %64 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %67 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %68 = arith.muli %arg5, %c4 : index
      ascendc.pipe.init_buffer %0, %20, %68 : !ascendc.tbuf<vecout>, index
      ascendc.pipe.init_queue %0, %8, %c1_i32, %68 : !ascendc.queue<vecout, 1>, i32, index
      %69 = arith.muli %arg5, %dim_12 : index
      %70 = arith.muli %arg5, %dim_12 : index
      %71 = arith.muli %70, %c4 : index
      ascendc.pipe.init_buffer %0, %19, %71 : !ascendc.tbuf<veccalc>, index
      %72 = ascendc.tbuf.get_tensor %19 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.duplicate_l2 %72, %cst, %69 : !ascendc.local_tensor<*xf32>, f32, index
      %73 = arith.muli %arg5, %dim_12 : index
      %74 = arith.muli %73, %c4 : index
      ascendc.pipe.init_buffer %0, %18, %74 : !ascendc.tbuf<veccalc>, index
      %75 = ascendc.tbuf.get_tensor %18 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %75, %55, %61, %69 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %76 = arith.muli %arg5, %dim_12 : index
      %77 = arith.muli %76, %c4 : index
      ascendc.pipe.init_buffer %0, %17, %77 : !ascendc.tbuf<veccalc>, index
      %78 = ascendc.tbuf.get_tensor %17 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.mul_l2 %78, %75, %67, %69 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.add_l2 %72, %72, %78, %69 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %79 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.reduce_sum_2d_l2 %79, %72 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.enque_tensor %8, %79 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %80 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %81 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %81, %subview_17 : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
      ascendc.data_copy_l2 %81, %80, %arg5 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.que_bind.free_tensor %8, %80 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %5, %55 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %6, %61 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %7, %67 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      scf.yield %46 : memref<?xf32>
    } else {
      scf.yield %46 : memref<?xf32>
    }
    scf.yield %48 : memref<?xf32>
  }
  %expand_shape = memref.expand_shape %31 [[0, 1]] output_shape [%dim, %dim_0] : memref<?xf32> into memref<?x?xf32>
  return %expand_shape : memref<?x?xf32>
}

