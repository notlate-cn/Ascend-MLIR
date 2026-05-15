// -----// IR Dump After AscendCParallelizePass (ascendc-parallelize) //----- //
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
  %31 = ascendc.get_block_idx : index
  %32 = arith.muli %31, %arg4 : index
  %33 = arith.cmpi ult, %32, %9 : index
  scf.if %33 {
    %34 = arith.subi %9, %32 : index
    %35 = arith.minsi %arg4, %34 : index
    %36 = arith.divsi %35, %arg5 : index
    %37 = arith.muli %36, %arg5 : index
    %38 = arith.muli %arg5, %dim_6 : index
    %39 = arith.muli %arg5, %dim_7 : index
    %40 = arith.muli %arg5, %dim_8 : index
    %41 = arith.muli %arg5, %dim_6 : index
    %42 = arith.muli %arg5, %dim_6 : index
    %43 = arith.muli %42, %c4 : index
    %44 = arith.muli %arg5, %dim_6 : index
    %45 = arith.muli %44, %c4 : index
    %46 = arith.muli %arg5, %dim_6 : index
    %47 = arith.muli %46, %c4 : index
    ascendc.pipe.init_buffer %0, %24, %47 : !ascendc.tbuf<veccalc>, index
    ascendc.pipe.init_buffer %0, %25, %45 : !ascendc.tbuf<veccalc>, index
    ascendc.pipe.init_buffer %0, %26, %43 : !ascendc.tbuf<veccalc>, index
    %48 = scf.for %arg6 = %c0 to %37 step %arg5 iter_args(%arg7 = %collapse_shape_3) -> (memref<?xf32>) {
      %51 = arith.addi %32, %arg6 : index
      %subview = memref.subview %collapse_shape[%51, 0] [%arg5, %dim_9] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_12 = memref.subview %collapse_shape_1[%51, 0] [%arg5, %dim_10] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_13 = memref.subview %collapse_shape_2[%51, 0] [%arg5, %dim_11] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_14 = memref.subview %arg7[%51] [%arg5] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      %52 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %53 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %53, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
      ascendc.data_copy_l2 %52, %53, %38 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %1, %52 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %54 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %55 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %56 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %56, %subview_12 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
      ascendc.data_copy_l2 %55, %56, %39 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %2, %55 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %57 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %58 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %59 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %59, %subview_13 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
      ascendc.data_copy_l2 %58, %59, %40 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %3, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %60 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %61 = ascendc.tbuf.get_tensor %26 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.duplicate_l2 %61, %cst, %41 : !ascendc.local_tensor<*xf32>, f32, index
      %62 = ascendc.tbuf.get_tensor %25 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %62, %54, %57, %41 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %63 = ascendc.tbuf.get_tensor %24 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.mul_l2 %63, %62, %60, %41 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.add_l2 %61, %61, %63, %41 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %64 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.reduce_sum_2d_l2 %64, %61 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.enque_tensor %4, %64 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %65 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %66 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %66, %subview_14 : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
      ascendc.data_copy_l2 %66, %65, %arg5 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.que_bind.free_tensor %4, %65 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %1, %54 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %2, %57 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %3, %60 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      scf.yield %arg7 : memref<?xf32>
    }
    %49 = arith.cmpi slt, %37, %35 : index
    %50 = scf.if %49 -> (memref<?xf32>) {
      %51 = arith.subi %9, %arg5 : index
      %dim_12 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
      %subview = memref.subview %collapse_shape[%51, 0] [%arg5, %dim_12] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_13 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
      %subview_14 = memref.subview %collapse_shape_1[%51, 0] [%arg5, %dim_13] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_15 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
      %subview_16 = memref.subview %collapse_shape_2[%51, 0] [%arg5, %dim_15] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_17 = memref.subview %48[%51] [%arg5] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      %52 = arith.muli %arg5, %dim_12 : index
      %53 = arith.muli %52, %c4 : index
      ascendc.pipe.init_buffer %0, %23, %53 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %0, %5, %c1_i32, %53 : !ascendc.queue<vecin, 1>, i32, index
      %54 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %55 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %55, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
      %56 = arith.muli %arg5, %dim_12 : index
      ascendc.data_copy_l2 %54, %55, %56 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %5, %54 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %57 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %58 = arith.muli %arg5, %dim_13 : index
      %59 = arith.muli %58, %c4 : index
      ascendc.pipe.init_buffer %0, %22, %59 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %0, %6, %c1_i32, %59 : !ascendc.queue<vecin, 1>, i32, index
      %60 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %61 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %61, %subview_14 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
      %62 = arith.muli %arg5, %dim_13 : index
      ascendc.data_copy_l2 %60, %61, %62 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %6, %60 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %63 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %64 = arith.muli %arg5, %dim_15 : index
      %65 = arith.muli %64, %c4 : index
      ascendc.pipe.init_buffer %0, %21, %65 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %0, %7, %c1_i32, %65 : !ascendc.queue<vecin, 1>, i32, index
      %66 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %67 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %67, %subview_16 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
      %68 = arith.muli %arg5, %dim_15 : index
      ascendc.data_copy_l2 %66, %67, %68 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %7, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %69 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %70 = arith.muli %arg5, %c4 : index
      ascendc.pipe.init_buffer %0, %20, %70 : !ascendc.tbuf<vecout>, index
      ascendc.pipe.init_queue %0, %8, %c1_i32, %70 : !ascendc.queue<vecout, 1>, i32, index
      %71 = arith.muli %arg5, %dim_12 : index
      %72 = arith.muli %arg5, %dim_12 : index
      %73 = arith.muli %72, %c4 : index
      ascendc.pipe.init_buffer %0, %19, %73 : !ascendc.tbuf<veccalc>, index
      %74 = ascendc.tbuf.get_tensor %19 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.duplicate_l2 %74, %cst, %71 : !ascendc.local_tensor<*xf32>, f32, index
      %75 = arith.muli %arg5, %dim_12 : index
      %76 = arith.muli %75, %c4 : index
      ascendc.pipe.init_buffer %0, %18, %76 : !ascendc.tbuf<veccalc>, index
      %77 = ascendc.tbuf.get_tensor %18 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %77, %57, %63, %71 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %78 = arith.muli %arg5, %dim_12 : index
      %79 = arith.muli %78, %c4 : index
      ascendc.pipe.init_buffer %0, %17, %79 : !ascendc.tbuf<veccalc>, index
      %80 = ascendc.tbuf.get_tensor %17 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.mul_l2 %80, %77, %69, %71 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.add_l2 %74, %74, %80, %71 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %81 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.reduce_sum_2d_l2 %81, %74 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.enque_tensor %8, %81 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %82 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %83 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      ascendc.global_tensor.set_global_buffer %83, %subview_17 : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
      ascendc.data_copy_l2 %83, %82, %arg5 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.que_bind.free_tensor %8, %82 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %5, %57 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %6, %63 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %7, %69 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      scf.yield %48 : memref<?xf32>
    } else {
      scf.yield %48 : memref<?xf32>
    }
  }
  %expand_shape = memref.expand_shape %collapse_shape_3 [[0, 1]] output_shape [%dim, %dim_0] : memref<?xf32> into memref<?x?xf32>
  return %expand_shape : memref<?x?xf32>
}

