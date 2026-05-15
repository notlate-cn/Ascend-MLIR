// -----// IR Dump After AscendCFlattenGMPtrPass (ascendc-flatten-gm-ptr) //----- //
func.func private @kernel_group1__v0(%arg0: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg1: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg2: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg3: memref<?x?xf32> {afir.symbolic_shape = "s0,s1"}, %arg4: index, %arg5: index, %arg6: memref<?x?xf32, strided<[?, 1], offset: ?>>) -> memref<?x?xf32> attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}]} {
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
  %cast = memref.cast %arg6 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32>
  %collapse_shape = memref.collapse_shape %cast [[0, 1]] : memref<?x?xf32> into memref<?xf32>
  %dim_1 = memref.dim %arg0, %c0 : memref<?x?x?xf32>
  %dim_2 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
  %9 = arith.muli %dim_1, %dim_2 : index
  %c2 = arith.constant 2 : index
  %dim_3 = memref.dim %arg0, %c2 : memref<?x?x?xf32>
  %10 = arith.muli %arg5, %dim_3 : index
  %11 = arith.muli %10, %c4 : index
  %c2_4 = arith.constant 2 : index
  %dim_5 = memref.dim %arg1, %c2_4 : memref<?x?x?xf32>
  %12 = arith.muli %arg5, %dim_5 : index
  %13 = arith.muli %12, %c4 : index
  %c2_6 = arith.constant 2 : index
  %dim_7 = memref.dim %arg2, %c2_6 : memref<?x?x?xf32>
  %14 = arith.muli %arg5, %dim_7 : index
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
    scf.for %arg7 = %c0 to %37 step %arg5 {
      %39 = arith.addi %32, %arg7 : index
      %40 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %41 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_8 = arith.constant 0 : index
      %c0_9 = arith.constant 0 : index
      %c1_10 = arith.constant 1 : index
      %c1_11 = arith.constant 1 : index
      %c2_12 = arith.constant 2 : index
      %dim_13 = memref.dim %arg0, %c2_12 : memref<?x?x?xf32>
      %42 = arith.muli %c1_10, %dim_13 : index
      %43 = arith.muli %39, %42 : index
      %44 = arith.addi %c0_9, %43 : index
      %c1_14 = arith.constant 1 : index
      %c0_15 = arith.constant 0 : index
      %45 = arith.muli %c0_15, %c1_14 : index
      %46 = arith.addi %44, %45 : index
      %47 = arith.addi %c0_8, %46 : index
      %48 = arith.index_cast %47 : index to i32
      %49 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %41, %49, %48 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %40, %41, %10 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %1, %40 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %50 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %51 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %52 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_16 = arith.constant 0 : index
      %c0_17 = arith.constant 0 : index
      %c1_18 = arith.constant 1 : index
      %c1_19 = arith.constant 1 : index
      %c2_20 = arith.constant 2 : index
      %dim_21 = memref.dim %arg1, %c2_20 : memref<?x?x?xf32>
      %53 = arith.muli %c1_18, %dim_21 : index
      %54 = arith.muli %39, %53 : index
      %55 = arith.addi %c0_17, %54 : index
      %c1_22 = arith.constant 1 : index
      %c0_23 = arith.constant 0 : index
      %56 = arith.muli %c0_23, %c1_22 : index
      %57 = arith.addi %55, %56 : index
      %58 = arith.addi %c0_16, %57 : index
      %59 = arith.index_cast %58 : index to i32
      %60 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %52, %60, %59 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %51, %52, %12 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %2, %51 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %61 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %62 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %63 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_24 = arith.constant 0 : index
      %c0_25 = arith.constant 0 : index
      %c1_26 = arith.constant 1 : index
      %c1_27 = arith.constant 1 : index
      %c2_28 = arith.constant 2 : index
      %dim_29 = memref.dim %arg2, %c2_28 : memref<?x?x?xf32>
      %64 = arith.muli %c1_26, %dim_29 : index
      %65 = arith.muli %39, %64 : index
      %66 = arith.addi %c0_25, %65 : index
      %c1_30 = arith.constant 1 : index
      %c0_31 = arith.constant 0 : index
      %67 = arith.muli %c0_31, %c1_30 : index
      %68 = arith.addi %66, %67 : index
      %69 = arith.addi %c0_24, %68 : index
      %70 = arith.index_cast %69 : index to i32
      %71 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %63, %71, %70 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %62, %63, %14 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %3, %62 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %72 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %73 = ascendc.tbuf.get_tensor %26 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.duplicate_l2 %73, %cst, %10 : !ascendc.local_tensor<*xf32>, f32, index
      %74 = ascendc.tbuf.get_tensor %25 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %74, %50, %61, %10 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %75 = ascendc.tbuf.get_tensor %24 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.mul_l2 %75, %74, %72, %10 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.add_l2 %73, %73, %75, %10 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %76 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.reduce_sum_2d_l2 %76, %73 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.enque_tensor %4, %76 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %77 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %78 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_32 = arith.constant 0 : index
      %79 = arith.addi %c0_32, %39 : index
      %80 = arith.index_cast %79 : index to i32
      %81 = emitasc.reinterpret_cast %arg6 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %78, %81, %80 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %78, %77, %arg5 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.que_bind.free_tensor %4, %77 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %1, %50 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %2, %61 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %3, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
    }
    %38 = arith.cmpi slt, %37, %35 : index
    scf.if %38 {
      %39 = arith.subi %9, %arg5 : index
      ascendc.pipe.init_buffer %0, %23, %11 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %0, %5, %c1_i32, %11 : !ascendc.queue<vecin, 1>, i32, index
      %40 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %41 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_8 = arith.constant 0 : index
      %c0_9 = arith.constant 0 : index
      %c1_10 = arith.constant 1 : index
      %c1_11 = arith.constant 1 : index
      %c2_12 = arith.constant 2 : index
      %dim_13 = memref.dim %arg0, %c2_12 : memref<?x?x?xf32>
      %42 = arith.muli %c1_10, %dim_13 : index
      %43 = arith.muli %39, %42 : index
      %44 = arith.addi %c0_9, %43 : index
      %c1_14 = arith.constant 1 : index
      %c0_15 = arith.constant 0 : index
      %45 = arith.muli %c0_15, %c1_14 : index
      %46 = arith.addi %44, %45 : index
      %47 = arith.addi %c0_8, %46 : index
      %48 = arith.index_cast %47 : index to i32
      %49 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %41, %49, %48 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %40, %41, %10 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %5, %40 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %50 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.pipe.init_buffer %0, %22, %13 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %0, %6, %c1_i32, %13 : !ascendc.queue<vecin, 1>, i32, index
      %51 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %52 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_16 = arith.constant 0 : index
      %c0_17 = arith.constant 0 : index
      %c1_18 = arith.constant 1 : index
      %c1_19 = arith.constant 1 : index
      %c2_20 = arith.constant 2 : index
      %dim_21 = memref.dim %arg1, %c2_20 : memref<?x?x?xf32>
      %53 = arith.muli %c1_18, %dim_21 : index
      %54 = arith.muli %39, %53 : index
      %55 = arith.addi %c0_17, %54 : index
      %c1_22 = arith.constant 1 : index
      %c0_23 = arith.constant 0 : index
      %56 = arith.muli %c0_23, %c1_22 : index
      %57 = arith.addi %55, %56 : index
      %58 = arith.addi %c0_16, %57 : index
      %59 = arith.index_cast %58 : index to i32
      %60 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %52, %60, %59 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %51, %52, %12 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %6, %51 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %61 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.pipe.init_buffer %0, %21, %15 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %0, %7, %c1_i32, %15 : !ascendc.queue<vecin, 1>, i32, index
      %62 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %63 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_24 = arith.constant 0 : index
      %c0_25 = arith.constant 0 : index
      %c1_26 = arith.constant 1 : index
      %c1_27 = arith.constant 1 : index
      %c2_28 = arith.constant 2 : index
      %dim_29 = memref.dim %arg2, %c2_28 : memref<?x?x?xf32>
      %64 = arith.muli %c1_26, %dim_29 : index
      %65 = arith.muli %39, %64 : index
      %66 = arith.addi %c0_25, %65 : index
      %c1_30 = arith.constant 1 : index
      %c0_31 = arith.constant 0 : index
      %67 = arith.muli %c0_31, %c1_30 : index
      %68 = arith.addi %66, %67 : index
      %69 = arith.addi %c0_24, %68 : index
      %70 = arith.index_cast %69 : index to i32
      %71 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %63, %71, %70 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %62, %63, %14 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %7, %62 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %72 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.pipe.init_buffer %0, %20, %16 : !ascendc.tbuf<vecout>, index
      ascendc.pipe.init_queue %0, %8, %c1_i32, %16 : !ascendc.queue<vecout, 1>, i32, index
      ascendc.pipe.init_buffer %0, %19, %11 : !ascendc.tbuf<veccalc>, index
      %73 = ascendc.tbuf.get_tensor %19 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.duplicate_l2 %73, %cst, %10 : !ascendc.local_tensor<*xf32>, f32, index
      ascendc.pipe.init_buffer %0, %18, %11 : !ascendc.tbuf<veccalc>, index
      %74 = ascendc.tbuf.get_tensor %18 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %74, %50, %61, %10 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.pipe.init_buffer %0, %17, %11 : !ascendc.tbuf<veccalc>, index
      %75 = ascendc.tbuf.get_tensor %17 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.mul_l2 %75, %74, %72, %10 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.add_l2 %73, %73, %75, %10 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %76 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.reduce_sum_2d_l2 %76, %73 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.enque_tensor %8, %76 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %77 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %78 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_32 = arith.constant 0 : index
      %79 = arith.addi %c0_32, %39 : index
      %80 = arith.index_cast %79 : index to i32
      %81 = emitasc.reinterpret_cast %arg6 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %78, %81, %80 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %78, %77, %arg5 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.que_bind.free_tensor %8, %77 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %5, %50 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %6, %61 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %7, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
    }
  }
  %expand_shape = memref.expand_shape %collapse_shape [[0, 1]] output_shape [%dim, %dim_0] : memref<?xf32> into memref<?x?xf32>
  return %expand_shape : memref<?x?xf32>
}

