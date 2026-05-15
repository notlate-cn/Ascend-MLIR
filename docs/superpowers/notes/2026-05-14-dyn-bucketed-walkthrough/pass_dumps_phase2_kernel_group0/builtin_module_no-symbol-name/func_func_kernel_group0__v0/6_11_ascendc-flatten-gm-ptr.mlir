// -----// IR Dump After AscendCFlattenGMPtrPass (ascendc-flatten-gm-ptr) //----- //
func.func private @kernel_group0__v0(%arg0: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg1: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg2: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg3: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg4: memref<?x?xf32> {afir.symbolic_shape = "s0,s1"}, %arg5: index, %arg6: index, %arg7: memref<?x?xf32, strided<[?, 1], offset: ?>>) -> memref<?x?xf32> attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}]} {
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
  %cast = memref.cast %arg7 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32>
  %collapse_shape = memref.collapse_shape %cast [[0, 1]] : memref<?x?xf32> into memref<?xf32>
  %dim_1 = memref.dim %arg0, %c0 : memref<?x?x?xf32>
  %dim_2 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
  %11 = arith.muli %dim_1, %dim_2 : index
  %c2 = arith.constant 2 : index
  %dim_3 = memref.dim %arg0, %c2 : memref<?x?x?xf32>
  %12 = arith.muli %arg6, %dim_3 : index
  %13 = arith.muli %12, %c4 : index
  %c2_4 = arith.constant 2 : index
  %dim_5 = memref.dim %arg1, %c2_4 : memref<?x?x?xf32>
  %14 = arith.muli %arg6, %dim_5 : index
  %15 = arith.muli %14, %c4 : index
  %c2_6 = arith.constant 2 : index
  %dim_7 = memref.dim %arg2, %c2_6 : memref<?x?x?xf32>
  %16 = arith.muli %arg6, %dim_7 : index
  %17 = arith.muli %16, %c4 : index
  %c2_8 = arith.constant 2 : index
  %dim_9 = memref.dim %arg3, %c2_8 : memref<?x?x?xf32>
  %18 = arith.muli %arg6, %dim_9 : index
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
    scf.for %arg8 = %c0 to %45 step %arg6 {
      %47 = arith.addi %40, %arg8 : index
      %48 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %49 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_10 = arith.constant 0 : index
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c1_13 = arith.constant 1 : index
      %c2_14 = arith.constant 2 : index
      %dim_15 = memref.dim %arg0, %c2_14 : memref<?x?x?xf32>
      %50 = arith.muli %c1_12, %dim_15 : index
      %51 = arith.muli %47, %50 : index
      %52 = arith.addi %c0_11, %51 : index
      %c1_16 = arith.constant 1 : index
      %c0_17 = arith.constant 0 : index
      %53 = arith.muli %c0_17, %c1_16 : index
      %54 = arith.addi %52, %53 : index
      %55 = arith.addi %c0_10, %54 : index
      %56 = arith.index_cast %55 : index to i32
      %57 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %49, %57, %56 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %48, %49, %12 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %1, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %58 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %59 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %60 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_18 = arith.constant 0 : index
      %c0_19 = arith.constant 0 : index
      %c1_20 = arith.constant 1 : index
      %c1_21 = arith.constant 1 : index
      %c2_22 = arith.constant 2 : index
      %dim_23 = memref.dim %arg1, %c2_22 : memref<?x?x?xf32>
      %61 = arith.muli %c1_20, %dim_23 : index
      %62 = arith.muli %47, %61 : index
      %63 = arith.addi %c0_19, %62 : index
      %c1_24 = arith.constant 1 : index
      %c0_25 = arith.constant 0 : index
      %64 = arith.muli %c0_25, %c1_24 : index
      %65 = arith.addi %63, %64 : index
      %66 = arith.addi %c0_18, %65 : index
      %67 = arith.index_cast %66 : index to i32
      %68 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %60, %68, %67 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %59, %60, %14 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %2, %59 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %69 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %70 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %71 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_26 = arith.constant 0 : index
      %c0_27 = arith.constant 0 : index
      %c1_28 = arith.constant 1 : index
      %c1_29 = arith.constant 1 : index
      %c2_30 = arith.constant 2 : index
      %dim_31 = memref.dim %arg2, %c2_30 : memref<?x?x?xf32>
      %72 = arith.muli %c1_28, %dim_31 : index
      %73 = arith.muli %47, %72 : index
      %74 = arith.addi %c0_27, %73 : index
      %c1_32 = arith.constant 1 : index
      %c0_33 = arith.constant 0 : index
      %75 = arith.muli %c0_33, %c1_32 : index
      %76 = arith.addi %74, %75 : index
      %77 = arith.addi %c0_26, %76 : index
      %78 = arith.index_cast %77 : index to i32
      %79 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %71, %79, %78 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %70, %71, %16 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %3, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %80 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %81 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %82 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_34 = arith.constant 0 : index
      %c0_35 = arith.constant 0 : index
      %c1_36 = arith.constant 1 : index
      %c1_37 = arith.constant 1 : index
      %c2_38 = arith.constant 2 : index
      %dim_39 = memref.dim %arg3, %c2_38 : memref<?x?x?xf32>
      %83 = arith.muli %c1_36, %dim_39 : index
      %84 = arith.muli %47, %83 : index
      %85 = arith.addi %c0_35, %84 : index
      %c1_40 = arith.constant 1 : index
      %c0_41 = arith.constant 0 : index
      %86 = arith.muli %c0_41, %c1_40 : index
      %87 = arith.addi %85, %86 : index
      %88 = arith.addi %c0_34, %87 : index
      %89 = arith.index_cast %88 : index to i32
      %90 = emitasc.reinterpret_cast %arg3 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %82, %90, %89 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %81, %82, %18 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %4, %81 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %91 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %92 = ascendc.tbuf.get_tensor %33 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.duplicate_l2 %92, %cst, %12 : !ascendc.local_tensor<*xf32>, f32, index
      %93 = ascendc.tbuf.get_tensor %32 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %93, %58, %69, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %94 = ascendc.tbuf.get_tensor %31 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.mul_l2 %94, %93, %80, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %95 = ascendc.tbuf.get_tensor %30 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %95, %94, %91, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.add_l2 %92, %92, %95, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %96 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.reduce_sum_2d_l2 %96, %92 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.enque_tensor %5, %96 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %97 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %98 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_42 = arith.constant 0 : index
      %99 = arith.addi %c0_42, %47 : index
      %100 = arith.index_cast %99 : index to i32
      %101 = emitasc.reinterpret_cast %arg7 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %98, %101, %100 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %98, %97, %arg6 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.que_bind.free_tensor %5, %97 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %1, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %2, %69 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %3, %80 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %4, %91 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
    }
    %46 = arith.cmpi slt, %45, %43 : index
    scf.if %46 {
      %47 = arith.subi %11, %arg6 : index
      ascendc.pipe.init_buffer %0, %29, %13 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %0, %6, %c1_i32, %13 : !ascendc.queue<vecin, 1>, i32, index
      %48 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %49 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_10 = arith.constant 0 : index
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c1_13 = arith.constant 1 : index
      %c2_14 = arith.constant 2 : index
      %dim_15 = memref.dim %arg0, %c2_14 : memref<?x?x?xf32>
      %50 = arith.muli %c1_12, %dim_15 : index
      %51 = arith.muli %47, %50 : index
      %52 = arith.addi %c0_11, %51 : index
      %c1_16 = arith.constant 1 : index
      %c0_17 = arith.constant 0 : index
      %53 = arith.muli %c0_17, %c1_16 : index
      %54 = arith.addi %52, %53 : index
      %55 = arith.addi %c0_10, %54 : index
      %56 = arith.index_cast %55 : index to i32
      %57 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %49, %57, %56 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %48, %49, %12 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %6, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %58 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.pipe.init_buffer %0, %28, %15 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %0, %7, %c1_i32, %15 : !ascendc.queue<vecin, 1>, i32, index
      %59 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %60 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_18 = arith.constant 0 : index
      %c0_19 = arith.constant 0 : index
      %c1_20 = arith.constant 1 : index
      %c1_21 = arith.constant 1 : index
      %c2_22 = arith.constant 2 : index
      %dim_23 = memref.dim %arg1, %c2_22 : memref<?x?x?xf32>
      %61 = arith.muli %c1_20, %dim_23 : index
      %62 = arith.muli %47, %61 : index
      %63 = arith.addi %c0_19, %62 : index
      %c1_24 = arith.constant 1 : index
      %c0_25 = arith.constant 0 : index
      %64 = arith.muli %c0_25, %c1_24 : index
      %65 = arith.addi %63, %64 : index
      %66 = arith.addi %c0_18, %65 : index
      %67 = arith.index_cast %66 : index to i32
      %68 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %60, %68, %67 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %59, %60, %14 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %7, %59 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %69 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.pipe.init_buffer %0, %27, %17 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %0, %8, %c1_i32, %17 : !ascendc.queue<vecin, 1>, i32, index
      %70 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %71 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_26 = arith.constant 0 : index
      %c0_27 = arith.constant 0 : index
      %c1_28 = arith.constant 1 : index
      %c1_29 = arith.constant 1 : index
      %c2_30 = arith.constant 2 : index
      %dim_31 = memref.dim %arg2, %c2_30 : memref<?x?x?xf32>
      %72 = arith.muli %c1_28, %dim_31 : index
      %73 = arith.muli %47, %72 : index
      %74 = arith.addi %c0_27, %73 : index
      %c1_32 = arith.constant 1 : index
      %c0_33 = arith.constant 0 : index
      %75 = arith.muli %c0_33, %c1_32 : index
      %76 = arith.addi %74, %75 : index
      %77 = arith.addi %c0_26, %76 : index
      %78 = arith.index_cast %77 : index to i32
      %79 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %71, %79, %78 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %70, %71, %16 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %8, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %80 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.pipe.init_buffer %0, %26, %19 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %0, %9, %c1_i32, %19 : !ascendc.queue<vecin, 1>, i32, index
      %81 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %82 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_34 = arith.constant 0 : index
      %c0_35 = arith.constant 0 : index
      %c1_36 = arith.constant 1 : index
      %c1_37 = arith.constant 1 : index
      %c2_38 = arith.constant 2 : index
      %dim_39 = memref.dim %arg3, %c2_38 : memref<?x?x?xf32>
      %83 = arith.muli %c1_36, %dim_39 : index
      %84 = arith.muli %47, %83 : index
      %85 = arith.addi %c0_35, %84 : index
      %c1_40 = arith.constant 1 : index
      %c0_41 = arith.constant 0 : index
      %86 = arith.muli %c0_41, %c1_40 : index
      %87 = arith.addi %85, %86 : index
      %88 = arith.addi %c0_34, %87 : index
      %89 = arith.index_cast %88 : index to i32
      %90 = emitasc.reinterpret_cast %arg3 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %82, %90, %89 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %81, %82, %18 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %9, %81 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %91 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.pipe.init_buffer %0, %25, %20 : !ascendc.tbuf<vecout>, index
      ascendc.pipe.init_queue %0, %10, %c1_i32, %20 : !ascendc.queue<vecout, 1>, i32, index
      ascendc.pipe.init_buffer %0, %24, %13 : !ascendc.tbuf<veccalc>, index
      %92 = ascendc.tbuf.get_tensor %24 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.duplicate_l2 %92, %cst, %12 : !ascendc.local_tensor<*xf32>, f32, index
      ascendc.pipe.init_buffer %0, %23, %13 : !ascendc.tbuf<veccalc>, index
      %93 = ascendc.tbuf.get_tensor %23 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %93, %58, %69, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.pipe.init_buffer %0, %22, %13 : !ascendc.tbuf<veccalc>, index
      %94 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.mul_l2 %94, %93, %80, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.pipe.init_buffer %0, %21, %13 : !ascendc.tbuf<veccalc>, index
      %95 = ascendc.tbuf.get_tensor %21 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %95, %94, %91, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.add_l2 %92, %92, %95, %12 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %96 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.reduce_sum_2d_l2 %96, %92 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.enque_tensor %10, %96 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %97 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %98 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_42 = arith.constant 0 : index
      %99 = arith.addi %c0_42, %47 : index
      %100 = arith.index_cast %99 : index to i32
      %101 = emitasc.reinterpret_cast %arg7 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %98, %101, %100 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %98, %97, %arg6 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.que_bind.free_tensor %10, %97 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %6, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %7, %69 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %8, %80 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %9, %91 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
    }
  }
  %expand_shape = memref.expand_shape %collapse_shape [[0, 1]] output_shape [%dim, %dim_0] : memref<?xf32> into memref<?x?xf32>
  return %expand_shape : memref<?x?xf32>
}

