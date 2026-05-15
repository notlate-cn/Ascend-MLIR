// -----// IR Dump After AscendCPackTilingDataPass (ascendc-pack-tiling-data) //----- //
func.func private @kernel_group1__v0(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg5: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, 22 : i32>) -> memref<?x?xf32> attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}]} {
  %0 = emitasc.copy_struct %arg5 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>
  %1 = emitasc.member %0 "XBLOCK" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
  %2 = emitasc.member %0 "XBLOCK_SUB" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
  %3 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
  %4 = emitasc.member %0 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
  %5 = emitasc.member %0 "dim_arg0_2" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
  %6 = arith.index_cast %1 : i64 to index
  %7 = arith.index_cast %2 : i64 to index
  %cst = arith.constant 0.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %8 = ascendc.pipe
  %9 = ascendc.queue : <vecin, 1>
  %10 = ascendc.queue : <vecin, 1>
  %11 = ascendc.queue : <vecin, 1>
  %12 = ascendc.queue : <vecout, 1>
  %13 = ascendc.queue : <vecin, 1>
  %14 = ascendc.queue : <vecin, 1>
  %15 = ascendc.queue : <vecin, 1>
  %16 = ascendc.queue : <vecout, 1>
  %17 = arith.index_cast %3 : i64 to index
  %18 = arith.index_cast %4 : i64 to index
  %cast = memref.cast %arg4 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32>
  %collapse_shape = memref.collapse_shape %cast [[0, 1]] : memref<?x?xf32> into memref<?xf32>
  %19 = arith.index_cast %3 : i64 to index
  %20 = arith.index_cast %4 : i64 to index
  %21 = arith.muli %19, %20 : index
  %c2 = arith.constant 2 : index
  %22 = arith.index_cast %5 : i64 to index
  %23 = arith.muli %7, %22 : index
  %24 = arith.muli %23, %c4 : index
  %c2_0 = arith.constant 2 : index
  %25 = arith.index_cast %5 : i64 to index
  %26 = arith.muli %7, %25 : index
  %27 = arith.muli %26, %c4 : index
  %c2_1 = arith.constant 2 : index
  %28 = arith.index_cast %5 : i64 to index
  %29 = arith.muli %7, %28 : index
  %30 = arith.muli %29, %c4 : index
  %31 = arith.muli %7, %c4 : index
  %32 = ascendc.tbuf : <veccalc>
  %33 = ascendc.tbuf : <veccalc>
  %34 = ascendc.tbuf : <veccalc>
  %35 = ascendc.tbuf : <vecout>
  %36 = ascendc.tbuf : <vecin>
  %37 = ascendc.tbuf : <vecin>
  %38 = ascendc.tbuf : <vecin>
  %39 = ascendc.tbuf : <veccalc>
  %40 = ascendc.tbuf : <veccalc>
  %41 = ascendc.tbuf : <veccalc>
  %42 = ascendc.tbuf : <vecout>
  ascendc.pipe.init_buffer %8, %42, %31 : !ascendc.tbuf<vecout>, index
  %43 = ascendc.tbuf : <vecin>
  ascendc.pipe.init_buffer %8, %43, %30 : !ascendc.tbuf<vecin>, index
  %44 = ascendc.tbuf : <vecin>
  ascendc.pipe.init_buffer %8, %44, %27 : !ascendc.tbuf<vecin>, index
  ascendc.pipe.init_queue %8, %9, %c1_i32, %24 : !ascendc.queue<vecin, 1>, i32, index
  %45 = ascendc.tbuf : <vecin>
  ascendc.pipe.init_buffer %8, %45, %24 : !ascendc.tbuf<vecin>, index
  ascendc.pipe.init_queue %8, %12, %c1_i32, %31 : !ascendc.queue<vecout, 1>, i32, index
  ascendc.pipe.init_queue %8, %11, %c1_i32, %30 : !ascendc.queue<vecin, 1>, i32, index
  ascendc.pipe.init_queue %8, %10, %c1_i32, %27 : !ascendc.queue<vecin, 1>, i32, index
  %46 = ascendc.get_block_idx : index
  %47 = arith.muli %46, %6 : index
  %48 = arith.cmpi ult, %47, %21 : index
  scf.if %48 {
    %49 = arith.subi %21, %47 : index
    %50 = arith.minsi %6, %49 : index
    %51 = arith.divsi %50, %7 : index
    %52 = arith.muli %51, %7 : index
    ascendc.pipe.init_buffer %8, %39, %24 : !ascendc.tbuf<veccalc>, index
    ascendc.pipe.init_buffer %8, %40, %24 : !ascendc.tbuf<veccalc>, index
    ascendc.pipe.init_buffer %8, %41, %24 : !ascendc.tbuf<veccalc>, index
    scf.for %arg6 = %c0 to %52 step %7 {
      %54 = arith.addi %47, %arg6 : index
      %55 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %56 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_2 = arith.constant 0 : index
      %c0_3 = arith.constant 0 : index
      %c1_4 = arith.constant 1 : index
      %c1_5 = arith.constant 1 : index
      %c2_6 = arith.constant 2 : index
      %57 = arith.index_cast %5 : i64 to index
      %58 = arith.muli %c1_4, %57 : index
      %59 = arith.muli %54, %58 : index
      %60 = arith.addi %c0_3, %59 : index
      %c1_7 = arith.constant 1 : index
      %c0_8 = arith.constant 0 : index
      %61 = arith.muli %c0_8, %c1_7 : index
      %62 = arith.addi %60, %61 : index
      %63 = arith.addi %c0_2, %62 : index
      %64 = arith.index_cast %63 : index to i32
      %65 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %56, %65, %64 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %55, %56, %23 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %9, %55 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %66 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %67 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %68 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_9 = arith.constant 0 : index
      %c0_10 = arith.constant 0 : index
      %c1_11 = arith.constant 1 : index
      %c1_12 = arith.constant 1 : index
      %c2_13 = arith.constant 2 : index
      %69 = arith.index_cast %5 : i64 to index
      %70 = arith.muli %c1_11, %69 : index
      %71 = arith.muli %54, %70 : index
      %72 = arith.addi %c0_10, %71 : index
      %c1_14 = arith.constant 1 : index
      %c0_15 = arith.constant 0 : index
      %73 = arith.muli %c0_15, %c1_14 : index
      %74 = arith.addi %72, %73 : index
      %75 = arith.addi %c0_9, %74 : index
      %76 = arith.index_cast %75 : index to i32
      %77 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %68, %77, %76 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %67, %68, %26 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %10, %67 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %78 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %79 = ascendc.que_bind.alloc_tensor %11 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %80 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_16 = arith.constant 0 : index
      %c0_17 = arith.constant 0 : index
      %c1_18 = arith.constant 1 : index
      %c1_19 = arith.constant 1 : index
      %c2_20 = arith.constant 2 : index
      %81 = arith.index_cast %5 : i64 to index
      %82 = arith.muli %c1_18, %81 : index
      %83 = arith.muli %54, %82 : index
      %84 = arith.addi %c0_17, %83 : index
      %c1_21 = arith.constant 1 : index
      %c0_22 = arith.constant 0 : index
      %85 = arith.muli %c0_22, %c1_21 : index
      %86 = arith.addi %84, %85 : index
      %87 = arith.addi %c0_16, %86 : index
      %88 = arith.index_cast %87 : index to i32
      %89 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %80, %89, %88 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %79, %80, %29 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %11, %79 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %90 = ascendc.que_bind.deque_tensor %11 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %91 = ascendc.tbuf.get_tensor %41 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.duplicate_l2 %91, %cst, %23 : !ascendc.local_tensor<*xf32>, f32, index
      %92 = ascendc.tbuf.get_tensor %40 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %92, %66, %78, %23 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %93 = ascendc.tbuf.get_tensor %39 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.mul_l2 %93, %92, %90, %23 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.add_l2 %91, %91, %93, %23 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %94 = ascendc.que_bind.alloc_tensor %12 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.reduce_sum_2d_l2 %94, %91 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.enque_tensor %12, %94 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %95 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %96 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_23 = arith.constant 0 : index
      %97 = arith.addi %c0_23, %54 : index
      %98 = arith.index_cast %97 : index to i32
      %99 = emitasc.reinterpret_cast %arg4 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %96, %99, %98 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %96, %95, %7 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.que_bind.free_tensor %12, %95 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %9, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %10, %78 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %11, %90 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
    }
    %53 = arith.cmpi slt, %52, %50 : index
    scf.if %53 {
      %54 = arith.subi %21, %7 : index
      ascendc.pipe.init_buffer %8, %38, %24 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %8, %13, %c1_i32, %24 : !ascendc.queue<vecin, 1>, i32, index
      %55 = ascendc.que_bind.alloc_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %56 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_2 = arith.constant 0 : index
      %c0_3 = arith.constant 0 : index
      %c1_4 = arith.constant 1 : index
      %c1_5 = arith.constant 1 : index
      %c2_6 = arith.constant 2 : index
      %57 = arith.index_cast %5 : i64 to index
      %58 = arith.muli %c1_4, %57 : index
      %59 = arith.muli %54, %58 : index
      %60 = arith.addi %c0_3, %59 : index
      %c1_7 = arith.constant 1 : index
      %c0_8 = arith.constant 0 : index
      %61 = arith.muli %c0_8, %c1_7 : index
      %62 = arith.addi %60, %61 : index
      %63 = arith.addi %c0_2, %62 : index
      %64 = arith.index_cast %63 : index to i32
      %65 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %56, %65, %64 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %55, %56, %23 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %13, %55 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %66 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.pipe.init_buffer %8, %37, %27 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %8, %14, %c1_i32, %27 : !ascendc.queue<vecin, 1>, i32, index
      %67 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %68 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_9 = arith.constant 0 : index
      %c0_10 = arith.constant 0 : index
      %c1_11 = arith.constant 1 : index
      %c1_12 = arith.constant 1 : index
      %c2_13 = arith.constant 2 : index
      %69 = arith.index_cast %5 : i64 to index
      %70 = arith.muli %c1_11, %69 : index
      %71 = arith.muli %54, %70 : index
      %72 = arith.addi %c0_10, %71 : index
      %c1_14 = arith.constant 1 : index
      %c0_15 = arith.constant 0 : index
      %73 = arith.muli %c0_15, %c1_14 : index
      %74 = arith.addi %72, %73 : index
      %75 = arith.addi %c0_9, %74 : index
      %76 = arith.index_cast %75 : index to i32
      %77 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %68, %77, %76 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %67, %68, %26 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %14, %67 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %78 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.pipe.init_buffer %8, %36, %30 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %8, %15, %c1_i32, %30 : !ascendc.queue<vecin, 1>, i32, index
      %79 = ascendc.que_bind.alloc_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %80 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_16 = arith.constant 0 : index
      %c0_17 = arith.constant 0 : index
      %c1_18 = arith.constant 1 : index
      %c1_19 = arith.constant 1 : index
      %c2_20 = arith.constant 2 : index
      %81 = arith.index_cast %5 : i64 to index
      %82 = arith.muli %c1_18, %81 : index
      %83 = arith.muli %54, %82 : index
      %84 = arith.addi %c0_17, %83 : index
      %c1_21 = arith.constant 1 : index
      %c0_22 = arith.constant 0 : index
      %85 = arith.muli %c0_22, %c1_21 : index
      %86 = arith.addi %84, %85 : index
      %87 = arith.addi %c0_16, %86 : index
      %88 = arith.index_cast %87 : index to i32
      %89 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %80, %89, %88 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %79, %80, %29 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %15, %79 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %90 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.pipe.init_buffer %8, %35, %31 : !ascendc.tbuf<vecout>, index
      ascendc.pipe.init_queue %8, %16, %c1_i32, %31 : !ascendc.queue<vecout, 1>, i32, index
      ascendc.pipe.init_buffer %8, %34, %24 : !ascendc.tbuf<veccalc>, index
      %91 = ascendc.tbuf.get_tensor %34 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.duplicate_l2 %91, %cst, %23 : !ascendc.local_tensor<*xf32>, f32, index
      ascendc.pipe.init_buffer %8, %33, %24 : !ascendc.tbuf<veccalc>, index
      %92 = ascendc.tbuf.get_tensor %33 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %92, %66, %78, %23 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.pipe.init_buffer %8, %32, %24 : !ascendc.tbuf<veccalc>, index
      %93 = ascendc.tbuf.get_tensor %32 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.mul_l2 %93, %92, %90, %23 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.add_l2 %91, %91, %93, %23 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %94 = ascendc.que_bind.alloc_tensor %16 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.reduce_sum_2d_l2 %94, %91 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.enque_tensor %16, %94 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %95 = ascendc.que_bind.deque_tensor %16 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %96 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_23 = arith.constant 0 : index
      %97 = arith.addi %c0_23, %54 : index
      %98 = arith.index_cast %97 : index to i32
      %99 = emitasc.reinterpret_cast %arg4 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %96, %99, %98 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %96, %95, %7 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.que_bind.free_tensor %16, %95 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %13, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %14, %78 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %15, %90 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
    }
  }
  %expand_shape = memref.expand_shape %collapse_shape [[0, 1]] output_shape [%17, %18] : memref<?xf32> into memref<?x?xf32>
  return %expand_shape : memref<?x?xf32>
}

