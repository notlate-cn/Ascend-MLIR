// -----// IR Dump After AscendCFinalizeKernelPass (ascendc-finalize-kernel) //----- //
func.func private @kernel_group0__v0(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>, %arg3: memref<?x?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg6: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, 22 : i32>) attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}], ascendc.aicore, ascendc.global} {
  %0 = emitasc.copy_struct %arg6 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>
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
  %12 = ascendc.queue : <vecin, 1>
  %13 = ascendc.queue : <vecout, 1>
  %14 = ascendc.queue : <vecin, 1>
  %15 = ascendc.queue : <vecin, 1>
  %16 = ascendc.queue : <vecin, 1>
  %17 = ascendc.queue : <vecin, 1>
  %18 = ascendc.queue : <vecout, 1>
  %19 = arith.index_cast %3 : i64 to index
  %20 = arith.index_cast %4 : i64 to index
  %cast = memref.cast %arg5 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32>
  %collapse_shape = memref.collapse_shape %cast [[0, 1]] : memref<?x?xf32> into memref<?xf32>
  %21 = arith.index_cast %3 : i64 to index
  %22 = arith.index_cast %4 : i64 to index
  %23 = arith.muli %21, %22 : index
  %c2 = arith.constant 2 : index
  %24 = arith.index_cast %5 : i64 to index
  %25 = arith.muli %7, %24 : index
  %26 = arith.muli %25, %c4 : index
  %c2_0 = arith.constant 2 : index
  %27 = arith.index_cast %5 : i64 to index
  %28 = arith.muli %7, %27 : index
  %29 = arith.muli %28, %c4 : index
  %c2_1 = arith.constant 2 : index
  %30 = arith.index_cast %5 : i64 to index
  %31 = arith.muli %7, %30 : index
  %32 = arith.muli %31, %c4 : index
  %c2_2 = arith.constant 2 : index
  %33 = arith.index_cast %5 : i64 to index
  %34 = arith.muli %7, %33 : index
  %35 = arith.muli %34, %c4 : index
  %36 = arith.muli %7, %c4 : index
  %37 = ascendc.tbuf : <veccalc>
  %38 = ascendc.tbuf : <veccalc>
  %39 = ascendc.tbuf : <veccalc>
  %40 = ascendc.tbuf : <veccalc>
  %41 = ascendc.tbuf : <vecout>
  %42 = ascendc.tbuf : <vecin>
  %43 = ascendc.tbuf : <vecin>
  %44 = ascendc.tbuf : <vecin>
  %45 = ascendc.tbuf : <vecin>
  %46 = ascendc.tbuf : <veccalc>
  %47 = ascendc.tbuf : <veccalc>
  %48 = ascendc.tbuf : <veccalc>
  %49 = ascendc.tbuf : <veccalc>
  %50 = ascendc.tbuf : <vecout>
  ascendc.pipe.init_buffer %8, %50, %36 : !ascendc.tbuf<vecout>, index
  %51 = ascendc.tbuf : <vecin>
  ascendc.pipe.init_buffer %8, %51, %35 : !ascendc.tbuf<vecin>, index
  %52 = ascendc.tbuf : <vecin>
  ascendc.pipe.init_buffer %8, %52, %32 : !ascendc.tbuf<vecin>, index
  %53 = ascendc.tbuf : <vecin>
  ascendc.pipe.init_buffer %8, %53, %29 : !ascendc.tbuf<vecin>, index
  ascendc.pipe.init_queue %8, %9, %c1_i32, %26 : !ascendc.queue<vecin, 1>, i32, index
  %54 = ascendc.tbuf : <vecin>
  ascendc.pipe.init_buffer %8, %54, %26 : !ascendc.tbuf<vecin>, index
  ascendc.pipe.init_queue %8, %13, %c1_i32, %36 : !ascendc.queue<vecout, 1>, i32, index
  ascendc.pipe.init_queue %8, %12, %c1_i32, %35 : !ascendc.queue<vecin, 1>, i32, index
  ascendc.pipe.init_queue %8, %11, %c1_i32, %32 : !ascendc.queue<vecin, 1>, i32, index
  ascendc.pipe.init_queue %8, %10, %c1_i32, %29 : !ascendc.queue<vecin, 1>, i32, index
  %55 = ascendc.get_block_idx : index
  %56 = arith.muli %55, %6 : index
  %57 = arith.cmpi ult, %56, %23 : index
  scf.if %57 {
    %58 = arith.subi %23, %56 : index
    %59 = arith.minsi %6, %58 : index
    %60 = arith.divsi %59, %7 : index
    %61 = arith.muli %60, %7 : index
    ascendc.pipe.init_buffer %8, %46, %26 : !ascendc.tbuf<veccalc>, index
    ascendc.pipe.init_buffer %8, %47, %26 : !ascendc.tbuf<veccalc>, index
    ascendc.pipe.init_buffer %8, %48, %26 : !ascendc.tbuf<veccalc>, index
    ascendc.pipe.init_buffer %8, %49, %26 : !ascendc.tbuf<veccalc>, index
    scf.for %arg7 = %c0 to %61 step %7 {
      %63 = arith.addi %56, %arg7 : index
      %64 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %65 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_3 = arith.constant 0 : index
      %c0_4 = arith.constant 0 : index
      %c1_5 = arith.constant 1 : index
      %c1_6 = arith.constant 1 : index
      %c2_7 = arith.constant 2 : index
      %66 = arith.index_cast %5 : i64 to index
      %67 = arith.muli %c1_5, %66 : index
      %68 = arith.muli %63, %67 : index
      %69 = arith.addi %c0_4, %68 : index
      %c1_8 = arith.constant 1 : index
      %c0_9 = arith.constant 0 : index
      %70 = arith.muli %c0_9, %c1_8 : index
      %71 = arith.addi %69, %70 : index
      %72 = arith.addi %c0_3, %71 : index
      %73 = arith.index_cast %72 : index to i32
      %74 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %65, %74, %73 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %64, %65, %25 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %9, %64 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %75 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %76 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %77 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_10 = arith.constant 0 : index
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c1_13 = arith.constant 1 : index
      %c2_14 = arith.constant 2 : index
      %78 = arith.index_cast %5 : i64 to index
      %79 = arith.muli %c1_12, %78 : index
      %80 = arith.muli %63, %79 : index
      %81 = arith.addi %c0_11, %80 : index
      %c1_15 = arith.constant 1 : index
      %c0_16 = arith.constant 0 : index
      %82 = arith.muli %c0_16, %c1_15 : index
      %83 = arith.addi %81, %82 : index
      %84 = arith.addi %c0_10, %83 : index
      %85 = arith.index_cast %84 : index to i32
      %86 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %77, %86, %85 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %76, %77, %28 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %10, %76 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %87 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %88 = ascendc.que_bind.alloc_tensor %11 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %89 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_17 = arith.constant 0 : index
      %c0_18 = arith.constant 0 : index
      %c1_19 = arith.constant 1 : index
      %c1_20 = arith.constant 1 : index
      %c2_21 = arith.constant 2 : index
      %90 = arith.index_cast %5 : i64 to index
      %91 = arith.muli %c1_19, %90 : index
      %92 = arith.muli %63, %91 : index
      %93 = arith.addi %c0_18, %92 : index
      %c1_22 = arith.constant 1 : index
      %c0_23 = arith.constant 0 : index
      %94 = arith.muli %c0_23, %c1_22 : index
      %95 = arith.addi %93, %94 : index
      %96 = arith.addi %c0_17, %95 : index
      %97 = arith.index_cast %96 : index to i32
      %98 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %89, %98, %97 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %88, %89, %31 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %11, %88 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %99 = ascendc.que_bind.deque_tensor %11 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %100 = ascendc.que_bind.alloc_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %101 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_24 = arith.constant 0 : index
      %c0_25 = arith.constant 0 : index
      %c1_26 = arith.constant 1 : index
      %c1_27 = arith.constant 1 : index
      %c2_28 = arith.constant 2 : index
      %102 = arith.index_cast %5 : i64 to index
      %103 = arith.muli %c1_26, %102 : index
      %104 = arith.muli %63, %103 : index
      %105 = arith.addi %c0_25, %104 : index
      %c1_29 = arith.constant 1 : index
      %c0_30 = arith.constant 0 : index
      %106 = arith.muli %c0_30, %c1_29 : index
      %107 = arith.addi %105, %106 : index
      %108 = arith.addi %c0_24, %107 : index
      %109 = arith.index_cast %108 : index to i32
      %110 = emitasc.reinterpret_cast %arg3 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %101, %110, %109 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %100, %101, %34 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %12, %100 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %111 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %112 = ascendc.tbuf.get_tensor %49 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.duplicate_l2 %112, %cst, %25 : !ascendc.local_tensor<*xf32>, f32, index
      %113 = ascendc.tbuf.get_tensor %48 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %113, %75, %87, %25 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %114 = ascendc.tbuf.get_tensor %47 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.mul_l2 %114, %113, %99, %25 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %115 = ascendc.tbuf.get_tensor %46 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %115, %114, %111, %25 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.add_l2 %112, %112, %115, %25 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %116 = ascendc.que_bind.alloc_tensor %13 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.reduce_sum_2d_l2 %116, %112 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.enque_tensor %13, %116 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %117 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %118 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_31 = arith.constant 0 : index
      %119 = arith.addi %c0_31, %63 : index
      %120 = arith.index_cast %119 : index to i32
      %121 = emitasc.reinterpret_cast %arg5 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %118, %121, %120 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %118, %117, %7 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.que_bind.free_tensor %13, %117 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %9, %75 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %10, %87 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %11, %99 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %12, %111 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
    }
    %62 = arith.cmpi slt, %61, %59 : index
    scf.if %62 {
      %63 = arith.subi %23, %7 : index
      ascendc.pipe.init_buffer %8, %45, %26 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %8, %14, %c1_i32, %26 : !ascendc.queue<vecin, 1>, i32, index
      %64 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %65 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_3 = arith.constant 0 : index
      %c0_4 = arith.constant 0 : index
      %c1_5 = arith.constant 1 : index
      %c1_6 = arith.constant 1 : index
      %c2_7 = arith.constant 2 : index
      %66 = arith.index_cast %5 : i64 to index
      %67 = arith.muli %c1_5, %66 : index
      %68 = arith.muli %63, %67 : index
      %69 = arith.addi %c0_4, %68 : index
      %c1_8 = arith.constant 1 : index
      %c0_9 = arith.constant 0 : index
      %70 = arith.muli %c0_9, %c1_8 : index
      %71 = arith.addi %69, %70 : index
      %72 = arith.addi %c0_3, %71 : index
      %73 = arith.index_cast %72 : index to i32
      %74 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %65, %74, %73 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %64, %65, %25 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %14, %64 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %75 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.pipe.init_buffer %8, %44, %29 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %8, %15, %c1_i32, %29 : !ascendc.queue<vecin, 1>, i32, index
      %76 = ascendc.que_bind.alloc_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %77 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_10 = arith.constant 0 : index
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c1_13 = arith.constant 1 : index
      %c2_14 = arith.constant 2 : index
      %78 = arith.index_cast %5 : i64 to index
      %79 = arith.muli %c1_12, %78 : index
      %80 = arith.muli %63, %79 : index
      %81 = arith.addi %c0_11, %80 : index
      %c1_15 = arith.constant 1 : index
      %c0_16 = arith.constant 0 : index
      %82 = arith.muli %c0_16, %c1_15 : index
      %83 = arith.addi %81, %82 : index
      %84 = arith.addi %c0_10, %83 : index
      %85 = arith.index_cast %84 : index to i32
      %86 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %77, %86, %85 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %76, %77, %28 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %15, %76 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %87 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.pipe.init_buffer %8, %43, %32 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %8, %16, %c1_i32, %32 : !ascendc.queue<vecin, 1>, i32, index
      %88 = ascendc.que_bind.alloc_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %89 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_17 = arith.constant 0 : index
      %c0_18 = arith.constant 0 : index
      %c1_19 = arith.constant 1 : index
      %c1_20 = arith.constant 1 : index
      %c2_21 = arith.constant 2 : index
      %90 = arith.index_cast %5 : i64 to index
      %91 = arith.muli %c1_19, %90 : index
      %92 = arith.muli %63, %91 : index
      %93 = arith.addi %c0_18, %92 : index
      %c1_22 = arith.constant 1 : index
      %c0_23 = arith.constant 0 : index
      %94 = arith.muli %c0_23, %c1_22 : index
      %95 = arith.addi %93, %94 : index
      %96 = arith.addi %c0_17, %95 : index
      %97 = arith.index_cast %96 : index to i32
      %98 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %89, %98, %97 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %88, %89, %31 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %16, %88 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %99 = ascendc.que_bind.deque_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.pipe.init_buffer %8, %42, %35 : !ascendc.tbuf<vecin>, index
      ascendc.pipe.init_queue %8, %17, %c1_i32, %35 : !ascendc.queue<vecin, 1>, i32, index
      %100 = ascendc.que_bind.alloc_tensor %17 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %101 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_24 = arith.constant 0 : index
      %c0_25 = arith.constant 0 : index
      %c1_26 = arith.constant 1 : index
      %c1_27 = arith.constant 1 : index
      %c2_28 = arith.constant 2 : index
      %102 = arith.index_cast %5 : i64 to index
      %103 = arith.muli %c1_26, %102 : index
      %104 = arith.muli %63, %103 : index
      %105 = arith.addi %c0_25, %104 : index
      %c1_29 = arith.constant 1 : index
      %c0_30 = arith.constant 0 : index
      %106 = arith.muli %c0_30, %c1_29 : index
      %107 = arith.addi %105, %106 : index
      %108 = arith.addi %c0_24, %107 : index
      %109 = arith.index_cast %108 : index to i32
      %110 = emitasc.reinterpret_cast %arg3 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %101, %110, %109 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %100, %101, %34 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
      ascendc.que_bind.enque_tensor %17, %100 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      %111 = ascendc.que_bind.deque_tensor %17 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.pipe.init_buffer %8, %41, %36 : !ascendc.tbuf<vecout>, index
      ascendc.pipe.init_queue %8, %18, %c1_i32, %36 : !ascendc.queue<vecout, 1>, i32, index
      ascendc.pipe.init_buffer %8, %40, %26 : !ascendc.tbuf<veccalc>, index
      %112 = ascendc.tbuf.get_tensor %40 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.duplicate_l2 %112, %cst, %25 : !ascendc.local_tensor<*xf32>, f32, index
      ascendc.pipe.init_buffer %8, %39, %26 : !ascendc.tbuf<veccalc>, index
      %113 = ascendc.tbuf.get_tensor %39 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %113, %75, %87, %25 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.pipe.init_buffer %8, %38, %26 : !ascendc.tbuf<veccalc>, index
      %114 = ascendc.tbuf.get_tensor %38 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.mul_l2 %114, %113, %99, %25 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.pipe.init_buffer %8, %37, %26 : !ascendc.tbuf<veccalc>, index
      %115 = ascendc.tbuf.get_tensor %37 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
      ascendc.add_l2 %115, %114, %111, %25 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.add_l2 %112, %112, %115, %25 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      %116 = ascendc.que_bind.alloc_tensor %18 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.reduce_sum_2d_l2 %116, %112 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.enque_tensor %18, %116 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %117 = ascendc.que_bind.deque_tensor %18 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      %118 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
      %c0_31 = arith.constant 0 : index
      %119 = arith.addi %c0_31, %63 : index
      %120 = arith.index_cast %119 : index to i32
      %121 = emitasc.reinterpret_cast %arg5 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
      ascendc.global_tensor.set_global_buffer %118, %121, %120 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
      ascendc.data_copy_l2 %118, %117, %7 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
      ascendc.que_bind.free_tensor %18, %117 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %14, %75 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %15, %87 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %16, %99 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      ascendc.que_bind.free_tensor %17, %111 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
    }
  }
  %expand_shape = memref.expand_shape %collapse_shape [[0, 1]] output_shape [%19, %20] : memref<?xf32> into memref<?x?xf32>
  return
}

