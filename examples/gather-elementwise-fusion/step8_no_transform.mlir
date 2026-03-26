module {
  func.func @relu_index_select_add(%arg0: memref<?x?xf16>, %arg1: memref<?xi64>, %arg2: memref<?xf16>, %arg3: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, 22 : i32>, %arg4: memref<?x?xf16, strided<[1, 1], offset: ?>>) attributes {ascendc.aicore, ascendc.global} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %0 = emitasc.copy_struct %arg3 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>
    %1 = emitasc.member %0 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %2 = emitasc.member %0 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %3 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %4 = emitasc.member %0 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %5 = emitasc.member %0 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %6 = emitasc.member %0 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %7 = ascendc.pipe
    %8 = ascendc.queue : <vecin, 1>
    %9 = ascendc.queue : <vecout, 1>
    %10 = arith.index_cast %2 : i64 to index
    %11 = arith.index_cast %1 : i64 to index
    %12 = arith.index_cast %3 : i64 to index
    %13 = arith.index_cast %4 : i64 to index
    %14 = ascendc.tbuf : <gm>
    %15 = ascendc.tbuf : <veccalc>
    %16 = ascendc.tbuf : <veccalc>
    %17 = ascendc.tbuf : <vecout>
    %18 = ascendc.tbuf : <vecin>
    %19 = ascendc.get_block_idx : index
    %20 = arith.muli %19, %11 : index
    %21 = arith.cmpi ult, %20, %12 : index
    scf.if %21 {
      %22 = arith.subi %12, %20 : index
      %23 = arith.minsi %11, %22 : index
      scf.for %arg5 = %c0 to %23 step %10 {
        %24 = arith.subi %23, %arg5 : index
        %25 = arith.minsi %24, %10 : index
        %26 = arith.muli %13, %c8 : index
        ascendc.pipe.init_buffer %7, %18, %26 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %7, %8, %c1_i32, %26 : !ascendc.queue<vecin, 1>, i32, index
        %27 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %28 = ascendc.global_tensor : !ascendc.global_tensor<*xi64>
        %29 = emitasc.reinterpret_cast %arg1 : memref<?xi64> to memref<?xi64, 22 : i32>
        ascendc.global_tensor.set_global_buffer %28, %29, %c0_i32 : !ascendc.global_tensor<*xi64>, memref<?xi64, 22 : i32>, i32
        ascendc.data_copy_l2 %27, %28, %13 : !ascendc.local_tensor<*xi64>, !ascendc.global_tensor<*xi64>, index
        ascendc.que_bind.enque_tensor %8, %27 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %30 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %31 = arith.muli %25, %13 : index
        %32 = arith.muli %31, %c2 : index
        ascendc.pipe.init_buffer %7, %17, %32 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %7, %9, %c1_i32, %32 : !ascendc.queue<vecout, 1>, i32, index
        %33 = arith.index_cast %5 : i64 to index
        %34 = arith.index_cast %13 : index to i32
        %35 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %36 = arith.muli %13, %c2 : index
        %37 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %37, %arg0 : !ascendc.global_tensor<*xf16>, memref<?x?xf16>
        %38 = arith.muli %33, %c2 : index
        ascendc.pipe.init_buffer %7, %16, %38 : !ascendc.tbuf<veccalc>, index
        %39 = ascendc.tbuf.get_tensor %16 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        scf.for %arg6 = %c0 to %25 step %c1 {
          %46 = arith.addi %arg6, %arg5 : index
          %47 = arith.addi %46, %20 : index
          %48 = arith.muli %47, %33 : index
          %49 = ascendc.global_tensor.bracket %37(%48) : !ascendc.global_tensor<*xf16>, index, !ascendc.global_tensor<*xf16>
          %50 = ascendc.tbuf.get_tensor %16 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
          ascendc.data_copy_l2 %50, %49, %33 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
          %51 = arith.muli %arg6, %36 : index
          %52 = ascendc.tbuf.get_with_offset %17, %36, %51 : !ascendc.tbuf<vecout>, index, index, !ascendc.local_tensor<*xf16>
          ascendc.gather_l2 %52, %50, %30, %c0_i32, %34 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xi64>, i32, i32
          ascendc.pipe.init_buffer %7, %15, %36 : !ascendc.tbuf<veccalc>, index
          %53 = ascendc.tbuf.get_tensor %15 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
          ascendc.duplicate_l2 %53, %cst, %34 : !ascendc.local_tensor<*xf16>, f16, i32
          ascendc.max_l2 %52, %52, %53, %34 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
          %54 = ascendc.tbuf.get_tensor %14 : !ascendc.tbuf<gm>, !ascendc.local_tensor<*xf16>
          ascendc.add_l2 %52, %52, %54, %34 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
        }
        ascendc.que_bind.enque_tensor %9, %35 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %40 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %41 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %42 = arith.addi %arg5, %20 : index
        %43 = arith.muli %42, %13 : index
        %44 = arith.index_cast %43 : index to i32
        %45 = emitasc.reinterpret_cast %arg4 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %41, %45, %44 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %41, %40, %31 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %9, %40 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %8, %30 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
      }
    }
    return
  }
}

