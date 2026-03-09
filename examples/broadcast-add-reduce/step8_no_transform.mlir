module {
  func.func @broadcast_add_reducesum(%arg0: memref<?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, 22 : i32>, %arg3: memref<?xf16, strided<[1], offset: ?>>) attributes {ascendc.aicore, ascendc.global} {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = emitasc.copy_struct %arg2 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>
    %1 = emitasc.member %0 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, i64
    %2 = emitasc.member %0 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, i64
    %3 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, i64
    %4 = emitasc.member %0 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, i64
    %5 = ascendc.pipe
    %6 = ascendc.queue : <vecin, 1>
    %7 = ascendc.queue : <vecout, 1>
    %8 = arith.index_cast %2 : i64 to index
    %9 = arith.index_cast %1 : i64 to index
    %10 = arith.index_cast %3 : i64 to index
    %11 = arith.index_cast %4 : i64 to index
    %12 = ascendc.tbuf : <veccalc>
    %13 = ascendc.tbuf : <veccalc>
    %14 = ascendc.tbuf : <veccalc>
    %15 = ascendc.tbuf : <vecout>
    %16 = ascendc.tbuf : <vecin>
    %17 = ascendc.get_block_idx : index
    %18 = arith.muli %17, %9 : index
    %19 = arith.cmpi ult, %18, %10 : index
    scf.if %19 {
      %20 = arith.subi %10, %18 : index
      %21 = arith.minsi %9, %20 : index
      scf.for %arg4 = %c0 to %21 step %8 {
        %22 = arith.subi %21, %arg4 : index
        %23 = arith.minsi %22, %8 : index
        %24 = arith.muli %23, %c2 : index
        ascendc.pipe.init_buffer %5, %16, %24 : !ascendc.tbuf<vecin>, index
        %25 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %26 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %27 = arith.addi %arg4, %18 : index
        %28 = arith.index_cast %27 : index to i32
        %29 = emitasc.reinterpret_cast %arg0 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %26, %29, %28 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %25, %26, %23 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %6, %25 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %30 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %5, %15, %24 : !ascendc.tbuf<vecout>, index
        %31 = arith.muli %23, %11 : index
        %32 = arith.muli %31, %c2 : index
        ascendc.pipe.init_buffer %5, %14, %32 : !ascendc.tbuf<veccalc>, index
        %33 = ascendc.tbuf.get_tensor %14 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %34 = arith.index_cast %23 : index to i32
        %35 = arith.index_cast %11 : index to i32
        ascendc.pipe.init_buffer %5, %13, %32 : !ascendc.tbuf<veccalc>, index
        %36 = ascendc.tbuf.get_tensor %13 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %36, %30, %34, %35, %34, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %5, %12, %32 : !ascendc.tbuf<veccalc>, index
        %37 = ascendc.tbuf.get_tensor %12 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %38 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %39 = arith.muli %27, %11 : index
        %40 = arith.index_cast %39 : index to i32
        %41 = emitasc.reinterpret_cast %arg1 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %38, %41, %40 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %37, %38, %31 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.add_l2 %33, %36, %37, %31 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %33, %33, %33, %31 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        %42 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.reduce_sum_2d_l2 %42, %33 {layout = 0 : i32} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.enque_tensor %7, %42 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %43 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %44 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %45 = emitasc.reinterpret_cast %arg3 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %44, %45, %28 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %44, %43, %23 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %7, %43 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %6, %30 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return
  }
}

