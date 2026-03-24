module attributes {transform.with_named_sequence} {
  func.func @broadcast_add_reducesum(%arg0: memref<?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?xf16, strided<[1], offset: ?>>, %arg3: memref<ui8>, %arg4: !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 2 : i32} {
    %cst = arith.constant 0.000000e+00 : f16
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = emitasc.member %arg4 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>, i64
    %1 = emitasc.member %arg4 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>, i64
    %2 = emitasc.member %arg4 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>, i64
    %3 = emitasc.member %arg4 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>, i64
    %4 = emitasc.member %arg4 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>, i64
    %5 = emitasc.member %arg4 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>, i64
    %6 = ascendc.pipe
    %7 = ascendc.queue : <vecin, 1>
    %8 = ascendc.queue : <vecout, 1>
    %9 = arith.index_cast %1 : i64 to index
    %10 = arith.index_cast %0 : i64 to index
    %11 = arith.index_cast %2 : i64 to index
    %12 = arith.index_cast %3 : i64 to index
    %13 = ascendc.tbuf : <veccalc>
    %14 = ascendc.tbuf : <veccalc>
    %15 = ascendc.tbuf : <veccalc>
    %16 = ascendc.tbuf : <veccalc>
    %17 = ascendc.tbuf : <vecout>
    %18 = ascendc.tbuf : <vecin>
    %19 = ascendc.get_block_idx : index
    %20 = arith.muli %19, %10 : index
    %21 = arith.cmpi ult, %20, %11 : index
    scf.if %21 {
      %22 = arith.subi %11, %20 : index
      %23 = arith.minsi %10, %22 : index
      scf.for %arg5 = %c0 to %23 step %9 {
        %24 = arith.subi %23, %arg5 : index
        %25 = arith.minsi %24, %9 : index
        %26 = arith.muli %25, %c2 : index
        ascendc.pipe.init_buffer %6, %18, %26 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %6, %7, %c1_i32, %26 : !ascendc.queue<vecin, 1>, i32, index
        %27 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %28 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %29 = arith.addi %arg5, %20 : index
        %30 = arith.index_cast %29 : index to i32
        %31 = emitasc.reinterpret_cast %arg0 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %28, %31, %30 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %27, %28, %25 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %7, %27 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %32 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %6, %17, %26 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %6, %8, %c1_i32, %26 : !ascendc.queue<vecout, 1>, i32, index
        %33 = arith.muli %25, %12 : index
        %34 = arith.muli %33, %c2 : index
        ascendc.pipe.init_buffer %6, %16, %34 : !ascendc.tbuf<veccalc>, index
        %35 = ascendc.tbuf.get_tensor %16 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %35, %cst, %33 : !ascendc.local_tensor<*xf16>, f16, index
        %36 = arith.index_cast %25 : index to i32
        %37 = arith.index_cast %12 : index to i32
        ascendc.pipe.init_buffer %6, %15, %34 : !ascendc.tbuf<veccalc>, index
        %38 = ascendc.tbuf.get_tensor %15 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %38, %32, %36, %37, %36, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %6, %14, %34 : !ascendc.tbuf<veccalc>, index
        %39 = ascendc.tbuf.get_tensor %14 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %40 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %41 = arith.muli %29, %12 : index
        %42 = arith.index_cast %41 : index to i32
        %43 = emitasc.reinterpret_cast %arg1 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %40, %43, %42 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %39, %40, %33 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.pipe.init_buffer %6, %13, %34 : !ascendc.tbuf<veccalc>, index
        %44 = ascendc.tbuf.get_tensor %13 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.add_l2 %44, %38, %39, %33 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %35, %35, %44, %33 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        %45 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.reduce_sum_2d_l2 %45, %35 {layout = 0 : i32} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.enque_tensor %8, %45 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %46 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %47 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %48 = emitasc.reinterpret_cast %arg2 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %47, %48, %30 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %47, %46, %25 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %8, %46 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %7, %32 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return
  }
}

