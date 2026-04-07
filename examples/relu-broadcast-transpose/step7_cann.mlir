module attributes {transform.with_named_sequence} {
  func.func @relu_transpose_broadcast_add(%arg0: memref<?x1xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?x?xf16, strided<[1, 1], offset: ?>>, %arg3: memref<ui8>, %arg4: !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 2 : i32} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = emitasc.member %arg4 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %1 = emitasc.member %arg4 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %2 = emitasc.member %arg4 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %3 = emitasc.member %arg4 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %4 = emitasc.member %arg4 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %5 = emitasc.member %arg4 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %6 = ascendc.pipe
    %7 = ascendc.queue : <vecin, 1>
    %8 = ascendc.queue : <vecout, 1>
    %9 = arith.index_cast %1 : i64 to index
    %10 = arith.index_cast %0 : i64 to index
    %11 = arith.index_cast %2 : i64 to index
    %12 = arith.index_cast %3 : i64 to index
    %13 = ascendc.tbuf : <veccalc>
    %14 = ascendc.queue : <vecin, 1>
    %15 = ascendc.tbuf : <vecin>
    %16 = ascendc.tbuf : <veccalc>
    %17 = ascendc.tbuf : <veccalc>
    %18 = ascendc.tbuf : <vecout>
    %19 = ascendc.tbuf : <vecin>
    %20 = ascendc.get_block_idx : index
    %21 = arith.muli %20, %10 : index
    %22 = arith.cmpi ult, %21, %12 : index
    scf.if %22 {
      %23 = arith.subi %12, %21 : index
      %24 = arith.minsi %10, %23 : index
      scf.for %arg5 = %c0 to %24 step %9 {
        %25 = arith.subi %24, %arg5 : index
        %26 = arith.minsi %25, %9 : index
        %27 = arith.muli %11, %c2 : index
        ascendc.pipe.init_buffer %6, %19, %27 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %6, %7, %c1_i32, %27 : !ascendc.queue<vecin, 1>, i32, index
        %28 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %29 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %30 = emitasc.reinterpret_cast %arg0 : memref<?x1xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %29, %30, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %28, %29, %11 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %7, %28 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %31 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %32 = arith.muli %26, %11 : index
        %33 = arith.muli %32, %c2 : index
        ascendc.pipe.init_buffer %6, %18, %33 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %6, %8, %c1_i32, %33 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %6, %17, %33 : !ascendc.tbuf<veccalc>, index
        %34 = ascendc.tbuf.get_tensor %17 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %35 = arith.index_cast %26 : index to i32
        %36 = arith.index_cast %11 : index to i32
        ascendc.pipe.init_buffer %6, %16, %33 : !ascendc.tbuf<veccalc>, index
        %37 = ascendc.tbuf.get_tensor %16 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %37, %31, %35, %36, %c1_i32, %36 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %38 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %39 = arith.addi %arg5, %21 : index
        %40 = arith.index_cast %5 : i64 to index
        %41 = arith.muli %39, %40 : index
        %42 = arith.index_cast %41 : index to i32
        %43 = emitasc.reinterpret_cast %arg1 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %38, %43, %42 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.pipe.init_buffer %6, %15, %33 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %6, %14, %c1_i32, %33 : !ascendc.queue<vecin, 1>, i32, index
        %44 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %44, %38, %32 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %14, %44 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %45 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %6, %13, %33 : !ascendc.tbuf<veccalc>, index
        %46 = ascendc.tbuf.get_tensor %13 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %46, %cst, %32 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %34, %37, %46, %32 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %34, %34, %45, %32 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %8, %34 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %47 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %48 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %49 = arith.muli %39, %11 : index
        %50 = arith.index_cast %49 : index to i32
        %51 = emitasc.reinterpret_cast %arg2 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %48, %51, %50 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %48, %47, %32 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %8, %47 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %7, %31 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return
  }
}

