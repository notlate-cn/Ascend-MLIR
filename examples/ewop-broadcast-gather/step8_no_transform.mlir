module {
  func.func @ewop_broadcast_gather(%arg0: memref<?x?xf16>, %arg1: memref<?xi32>, %arg2: memref<?xf16>, %arg3: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg0_1", "dim_arg2_0", "dim_arg2_1"]>, 22 : i32>, %arg4: memref<?x?xf16, strided<[1, 1], offset: ?>>, %arg5: memref<?x?xf16, strided<[1, 1], offset: ?>>) attributes {ascendc.aicore, ascendc.global} {
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = emitasc.copy_struct %arg3 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg0_1", "dim_arg2_0", "dim_arg2_1"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg0_1", "dim_arg2_0", "dim_arg2_1"]>
    %1 = emitasc.member %0 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg0_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %2 = emitasc.member %0 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg0_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %3 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg0_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %4 = emitasc.member %0 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg0_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %5 = emitasc.member %0 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg0_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %6 = emitasc.member %0 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg0_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %7 = emitasc.member %0 "dim_arg2_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg0_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %8 = emitasc.member %0 "dim_arg2_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg0_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %9 = ascendc.pipe
    %10 = ascendc.queue : <vecin, 1>
    %11 = ascendc.queue : <vecout, 1>
    %12 = ascendc.queue : <vecin, 1>
    %13 = ascendc.queue : <vecout, 1>
    %14 = arith.index_cast %2 : i64 to index
    %15 = arith.index_cast %1 : i64 to index
    %16 = arith.index_cast %3 : i64 to index
    %17 = arith.index_cast %4 : i64 to index
    %18 = ascendc.tbuf : <veccalc>
    %19 = ascendc.tbuf : <veccalc>
    %20 = ascendc.tbuf : <veccalc>
    %21 = ascendc.tbuf : <vecout>
    %22 = ascendc.tbuf : <vecin>
    %23 = ascendc.get_block_idx : index
    %24 = arith.muli %23, %15 : index
    %25 = arith.cmpi ult, %24, %16 : index
    scf.if %25 {
      %31 = arith.subi %16, %24 : index
      %32 = arith.minsi %15, %31 : index
      scf.for %arg6 = %c0 to %32 step %14 {
        %33 = arith.subi %32, %arg6 : index
        %34 = arith.minsi %33, %14 : index
        %35 = arith.muli %17, %c4 : index
        ascendc.pipe.init_buffer %9, %22, %35 : !ascendc.tbuf<vecin>, index
        %36 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi32>
        %37 = ascendc.global_tensor : !ascendc.global_tensor<*xi32>
        %38 = emitasc.reinterpret_cast %arg1 : memref<?xi32> to memref<?xi32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %37, %38, %c0_i32 : !ascendc.global_tensor<*xi32>, memref<?xi32, 22 : i32>, i32
        ascendc.data_copy_l2 %36, %37, %17 : !ascendc.local_tensor<*xi32>, !ascendc.global_tensor<*xi32>, index
        ascendc.que_bind.enque_tensor %10, %36 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi32>
        %39 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi32>
        %40 = arith.muli %34, %17 : index
        %41 = arith.muli %40, %c2 : index
        ascendc.pipe.init_buffer %9, %21, %41 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %9, %20, %41 : !ascendc.tbuf<veccalc>, index
        %42 = ascendc.tbuf.get_tensor %20 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %43 = arith.index_cast %34 : index to i32
        %44 = arith.index_cast %17 : index to i32
        ascendc.pipe.init_buffer %9, %19, %41 : !ascendc.tbuf<veccalc>, index
        %45 = ascendc.tbuf.get_tensor %19 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %45, %39, %43, %44, %c1_i32, %44 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xi32>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %9, %18, %41 : !ascendc.tbuf<veccalc>, index
        %46 = ascendc.tbuf.get_tensor %18 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %47 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %48 = arith.addi %arg6, %24 : index
        %49 = arith.index_cast %6 : i64 to index
        %50 = arith.muli %48, %49 : index
        %51 = arith.index_cast %50 : index to i32
        %52 = emitasc.reinterpret_cast %arg0 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %47, %52, %51 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %46, %47, %40 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %11, %42 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %53 = ascendc.que_bind.deque_tensor %11 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %55 = arith.muli %48, %17 : index
        %56 = arith.index_cast %55 : index to i32
        %57 = emitasc.reinterpret_cast %arg5 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %54, %57, %56 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %54, %53, %40 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %11, %53 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %10, %39 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi32>
      }
    }
    %26 = ascendc.tbuf : <veccalc>
    %27 = ascendc.tbuf : <veccalc>
    %28 = ascendc.tbuf : <veccalc>
    %29 = ascendc.tbuf : <vecout>
    %30 = ascendc.tbuf : <vecin>
    scf.if %25 {
      %31 = arith.subi %16, %24 : index
      %32 = arith.minsi %15, %31 : index
      scf.for %arg6 = %c0 to %32 step %14 {
        %33 = arith.subi %32, %arg6 : index
        %34 = arith.minsi %33, %14 : index
        %35 = arith.muli %34, %17 : index
        %36 = arith.muli %35, %c2 : index
        ascendc.pipe.init_buffer %9, %30, %36 : !ascendc.tbuf<vecin>, index
        %37 = ascendc.que_bind.alloc_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %38 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %39 = arith.addi %arg6, %24 : index
        %40 = arith.muli %39, %17 : index
        %41 = arith.index_cast %40 : index to i32
        %42 = emitasc.reinterpret_cast %arg5 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %38, %42, %41 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %37, %38, %35 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %12, %37 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %43 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %9, %29, %36 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %9, %28, %36 : !ascendc.tbuf<veccalc>, index
        %44 = ascendc.tbuf.get_tensor %28 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %45 = arith.muli %34, %c2 : index
        ascendc.pipe.init_buffer %9, %27, %45 : !ascendc.tbuf<veccalc>, index
        %46 = ascendc.tbuf.get_tensor %27 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %47 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %48 = arith.index_cast %39 : index to i32
        %49 = emitasc.reinterpret_cast %arg2 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %47, %49, %48 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %46, %47, %34 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %50 = arith.index_cast %34 : index to i32
        %51 = arith.index_cast %17 : index to i32
        ascendc.pipe.init_buffer %9, %26, %36 : !ascendc.tbuf<veccalc>, index
        %52 = ascendc.tbuf.get_tensor %26 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %52, %46, %50, %51, %50, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.add_l2 %44, %43, %52, %35 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %13, %44 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %53 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %55 = emitasc.reinterpret_cast %arg4 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %54, %55, %41 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %54, %53, %35 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %13, %53 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %12, %43 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return
  }
}

