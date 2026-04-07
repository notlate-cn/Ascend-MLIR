module {
  func.func @relu_index_select_add(%arg0: memref<?x?xf16>, %arg1: memref<?xi64>, %arg2: memref<?xf16>, %arg3: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, 22 : i32>, %arg4: memref<?x?xf16, strided<[1, 1], offset: ?>>) attributes {ascendc.aicore, ascendc.global} {
    %cst = arith.constant 0.000000e+00 : f16
    %c8 = arith.constant 8 : index
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = emitasc.copy_struct %arg3 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>
    %1 = emitasc.member %0 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %2 = emitasc.member %0 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %3 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %4 = emitasc.member %0 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %5 = emitasc.member %0 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %6 = emitasc.member %0 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %7 = emitasc.member %0 "dim_arg2_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %8 = emitasc.member %0 "dim_arg2_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %9 = ascendc.pipe
    %10 = ascendc.queue : <vecin, 1>
    %11 = ascendc.queue : <vecout, 1>
    %12 = arith.index_cast %2 : i64 to index
    %13 = arith.index_cast %1 : i64 to index
    %14 = arith.index_cast %3 : i64 to index
    %15 = arith.index_cast %4 : i64 to index
    %16 = ascendc.tbuf : <veccalc>
    %17 = ascendc.tbuf : <veccalc>
    %18 = ascendc.tbuf : <veccalc>
    %19 = ascendc.tbuf : <vecout>
    %20 = ascendc.tbuf : <vecin>
    %21 = ascendc.get_block_idx : index
    %22 = arith.muli %21, %13 : index
    %23 = arith.cmpi ult, %22, %14 : index
    scf.if %23 {
      %24 = arith.subi %14, %22 : index
      %25 = arith.minsi %13, %24 : index
      scf.for %arg5 = %c0 to %25 step %12 {
        %26 = arith.subi %25, %arg5 : index
        %27 = arith.minsi %26, %12 : index
        %28 = arith.muli %15, %c8 : index
        ascendc.pipe.init_buffer %9, %20, %28 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %9, %10, %c1_i32, %28 : !ascendc.queue<vecin, 1>, i32, index
        %29 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %30 = ascendc.global_tensor : !ascendc.global_tensor<*xi64>
        %31 = emitasc.reinterpret_cast %arg1 : memref<?xi64> to memref<?xi64, 22 : i32>
        ascendc.global_tensor.set_global_buffer %30, %31, %c0_i32 : !ascendc.global_tensor<*xi64>, memref<?xi64, 22 : i32>, i32
        ascendc.data_copy_l2 %29, %30, %15 : !ascendc.local_tensor<*xi64>, !ascendc.global_tensor<*xi64>, index
        ascendc.que_bind.enque_tensor %10, %29 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %32 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %33 = arith.muli %27, %15 : index
        %34 = arith.muli %33, %c2 : index
        ascendc.pipe.init_buffer %9, %19, %34 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %9, %11, %c1_i32, %34 : !ascendc.queue<vecout, 1>, i32, index
        %35 = arith.index_cast %5 : i64 to index
        %36 = arith.index_cast %15 : index to i32
        %37 = ascendc.que_bind.alloc_tensor %11 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %38 = arith.muli %15, %c2 : index
        %39 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %39, %arg0 : !ascendc.global_tensor<*xf16>, memref<?x?xf16>
        %40 = arith.muli %35, %c2 : index
        ascendc.pipe.init_buffer %9, %18, %40 : !ascendc.tbuf<veccalc>, index
        %41 = ascendc.tbuf.get_tensor %18 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        scf.for %arg6 = %c0 to %27 step %c1 {
          %48 = arith.addi %arg6, %arg5 : index
          %49 = arith.addi %48, %22 : index
          %50 = arith.muli %49, %35 : index
          %51 = ascendc.global_tensor.bracket %39(%50) : !ascendc.global_tensor<*xf16>, index, !ascendc.global_tensor<*xf16>
          %52 = ascendc.tbuf.get_tensor %18 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
          ascendc.data_copy_l2 %52, %51, %35 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
          %53 = arith.muli %arg6, %38 : index
          %54 = ascendc.tbuf.get_with_offset %19, %38, %53 : !ascendc.tbuf<vecout>, index, index, !ascendc.local_tensor<*xf16>
          ascendc.gather_l2 %54, %52, %32, %c0_i32, %36 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xi64>, i32, i32
          ascendc.pipe.init_buffer %9, %17, %38 : !ascendc.tbuf<veccalc>, index
          %55 = ascendc.tbuf.get_tensor %17 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
          ascendc.duplicate_l2 %55, %cst, %36 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, f16, i32
          ascendc.max_l2 %54, %54, %55, %36 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
          ascendc.pipe.init_buffer %9, %16, %38 : !ascendc.tbuf<veccalc>, index
          %56 = ascendc.tbuf.get_tensor %16 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
          %57 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
          %58 = emitasc.reinterpret_cast %arg2 : memref<?xf16> to memref<?xf16, 22 : i32>
          ascendc.global_tensor.set_global_buffer %57, %58, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
          ascendc.data_copy_l2 %56, %57, %15 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
          ascendc.add_l2 %54, %54, %56, %36 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
        }
        ascendc.que_bind.enque_tensor %11, %37 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %42 = ascendc.que_bind.deque_tensor %11 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %43 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %44 = arith.addi %arg5, %22 : index
        %45 = arith.muli %44, %15 : index
        %46 = arith.index_cast %45 : index to i32
        %47 = emitasc.reinterpret_cast %arg4 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %43, %47, %46 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %43, %42, %33 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %11, %42 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %10, %32 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
      }
    }
    return
  }
}

