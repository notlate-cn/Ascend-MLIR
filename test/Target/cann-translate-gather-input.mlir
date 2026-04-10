module {
  func.func @relu_index_select_add(%arg0: memref<?x?xf16>, %arg1: memref<?xi64>, %arg2: memref<?xf16>, %arg3: memref<?x?xf16, strided<[1, 1], offset: ?>>, %arg4: memref<ui8>, %arg5: !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 3 : i32} {
    %cst = arith.constant 0.000000e+00 : f16
    %c8 = arith.constant 8 : index
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = emitasc.member %arg5 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %1 = emitasc.member %arg5 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %2 = emitasc.member %arg5 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %3 = emitasc.member %arg5 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %4 = emitasc.member %arg5 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %5 = emitasc.member %arg5 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %6 = emitasc.member %arg5 "dim_arg2_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %7 = emitasc.member %arg5 "dim_arg2_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %8 = ascendc.pipe
    %9 = ascendc.queue : <vecin, 1>
    %10 = ascendc.queue : <vecout, 1>
    %11 = arith.index_cast %1 : i64 to index
    %12 = arith.index_cast %0 : i64 to index
    %13 = arith.index_cast %2 : i64 to index
    %14 = arith.index_cast %3 : i64 to index
    %15 = ascendc.tbuf : <veccalc>
    %16 = ascendc.tbuf : <veccalc>
    %17 = ascendc.tbuf : <veccalc>
    %18 = ascendc.tbuf : <vecout>
    %19 = ascendc.tbuf : <vecin>
    %20 = ascendc.get_block_idx : index
    %21 = arith.muli %20, %12 : index
    %22 = arith.cmpi ult, %21, %13 : index
    scf.if %22 {
      %23 = arith.subi %13, %21 : index
      %24 = arith.minsi %12, %23 : index
      scf.for %arg6 = %c0 to %24 step %11 {
        %25 = arith.subi %24, %arg6 : index
        %26 = arith.minsi %25, %11 : index
        %27 = arith.muli %14, %c8 : index
        ascendc.pipe.init_buffer %8, %19, %27 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %8, %9, %c1_i32, %27 : !ascendc.queue<vecin, 1>, i32, index
        %28 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %29 = ascendc.global_tensor : !ascendc.global_tensor<*xi64>
        %30 = emitasc.reinterpret_cast %arg1 : memref<?xi64> to memref<?xi64, 22 : i32>
        ascendc.global_tensor.set_global_buffer %29, %30, %c0_i32 : !ascendc.global_tensor<*xi64>, memref<?xi64, 22 : i32>, i32
        ascendc.data_copy_l2 %28, %29, %14 : !ascendc.local_tensor<*xi64>, !ascendc.global_tensor<*xi64>, index
        ascendc.que_bind.enque_tensor %9, %28 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %31 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %32 = arith.muli %26, %14 : index
        %33 = arith.muli %32, %c2 : index
        ascendc.pipe.init_buffer %8, %18, %33 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %8, %10, %c1_i32, %33 : !ascendc.queue<vecout, 1>, i32, index
        %34 = arith.index_cast %4 : i64 to index
        %35 = arith.index_cast %14 : index to i32
        %36 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %37 = arith.muli %14, %c2 : index
        %38 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %38, %arg0 : !ascendc.global_tensor<*xf16>, memref<?x?xf16>
        %39 = arith.muli %34, %c2 : index
        ascendc.pipe.init_buffer %8, %17, %39 : !ascendc.tbuf<veccalc>, index
        %40 = ascendc.tbuf.get_tensor %17 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        scf.for %arg7 = %c0 to %26 step %c1 {
          %47 = arith.addi %arg7, %arg6 : index
          %48 = arith.addi %47, %21 : index
          %49 = arith.muli %48, %34 : index
          %50 = ascendc.global_tensor.bracket %38(%49) : !ascendc.global_tensor<*xf16>, index, !ascendc.global_tensor<*xf16>
          %51 = ascendc.tbuf.get_tensor %17 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
          ascendc.data_copy_l2 %51, %50, %34 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
          %52 = arith.muli %arg7, %37 : index
          %53 = ascendc.tbuf.get_with_offset %18, %37, %52 : !ascendc.tbuf<vecout>, index, index, !ascendc.local_tensor<*xf16>
          ascendc.gather_l2 %53, %51, %31, %c0_i32, %35 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xi64>, i32, i32
          ascendc.pipe.init_buffer %8, %16, %37 : !ascendc.tbuf<veccalc>, index
          %54 = ascendc.tbuf.get_tensor %16 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
          ascendc.duplicate_l2 %54, %cst, %35 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, f16, i32
          ascendc.max_l2 %53, %53, %54, %35 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
          ascendc.pipe.init_buffer %8, %15, %37 : !ascendc.tbuf<veccalc>, index
          %55 = ascendc.tbuf.get_tensor %15 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
          %56 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
          %57 = emitasc.reinterpret_cast %arg2 : memref<?xf16> to memref<?xf16, 22 : i32>
          ascendc.global_tensor.set_global_buffer %56, %57, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
          ascendc.data_copy_l2 %55, %56, %14 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
          ascendc.add_l2 %53, %53, %55, %35 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
        }
        ascendc.que_bind.enque_tensor %10, %36 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %41 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %42 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %43 = arith.addi %arg6, %21 : index
        %44 = arith.muli %43, %14 : index
        %45 = arith.index_cast %44 : index to i32
        %46 = emitasc.reinterpret_cast %arg3 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %42, %46, %45 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %42, %41, %32 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %10, %41 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %9, %31 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
      }
    }
    return
  }
}
