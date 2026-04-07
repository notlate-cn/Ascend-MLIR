module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_concat(%arg0: memref<?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?xf16>, %arg3: memref<?x?xf16>, %arg4: memref<?x?xf16, strided<[1, 1], offset: ?>>, %arg5: memref<ui8>, %arg6: !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 4 : i32} {
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = emitasc.member %arg6 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %1 = emitasc.member %arg6 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %2 = emitasc.member %arg6 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %3 = emitasc.member %arg6 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %4 = emitasc.member %arg6 "dim_arg2_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %5 = emitasc.member %arg6 "dim_arg3_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %6 = emitasc.member %arg6 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %7 = emitasc.member %arg6 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %8 = emitasc.member %arg6 "dim_arg2_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %9 = emitasc.member %arg6 "dim_arg3_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %10 = ascendc.pipe
    %11 = ascendc.queue : <vecin, 1>
    %12 = ascendc.queue : <vecout, 1>
    %13 = ascendc.queue : <vecin, 1>
    %14 = ascendc.queue : <vecout, 1>
    %15 = arith.index_cast %1 : i64 to index
    %16 = arith.index_cast %0 : i64 to index
    %17 = arith.index_cast %2 : i64 to index
    %18 = arith.index_cast %3 : i64 to index
    %19 = ascendc.queue : <vecin, 1>
    %20 = ascendc.tbuf : <vecin>
    %21 = ascendc.tbuf : <veccalc>
    %22 = ascendc.tbuf : <veccalc>
    %23 = ascendc.tbuf : <vecout>
    %24 = ascendc.tbuf : <vecin>
    %25 = ascendc.get_block_idx : index
    %26 = arith.muli %25, %16 : index
    %27 = arith.cmpi ult, %26, %17 : index
    scf.if %27 {
      %37 = arith.subi %17, %26 : index
      %38 = arith.minsi %16, %37 : index
      scf.for %arg7 = %c0 to %38 step %15 {
        %39 = arith.subi %38, %arg7 : index
        %40 = arith.minsi %39, %15 : index
        %41 = arith.muli %40, %c2 : index
        ascendc.pipe.init_buffer %10, %24, %41 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %10, %11, %c1_i32, %41 : !ascendc.queue<vecin, 1>, i32, index
        %42 = ascendc.que_bind.alloc_tensor %11 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %43 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %44 = arith.addi %arg7, %26 : index
        %45 = arith.index_cast %44 : index to i32
        %46 = emitasc.reinterpret_cast %arg0 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %43, %46, %45 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %42, %43, %40 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %11, %42 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %47 = ascendc.que_bind.deque_tensor %11 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %48 = arith.muli %40, %18 : index
        %49 = arith.muli %48, %c2 : index
        ascendc.pipe.init_buffer %10, %23, %49 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %10, %12, %c1_i32, %49 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %10, %22, %49 : !ascendc.tbuf<veccalc>, index
        %50 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %51 = arith.index_cast %40 : index to i32
        %52 = arith.index_cast %18 : index to i32
        ascendc.pipe.init_buffer %10, %21, %49 : !ascendc.tbuf<veccalc>, index
        %53 = ascendc.tbuf.get_tensor %21 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %53, %47, %51, %52, %51, %c1_i32 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %55 = arith.muli %44, %18 : index
        %56 = arith.index_cast %55 : index to i32
        %57 = emitasc.reinterpret_cast %arg1 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %54, %57, %56 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.pipe.init_buffer %10, %20, %49 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %10, %19, %c1_i32, %49 : !ascendc.queue<vecin, 1>, i32, index
        %58 = ascendc.que_bind.alloc_tensor %19 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %58, %54, %48 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %19, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %59 = ascendc.que_bind.deque_tensor %19 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.add_l2 %50, %53, %59, %48 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %12, %50 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %60 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %61 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %62 = emitasc.reinterpret_cast %arg4 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %61, %62, %56 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %61, %60, %48 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %12, %60 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %11, %47 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %28 = arith.index_cast %4 : i64 to index
    %29 = arith.index_cast %5 : i64 to index
    %30 = ascendc.queue : <vecin, 1>
    %31 = ascendc.tbuf : <vecin>
    %32 = ascendc.tbuf : <veccalc>
    %33 = ascendc.tbuf : <veccalc>
    %34 = ascendc.tbuf : <vecout>
    %35 = ascendc.tbuf : <vecin>
    %36 = arith.cmpi ult, %26, %28 : index
    scf.if %36 {
      %37 = arith.subi %28, %26 : index
      %38 = arith.minsi %16, %37 : index
      scf.for %arg7 = %c0 to %38 step %15 {
        %39 = arith.subi %38, %arg7 : index
        %40 = arith.minsi %39, %15 : index
        %41 = arith.muli %40, %c2 : index
        ascendc.pipe.init_buffer %10, %35, %41 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %10, %13, %c1_i32, %41 : !ascendc.queue<vecin, 1>, i32, index
        %42 = ascendc.que_bind.alloc_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %43 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %44 = arith.addi %arg7, %26 : index
        %45 = arith.index_cast %44 : index to i32
        %46 = emitasc.reinterpret_cast %arg2 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %43, %46, %45 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %42, %43, %40 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %13, %42 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %47 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %48 = arith.muli %40, %29 : index
        %49 = arith.muli %48, %c2 : index
        ascendc.pipe.init_buffer %10, %34, %49 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %10, %14, %c1_i32, %49 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %10, %33, %49 : !ascendc.tbuf<veccalc>, index
        %50 = ascendc.tbuf.get_tensor %33 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %51 = arith.index_cast %40 : index to i32
        %52 = arith.index_cast %29 : index to i32
        ascendc.pipe.init_buffer %10, %32, %49 : !ascendc.tbuf<veccalc>, index
        %53 = ascendc.tbuf.get_tensor %32 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %53, %47, %51, %52, %51, %c1_i32 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %55 = arith.muli %44, %29 : index
        %56 = arith.index_cast %55 : index to i32
        %57 = emitasc.reinterpret_cast %arg3 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %54, %57, %56 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.pipe.init_buffer %10, %31, %49 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %10, %30, %c1_i32, %49 : !ascendc.queue<vecin, 1>, i32, index
        %58 = ascendc.que_bind.alloc_tensor %30 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %58, %54, %48 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %30, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %59 = ascendc.que_bind.deque_tensor %30 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.mul_l2 %50, %53, %59, %48 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %14, %50 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %60 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %61 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %62 = arith.addi %44, %17 : index
        %63 = arith.muli %62, %18 : index
        %64 = arith.index_cast %63 : index to i32
        %65 = emitasc.reinterpret_cast %arg4 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %61, %65, %64 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %61, %60, %48 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %14, %60 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %13, %47 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return
  }
}

