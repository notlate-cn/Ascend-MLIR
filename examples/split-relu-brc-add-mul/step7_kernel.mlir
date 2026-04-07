module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_split(%arg0: memref<?x?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, 22 : i32>, %arg6: memref<?x?xf16, strided<[1, 1], offset: ?>>) attributes {ascendc.aicore, ascendc.global} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = emitasc.copy_struct %arg5 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>
    %1 = emitasc.member %0 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %2 = emitasc.member %0 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %3 = emitasc.member %0 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %4 = emitasc.member %0 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %5 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %6 = emitasc.member %0 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %7 = emitasc.member %0 "dim_arg3_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %8 = emitasc.member %0 "dim_arg3_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %9 = emitasc.member %0 "dim_arg2_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %10 = emitasc.member %0 "dim_arg2_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %11 = emitasc.member %0 "dim_arg4_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %12 = emitasc.member %0 "dim_arg4_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %13 = ascendc.pipe
    %14 = ascendc.queue : <vecin, 1>
    %15 = ascendc.queue : <vecout, 1>
    %16 = ascendc.queue : <vecin, 1>
    %17 = ascendc.queue : <vecout, 1>
    %18 = arith.index_cast %2 : i64 to index
    %19 = arith.index_cast %1 : i64 to index
    %20 = arith.index_cast %3 : i64 to index
    %21 = arith.index_cast %4 : i64 to index
    %22 = ascendc.tbuf : <veccalc>
    %23 = ascendc.tbuf : <veccalc>
    %24 = ascendc.queue : <vecin, 1>
    %25 = ascendc.tbuf : <vecin>
    %26 = ascendc.tbuf : <veccalc>
    %27 = ascendc.queue : <vecin, 1>
    %28 = ascendc.tbuf : <vecin>
    %29 = ascendc.tbuf : <veccalc>
    %30 = ascendc.tbuf : <vecout>
    %31 = ascendc.tbuf : <vecin>
    %32 = ascendc.get_block_idx : index
    %33 = arith.muli %32, %19 : index
    %34 = arith.cmpi ult, %33, %21 : index
    scf.if %34 {
      %45 = arith.subi %21, %33 : index
      %46 = arith.minsi %19, %45 : index
      scf.for %arg7 = %c0 to %46 step %18 {
        %47 = arith.subi %46, %arg7 : index
        %48 = arith.minsi %47, %18 : index
        %49 = arith.muli %48, %20 : index
        %50 = arith.muli %49, %c2 : index
        ascendc.pipe.init_buffer %13, %31, %50 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %13, %14, %c1_i32, %50 : !ascendc.queue<vecin, 1>, i32, index
        %51 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %52 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %53 = arith.addi %arg7, %33 : index
        %54 = arith.muli %53, %20 : index
        %55 = arith.index_cast %54 : index to i32
        %56 = emitasc.reinterpret_cast %arg0 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %52, %56, %55 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %51, %52, %49 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %14, %51 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %57 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %13, %30, %50 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %13, %15, %c1_i32, %50 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %13, %29, %50 : !ascendc.tbuf<veccalc>, index
        %58 = ascendc.tbuf.get_tensor %29 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %59 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %60 = arith.index_cast %53 : index to i32
        %61 = emitasc.reinterpret_cast %arg1 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %59, %61, %60 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        %62 = arith.muli %48, %c2 : index
        ascendc.pipe.init_buffer %13, %28, %62 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %13, %27, %c1_i32, %62 : !ascendc.queue<vecin, 1>, i32, index
        %63 = ascendc.que_bind.alloc_tensor %27 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %63, %59, %48 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %27, %63 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %64 = ascendc.que_bind.deque_tensor %27 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %65 = arith.index_cast %48 : index to i32
        %66 = arith.index_cast %20 : index to i32
        ascendc.pipe.init_buffer %13, %26, %50 : !ascendc.tbuf<veccalc>, index
        %67 = ascendc.tbuf.get_tensor %26 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %67, %64, %65, %66, %65, %c1_i32 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %68 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %69 = emitasc.reinterpret_cast %arg3 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %68, %69, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        %70 = arith.muli %20, %c2 : index
        ascendc.pipe.init_buffer %13, %25, %70 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %13, %24, %c1_i32, %70 : !ascendc.queue<vecin, 1>, i32, index
        %71 = ascendc.que_bind.alloc_tensor %24 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %71, %68, %20 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %24, %71 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %72 = ascendc.que_bind.deque_tensor %24 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %13, %23, %50 : !ascendc.tbuf<veccalc>, index
        %73 = ascendc.tbuf.get_tensor %23 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %73, %72, %65, %66, %c1_i32, %66 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %13, %22, %50 : !ascendc.tbuf<veccalc>, index
        %74 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %74, %cst, %49 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %58, %57, %74, %49 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %58, %58, %67, %49 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.mul_l2 %58, %58, %73, %49 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %15, %58 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %75 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %76 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %77 = emitasc.reinterpret_cast %arg6 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %76, %77, %55 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %76, %75, %49 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %15, %75 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %14, %57 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %35 = ascendc.tbuf : <veccalc>
    %36 = ascendc.tbuf : <veccalc>
    %37 = ascendc.queue : <vecin, 1>
    %38 = ascendc.tbuf : <vecin>
    %39 = ascendc.tbuf : <veccalc>
    %40 = ascendc.queue : <vecin, 1>
    %41 = ascendc.tbuf : <vecin>
    %42 = ascendc.tbuf : <veccalc>
    %43 = ascendc.tbuf : <vecout>
    %44 = ascendc.tbuf : <vecin>
    scf.if %34 {
      %45 = arith.subi %21, %33 : index
      %46 = arith.minsi %19, %45 : index
      scf.for %arg7 = %c0 to %46 step %18 {
        %47 = arith.subi %46, %arg7 : index
        %48 = arith.minsi %47, %18 : index
        %49 = arith.muli %48, %20 : index
        %50 = arith.muli %49, %c2 : index
        ascendc.pipe.init_buffer %13, %44, %50 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %13, %16, %c1_i32, %50 : !ascendc.queue<vecin, 1>, i32, index
        %51 = ascendc.que_bind.alloc_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %52 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %53 = arith.addi %arg7, %33 : index
        %54 = arith.addi %53, %21 : index
        %55 = arith.muli %54, %20 : index
        %56 = arith.index_cast %55 : index to i32
        %57 = emitasc.reinterpret_cast %arg0 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %52, %57, %56 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %51, %52, %49 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %16, %51 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %58 = ascendc.que_bind.deque_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %13, %43, %50 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %13, %17, %c1_i32, %50 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %13, %42, %50 : !ascendc.tbuf<veccalc>, index
        %59 = ascendc.tbuf.get_tensor %42 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %60 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %61 = arith.index_cast %53 : index to i32
        %62 = emitasc.reinterpret_cast %arg2 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %60, %62, %61 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        %63 = arith.muli %48, %c2 : index
        ascendc.pipe.init_buffer %13, %41, %63 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %13, %40, %c1_i32, %63 : !ascendc.queue<vecin, 1>, i32, index
        %64 = ascendc.que_bind.alloc_tensor %40 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %64, %60, %48 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %40, %64 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %65 = ascendc.que_bind.deque_tensor %40 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %66 = arith.index_cast %48 : index to i32
        %67 = arith.index_cast %20 : index to i32
        ascendc.pipe.init_buffer %13, %39, %50 : !ascendc.tbuf<veccalc>, index
        %68 = ascendc.tbuf.get_tensor %39 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %68, %65, %66, %67, %66, %c1_i32 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %69 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %70 = emitasc.reinterpret_cast %arg4 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %69, %70, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        %71 = arith.muli %20, %c2 : index
        ascendc.pipe.init_buffer %13, %38, %71 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %13, %37, %c1_i32, %71 : !ascendc.queue<vecin, 1>, i32, index
        %72 = ascendc.que_bind.alloc_tensor %37 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %72, %69, %20 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %37, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %73 = ascendc.que_bind.deque_tensor %37 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %13, %36, %50 : !ascendc.tbuf<veccalc>, index
        %74 = ascendc.tbuf.get_tensor %36 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %74, %73, %66, %67, %c1_i32, %67 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %13, %35, %50 : !ascendc.tbuf<veccalc>, index
        %75 = ascendc.tbuf.get_tensor %35 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %75, %cst, %49 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %59, %58, %75, %49 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %59, %59, %68, %49 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.mul_l2 %59, %59, %74, %49 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %17, %59 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %76 = ascendc.que_bind.deque_tensor %17 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %77 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %78 = emitasc.reinterpret_cast %arg6 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %77, %78, %56 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %77, %76, %49 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %17, %76 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %16, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.tensor.decompose_concat
    } : !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.param.constant true -> !transform.any_param
    %3 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %4 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %5 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.foreach %1 : !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      %tiled_linalg_op, %loops = transform.structured.tile_using_for %arg1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.annotate %loops "ascendc.parallel" = %2 : !transform.any_op, !transform.any_param
      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.annotate %loops_1 "ascendc.prologue" = %3 : !transform.any_op, !transform.any_param
      transform.annotate %loops_1 "ascendc.epilogue" = %4 : !transform.any_op, !transform.any_param
      transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %5 : !transform.any_op, !transform.any_param
      transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    }
    transform.yield 
  }
}

