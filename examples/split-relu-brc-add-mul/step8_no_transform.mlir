module {
  func.func @ewop_broadcast_split(%arg0: memref<?x?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg2_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, 22 : i32>, %arg5: memref<?x?xf16, strided<[1, 1], offset: ?>>, %arg6: memref<?x?xf16, strided<[1, 1], offset: ?>>) attributes {ascendc.aicore, ascendc.global} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = emitasc.copy_struct %arg4 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg2_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg2_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>
    %1 = emitasc.member %0 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg2_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %2 = emitasc.member %0 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg2_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %3 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg2_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %4 = emitasc.member %0 "dim_arg2_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg2_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %5 = emitasc.member %0 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg2_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %6 = emitasc.member %0 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg2_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %7 = emitasc.member %0 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg2_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %8 = emitasc.member %0 "dim_arg2_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg2_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %9 = emitasc.member %0 "dim_arg3_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg2_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %10 = emitasc.member %0 "dim_arg3_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg2_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %11 = ascendc.pipe
    %12 = ascendc.queue : <vecin, 1>
    %13 = ascendc.queue : <vecout, 1>
    %14 = ascendc.queue : <vecin, 1>
    %15 = ascendc.queue : <vecout, 1>
    %16 = arith.index_cast %2 : i64 to index
    %17 = arith.index_cast %1 : i64 to index
    %18 = arith.index_cast %3 : i64 to index
    %19 = arith.index_cast %4 : i64 to index
    %20 = ascendc.tbuf : <veccalc>
    %21 = ascendc.tbuf : <veccalc>
    %22 = ascendc.tbuf : <veccalc>
    %23 = ascendc.tbuf : <veccalc>
    %24 = ascendc.tbuf : <veccalc>
    %25 = ascendc.tbuf : <veccalc>
    %26 = ascendc.tbuf : <vecout>
    %27 = ascendc.tbuf : <vecin>
    %28 = ascendc.get_block_idx : index
    %29 = arith.muli %28, %17 : index
    %30 = arith.cmpi ult, %29, %18 : index
    scf.if %30 {
      %39 = arith.subi %18, %29 : index
      %40 = arith.minsi %17, %39 : index
      scf.for %arg7 = %c0 to %40 step %16 {
        %41 = arith.subi %40, %arg7 : index
        %42 = arith.minsi %41, %16 : index
        %43 = arith.muli %42, %19 : index
        %44 = arith.muli %43, %c2 : index
        ascendc.pipe.init_buffer %11, %27, %44 : !ascendc.tbuf<vecin>, index
        %45 = ascendc.que_bind.alloc_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %46 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %47 = arith.addi %arg7, %29 : index
        %48 = arith.index_cast %5 : i64 to index
        %49 = arith.muli %47, %48 : index
        %50 = arith.index_cast %49 : index to i32
        %51 = emitasc.reinterpret_cast %arg0 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %46, %51, %50 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %45, %46, %43 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %12, %45 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %52 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %11, %26, %44 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %11, %25, %44 : !ascendc.tbuf<veccalc>, index
        %53 = ascendc.tbuf.get_tensor %25 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %54 = arith.muli %42, %c2 : index
        ascendc.pipe.init_buffer %11, %24, %54 : !ascendc.tbuf<veccalc>, index
        %55 = ascendc.tbuf.get_tensor %24 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %56 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %57 = arith.index_cast %47 : index to i32
        %58 = emitasc.reinterpret_cast %arg1 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %56, %58, %57 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %55, %56, %42 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %59 = arith.index_cast %42 : index to i32
        %60 = arith.index_cast %19 : index to i32
        ascendc.pipe.init_buffer %11, %23, %44 : !ascendc.tbuf<veccalc>, index
        %61 = ascendc.tbuf.get_tensor %23 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %61, %55, %59, %60, %59, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %62 = arith.muli %19, %c2 : index
        ascendc.pipe.init_buffer %11, %22, %62 : !ascendc.tbuf<veccalc>, index
        %63 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %64 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %65 = emitasc.reinterpret_cast %arg2 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %64, %65, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %63, %64, %19 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.pipe.init_buffer %11, %21, %44 : !ascendc.tbuf<veccalc>, index
        %66 = ascendc.tbuf.get_tensor %21 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %66, %63, %59, %60, %c1_i32, %60 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %11, %20, %44 : !ascendc.tbuf<veccalc>, index
        %67 = ascendc.tbuf.get_tensor %20 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %67, %cst, %43 : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %53, %52, %67, %43 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %53, %53, %61, %43 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.mul_l2 %53, %53, %66, %43 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %13, %53 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %68 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %69 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %70 = arith.muli %47, %19 : index
        %71 = arith.index_cast %70 : index to i32
        %72 = emitasc.reinterpret_cast %arg6 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %69, %72, %71 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %69, %68, %43 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %13, %68 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %12, %52 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %31 = ascendc.tbuf : <veccalc>
    %32 = ascendc.tbuf : <veccalc>
    %33 = ascendc.tbuf : <veccalc>
    %34 = ascendc.tbuf : <veccalc>
    %35 = ascendc.tbuf : <veccalc>
    %36 = ascendc.tbuf : <veccalc>
    %37 = ascendc.tbuf : <vecout>
    %38 = ascendc.tbuf : <vecin>
    scf.if %30 {
      %39 = arith.subi %18, %29 : index
      %40 = arith.minsi %17, %39 : index
      scf.for %arg7 = %c0 to %40 step %16 {
        %41 = arith.subi %40, %arg7 : index
        %42 = arith.minsi %41, %16 : index
        %43 = arith.muli %42, %19 : index
        %44 = arith.muli %43, %c2 : index
        ascendc.pipe.init_buffer %11, %38, %44 : !ascendc.tbuf<vecin>, index
        %45 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %46 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %47 = arith.addi %arg7, %29 : index
        %48 = arith.index_cast %5 : i64 to index
        %49 = arith.muli %47, %48 : index
        %50 = arith.addi %49, %19 : index
        %51 = arith.index_cast %50 : index to i32
        %52 = emitasc.reinterpret_cast %arg0 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %46, %52, %51 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %45, %46, %43 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %14, %45 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %53 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %11, %37, %44 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %11, %36, %44 : !ascendc.tbuf<veccalc>, index
        %54 = ascendc.tbuf.get_tensor %36 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %55 = arith.muli %42, %c2 : index
        ascendc.pipe.init_buffer %11, %35, %55 : !ascendc.tbuf<veccalc>, index
        %56 = ascendc.tbuf.get_tensor %35 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %57 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %58 = arith.index_cast %47 : index to i32
        %59 = emitasc.reinterpret_cast %arg1 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %57, %59, %58 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %56, %57, %42 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %60 = arith.index_cast %42 : index to i32
        %61 = arith.index_cast %19 : index to i32
        ascendc.pipe.init_buffer %11, %34, %44 : !ascendc.tbuf<veccalc>, index
        %62 = ascendc.tbuf.get_tensor %34 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %62, %56, %60, %61, %60, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %63 = arith.muli %19, %c2 : index
        ascendc.pipe.init_buffer %11, %33, %63 : !ascendc.tbuf<veccalc>, index
        %64 = ascendc.tbuf.get_tensor %33 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %65 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %66 = emitasc.reinterpret_cast %arg3 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %65, %66, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %64, %65, %19 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.pipe.init_buffer %11, %32, %44 : !ascendc.tbuf<veccalc>, index
        %67 = ascendc.tbuf.get_tensor %32 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %67, %64, %60, %61, %c1_i32, %61 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %11, %31, %44 : !ascendc.tbuf<veccalc>, index
        %68 = ascendc.tbuf.get_tensor %31 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %68, %cst, %43 : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %54, %53, %68, %43 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %54, %54, %62, %43 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.mul_l2 %54, %54, %67, %43 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %15, %54 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %69 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %70 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %71 = arith.muli %47, %19 : index
        %72 = arith.index_cast %71 : index to i32
        %73 = emitasc.reinterpret_cast %arg5 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %70, %73, %72 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %70, %69, %43 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %15, %69 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %14, %53 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return
  }
}

