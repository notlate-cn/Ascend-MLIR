module {
  func.func @ewop_broadcast_split(%arg0: memref<?x?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg2_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, 22 : i32>, %arg5: memref<?x?xf16, strided<[1, 1], offset: ?>>, %arg6: memref<?x?xf16, strided<[1, 1], offset: ?>>, %arg7: memref<?x?xf16, strided<[1, 1], offset: ?>>) attributes {ascendc.aicore, ascendc.global} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = emitasc.copy_struct %arg4 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg2_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg2_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>
    %1 = emitasc.member %0 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg2_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %2 = emitasc.member %0 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg2_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %3 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg2_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %4 = emitasc.member %0 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg2_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %5 = emitasc.member %0 "dim_arg2_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg2_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %6 = emitasc.member %0 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg2_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %7 = emitasc.member %0 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg2_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %8 = emitasc.member %0 "dim_arg2_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg2_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %9 = emitasc.member %0 "dim_arg3_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg2_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %10 = emitasc.member %0 "dim_arg3_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg2_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_1", "dim_arg3_0", "dim_arg3_1"]>, i64
    %11 = ascendc.pipe
    %12 = ascendc.queue : <vecin, 1>
    %13 = ascendc.queue : <vecout, 1>
    %14 = ascendc.queue : <vecin, 1>
    %15 = ascendc.queue : <vecout, 1>
    %16 = ascendc.queue : <vecin, 1>
    %17 = ascendc.queue : <vecout, 1>
    %18 = arith.index_cast %2 : i64 to index
    %19 = arith.index_cast %1 : i64 to index
    %20 = arith.index_cast %3 : i64 to index
    %21 = arith.index_cast %4 : i64 to index
    %22 = arith.index_cast %5 : i64 to index
    %23 = ascendc.tbuf : <veccalc>
    %24 = ascendc.tbuf : <veccalc>
    %25 = ascendc.tbuf : <veccalc>
    %26 = ascendc.tbuf : <veccalc>
    %27 = ascendc.tbuf : <vecout>
    %28 = ascendc.tbuf : <vecin>
    %29 = ascendc.get_block_idx : index
    %30 = arith.muli %29, %19 : index
    %31 = arith.cmpi ult, %30, %20 : index
    scf.if %31 {
      %42 = arith.subi %20, %30 : index
      %43 = arith.minsi %19, %42 : index
      scf.for %arg8 = %c0 to %43 step %18 {
        %44 = arith.subi %43, %arg8 : index
        %45 = arith.minsi %44, %18 : index
        %46 = arith.muli %45, %21 : index
        %47 = arith.muli %46, %c2 : index
        ascendc.pipe.init_buffer %11, %28, %47 : !ascendc.tbuf<vecin>, index
        %48 = ascendc.que_bind.alloc_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %49 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %50 = arith.addi %arg8, %30 : index
        %51 = arith.muli %50, %21 : index
        %52 = arith.index_cast %51 : index to i32
        %53 = emitasc.reinterpret_cast %arg0 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %49, %53, %52 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %48, %49, %46 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %12, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %54 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %11, %27, %47 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %11, %26, %47 : !ascendc.tbuf<veccalc>, index
        %55 = ascendc.tbuf.get_tensor %26 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %56 = arith.muli %45, %c2 : index
        ascendc.pipe.init_buffer %11, %25, %56 : !ascendc.tbuf<veccalc>, index
        %57 = ascendc.tbuf.get_tensor %25 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %58 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %59 = arith.index_cast %50 : index to i32
        %60 = emitasc.reinterpret_cast %arg1 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %58, %60, %59 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %57, %58, %45 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %61 = arith.index_cast %45 : index to i32
        %62 = arith.index_cast %21 : index to i32
        ascendc.pipe.init_buffer %11, %24, %47 : !ascendc.tbuf<veccalc>, index
        %63 = ascendc.tbuf.get_tensor %24 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %63, %57, %61, %62, %61, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %11, %23, %47 : !ascendc.tbuf<veccalc>, index
        %64 = ascendc.tbuf.get_tensor %23 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %64, %cst, %46 : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %55, %54, %64, %46 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %55, %55, %63, %46 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %13, %55 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %65 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %66 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %67 = emitasc.reinterpret_cast %arg5 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %66, %67, %52 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %66, %65, %46 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %13, %65 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %12, %54 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %32 = ascendc.tbuf : <veccalc>
    %33 = ascendc.tbuf : <veccalc>
    %34 = ascendc.tbuf : <veccalc>
    %35 = ascendc.tbuf : <vecout>
    %36 = ascendc.tbuf : <vecin>
    scf.if %31 {
      %42 = arith.subi %20, %30 : index
      %43 = arith.minsi %19, %42 : index
      scf.for %arg8 = %c0 to %43 step %18 {
        %44 = arith.subi %43, %arg8 : index
        %45 = arith.minsi %44, %18 : index
        %46 = arith.muli %45, %22 : index
        %47 = arith.muli %46, %c2 : index
        ascendc.pipe.init_buffer %11, %36, %47 : !ascendc.tbuf<vecin>, index
        %48 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %49 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %50 = arith.addi %arg8, %30 : index
        %51 = arith.muli %50, %21 : index
        %52 = arith.index_cast %51 : index to i32
        %53 = emitasc.reinterpret_cast %arg5 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %49, %53, %52 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %48, %49, %46 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %14, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %54 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %11, %35, %47 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %11, %34, %47 : !ascendc.tbuf<veccalc>, index
        %55 = ascendc.tbuf.get_tensor %34 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %56 = arith.muli %22, %c2 : index
        ascendc.pipe.init_buffer %11, %33, %56 : !ascendc.tbuf<veccalc>, index
        %57 = ascendc.tbuf.get_tensor %33 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %58 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %59 = emitasc.reinterpret_cast %arg2 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %58, %59, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %57, %58, %22 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %60 = arith.index_cast %45 : index to i32
        %61 = arith.index_cast %22 : index to i32
        ascendc.pipe.init_buffer %11, %32, %47 : !ascendc.tbuf<veccalc>, index
        %62 = ascendc.tbuf.get_tensor %32 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %62, %57, %60, %61, %c1_i32, %61 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.mul_l2 %55, %54, %62, %46 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %15, %55 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %63 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %64 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %65 = arith.muli %50, %22 : index
        %66 = arith.index_cast %65 : index to i32
        %67 = emitasc.reinterpret_cast %arg7 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %64, %67, %66 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %64, %63, %46 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %15, %63 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %14, %54 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %37 = ascendc.tbuf : <veccalc>
    %38 = ascendc.tbuf : <veccalc>
    %39 = ascendc.tbuf : <veccalc>
    %40 = ascendc.tbuf : <vecout>
    %41 = ascendc.tbuf : <vecin>
    scf.if %31 {
      %42 = arith.subi %20, %30 : index
      %43 = arith.minsi %19, %42 : index
      scf.for %arg8 = %c0 to %43 step %18 {
        %44 = arith.subi %43, %arg8 : index
        %45 = arith.minsi %44, %18 : index
        %46 = arith.muli %45, %22 : index
        %47 = arith.muli %46, %c2 : index
        ascendc.pipe.init_buffer %11, %41, %47 : !ascendc.tbuf<vecin>, index
        %48 = ascendc.que_bind.alloc_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %49 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %50 = arith.addi %arg8, %30 : index
        %51 = arith.muli %50, %21 : index
        %52 = arith.addi %51, %22 : index
        %53 = arith.index_cast %52 : index to i32
        %54 = emitasc.reinterpret_cast %arg5 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %49, %54, %53 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %48, %49, %46 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %16, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %55 = ascendc.que_bind.deque_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %11, %40, %47 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %11, %39, %47 : !ascendc.tbuf<veccalc>, index
        %56 = ascendc.tbuf.get_tensor %39 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %57 = arith.muli %22, %c2 : index
        ascendc.pipe.init_buffer %11, %38, %57 : !ascendc.tbuf<veccalc>, index
        %58 = ascendc.tbuf.get_tensor %38 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %59 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %60 = emitasc.reinterpret_cast %arg3 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %59, %60, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %58, %59, %22 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %61 = arith.index_cast %45 : index to i32
        %62 = arith.index_cast %22 : index to i32
        ascendc.pipe.init_buffer %11, %37, %47 : !ascendc.tbuf<veccalc>, index
        %63 = ascendc.tbuf.get_tensor %37 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %63, %58, %61, %62, %c1_i32, %62 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.mul_l2 %56, %55, %63, %46 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %17, %56 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %64 = ascendc.que_bind.deque_tensor %17 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %65 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %66 = arith.muli %50, %22 : index
        %67 = arith.index_cast %66 : index to i32
        %68 = emitasc.reinterpret_cast %arg6 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %65, %68, %67 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %65, %64, %46 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %17, %64 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %16, %55 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return
  }
}

