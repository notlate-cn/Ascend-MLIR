module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_split(%arg0: memref<?x?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: memref<?x?xf16, strided<[1, 1], offset: ?>>, %arg6: memref<ui8>, %arg7: !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 5 : i32} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = emitasc.member %arg7 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %1 = emitasc.member %arg7 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %2 = emitasc.member %arg7 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %3 = emitasc.member %arg7 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %4 = emitasc.member %arg7 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %5 = emitasc.member %arg7 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %6 = emitasc.member %arg7 "dim_arg3_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %7 = emitasc.member %arg7 "dim_arg3_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %8 = emitasc.member %arg7 "dim_arg2_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %9 = emitasc.member %arg7 "dim_arg2_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %10 = emitasc.member %arg7 "dim_arg4_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %11 = emitasc.member %arg7 "dim_arg4_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_1", "dim_arg1_0", "dim_arg0_0", "dim_arg1_1", "dim_arg3_0", "dim_arg3_1", "dim_arg2_0", "dim_arg2_1", "dim_arg4_0", "dim_arg4_1"]>, i64
    %12 = ascendc.pipe
    %13 = ascendc.queue : <vecin, 1>
    %14 = ascendc.queue : <vecout, 1>
    %15 = ascendc.queue : <vecin, 1>
    %16 = ascendc.queue : <vecout, 1>
    %17 = arith.index_cast %1 : i64 to index
    %18 = arith.index_cast %0 : i64 to index
    %19 = arith.index_cast %2 : i64 to index
    %20 = arith.index_cast %3 : i64 to index
    %21 = ascendc.tbuf : <veccalc>
    %22 = ascendc.tbuf : <veccalc>
    %23 = ascendc.queue : <vecin, 1>
    %24 = ascendc.tbuf : <vecin>
    %25 = ascendc.tbuf : <veccalc>
    %26 = ascendc.queue : <vecin, 1>
    %27 = ascendc.tbuf : <vecin>
    %28 = ascendc.tbuf : <veccalc>
    %29 = ascendc.tbuf : <vecout>
    %30 = ascendc.tbuf : <vecin>
    %31 = ascendc.get_block_idx : index
    %32 = arith.muli %31, %18 : index
    %33 = arith.cmpi ult, %32, %20 : index
    scf.if %33 {
      %44 = arith.subi %20, %32 : index
      %45 = arith.minsi %18, %44 : index
      scf.for %arg8 = %c0 to %45 step %17 {
        %46 = arith.subi %45, %arg8 : index
        %47 = arith.minsi %46, %17 : index
        %48 = arith.muli %47, %19 : index
        %49 = arith.muli %48, %c2 : index
        ascendc.pipe.init_buffer %12, %30, %49 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %12, %13, %c1_i32, %49 : !ascendc.queue<vecin, 1>, i32, index
        %50 = ascendc.que_bind.alloc_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %51 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %52 = arith.addi %arg8, %32 : index
        %53 = arith.muli %52, %19 : index
        %54 = arith.index_cast %53 : index to i32
        %55 = emitasc.reinterpret_cast %arg0 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %51, %55, %54 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %50, %51, %48 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %13, %50 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %56 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %12, %29, %49 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %12, %14, %c1_i32, %49 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %12, %28, %49 : !ascendc.tbuf<veccalc>, index
        %57 = ascendc.tbuf.get_tensor %28 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %58 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %59 = arith.index_cast %52 : index to i32
        %60 = emitasc.reinterpret_cast %arg1 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %58, %60, %59 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        %61 = arith.muli %47, %c2 : index
        ascendc.pipe.init_buffer %12, %27, %61 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %12, %26, %c1_i32, %61 : !ascendc.queue<vecin, 1>, i32, index
        %62 = ascendc.que_bind.alloc_tensor %26 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %62, %58, %47 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %26, %62 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %63 = ascendc.que_bind.deque_tensor %26 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %64 = arith.index_cast %47 : index to i32
        %65 = arith.index_cast %19 : index to i32
        ascendc.pipe.init_buffer %12, %25, %49 : !ascendc.tbuf<veccalc>, index
        %66 = ascendc.tbuf.get_tensor %25 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %66, %63, %64, %65, %64, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %67 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %68 = emitasc.reinterpret_cast %arg3 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %67, %68, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        %69 = arith.muli %19, %c2 : index
        ascendc.pipe.init_buffer %12, %24, %69 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %12, %23, %c1_i32, %69 : !ascendc.queue<vecin, 1>, i32, index
        %70 = ascendc.que_bind.alloc_tensor %23 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %70, %67, %19 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %23, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %71 = ascendc.que_bind.deque_tensor %23 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %12, %22, %49 : !ascendc.tbuf<veccalc>, index
        %72 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %72, %71, %64, %65, %c1_i32, %65 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %12, %21, %49 : !ascendc.tbuf<veccalc>, index
        %73 = ascendc.tbuf.get_tensor %21 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %73, %cst, %48 : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %57, %56, %73, %48 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %57, %57, %66, %48 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.mul_l2 %57, %57, %72, %48 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %14, %57 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %74 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %75 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %76 = emitasc.reinterpret_cast %arg5 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %75, %76, %54 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %75, %74, %48 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %14, %74 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %13, %56 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %34 = ascendc.tbuf : <veccalc>
    %35 = ascendc.tbuf : <veccalc>
    %36 = ascendc.queue : <vecin, 1>
    %37 = ascendc.tbuf : <vecin>
    %38 = ascendc.tbuf : <veccalc>
    %39 = ascendc.queue : <vecin, 1>
    %40 = ascendc.tbuf : <vecin>
    %41 = ascendc.tbuf : <veccalc>
    %42 = ascendc.tbuf : <vecout>
    %43 = ascendc.tbuf : <vecin>
    scf.if %33 {
      %44 = arith.subi %20, %32 : index
      %45 = arith.minsi %18, %44 : index
      scf.for %arg8 = %c0 to %45 step %17 {
        %46 = arith.subi %45, %arg8 : index
        %47 = arith.minsi %46, %17 : index
        %48 = arith.muli %47, %19 : index
        %49 = arith.muli %48, %c2 : index
        ascendc.pipe.init_buffer %12, %43, %49 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %12, %15, %c1_i32, %49 : !ascendc.queue<vecin, 1>, i32, index
        %50 = ascendc.que_bind.alloc_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %51 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %52 = arith.addi %arg8, %32 : index
        %53 = arith.addi %52, %20 : index
        %54 = arith.muli %53, %19 : index
        %55 = arith.index_cast %54 : index to i32
        %56 = emitasc.reinterpret_cast %arg0 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %51, %56, %55 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %50, %51, %48 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %15, %50 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %57 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %12, %42, %49 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %12, %16, %c1_i32, %49 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %12, %41, %49 : !ascendc.tbuf<veccalc>, index
        %58 = ascendc.tbuf.get_tensor %41 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %59 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %60 = arith.index_cast %52 : index to i32
        %61 = emitasc.reinterpret_cast %arg2 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %59, %61, %60 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        %62 = arith.muli %47, %c2 : index
        ascendc.pipe.init_buffer %12, %40, %62 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %12, %39, %c1_i32, %62 : !ascendc.queue<vecin, 1>, i32, index
        %63 = ascendc.que_bind.alloc_tensor %39 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %63, %59, %47 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %39, %63 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %64 = ascendc.que_bind.deque_tensor %39 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %65 = arith.index_cast %47 : index to i32
        %66 = arith.index_cast %19 : index to i32
        ascendc.pipe.init_buffer %12, %38, %49 : !ascendc.tbuf<veccalc>, index
        %67 = ascendc.tbuf.get_tensor %38 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %67, %64, %65, %66, %65, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %68 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %69 = emitasc.reinterpret_cast %arg4 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %68, %69, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        %70 = arith.muli %19, %c2 : index
        ascendc.pipe.init_buffer %12, %37, %70 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %12, %36, %c1_i32, %70 : !ascendc.queue<vecin, 1>, i32, index
        %71 = ascendc.que_bind.alloc_tensor %36 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %71, %68, %19 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %36, %71 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %72 = ascendc.que_bind.deque_tensor %36 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %12, %35, %49 : !ascendc.tbuf<veccalc>, index
        %73 = ascendc.tbuf.get_tensor %35 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %73, %72, %65, %66, %c1_i32, %66 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %12, %34, %49 : !ascendc.tbuf<veccalc>, index
        %74 = ascendc.tbuf.get_tensor %34 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %74, %cst, %48 : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %58, %57, %74, %48 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %58, %58, %67, %48 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.mul_l2 %58, %58, %73, %48 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %16, %58 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %75 = ascendc.que_bind.deque_tensor %16 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %76 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %77 = emitasc.reinterpret_cast %arg5 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %76, %77, %55 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %76, %75, %48 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %16, %75 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %15, %57 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return
  }
}

