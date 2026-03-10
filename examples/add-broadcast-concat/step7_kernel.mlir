module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_concat(%arg0: memref<?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?xf16>, %arg3: memref<?x?xf16>, %arg4: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, 22 : i32>, %arg5: memref<?x?xf16, strided<[1, 1], offset: ?>>) attributes {ascendc.aicore, ascendc.global} {
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = emitasc.copy_struct %arg4 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>
    %1 = emitasc.member %0 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %2 = emitasc.member %0 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %3 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %4 = emitasc.member %0 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %5 = emitasc.member %0 "dim_arg2_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %6 = emitasc.member %0 "dim_arg3_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %7 = emitasc.member %0 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %8 = emitasc.member %0 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %9 = emitasc.member %0 "dim_arg2_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
    %10 = emitasc.member %0 "dim_arg3_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg2_0", "dim_arg3_1", "dim_arg0_1", "dim_arg1_0", "dim_arg2_1", "dim_arg3_0"]>, i64
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
    %23 = ascendc.tbuf : <vecout>
    %24 = ascendc.tbuf : <vecin>
    %25 = ascendc.get_block_idx : index
    %26 = arith.muli %25, %17 : index
    %27 = arith.cmpi ult, %26, %18 : index
    scf.if %27 {
      %36 = arith.subi %18, %26 : index
      %37 = arith.minsi %17, %36 : index
      scf.for %arg6 = %c0 to %37 step %16 {
        %38 = arith.subi %37, %arg6 : index
        %39 = arith.minsi %38, %16 : index
        %40 = arith.muli %39, %c2 : index
        ascendc.pipe.init_buffer %11, %24, %40 : !ascendc.tbuf<vecin>, index
        %41 = ascendc.que_bind.alloc_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %42 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %43 = arith.addi %arg6, %26 : index
        %44 = arith.index_cast %43 : index to i32
        %45 = emitasc.reinterpret_cast %arg0 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %42, %45, %44 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %41, %42, %39 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %12, %41 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %46 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %47 = arith.muli %39, %19 : index
        %48 = arith.muli %47, %c2 : index
        ascendc.pipe.init_buffer %11, %23, %48 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %11, %22, %48 : !ascendc.tbuf<veccalc>, index
        %49 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %50 = arith.index_cast %39 : index to i32
        %51 = arith.index_cast %19 : index to i32
        ascendc.pipe.init_buffer %11, %21, %48 : !ascendc.tbuf<veccalc>, index
        %52 = ascendc.tbuf.get_tensor %21 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %52, %46, %50, %51, %50, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %11, %20, %48 : !ascendc.tbuf<veccalc>, index
        %53 = ascendc.tbuf.get_tensor %20 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %55 = arith.muli %43, %19 : index
        %56 = arith.index_cast %55 : index to i32
        %57 = emitasc.reinterpret_cast %arg1 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %54, %57, %56 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %53, %54, %47 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.add_l2 %49, %52, %53, %47 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %13, %49 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %58 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %59 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %60 = emitasc.reinterpret_cast %arg5 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %59, %60, %56 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %59, %58, %47 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %13, %58 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %12, %46 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %28 = arith.index_cast %5 : i64 to index
    %29 = arith.index_cast %6 : i64 to index
    %30 = ascendc.tbuf : <veccalc>
    %31 = ascendc.tbuf : <veccalc>
    %32 = ascendc.tbuf : <veccalc>
    %33 = ascendc.tbuf : <vecout>
    %34 = ascendc.tbuf : <vecin>
    %35 = arith.cmpi ult, %26, %28 : index
    scf.if %35 {
      %36 = arith.subi %28, %26 : index
      %37 = arith.minsi %17, %36 : index
      scf.for %arg6 = %c0 to %37 step %16 {
        %38 = arith.subi %37, %arg6 : index
        %39 = arith.minsi %38, %16 : index
        %40 = arith.muli %39, %c2 : index
        ascendc.pipe.init_buffer %11, %34, %40 : !ascendc.tbuf<vecin>, index
        %41 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %42 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %43 = arith.addi %arg6, %26 : index
        %44 = arith.index_cast %43 : index to i32
        %45 = emitasc.reinterpret_cast %arg2 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %42, %45, %44 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %41, %42, %39 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %14, %41 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %46 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %47 = arith.muli %39, %29 : index
        %48 = arith.muli %47, %c2 : index
        ascendc.pipe.init_buffer %11, %33, %48 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %11, %32, %48 : !ascendc.tbuf<veccalc>, index
        %49 = ascendc.tbuf.get_tensor %32 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %50 = arith.index_cast %39 : index to i32
        %51 = arith.index_cast %29 : index to i32
        ascendc.pipe.init_buffer %11, %31, %48 : !ascendc.tbuf<veccalc>, index
        %52 = ascendc.tbuf.get_tensor %31 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %52, %46, %50, %51, %50, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %11, %30, %48 : !ascendc.tbuf<veccalc>, index
        %53 = ascendc.tbuf.get_tensor %30 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %55 = arith.muli %43, %29 : index
        %56 = arith.index_cast %55 : index to i32
        %57 = emitasc.reinterpret_cast %arg3 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %54, %57, %56 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %53, %54, %47 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.mul_l2 %49, %52, %53, %47 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %15, %49 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %58 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %59 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %60 = arith.addi %43, %18 : index
        %61 = arith.muli %60, %19 : index
        %62 = arith.index_cast %61 : index to i32
        %63 = emitasc.reinterpret_cast %arg5 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %59, %63, %62 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %59, %58, %47 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %15, %58 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %14, %46 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "broadcast_add"} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "broadcast_mul"} in %transformed : (!transform.any_op) -> !transform.any_op
    %3 = transform.param.constant true -> !transform.any_param
    %4 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %5 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %6 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops "ascendc.parallel" = %3 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_1 "ascendc.prologue" = %4 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %6 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %2 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_3 "ascendc.parallel" = %3 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_5 "ascendc.prologue" = %4 : !transform.any_op, !transform.any_param
    transform.annotate %loops_5 "ascendc.epilogue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_4 "ascendc.unit" = %6 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_5 : !transform.any_op
    transform.yield 
  }
}

