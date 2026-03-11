module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_transpose(%arg0: memref<?x?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, 22 : i32>, %arg4: memref<?x?xf16, strided<[1, 1], offset: ?>>, %arg5: memref<?x?xf16, strided<[1, 1], offset: ?>>, %arg6: memref<?x?xf16, strided<[1, 1], offset: ?>>) attributes {ascendc.aicore, ascendc.global} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = emitasc.copy_struct %arg3 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>
    %1 = emitasc.member %0 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %2 = emitasc.member %0 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %3 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %4 = emitasc.member %0 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %5 = emitasc.member %0 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %6 = emitasc.member %0 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %7 = emitasc.member %0 "dim_arg2_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %8 = emitasc.member %0 "dim_arg2_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg0_1", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %9 = ascendc.pipe
    %10 = ascendc.queue : <vecin, 1>
    %11 = ascendc.queue : <vecout, 1>
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
    %24 = ascendc.tbuf : <vecout>
    %25 = ascendc.tbuf : <vecin>
    %26 = ascendc.get_block_idx : index
    %27 = arith.muli %26, %17 : index
    %28 = arith.cmpi ult, %27, %18 : index
    scf.if %28 {
      %38 = arith.subi %18, %27 : index
      %39 = arith.minsi %17, %38 : index
      scf.for %arg7 = %c0 to %39 step %16 {
        %40 = arith.subi %39, %arg7 : index
        %41 = arith.minsi %40, %16 : index
        %42 = arith.muli %41, %19 : index
        %43 = arith.muli %42, %c2 : index
        ascendc.pipe.init_buffer %9, %25, %43 : !ascendc.tbuf<vecin>, index
        %44 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %45 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %46 = arith.addi %arg7, %27 : index
        %47 = arith.muli %46, %19 : index
        %48 = arith.index_cast %47 : index to i32
        %49 = emitasc.reinterpret_cast %arg0 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %45, %49, %48 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %44, %45, %42 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %10, %44 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %50 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %9, %24, %43 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %9, %23, %43 : !ascendc.tbuf<veccalc>, index
        %51 = ascendc.tbuf.get_tensor %23 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %52 = arith.muli %19, %c2 : index
        ascendc.pipe.init_buffer %9, %22, %52 : !ascendc.tbuf<veccalc>, index
        %53 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %55 = emitasc.reinterpret_cast %arg1 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %54, %55, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %53, %54, %19 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %56 = arith.index_cast %41 : index to i32
        %57 = arith.index_cast %19 : index to i32
        ascendc.pipe.init_buffer %9, %21, %43 : !ascendc.tbuf<veccalc>, index
        %58 = ascendc.tbuf.get_tensor %21 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %58, %53, %56, %57, %c1_i32, %57 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %9, %20, %43 : !ascendc.tbuf<veccalc>, index
        %59 = ascendc.tbuf.get_tensor %20 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %59, %cst, %42 : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %51, %50, %59, %42 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %51, %51, %58, %42 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %11, %51 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %60 = ascendc.que_bind.deque_tensor %11 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %61 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %62 = emitasc.reinterpret_cast %arg4 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %61, %62, %48 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %61, %60, %42 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %11, %60 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %10, %50 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %29 = ascendc.tbuf : <veccalc>
    %30 = ascendc.tbuf : <vecout>
    %31 = ascendc.tbuf : <vecin>
    %32 = arith.cmpi ult, %27, %19 : index
    scf.if %32 {
      %38 = arith.subi %19, %27 : index
      %39 = arith.minsi %17, %38 : index
      scf.for %arg7 = %c0 to %39 step %16 {
        %40 = arith.subi %39, %arg7 : index
        %41 = arith.minsi %40, %16 : index
        %42 = arith.muli %18, %41 : index
        %43 = arith.muli %42, %c2 : index
        ascendc.pipe.init_buffer %9, %31, %43 : !ascendc.tbuf<vecin>, index
        %44 = ascendc.que_bind.alloc_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %45 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %46 = arith.addi %arg7, %27 : index
        %47 = arith.index_cast %46 : index to i32
        %48 = emitasc.reinterpret_cast %arg4 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %45, %48, %47 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %44, %45, %42 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %12, %44 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %49 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %9, %30, %43 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %9, %29, %43 : !ascendc.tbuf<veccalc>, index
        %50 = ascendc.tbuf.get_tensor %29 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.enque_tensor %13, %50 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %51 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %52 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %53 = arith.muli %46, %18 : index
        %54 = arith.index_cast %53 : index to i32
        %55 = emitasc.reinterpret_cast %arg6 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %52, %55, %54 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %52, %51, %42 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %13, %51 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %12, %49 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %33 = ascendc.tbuf : <veccalc>
    %34 = ascendc.tbuf : <veccalc>
    %35 = ascendc.tbuf : <veccalc>
    %36 = ascendc.tbuf : <vecout>
    %37 = ascendc.tbuf : <vecin>
    scf.if %32 {
      %38 = arith.subi %19, %27 : index
      %39 = arith.minsi %17, %38 : index
      scf.for %arg7 = %c0 to %39 step %16 {
        %40 = arith.subi %39, %arg7 : index
        %41 = arith.minsi %40, %16 : index
        %42 = arith.muli %41, %18 : index
        %43 = arith.muli %42, %c2 : index
        ascendc.pipe.init_buffer %9, %37, %43 : !ascendc.tbuf<vecin>, index
        %44 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %45 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %46 = arith.addi %arg7, %27 : index
        %47 = arith.muli %46, %18 : index
        %48 = arith.index_cast %47 : index to i32
        %49 = emitasc.reinterpret_cast %arg6 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %45, %49, %48 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %44, %45, %42 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %14, %44 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %50 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %9, %36, %43 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_buffer %9, %35, %43 : !ascendc.tbuf<veccalc>, index
        %51 = ascendc.tbuf.get_tensor %35 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %52 = arith.muli %18, %c2 : index
        ascendc.pipe.init_buffer %9, %34, %52 : !ascendc.tbuf<veccalc>, index
        %53 = ascendc.tbuf.get_tensor %34 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %55 = emitasc.reinterpret_cast %arg2 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %54, %55, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %53, %54, %18 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        %56 = arith.index_cast %41 : index to i32
        %57 = arith.index_cast %18 : index to i32
        ascendc.pipe.init_buffer %9, %33, %43 : !ascendc.tbuf<veccalc>, index
        %58 = ascendc.tbuf.get_tensor %33 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %58, %53, %56, %57, %c1_i32, %57 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.mul_l2 %51, %50, %58, %42 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %15, %51 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %59 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %60 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %61 = emitasc.reinterpret_cast %arg5 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %60, %61, %48 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %60, %59, %42 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %15, %59 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %14, %50 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "relu_bias_add"} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "transpose"} in %transformed : (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "scale_mul"} in %transformed : (!transform.any_op) -> !transform.any_op
    %4 = transform.param.constant true -> !transform.any_param
    %5 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %6 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %7 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_1 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %2 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_3 "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_5 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_5 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_4 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_5 : !transform.any_op
    %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %3 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_7 "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_9 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_9 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_8 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_9 : !transform.any_op
    transform.yield 
  }
}

