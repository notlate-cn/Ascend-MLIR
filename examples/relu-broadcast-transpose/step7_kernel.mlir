module attributes {transform.with_named_sequence} {
  func.func @relu_transpose_broadcast_add(%arg0: memref<?x1xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, 22 : i32>, %arg3: memref<?x?xf16, strided<[1, 1], offset: ?>>) attributes {ascendc.aicore, ascendc.global} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = emitasc.copy_struct %arg2 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>
    %1 = emitasc.member %0 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %2 = emitasc.member %0 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %3 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %4 = emitasc.member %0 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %5 = emitasc.member %0 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %6 = emitasc.member %0 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]>, i64
    %7 = ascendc.pipe
    %8 = ascendc.queue : <vecin, 1>
    %9 = ascendc.queue : <vecout, 1>
    %10 = arith.index_cast %2 : i64 to index
    %11 = arith.index_cast %1 : i64 to index
    %12 = arith.index_cast %3 : i64 to index
    %13 = arith.index_cast %4 : i64 to index
    %14 = ascendc.tbuf : <veccalc>
    %15 = ascendc.queue : <vecin, 1>
    %16 = ascendc.tbuf : <vecin>
    %17 = ascendc.tbuf : <veccalc>
    %18 = ascendc.tbuf : <veccalc>
    %19 = ascendc.tbuf : <veccalc>
    %20 = ascendc.tbuf : <vecout>
    %21 = ascendc.tbuf : <vecin>
    %22 = ascendc.get_block_idx : index
    %23 = arith.muli %22, %11 : index
    %24 = arith.cmpi ult, %23, %13 : index
    scf.if %24 {
      %25 = arith.subi %13, %23 : index
      %26 = arith.minsi %11, %25 : index
      scf.for %arg4 = %c0 to %26 step %10 {
        %27 = arith.subi %26, %arg4 : index
        %28 = arith.minsi %27, %10 : index
        %29 = arith.muli %12, %c2 : index
        ascendc.pipe.init_buffer %7, %21, %29 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %7, %8, %c1_i32, %29 : !ascendc.queue<vecin, 1>, i32, index
        %30 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %31 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %32 = emitasc.reinterpret_cast %arg0 : memref<?x1xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %31, %32, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %30, %31, %12 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %8, %30 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %33 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %34 = arith.muli %28, %12 : index
        %35 = arith.muli %34, %c2 : index
        ascendc.pipe.init_buffer %7, %20, %35 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %7, %9, %c1_i32, %35 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %7, %19, %35 : !ascendc.tbuf<veccalc>, index
        %36 = ascendc.tbuf.get_tensor %19 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %37 = arith.index_cast %12 : index to i32
        %38 = arith.index_cast %28 : index to i32
        ascendc.pipe.init_buffer %7, %18, %35 : !ascendc.tbuf<veccalc>, index
        %39 = ascendc.tbuf.get_tensor %18 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %39, %33, %37, %38, %37, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %7, %17, %35 : !ascendc.tbuf<veccalc>, index
        %40 = ascendc.tbuf.get_tensor %17 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.transpose %40, %39 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>
        %41 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %42 = arith.addi %arg4, %23 : index
        %43 = arith.index_cast %6 : i64 to index
        %44 = arith.muli %42, %43 : index
        %45 = arith.index_cast %44 : index to i32
        %46 = emitasc.reinterpret_cast %arg1 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %41, %46, %45 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.pipe.init_buffer %7, %16, %35 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %7, %15, %c1_i32, %35 : !ascendc.queue<vecin, 1>, i32, index
        %47 = ascendc.que_bind.alloc_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %47, %41, %34 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %15, %47 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %48 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %7, %14, %35 : !ascendc.tbuf<veccalc>, index
        %49 = ascendc.tbuf.get_tensor %14 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %49, %cst, %34 : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %36, %40, %49, %34 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %36, %36, %48, %34 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %9, %36 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %50 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %51 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %52 = arith.muli %42, %12 : index
        %53 = arith.index_cast %52 : index to i32
        %54 = emitasc.reinterpret_cast %arg3 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %51, %54, %53 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %51, %50, %34 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %9, %50 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %8, %33 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.param.constant true -> !transform.any_param
    %3 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %4 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %5 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops "ascendc.parallel" = %2 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_1 "ascendc.prologue" = %3 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %4 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %5 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    transform.yield 
  }
}

