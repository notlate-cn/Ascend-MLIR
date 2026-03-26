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
    %18 = ascendc.tbuf : <vecout>
    %19 = ascendc.tbuf : <vecin>
    %20 = ascendc.get_block_idx : index
    %21 = arith.muli %20, %11 : index
    %22 = arith.cmpi ult, %21, %13 : index
    scf.if %22 {
      %23 = arith.subi %13, %21 : index
      %24 = arith.minsi %11, %23 : index
      scf.for %arg4 = %c0 to %24 step %10 {
        %25 = arith.subi %24, %arg4 : index
        %26 = arith.minsi %25, %10 : index
        %27 = arith.muli %12, %c2 : index
        ascendc.pipe.init_buffer %7, %19, %27 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %7, %8, %c1_i32, %27 : !ascendc.queue<vecin, 1>, i32, index
        %28 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %29 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %30 = emitasc.reinterpret_cast %arg0 : memref<?x1xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %29, %30, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %28, %29, %12 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %8, %28 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %31 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %32 = arith.muli %26, %12 : index
        %33 = arith.muli %32, %c2 : index
        ascendc.pipe.init_buffer %7, %18, %33 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %7, %9, %c1_i32, %33 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %7, %17, %27 : !ascendc.tbuf<veccalc>, index
        %34 = ascendc.tbuf.get_tensor %17 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %35 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %36 = arith.addi %arg4, %21 : index
        %37 = arith.index_cast %6 : i64 to index
        %38 = arith.muli %36, %37 : index
        %39 = arith.index_cast %38 : index to i32
        %40 = emitasc.reinterpret_cast %arg1 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %35, %40, %39 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.pipe.init_buffer %7, %16, %27 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %7, %15, %c1_i32, %27 : !ascendc.queue<vecin, 1>, i32, index
        %41 = ascendc.que_bind.alloc_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %41, %35, %12 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %15, %41 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %42 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %7, %14, %27 : !ascendc.tbuf<veccalc>, index
        %43 = ascendc.tbuf.get_tensor %14 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %43, %cst, %12 : !ascendc.local_tensor<*xf16>, f16, index
        ascendc.max_l2 %34, %31, %43, %12 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %34, %34, %42, %12 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %9, %34 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %44 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %45 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %46 = arith.muli %36, %12 : index
        %47 = arith.index_cast %46 : index to i32
        %48 = emitasc.reinterpret_cast %arg3 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %45, %48, %47 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %45, %44, %32 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %9, %44 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %8, %31 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
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

