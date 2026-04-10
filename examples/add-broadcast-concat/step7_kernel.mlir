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
    %20 = ascendc.queue : <vecin, 1>
    %21 = ascendc.tbuf : <vecin>
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
      scf.for %arg6 = %c0 to %39 step %16 {
        %40 = arith.subi %39, %arg6 : index
        %41 = arith.minsi %40, %16 : index
        %42 = arith.muli %41, %c2 : index
        ascendc.pipe.init_buffer %11, %25, %42 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %11, %12, %c1_i32, %42 : !ascendc.queue<vecin, 1>, i32, index
        %43 = ascendc.que_bind.alloc_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %44 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %45 = arith.addi %arg6, %27 : index
        %46 = arith.index_cast %45 : index to i32
        %47 = emitasc.reinterpret_cast %arg0 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %44, %47, %46 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %43, %44, %41 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %12, %43 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %48 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %49 = arith.muli %41, %19 : index
        %50 = arith.muli %49, %c2 : index
        ascendc.pipe.init_buffer %11, %24, %50 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %11, %13, %c1_i32, %50 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %11, %23, %50 : !ascendc.tbuf<veccalc>, index
        %51 = ascendc.tbuf.get_tensor %23 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %52 = arith.index_cast %41 : index to i32
        %53 = arith.index_cast %19 : index to i32
        ascendc.pipe.init_buffer %11, %22, %50 : !ascendc.tbuf<veccalc>, index
        %54 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %54, %48, %52, %53, %52, %c1_i32 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %55 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %56 = arith.muli %45, %19 : index
        %57 = arith.index_cast %56 : index to i32
        %58 = emitasc.reinterpret_cast %arg1 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %55, %58, %57 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.pipe.init_buffer %11, %21, %50 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %11, %20, %c1_i32, %50 : !ascendc.queue<vecin, 1>, i32, index
        %59 = ascendc.que_bind.alloc_tensor %20 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %59, %55, %49 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %20, %59 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %60 = ascendc.que_bind.deque_tensor %20 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.add_l2 %51, %54, %60, %49 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %13, %51 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %61 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %62 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %63 = emitasc.reinterpret_cast %arg5 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %62, %63, %57 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %62, %61, %49 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %13, %61 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %12, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    %29 = arith.index_cast %5 : i64 to index
    %30 = arith.index_cast %6 : i64 to index
    %31 = ascendc.queue : <vecin, 1>
    %32 = ascendc.tbuf : <vecin>
    %33 = ascendc.tbuf : <veccalc>
    %34 = ascendc.tbuf : <veccalc>
    %35 = ascendc.tbuf : <vecout>
    %36 = ascendc.tbuf : <vecin>
    %37 = arith.cmpi ult, %27, %29 : index
    scf.if %37 {
      %38 = arith.subi %29, %27 : index
      %39 = arith.minsi %17, %38 : index
      scf.for %arg6 = %c0 to %39 step %16 {
        %40 = arith.subi %39, %arg6 : index
        %41 = arith.minsi %40, %16 : index
        %42 = arith.muli %41, %c2 : index
        ascendc.pipe.init_buffer %11, %36, %42 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %11, %14, %c1_i32, %42 : !ascendc.queue<vecin, 1>, i32, index
        %43 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %44 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %45 = arith.addi %arg6, %27 : index
        %46 = arith.index_cast %45 : index to i32
        %47 = emitasc.reinterpret_cast %arg2 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %44, %47, %46 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %43, %44, %41 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %14, %43 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %48 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %49 = arith.muli %41, %30 : index
        %50 = arith.muli %49, %c2 : index
        ascendc.pipe.init_buffer %11, %35, %50 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %11, %15, %c1_i32, %50 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %11, %34, %50 : !ascendc.tbuf<veccalc>, index
        %51 = ascendc.tbuf.get_tensor %34 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %52 = arith.index_cast %41 : index to i32
        %53 = arith.index_cast %30 : index to i32
        ascendc.pipe.init_buffer %11, %33, %50 : !ascendc.tbuf<veccalc>, index
        %54 = ascendc.tbuf.get_tensor %33 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %54, %48, %52, %53, %52, %c1_i32 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %55 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %56 = arith.muli %45, %30 : index
        %57 = arith.index_cast %56 : index to i32
        %58 = emitasc.reinterpret_cast %arg3 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %55, %58, %57 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.pipe.init_buffer %11, %32, %50 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %11, %31, %c1_i32, %50 : !ascendc.queue<vecin, 1>, i32, index
        %59 = ascendc.que_bind.alloc_tensor %31 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %59, %55, %49 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %31, %59 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %60 = ascendc.que_bind.deque_tensor %31 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.mul_l2 %51, %54, %60, %49 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %15, %51 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %61 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %62 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %63 = arith.addi %45, %18 : index
        %64 = arith.muli %63, %19 : index
        %65 = arith.index_cast %64 : index to i32
        %66 = emitasc.reinterpret_cast %arg5 : memref<?x?xf16, strided<[1, 1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %62, %66, %65 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %62, %61, %49 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %15, %61 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %14, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
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

