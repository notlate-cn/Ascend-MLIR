module attributes {transform.with_named_sequence} {
  func.func @broadcast_add_reducesum(%arg0: memref<?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>, 22 : i32>, %arg3: memref<?xf16, strided<[1], offset: ?>>) attributes {ascendc.aicore, ascendc.global} {
    %cst = arith.constant 0.000000e+00 : f16
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = emitasc.copy_struct %arg2 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>
    %1 = emitasc.member %0 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>, i64
    %2 = emitasc.member %0 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>, i64
    %3 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>, i64
    %4 = emitasc.member %0 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>, i64
    %5 = emitasc.member %0 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>, i64
    %6 = emitasc.member %0 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"]>, i64
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
    %19 = ascendc.tbuf : <vecout>
    %20 = ascendc.tbuf : <vecin>
    %21 = ascendc.get_block_idx : index
    %22 = arith.muli %21, %11 : index
    %23 = arith.cmpi ult, %22, %12 : index
    scf.if %23 {
      %24 = arith.subi %12, %22 : index
      %25 = arith.minsi %11, %24 : index
      scf.for %arg4 = %c0 to %25 step %10 {
        %26 = arith.subi %25, %arg4 : index
        %27 = arith.minsi %26, %10 : index
        %28 = arith.muli %27, %c2 : index
        ascendc.pipe.init_buffer %7, %20, %28 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %7, %8, %c1_i32, %28 : !ascendc.queue<vecin, 1>, i32, index
        %29 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %30 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %31 = arith.addi %arg4, %22 : index
        %32 = arith.index_cast %31 : index to i32
        %33 = emitasc.reinterpret_cast %arg0 : memref<?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %30, %33, %32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %29, %30, %27 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %8, %29 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %34 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %7, %19, %28 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %7, %9, %c1_i32, %28 : !ascendc.queue<vecout, 1>, i32, index
        %35 = arith.muli %27, %13 : index
        %36 = arith.muli %35, %c2 : index
        ascendc.pipe.init_buffer %7, %18, %36 : !ascendc.tbuf<veccalc>, index
        %37 = ascendc.tbuf.get_tensor %18 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %37, %cst, %35 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, f16, index
        %38 = arith.index_cast %27 : index to i32
        %39 = arith.index_cast %13 : index to i32
        ascendc.pipe.init_buffer %7, %17, %36 : !ascendc.tbuf<veccalc>, index
        %40 = ascendc.tbuf.get_tensor %17 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %40, %34, %38, %39, %38, %c1_i32 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %41 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %42 = arith.muli %31, %13 : index
        %43 = arith.index_cast %42 : index to i32
        %44 = emitasc.reinterpret_cast %arg1 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %41, %44, %43 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.pipe.init_buffer %7, %16, %36 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %7, %15, %c1_i32, %36 : !ascendc.queue<vecin, 1>, i32, index
        %45 = ascendc.que_bind.alloc_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %45, %41, %35 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %15, %45 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %46 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %7, %14, %36 : !ascendc.tbuf<veccalc>, index
        %47 = ascendc.tbuf.get_tensor %14 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.add_l2 %47, %40, %46, %35 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %37, %37, %47, %35 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        %48 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.reduce_sum_2d_l2 %48, %37 {ascendc.unit = "AiCore.Vector", layout = 0 : i32} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.enque_tensor %9, %48 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %49 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %50 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %51 = emitasc.reinterpret_cast %arg3 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %50, %51, %32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        ascendc.data_copy_l2 %50, %49, %27 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %9, %49 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %8, %34 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} in %transformed : (!transform.any_op) -> !transform.any_op
    transform.print %1 {name = "--------------------------------  \E5\8E\9F\E5\A7\8B\E8\9E\8D\E5\90\88\E5\AD\90\E5\9B\BE --------------------------------"} : !transform.any_op
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %loops {name = "-------------------------------- \E9\A6\96\E6\AC\A1\E5\88\87\E5\87\BA\E5\A4\96\E5\B1\82\E5\BE\AA\E7\8E\AFTB --------------------------------"} : !transform.any_op
    transform.print %tiled_linalg_op {name = "-------------------------------- \E9\A6\96\E6\AC\A1\E5\88\87\E5\88\86\E5\90\8E\E7\9A\84\E5\86\85\E5\B1\82\E5\AD\90\E5\9B\BETb --------------------------------"} : !transform.any_op
    %2 = transform.param.constant true -> !transform.any_param
    transform.annotate %loops "ascendc.parallel" = %2 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %loops_1 {name = "-------------------------------- \E4\BA\8C\E6\AC\A1\E5\88\87\E5\87\BA\E5\BE\AA\E7\8E\AFTb--------------------------------"} : !transform.any_op
    transform.print %tiled_linalg_op_0 {name = "-------------------------------- \E4\BA\8C\E6\AC\A1\E5\88\87\E5\90\8E\E7\9A\84\E5\AD\90\E5\9B\BE --------------------------------"} : !transform.any_op
    %3 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %4 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    transform.annotate %loops_1 "ascendc.prologue" = %3 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %4 : !transform.any_op, !transform.any_param
    %5 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %5 : !transform.any_op, !transform.any_param
    transform.yield 
  }
}

