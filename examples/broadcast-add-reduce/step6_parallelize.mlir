#map = affine_map<()[s0, s1, s2] -> (s1, s0 - s2)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module attributes {transform.with_named_sequence} {
  func.func @broadcast_add_reducesum(%arg0: memref<?xf16>, %arg1: memref<?x?xf16>, %arg2: i64, %arg3: i64) -> memref<?xf16> {
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = ascendc.pipe
    %1 = ascendc.queue : <vecin, 1>
    %2 = ascendc.queue : <vecout, 1>
    %3 = arith.index_cast %arg3 : i64 to index
    %4 = arith.index_cast %arg2 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?xf16>
    %alloc = memref.alloc(%dim) {alignment = 64 : i64} : memref<?xf16>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf16>
    %5 = ascendc.tbuf : <veccalc>
    %6 = ascendc.tbuf : <veccalc>
    %7 = ascendc.tbuf : <veccalc>
    %8 = ascendc.tbuf : <veccalc>
    %9 = ascendc.tbuf : <vecout>
    %10 = ascendc.tbuf : <vecin>
    %11 = ascendc.get_block_idx : index
    %12 = arith.muli %11, %4 : index
    %13 = arith.cmpi ult, %12, %dim : index
    scf.if %13 {
      %14 = affine.min #map()[%dim, %4, %12]
      %subview = memref.subview %arg0[%12] [%14] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_1 = memref.subview %arg1[%12, 0] [%14, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_2 = memref.subview %alloc[%12] [%14] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      scf.for %arg4 = %c0 to %14 step %3 {
        %15 = affine.min #map1(%arg4)[%14, %3]
        %subview_3 = memref.subview %subview[%arg4] [%15] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %16 = arith.muli %15, %c2 : index
        ascendc.pipe.init_buffer %0, %10, %16 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %1, %c1_i32, %16 : !ascendc.queue<vecin, 1>, i32, index
        %17 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %18 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %18, %subview_3 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %17, %18, %15 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %1, %17 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %19 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_4 = memref.subview %subview_1[%arg4, 0] [%15, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_5 = memref.subview %subview_2[%arg4] [%15] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        ascendc.pipe.init_buffer %0, %9, %16 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %2, %c1_i32, %16 : !ascendc.queue<vecout, 1>, i32, index
        %20 = arith.muli %15, %dim_0 : index
        %21 = arith.muli %20, %c2 : index
        ascendc.pipe.init_buffer %0, %8, %21 : !ascendc.tbuf<veccalc>, index
        %22 = ascendc.tbuf.get_tensor %8 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %22, %cst, %20 : !ascendc.local_tensor<*xf16>, f16, index
        %23 = arith.index_cast %15 : index to i32
        %24 = arith.index_cast %dim_0 : index to i32
        ascendc.pipe.init_buffer %0, %7, %21 : !ascendc.tbuf<veccalc>, index
        %25 = ascendc.tbuf.get_tensor %7 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %25, %19, %23, %24, %23, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        ascendc.pipe.init_buffer %0, %6, %21 : !ascendc.tbuf<veccalc>, index
        %26 = ascendc.tbuf.get_tensor %6 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        %27 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %27, %subview_4 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %26, %27, %20 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.pipe.init_buffer %0, %5, %21 : !ascendc.tbuf<veccalc>, index
        %28 = ascendc.tbuf.get_tensor %5 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.add_l2 %28, %25, %26, %20 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %22, %22, %28, %20 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        %29 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.reduce_sum_2d_l2 %29, %22 {layout = 0 : i32} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.enque_tensor %2, %29 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %30 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %31 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %31, %subview_5 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %31, %30, %15 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %30 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %19 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
      }
    }
    return %alloc : memref<?xf16>
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

