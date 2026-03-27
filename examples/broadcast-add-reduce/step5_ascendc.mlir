#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
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
    %6 = ascendc.queue : <vecin, 1>
    %7 = ascendc.tbuf : <vecin>
    %8 = ascendc.tbuf : <veccalc>
    %9 = ascendc.tbuf : <veccalc>
    %10 = ascendc.tbuf : <vecout>
    %11 = ascendc.tbuf : <vecin>
    scf.for %arg4 = %c0 to %dim step %4 {
      %12 = affine.min #map(%arg4)[%dim, %4]
      %subview = memref.subview %arg0[%arg4] [%12] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_1 = memref.subview %arg1[%arg4, 0] [%12, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_2 = memref.subview %alloc[%arg4] [%12] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      scf.for %arg5 = %c0 to %12 step %3 {
        %13 = affine.min #map(%arg5)[%12, %3]
        %subview_3 = memref.subview %subview[%arg5] [%13] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %14 = arith.muli %13, %c2 : index
        ascendc.pipe.init_buffer %0, %11, %14 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %1, %c1_i32, %14 : !ascendc.queue<vecin, 1>, i32, index
        %15 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %16 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %16, %subview_3 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %15, %16, %13 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %1, %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %17 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %subview_4 = memref.subview %subview_1[%arg5, 0] [%13, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_5 = memref.subview %subview_2[%arg5] [%13] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        ascendc.pipe.init_buffer %0, %10, %14 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %2, %c1_i32, %14 : !ascendc.queue<vecout, 1>, i32, index
        %18 = arith.muli %13, %dim_0 : index
        %19 = arith.muli %18, %c2 : index
        ascendc.pipe.init_buffer %0, %9, %19 : !ascendc.tbuf<veccalc>, index
        %20 = ascendc.tbuf.get_tensor %9 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.duplicate_l2 %20, %cst, %18 : !ascendc.local_tensor<*xf16>, f16, index
        %21 = arith.index_cast %13 : index to i32
        %22 = arith.index_cast %dim_0 : index to i32
        ascendc.pipe.init_buffer %0, %8, %19 : !ascendc.tbuf<veccalc>, index
        %23 = ascendc.tbuf.get_tensor %8 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.broadcast_l2 %23, %17, %21, %22, %21, %c1_i32 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32
        %24 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %24, %subview_4 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.pipe.init_buffer %0, %7, %19 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %6, %c1_i32, %19 : !ascendc.queue<vecin, 1>, i32, index
        %25 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.data_copy_l2 %25, %24, %18 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
        ascendc.que_bind.enque_tensor %6, %25 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %26 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        ascendc.pipe.init_buffer %0, %5, %19 : !ascendc.tbuf<veccalc>, index
        %27 = ascendc.tbuf.get_tensor %5 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        ascendc.add_l2 %27, %23, %26, %18 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.add_l2 %20, %20, %27, %18 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        %28 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.reduce_sum_2d_l2 %28, %20 {layout = 0 : i32} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.enque_tensor %2, %28 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %29 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %30 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %30, %subview_5 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %30, %29, %13 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %29 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %17 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
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

