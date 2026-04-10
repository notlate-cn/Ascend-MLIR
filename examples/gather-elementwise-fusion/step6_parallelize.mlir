#map = affine_map<()[s0, s1, s2] -> (s1, s0 - s2)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module {
  func.func @relu_index_select_add(%arg0: memref<?x?xf16>, %arg1: memref<?xi64>, %arg2: memref<?xf16>, %arg3: i64, %arg4: i64) -> memref<?x?xf16> {
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = ascendc.pipe
    %1 = ascendc.queue : <vecin, 1>
    %2 = ascendc.queue : <vecout, 1>
    %3 = arith.index_cast %arg4 : i64 to index
    %4 = arith.index_cast %arg3 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf16>
    %dim_0 = memref.dim %arg1, %c0 : memref<?xi64>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf16>
    %5 = ascendc.tbuf : <veccalc>
    %6 = ascendc.tbuf : <veccalc>
    %7 = ascendc.tbuf : <veccalc>
    %8 = ascendc.tbuf : <vecout>
    %9 = ascendc.tbuf : <vecin>
    %10 = ascendc.get_block_idx : index
    %11 = arith.muli %10, %4 : index
    %12 = arith.cmpi ult, %11, %dim : index
    scf.if %12 {
      %13 = affine.min #map()[%dim, %4, %11]
      %subview = memref.subview %arg1[0] [%dim_0] [1] : memref<?xi64> to memref<?xi64, strided<[1]>>
      %subview_1 = memref.subview %arg2[0] [%dim_0] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_2 = memref.subview %alloc[%11, 0] [%13, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg5 = %c0 to %13 step %3 {
        %14 = affine.min #map1(%arg5)[%13, %3]
        %15 = arith.muli %dim_0, %c8 : index
        ascendc.pipe.init_buffer %0, %9, %15 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %1, %c1_i32, %15 : !ascendc.queue<vecin, 1>, i32, index
        %16 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %17 = ascendc.global_tensor : !ascendc.global_tensor<*xi64>
        ascendc.global_tensor.set_global_buffer %17, %subview : !ascendc.global_tensor<*xi64>, memref<?xi64, strided<[1]>>
        ascendc.data_copy_l2 %16, %17, %dim_0 : !ascendc.local_tensor<*xi64>, !ascendc.global_tensor<*xi64>, index
        ascendc.que_bind.enque_tensor %1, %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %18 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %subview_3 = memref.subview %subview_2[%arg5, 0] [%14, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %19 = arith.muli %14, %dim_0 : index
        %20 = arith.muli %19, %c2 : index
        ascendc.pipe.init_buffer %0, %8, %20 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %2, %c1_i32, %20 : !ascendc.queue<vecout, 1>, i32, index
        %dim_4 = memref.dim %arg0, %c1 : memref<?x?xf16>
        %21 = arith.index_cast %dim_0 : index to i32
        %22 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %23 = arith.muli %dim_0, %c2 : index
        %24 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %24, %arg0 : !ascendc.global_tensor<*xf16>, memref<?x?xf16>
        %25 = arith.muli %dim_4, %c2 : index
        ascendc.pipe.init_buffer %0, %7, %25 : !ascendc.tbuf<veccalc>, index
        %26 = ascendc.tbuf.get_tensor %7 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        scf.for %arg6 = %c0 to %14 step %c1 {
          %29 = arith.addi %arg6, %arg5 : index
          %30 = arith.addi %29, %11 : index
          %31 = arith.muli %30, %dim_4 : index
          %32 = ascendc.global_tensor.bracket %24(%31) : !ascendc.global_tensor<*xf16>, index, !ascendc.global_tensor<*xf16>
          %33 = ascendc.tbuf.get_tensor %7 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
          ascendc.data_copy_l2 %33, %32, %dim_4 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
          %34 = arith.muli %arg6, %23 : index
          %35 = ascendc.tbuf.get_with_offset %8, %23, %34 : !ascendc.tbuf<vecout>, index, index, !ascendc.local_tensor<*xf16>
          ascendc.gather_l2 %35, %33, %18, %c0_i32, %21 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xi64>, i32, i32
          ascendc.pipe.init_buffer %0, %6, %23 : !ascendc.tbuf<veccalc>, index
          %36 = ascendc.tbuf.get_tensor %6 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
          ascendc.duplicate_l2 %36, %cst, %21 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, f16, i32
          ascendc.max_l2 %35, %35, %36, %21 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
          ascendc.pipe.init_buffer %0, %5, %23 : !ascendc.tbuf<veccalc>, index
          %37 = ascendc.tbuf.get_tensor %5 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
          %38 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
          ascendc.global_tensor.set_global_buffer %38, %subview_1 : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1]>>
          ascendc.data_copy_l2 %37, %38, %dim_0 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
          ascendc.add_l2 %35, %35, %37, %21 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
        }
        ascendc.que_bind.enque_tensor %2, %22 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %27 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %28 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %28, %subview_3 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %28, %27, %19 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %27 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %18 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
      }
    }
    return %alloc : memref<?x?xf16>
  }
}

