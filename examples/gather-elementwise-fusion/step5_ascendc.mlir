#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
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
    %5 = ascendc.tbuf : <gm>
    %6 = ascendc.tbuf : <veccalc>
    %7 = ascendc.tbuf : <veccalc>
    %8 = ascendc.tbuf : <vecout>
    %9 = ascendc.tbuf : <vecin>
    scf.for %arg5 = %c0 to %dim step %4 {
      %10 = affine.min #map(%arg5)[%dim, %4]
      %subview = memref.subview %arg1[0] [%dim_0] [1] : memref<?xi64> to memref<?xi64, strided<[1]>>
      %subview_1 = memref.subview %alloc[%arg5, 0] [%10, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg6 = %c0 to %10 step %3 {
        %11 = affine.min #map(%arg6)[%10, %3]
        %12 = arith.muli %dim_0, %c8 : index
        ascendc.pipe.init_buffer %0, %9, %12 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %1, %c1_i32, %12 : !ascendc.queue<vecin, 1>, i32, index
        %13 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %14 = ascendc.global_tensor : !ascendc.global_tensor<*xi64>
        ascendc.global_tensor.set_global_buffer %14, %subview : !ascendc.global_tensor<*xi64>, memref<?xi64, strided<[1]>>
        ascendc.data_copy_l2 %13, %14, %dim_0 : !ascendc.local_tensor<*xi64>, !ascendc.global_tensor<*xi64>, index
        ascendc.que_bind.enque_tensor %1, %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %15 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
        %subview_2 = memref.subview %subview_1[%arg6, 0] [%11, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %16 = arith.muli %11, %dim_0 : index
        %17 = arith.muli %16, %c2 : index
        ascendc.pipe.init_buffer %0, %8, %17 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %0, %2, %c1_i32, %17 : !ascendc.queue<vecout, 1>, i32, index
        %dim_3 = memref.dim %arg0, %c1 : memref<?x?xf16>
        %18 = arith.index_cast %dim_0 : index to i32
        %19 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %20 = arith.muli %dim_0, %c2 : index
        %21 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %21, %arg0 : !ascendc.global_tensor<*xf16>, memref<?x?xf16>
        %22 = arith.muli %dim_3, %c2 : index
        ascendc.pipe.init_buffer %0, %7, %22 : !ascendc.tbuf<veccalc>, index
        %23 = ascendc.tbuf.get_tensor %7 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
        scf.for %arg7 = %c0 to %11 step %c1 {
          %26 = arith.addi %arg7, %arg6 : index
          %27 = arith.addi %26, %arg5 : index
          %28 = arith.muli %27, %dim_3 : index
          %29 = ascendc.global_tensor.bracket %21(%28) : !ascendc.global_tensor<*xf16>, index, !ascendc.global_tensor<*xf16>
          %30 = ascendc.tbuf.get_tensor %7 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
          ascendc.data_copy_l2 %30, %29, %dim_3 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
          %31 = arith.muli %arg7, %20 : index
          %32 = ascendc.tbuf.get_with_offset %8, %20, %31 : !ascendc.tbuf<vecout>, index, index, !ascendc.local_tensor<*xf16>
          ascendc.gather_l2 %32, %30, %15, %c0_i32, %18 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xi64>, i32, i32
          ascendc.pipe.init_buffer %0, %6, %20 : !ascendc.tbuf<veccalc>, index
          %33 = ascendc.tbuf.get_tensor %6 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>
          ascendc.duplicate_l2 %33, %cst, %18 : !ascendc.local_tensor<*xf16>, f16, i32
          ascendc.max_l2 %32, %32, %33, %18 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
          %34 = ascendc.tbuf.get_tensor %5 : !ascendc.tbuf<gm>, !ascendc.local_tensor<*xf16>
          ascendc.add_l2 %32, %32, %34, %18 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
        }
        ascendc.que_bind.enque_tensor %2, %19 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %24 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %25 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %25, %subview_2 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        ascendc.data_copy_l2 %25, %24, %16 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
        ascendc.que_bind.free_tensor %2, %24 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xi64>
      }
    }
    return %alloc : memref<?x?xf16>
  }
}

