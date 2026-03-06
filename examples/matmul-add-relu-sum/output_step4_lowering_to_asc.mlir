#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module {
  func.func @fc_relu(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg1: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg3: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {
    %true = arith.constant true
    %c1_i16 = arith.constant 1 : i16
    %c0_i8 = arith.constant 0 : i8
    %false = arith.constant false
    %c0_i16 = arith.constant 0 : i16
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = ascendc.pipe
    %1 = ascendc.queue : <a1, 1>
    %2 = ascendc.queue : <b1, 1>
    %3 = ascendc.queue : <vecin, 1>
    %4 = ascendc.queue : <vecout, 1>
    %5 = ascendc.queue : <co1, 1>
    %6 = ascendc.queue : <a2, 1>
    %7 = ascendc.queue : <b2, 1>
    %8 = ascendc.queue : <vecin, 1>
    %9 = ascendc.queue : <veccalc, 1>
    %10 = ascendc.queue : <vecin, 1>
    %11 = arith.index_cast %arg8 : i64 to index
    %12 = arith.index_cast %arg7 : i64 to index
    %13 = arith.index_cast %arg6 : i64 to index
    %14 = arith.index_cast %arg5 : i64 to index
    %15 = arith.index_cast %arg4 : i64 to index
    %dim = memref.dim %arg3, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %16 = ascendc.tbuf : <vecin>
    %17 = ascendc.tbuf : <veccalc>
    %18 = ascendc.tbuf : <vecin>
    %19 = ascendc.tbuf : <b2>
    %20 = ascendc.tbuf : <a2>
    %21 = ascendc.tbuf : <co1>
    %22 = ascendc.tbuf : <vecout>
    %23 = ascendc.tbuf : <vecin>
    %24 = ascendc.tbuf : <b1>
    %25 = ascendc.tbuf : <a1>
    scf.for %arg9 = %c0 to %dim step %15 {
      scf.for %arg10 = %c0 to %dim_0 step %14 {
        %27 = affine.min #map(%arg9)[%dim, %15]
        %28 = affine.min #map(%arg10)[%dim_0, %14]
        %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview = memref.subview %arg0[%arg9, 0] [%27, %dim_1] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %29 = arith.muli %27, %dim_1 : index
        %30 = arith.muli %29, %c4 : index
        ascendc.pipe.init_buffer %0, %25, %30 : !ascendc.tbuf<a1>, index
        %31 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %32 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %32, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        %33 = arith.divui %dim_1, %c16 : index
        %34 = arith.index_cast %27 : index to i16
        %35 = arith.index_cast %33 : index to i16
        %36 = arith.index_cast %dim_1 : index to i16
        %37 = ascendc.construct !ascendc.nd2nz_params(%34, %35, %34, %36, %34, %c0_i16, %34, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %31, %32, %37 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %1, %31 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %subview_2 = memref.subview %arg1[0, %arg10] [%dim_1, %28] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %38 = arith.muli %dim_1, %28 : index
        %39 = arith.muli %38, %c4 : index
        ascendc.pipe.init_buffer %0, %24, %39 : !ascendc.tbuf<b1>, index
        %40 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %41 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %41, %subview_2 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        %42 = arith.divui %28, %c16 : index
        %43 = arith.index_cast %42 : index to i16
        %44 = arith.index_cast %28 : index to i16
        %45 = ascendc.construct !ascendc.nd2nz_params(%36, %43, %36, %44, %36, %c0_i16, %36, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %40, %41, %45 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %2, %40 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %subview_3 = memref.subview %arg3[%arg9, %arg10] [%27, %28] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_4 = memref.subview %arg2[%arg9, %arg10] [%27, %28] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %46 = arith.muli %27, %28 : index
        %47 = arith.muli %46, %c4 : index
        ascendc.pipe.init_buffer %0, %23, %47 : !ascendc.tbuf<vecin>, index
        %48 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %49 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %49, %subview_4 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        ascendc.data_copy_l2 %48, %49, %46 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %3, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %50 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %0, %22, %47 : !ascendc.tbuf<vecout>, index
        %51 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %52 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        scf.for %arg11 = %c0 to %27 step %13 {
          %55 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          scf.for %arg12 = %c0 to %28 step %12 {
            %56 = affine.min #map(%arg11)[%27, %13]
            %57 = affine.min #map(%arg12)[%28, %12]
            %58 = arith.muli %56, %57 : index
            %59 = arith.muli %58, %c4 : index
            ascendc.pipe.init_buffer %0, %21, %59 : !ascendc.tbuf<co1>, index
            %60 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            scf.for %arg13 = %c0 to %dim_1 step %11 {
              %74 = affine.min #map(%arg13)[%dim_1, %11]
              %75 = arith.muli %56, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %20, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %56, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %51, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %57 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %19, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %52, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %56 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %57 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %60, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %60 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %18, %59 : !ascendc.tbuf<vecin>, index
            %61 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %62 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %63 = ascendc.construct !ascendc.data_copy_co12dst_params()
            ascendc.data_copy_co12dst %62, %61, %63 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.data_copy_co12dst_params
            ascendc.que_bind.enque_tensor %8, %62 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %61 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %17, %59 : !ascendc.tbuf<veccalc>, index
            %64 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %65 = arith.muli %arg11, %28 : index
            %66 = arith.addi %65, %arg12 : index
            %67 = arith.muli %66, %c4 : index
            %68 = ascendc.tbuf.get_with_offset %23, %59, %67 : !ascendc.tbuf<vecin>, index, index, !ascendc.local_tensor<*xf32>
            %69 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %69, %64, %68, %58 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %69 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %64 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %16, %59 : !ascendc.tbuf<vecin>, index
            %70 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %70, %cst, %58 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %72 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %73 = ascendc.tbuf.get_with_offset %22, %59, %67 : !ascendc.tbuf<vecout>, index, index, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %73, %71, %72, %58 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %71 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          ascendc.que_bind.enque_tensor %4, %55 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        }
        ascendc.que_bind.free_tensor %2, %52 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %1, %51 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %53 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %54, %subview_3 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        ascendc.data_copy_l2 %54, %53, %46 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %4, %53 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %3, %50 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    %26 = bufferization.clone %arg3 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    return %26 : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }
}

