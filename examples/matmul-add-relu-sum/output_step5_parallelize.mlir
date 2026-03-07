#map = affine_map<()[s0, s1, s2] -> (s1, s0 - s2)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
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
    %26 = ascendc.get_block_idx : index
    %27 = arith.muli %26, %15 : index
    %28 = arith.cmpi ult, %27, %dim : index
    scf.if %28 {
      scf.for %arg9 = %c0 to %dim_0 step %14 {
        %30 = affine.min #map()[%dim, %15, %27]
        %31 = affine.min #map1(%arg9)[%dim_0, %14]
        %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview = memref.subview %arg0[%27, 0] [%30, %dim_1] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %32 = arith.muli %30, %dim_1 : index
        %33 = arith.muli %32, %c4 : index
        ascendc.pipe.init_buffer %0, %25, %33 : !ascendc.tbuf<a1>, index
        %34 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %35 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %35, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        %36 = arith.divui %dim_1, %c16 : index
        %37 = arith.index_cast %30 : index to i16
        %38 = arith.index_cast %36 : index to i16
        %39 = arith.index_cast %dim_1 : index to i16
        %40 = ascendc.construct !ascendc.nd2nz_params(%37, %38, %37, %39, %37, %c0_i16, %37, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %34, %35, %40 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %1, %34 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %subview_2 = memref.subview %arg1[0, %arg9] [%dim_1, %31] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %41 = arith.muli %dim_1, %31 : index
        %42 = arith.muli %41, %c4 : index
        ascendc.pipe.init_buffer %0, %24, %42 : !ascendc.tbuf<b1>, index
        %43 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %44 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %44, %subview_2 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        %45 = arith.divui %31, %c16 : index
        %46 = arith.index_cast %45 : index to i16
        %47 = arith.index_cast %31 : index to i16
        %48 = ascendc.construct !ascendc.nd2nz_params(%39, %46, %39, %47, %39, %c0_i16, %39, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %43, %44, %48 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %2, %43 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %subview_3 = memref.subview %arg3[%27, %arg9] [%30, %31] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_4 = memref.subview %arg2[%27, %arg9] [%30, %31] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %49 = arith.muli %30, %31 : index
        %50 = arith.muli %49, %c4 : index
        ascendc.pipe.init_buffer %0, %23, %50 : !ascendc.tbuf<vecin>, index
        %51 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %52 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %52, %subview_4 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        ascendc.data_copy_l2 %51, %52, %49 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %3, %51 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %53 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %0, %22, %50 : !ascendc.tbuf<vecout>, index
        %54 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %55 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        scf.for %arg10 = %c0 to %30 step %13 {
          %58 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          scf.for %arg11 = %c0 to %31 step %12 {
            %59 = affine.min #map1(%arg10)[%30, %13]
            %60 = affine.min #map1(%arg11)[%31, %12]
            %61 = arith.muli %59, %60 : index
            %62 = arith.muli %61, %c4 : index
            ascendc.pipe.init_buffer %0, %21, %62 : !ascendc.tbuf<co1>, index
            %63 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            scf.for %arg12 = %c0 to %dim_1 step %11 {
              %77 = affine.min #map1(%arg12)[%dim_1, %11]
              %78 = arith.muli %59, %77 : index
              %79 = arith.muli %78, %c4 : index
              ascendc.pipe.init_buffer %0, %20, %79 : !ascendc.tbuf<a2>, index
              %80 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %81 = arith.divui %59, %c16 : index
              %82 = arith.index_cast %81 : index to i16
              %83 = arith.divui %77, %c16 : index
              %84 = arith.index_cast %83 : index to i64
              %85 = arith.trunci %84 : i64 to i8
              %86 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %85, %82, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %80, %54, %86 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %80 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %87 = arith.muli %77, %60 : index
              %88 = arith.muli %87, %c4 : index
              ascendc.pipe.init_buffer %0, %19, %88 : !ascendc.tbuf<b2>, index
              %89 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %85, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %89, %55, %90 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %91 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %92 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %93 = arith.index_cast %59 : index to i16
              %94 = arith.index_cast %77 : index to i16
              %95 = arith.index_cast %60 : index to i16
              %96 = ascendc.construct !ascendc.mmad_params(%93, %95, %94, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %91, %92, %96 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %91 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %92 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %63 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %18, %62 : !ascendc.tbuf<vecin>, index
            %64 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %65 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %66 = ascendc.construct !ascendc.data_copy_co12dst_params()
            ascendc.data_copy_co12dst %65, %64, %66 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.data_copy_co12dst_params
            ascendc.que_bind.enque_tensor %8, %65 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %64 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %17, %62 : !ascendc.tbuf<veccalc>, index
            %67 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %68 = arith.muli %arg10, %31 : index
            %69 = arith.addi %68, %arg11 : index
            %70 = arith.muli %69, %c4 : index
            %71 = ascendc.tbuf.get_with_offset %23, %62, %70 : !ascendc.tbuf<vecin>, index, index, !ascendc.local_tensor<*xf32>
            %72 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %72, %67, %71, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %72 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %67 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %16, %62 : !ascendc.tbuf<vecin>, index
            %73 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %73, %cst, %61 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %73 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %74 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %75 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %76 = ascendc.tbuf.get_with_offset %22, %62, %70 : !ascendc.tbuf<vecout>, index, index, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %76, %74, %75, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %74 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %75 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          ascendc.que_bind.enque_tensor %4, %58 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        }
        ascendc.que_bind.free_tensor %2, %55 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %1, %54 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %56 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %57 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %57, %subview_3 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        ascendc.data_copy_l2 %57, %56, %49 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %4, %56 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %3, %53 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    %29 = bufferization.clone %arg3 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    return %29 : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }
}

