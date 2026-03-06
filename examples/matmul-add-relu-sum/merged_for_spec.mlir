module {
  func.func @fc_relu(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg1: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg3: memref<?x?xf32, strided<[?, ?], offset: ?>>) -> memref<?x?xf32, strided<[?, ?], offset: ?>>  attributes {alwaysinline} {
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
    %11 = arith.constant 32 : index
    %12 = arith.constant 64 : index
    %13 = arith.constant 64 : index
    %14 = arith.constant 128 : index
    %15 = arith.constant 128 : index
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
        %27 = arith.subi %dim, %arg9 : index
        %28 = arith.minsi %27, %15 : index
        %29 = arith.subi %dim_0, %arg10 : index
        %30 = arith.minsi %29, %14 : index
        %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview = memref.subview %arg0[%arg9, 0] [%28, %dim_1] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %31 = arith.muli %28, %dim_1 : index
        %32 = arith.muli %31, %c4 : index
        ascendc.pipe.init_buffer %0, %25, %32 : !ascendc.tbuf<a1>, index
        %33 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %34 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %34, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        %35 = arith.divui %dim_1, %c16 : index
        %36 = arith.index_cast %28 : index to i16
        %37 = arith.index_cast %35 : index to i16
        %38 = arith.index_cast %dim_1 : index to i16
        %39 = ascendc.construct !ascendc.nd2nz_params(%36, %37, %36, %38, %36, %c0_i16, %36, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %33, %34, %39 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %1, %33 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %subview_2 = memref.subview %arg1[0, %arg10] [%dim_1, %30] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %40 = arith.muli %dim_1, %30 : index
        %41 = arith.muli %40, %c4 : index
        ascendc.pipe.init_buffer %0, %24, %41 : !ascendc.tbuf<b1>, index
        %42 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %43 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %43, %subview_2 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        %44 = arith.divui %30, %c16 : index
        %45 = arith.index_cast %44 : index to i16
        %46 = arith.index_cast %30 : index to i16
        %47 = ascendc.construct !ascendc.nd2nz_params(%38, %45, %38, %46, %38, %c0_i16, %38, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %42, %43, %47 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %2, %42 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %subview_3 = memref.subview %arg3[%arg9, %arg10] [%28, %30] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_4 = memref.subview %arg2[%arg9, %arg10] [%28, %30] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %48 = arith.muli %28, %30 : index
        %49 = arith.muli %48, %c4 : index
        ascendc.pipe.init_buffer %0, %23, %49 : !ascendc.tbuf<vecin>, index
        %50 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %51 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %51, %subview_4 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        ascendc.data_copy_l2 %50, %51, %48 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %3, %50 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %52 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %0, %22, %49 : !ascendc.tbuf<vecout>, index
        %53 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %54 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        scf.for %arg11 = %c0 to %28 step %13 {
          %57 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          scf.for %arg12 = %c0 to %30 step %12 {
            %58 = arith.subi %28, %arg11 : index
            %59 = arith.minsi %58, %13 : index
            %60 = arith.subi %30, %arg12 : index
            %61 = arith.minsi %60, %12 : index
            %62 = arith.muli %59, %61 : index
            %63 = arith.muli %62, %c4 : index
            ascendc.pipe.init_buffer %0, %21, %63 : !ascendc.tbuf<co1>, index
            %64 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            scf.for %arg13 = %c0 to %dim_1 step %11 {
              %73 = arith.subi %dim_1, %arg13 : index
              %74 = arith.minsi %73, %11 : index
              %75 = arith.muli %59, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %20, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %59, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %53, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %61 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %19, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %54, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %59 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %61 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %64, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %64 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %18, %63 : !ascendc.tbuf<vecin>, index
            %65 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %66 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %67 = ascendc.construct !ascendc.fixpipe_params<f32>()
            ascendc.fixpipe %66, %65, %67 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<f32>
            ascendc.que_bind.enque_tensor %8, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %65 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %17, %63 : !ascendc.tbuf<veccalc>, index
            %68 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %69 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %69, %68, %52, %62 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %69 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %68 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %16, %63 : !ascendc.tbuf<vecin>, index
            %70 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %70, %cst, %62 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %72 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %57, %71, %72, %62 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %71 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          ascendc.que_bind.enque_tensor %4, %57 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        }
        ascendc.que_bind.free_tensor %2, %54 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %1, %53 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %55 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %56 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %56, %subview_3 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        ascendc.data_copy_l2 %56, %55, %48 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %4, %55 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %3, %52 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    %26 = bufferization.clone %arg3 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    return %26 : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }
}

