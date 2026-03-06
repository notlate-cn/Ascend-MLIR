#map = affine_map<()[s0] -> ((s0 floordiv 128) * 128)>
#map1 = affine_map<()[s0] -> ((s0 floordiv 64) * 64)>
#map2 = affine_map<()[s0] -> ((s0 floordiv 32) * 32)>
module {
  func.func @fc_relu(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg1: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg3: memref<?x?xf32, strided<[?, ?], offset: ?>>) -> memref<?x?xf32, strided<[?, ?], offset: ?>> attributes {alwaysinline} {
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
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
    %dim = memref.dim %arg3, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %11 = ascendc.tbuf : <vecin>
    %12 = ascendc.tbuf : <veccalc>
    %13 = ascendc.tbuf : <vecin>
    %14 = ascendc.tbuf : <b2>
    %15 = ascendc.tbuf : <a2>
    %16 = ascendc.tbuf : <co1>
    %17 = ascendc.tbuf : <vecout>
    %18 = ascendc.tbuf : <vecin>
    %19 = ascendc.tbuf : <b1>
    %20 = ascendc.tbuf : <a1>
    %21 = affine.apply #map()[%dim]
    scf.for %arg4 = %c0 to %21 step %c128 {
      %23 = affine.apply #map()[%dim_0]
      scf.for %arg5 = %c0 to %23 step %c128 {
        %24 = arith.subi %dim, %arg4 : index
        %25 = arith.minsi %24, %c128 : index
        %26 = arith.subi %dim_0, %arg5 : index
        %27 = arith.minsi %26, %c128 : index
        %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview = memref.subview %arg0[%arg4, 0] [%25, %dim_1] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %28 = arith.muli %25, %dim_1 : index
        %29 = arith.muli %28, %c4 : index
        ascendc.pipe.init_buffer %0, %20, %29 : !ascendc.tbuf<a1>, index
        %30 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %31 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %31, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        %32 = arith.divui %dim_1, %c16 : index
        %33 = arith.index_cast %25 : index to i16
        %34 = arith.index_cast %32 : index to i16
        %35 = arith.index_cast %dim_1 : index to i16
        %36 = ascendc.construct !ascendc.nd2nz_params(%33, %34, %33, %35, %33, %c0_i16, %33, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %30, %31, %36 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %1, %30 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %subview_2 = memref.subview %arg1[0, %arg5] [%dim_1, %27] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %37 = arith.muli %dim_1, %27 : index
        %38 = arith.muli %37, %c4 : index
        ascendc.pipe.init_buffer %0, %19, %38 : !ascendc.tbuf<b1>, index
        %39 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %40 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %40, %subview_2 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        %41 = arith.divui %27, %c16 : index
        %42 = arith.index_cast %41 : index to i16
        %43 = arith.index_cast %27 : index to i16
        %44 = ascendc.construct !ascendc.nd2nz_params(%35, %42, %35, %43, %35, %c0_i16, %35, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %39, %40, %44 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %2, %39 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %subview_3 = memref.subview %arg3[%arg4, %arg5] [%25, %27] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_4 = memref.subview %arg2[%arg4, %arg5] [%25, %27] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %45 = arith.muli %25, %27 : index
        %46 = arith.muli %45, %c4 : index
        ascendc.pipe.init_buffer %0, %18, %46 : !ascendc.tbuf<vecin>, index
        %47 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %48 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %48, %subview_4 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        ascendc.data_copy_l2 %47, %48, %45 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %3, %47 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %49 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %0, %17, %46 : !ascendc.tbuf<vecout>, index
        %50 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %51 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %52 = affine.apply #map1()[%25]
        scf.for %arg6 = %c0 to %52 step %c64 {
          %55 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          %56 = affine.apply #map1()[%27]
          scf.for %arg7 = %c0 to %56 step %c64 {
            %57 = arith.subi %25, %arg6 : index
            %58 = arith.minsi %57, %c64 : index
            %59 = arith.subi %27, %arg7 : index
            %60 = arith.minsi %59, %c64 : index
            %61 = arith.muli %58, %60 : index
            %62 = arith.muli %61, %c4 : index
            ascendc.pipe.init_buffer %0, %16, %62 : !ascendc.tbuf<co1>, index
            %63 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %64 = affine.apply #map2()[%dim_1]
            scf.for %arg8 = %c0 to %64 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            scf.for %arg8 = %64 to %dim_1 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %63 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %13, %62 : !ascendc.tbuf<vecin>, index
            %65 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %66 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %67 = ascendc.construct !ascendc.fixpipe_params<f32>()
            ascendc.fixpipe %66, %65, %67 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<f32>
            ascendc.que_bind.enque_tensor %8, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %65 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %12, %62 : !ascendc.tbuf<veccalc>, index
            %68 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %69 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %69, %68, %49, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %69 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %68 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %11, %62 : !ascendc.tbuf<vecin>, index
            %70 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %70, %cst, %61 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %72 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %55, %71, %72, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %71 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          scf.for %arg7 = %56 to %27 step %c64 {
            %57 = arith.subi %25, %arg6 : index
            %58 = arith.minsi %57, %c64 : index
            %59 = arith.subi %27, %arg7 : index
            %60 = arith.minsi %59, %c64 : index
            %61 = arith.muli %58, %60 : index
            %62 = arith.muli %61, %c4 : index
            ascendc.pipe.init_buffer %0, %16, %62 : !ascendc.tbuf<co1>, index
            %63 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %64 = affine.apply #map2()[%dim_1]
            scf.for %arg8 = %c0 to %64 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            scf.for %arg8 = %64 to %dim_1 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %63 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %13, %62 : !ascendc.tbuf<vecin>, index
            %65 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %66 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %67 = ascendc.construct !ascendc.fixpipe_params<f32>()
            ascendc.fixpipe %66, %65, %67 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<f32>
            ascendc.que_bind.enque_tensor %8, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %65 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %12, %62 : !ascendc.tbuf<veccalc>, index
            %68 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %69 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %69, %68, %49, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %69 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %68 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %11, %62 : !ascendc.tbuf<vecin>, index
            %70 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %70, %cst, %61 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %72 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %55, %71, %72, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %71 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          ascendc.que_bind.enque_tensor %4, %55 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        }
        scf.for %arg6 = %52 to %25 step %c64 {
          %55 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          %56 = affine.apply #map1()[%27]
          scf.for %arg7 = %c0 to %56 step %c64 {
            %57 = arith.subi %25, %arg6 : index
            %58 = arith.minsi %57, %c64 : index
            %59 = arith.subi %27, %arg7 : index
            %60 = arith.minsi %59, %c64 : index
            %61 = arith.muli %58, %60 : index
            %62 = arith.muli %61, %c4 : index
            ascendc.pipe.init_buffer %0, %16, %62 : !ascendc.tbuf<co1>, index
            %63 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %64 = affine.apply #map2()[%dim_1]
            scf.for %arg8 = %c0 to %64 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            scf.for %arg8 = %64 to %dim_1 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %63 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %13, %62 : !ascendc.tbuf<vecin>, index
            %65 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %66 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %67 = ascendc.construct !ascendc.fixpipe_params<f32>()
            ascendc.fixpipe %66, %65, %67 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<f32>
            ascendc.que_bind.enque_tensor %8, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %65 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %12, %62 : !ascendc.tbuf<veccalc>, index
            %68 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %69 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %69, %68, %49, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %69 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %68 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %11, %62 : !ascendc.tbuf<vecin>, index
            %70 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %70, %cst, %61 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %72 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %55, %71, %72, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %71 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          scf.for %arg7 = %56 to %27 step %c64 {
            %57 = arith.subi %25, %arg6 : index
            %58 = arith.minsi %57, %c64 : index
            %59 = arith.subi %27, %arg7 : index
            %60 = arith.minsi %59, %c64 : index
            %61 = arith.muli %58, %60 : index
            %62 = arith.muli %61, %c4 : index
            ascendc.pipe.init_buffer %0, %16, %62 : !ascendc.tbuf<co1>, index
            %63 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %64 = affine.apply #map2()[%dim_1]
            scf.for %arg8 = %c0 to %64 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            scf.for %arg8 = %64 to %dim_1 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %63 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %13, %62 : !ascendc.tbuf<vecin>, index
            %65 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %66 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %67 = ascendc.construct !ascendc.fixpipe_params<f32>()
            ascendc.fixpipe %66, %65, %67 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<f32>
            ascendc.que_bind.enque_tensor %8, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %65 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %12, %62 : !ascendc.tbuf<veccalc>, index
            %68 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %69 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %69, %68, %49, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %69 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %68 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %11, %62 : !ascendc.tbuf<vecin>, index
            %70 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %70, %cst, %61 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %72 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %55, %71, %72, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %71 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          ascendc.que_bind.enque_tensor %4, %55 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        }
        ascendc.que_bind.free_tensor %2, %51 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %1, %50 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %53 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %54, %subview_3 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        ascendc.data_copy_l2 %54, %53, %45 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %4, %53 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %3, %49 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
      scf.for %arg5 = %23 to %dim_0 step %c128 {
        %24 = arith.subi %dim, %arg4 : index
        %25 = arith.minsi %24, %c128 : index
        %26 = arith.subi %dim_0, %arg5 : index
        %27 = arith.minsi %26, %c128 : index
        %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview = memref.subview %arg0[%arg4, 0] [%25, %dim_1] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %28 = arith.muli %25, %dim_1 : index
        %29 = arith.muli %28, %c4 : index
        ascendc.pipe.init_buffer %0, %20, %29 : !ascendc.tbuf<a1>, index
        %30 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %31 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %31, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        %32 = arith.divui %dim_1, %c16 : index
        %33 = arith.index_cast %25 : index to i16
        %34 = arith.index_cast %32 : index to i16
        %35 = arith.index_cast %dim_1 : index to i16
        %36 = ascendc.construct !ascendc.nd2nz_params(%33, %34, %33, %35, %33, %c0_i16, %33, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %30, %31, %36 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %1, %30 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %subview_2 = memref.subview %arg1[0, %arg5] [%dim_1, %27] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %37 = arith.muli %dim_1, %27 : index
        %38 = arith.muli %37, %c4 : index
        ascendc.pipe.init_buffer %0, %19, %38 : !ascendc.tbuf<b1>, index
        %39 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %40 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %40, %subview_2 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        %41 = arith.divui %27, %c16 : index
        %42 = arith.index_cast %41 : index to i16
        %43 = arith.index_cast %27 : index to i16
        %44 = ascendc.construct !ascendc.nd2nz_params(%35, %42, %35, %43, %35, %c0_i16, %35, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %39, %40, %44 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %2, %39 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %subview_3 = memref.subview %arg3[%arg4, %arg5] [%25, %27] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_4 = memref.subview %arg2[%arg4, %arg5] [%25, %27] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %45 = arith.muli %25, %27 : index
        %46 = arith.muli %45, %c4 : index
        ascendc.pipe.init_buffer %0, %18, %46 : !ascendc.tbuf<vecin>, index
        %47 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %48 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %48, %subview_4 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        ascendc.data_copy_l2 %47, %48, %45 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %3, %47 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %49 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %0, %17, %46 : !ascendc.tbuf<vecout>, index
        %50 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %51 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %52 = affine.apply #map1()[%25]
        scf.for %arg6 = %c0 to %52 step %c64 {
          %55 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          %56 = affine.apply #map1()[%27]
          scf.for %arg7 = %c0 to %56 step %c64 {
            %57 = arith.subi %25, %arg6 : index
            %58 = arith.minsi %57, %c64 : index
            %59 = arith.subi %27, %arg7 : index
            %60 = arith.minsi %59, %c64 : index
            %61 = arith.muli %58, %60 : index
            %62 = arith.muli %61, %c4 : index
            ascendc.pipe.init_buffer %0, %16, %62 : !ascendc.tbuf<co1>, index
            %63 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %64 = affine.apply #map2()[%dim_1]
            scf.for %arg8 = %c0 to %64 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            scf.for %arg8 = %64 to %dim_1 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %63 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %13, %62 : !ascendc.tbuf<vecin>, index
            %65 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %66 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %67 = ascendc.construct !ascendc.fixpipe_params<f32>()
            ascendc.fixpipe %66, %65, %67 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<f32>
            ascendc.que_bind.enque_tensor %8, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %65 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %12, %62 : !ascendc.tbuf<veccalc>, index
            %68 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %69 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %69, %68, %49, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %69 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %68 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %11, %62 : !ascendc.tbuf<vecin>, index
            %70 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %70, %cst, %61 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %72 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %55, %71, %72, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %71 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          scf.for %arg7 = %56 to %27 step %c64 {
            %57 = arith.subi %25, %arg6 : index
            %58 = arith.minsi %57, %c64 : index
            %59 = arith.subi %27, %arg7 : index
            %60 = arith.minsi %59, %c64 : index
            %61 = arith.muli %58, %60 : index
            %62 = arith.muli %61, %c4 : index
            ascendc.pipe.init_buffer %0, %16, %62 : !ascendc.tbuf<co1>, index
            %63 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %64 = affine.apply #map2()[%dim_1]
            scf.for %arg8 = %c0 to %64 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            scf.for %arg8 = %64 to %dim_1 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %63 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %13, %62 : !ascendc.tbuf<vecin>, index
            %65 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %66 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %67 = ascendc.construct !ascendc.fixpipe_params<f32>()
            ascendc.fixpipe %66, %65, %67 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<f32>
            ascendc.que_bind.enque_tensor %8, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %65 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %12, %62 : !ascendc.tbuf<veccalc>, index
            %68 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %69 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %69, %68, %49, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %69 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %68 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %11, %62 : !ascendc.tbuf<vecin>, index
            %70 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %70, %cst, %61 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %72 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %55, %71, %72, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %71 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          ascendc.que_bind.enque_tensor %4, %55 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        }
        scf.for %arg6 = %52 to %25 step %c64 {
          %55 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          %56 = affine.apply #map1()[%27]
          scf.for %arg7 = %c0 to %56 step %c64 {
            %57 = arith.subi %25, %arg6 : index
            %58 = arith.minsi %57, %c64 : index
            %59 = arith.subi %27, %arg7 : index
            %60 = arith.minsi %59, %c64 : index
            %61 = arith.muli %58, %60 : index
            %62 = arith.muli %61, %c4 : index
            ascendc.pipe.init_buffer %0, %16, %62 : !ascendc.tbuf<co1>, index
            %63 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %64 = affine.apply #map2()[%dim_1]
            scf.for %arg8 = %c0 to %64 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            scf.for %arg8 = %64 to %dim_1 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %63 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %13, %62 : !ascendc.tbuf<vecin>, index
            %65 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %66 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %67 = ascendc.construct !ascendc.fixpipe_params<f32>()
            ascendc.fixpipe %66, %65, %67 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<f32>
            ascendc.que_bind.enque_tensor %8, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %65 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %12, %62 : !ascendc.tbuf<veccalc>, index
            %68 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %69 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %69, %68, %49, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %69 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %68 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %11, %62 : !ascendc.tbuf<vecin>, index
            %70 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %70, %cst, %61 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %72 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %55, %71, %72, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %71 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          scf.for %arg7 = %56 to %27 step %c64 {
            %57 = arith.subi %25, %arg6 : index
            %58 = arith.minsi %57, %c64 : index
            %59 = arith.subi %27, %arg7 : index
            %60 = arith.minsi %59, %c64 : index
            %61 = arith.muli %58, %60 : index
            %62 = arith.muli %61, %c4 : index
            ascendc.pipe.init_buffer %0, %16, %62 : !ascendc.tbuf<co1>, index
            %63 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %64 = affine.apply #map2()[%dim_1]
            scf.for %arg8 = %c0 to %64 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            scf.for %arg8 = %64 to %dim_1 step %c32 {
              %73 = arith.subi %dim_1, %arg8 : index
              %74 = arith.minsi %73, %c32 : index
              %75 = arith.muli %58, %74 : index
              %76 = arith.muli %75, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %76 : !ascendc.tbuf<a2>, index
              %77 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %78 = arith.divui %58, %c16 : index
              %79 = arith.index_cast %78 : index to i16
              %80 = arith.divui %74, %c16 : index
              %81 = arith.index_cast %80 : index to i64
              %82 = arith.trunci %81 : i64 to i8
              %83 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %82, %79, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %77, %50, %83 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %77 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %84 = arith.muli %74, %60 : index
              %85 = arith.muli %84, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %85 : !ascendc.tbuf<b2>, index
              %86 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %82, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %86, %51, %87 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %86 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %89 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %90 = arith.index_cast %58 : index to i16
              %91 = arith.index_cast %74 : index to i16
              %92 = arith.index_cast %60 : index to i16
              %93 = ascendc.construct !ascendc.mmad_params(%90, %92, %91, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %63, %88, %89, %93 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %88 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %89 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %63 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %13, %62 : !ascendc.tbuf<vecin>, index
            %65 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %66 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %67 = ascendc.construct !ascendc.fixpipe_params<f32>()
            ascendc.fixpipe %66, %65, %67 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<f32>
            ascendc.que_bind.enque_tensor %8, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %65 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %12, %62 : !ascendc.tbuf<veccalc>, index
            %68 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %69 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %69, %68, %49, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %69 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %68 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %11, %62 : !ascendc.tbuf<vecin>, index
            %70 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %70, %cst, %61 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %72 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %55, %71, %72, %61 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %71 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          ascendc.que_bind.enque_tensor %4, %55 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        }
        ascendc.que_bind.free_tensor %2, %51 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %1, %50 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %53 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %54, %subview_3 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        ascendc.data_copy_l2 %54, %53, %45 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %4, %53 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %3, %49 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    scf.for %arg4 = %21 to %dim step %c128 {
      scf.for %arg5 = %c0 to %dim_0 step %c128 {
        %23 = arith.subi %dim, %arg4 : index
        %24 = arith.minsi %23, %c128 : index
        %25 = arith.subi %dim_0, %arg5 : index
        %26 = arith.minsi %25, %c128 : index
        %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview = memref.subview %arg0[%arg4, 0] [%24, %dim_1] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %27 = arith.muli %24, %dim_1 : index
        %28 = arith.muli %27, %c4 : index
        ascendc.pipe.init_buffer %0, %20, %28 : !ascendc.tbuf<a1>, index
        %29 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %30 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %30, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        %31 = arith.divui %dim_1, %c16 : index
        %32 = arith.index_cast %24 : index to i16
        %33 = arith.index_cast %31 : index to i16
        %34 = arith.index_cast %dim_1 : index to i16
        %35 = ascendc.construct !ascendc.nd2nz_params(%32, %33, %32, %34, %32, %c0_i16, %32, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %29, %30, %35 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %1, %29 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %subview_2 = memref.subview %arg1[0, %arg5] [%dim_1, %26] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %36 = arith.muli %dim_1, %26 : index
        %37 = arith.muli %36, %c4 : index
        ascendc.pipe.init_buffer %0, %19, %37 : !ascendc.tbuf<b1>, index
        %38 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %39 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %39, %subview_2 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        %40 = arith.divui %26, %c16 : index
        %41 = arith.index_cast %40 : index to i16
        %42 = arith.index_cast %26 : index to i16
        %43 = ascendc.construct !ascendc.nd2nz_params(%34, %41, %34, %42, %34, %c0_i16, %34, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %38, %39, %43 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %2, %38 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %subview_3 = memref.subview %arg3[%arg4, %arg5] [%24, %26] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_4 = memref.subview %arg2[%arg4, %arg5] [%24, %26] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %44 = arith.muli %24, %26 : index
        %45 = arith.muli %44, %c4 : index
        ascendc.pipe.init_buffer %0, %18, %45 : !ascendc.tbuf<vecin>, index
        %46 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %47 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %47, %subview_4 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        ascendc.data_copy_l2 %46, %47, %44 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %3, %46 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %48 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %0, %17, %45 : !ascendc.tbuf<vecout>, index
        %49 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %50 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %51 = affine.apply #map1()[%24]
        scf.for %arg6 = %c0 to %51 step %c64 {
          %54 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          %55 = affine.apply #map1()[%26]
          scf.for %arg7 = %c0 to %55 step %c64 {
            %56 = arith.subi %24, %arg6 : index
            %57 = arith.minsi %56, %c64 : index
            %58 = arith.subi %26, %arg7 : index
            %59 = arith.minsi %58, %c64 : index
            %60 = arith.muli %57, %59 : index
            %61 = arith.muli %60, %c4 : index
            ascendc.pipe.init_buffer %0, %16, %61 : !ascendc.tbuf<co1>, index
            %62 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %63 = affine.apply #map2()[%dim_1]
            scf.for %arg8 = %c0 to %63 step %c32 {
              %72 = arith.subi %dim_1, %arg8 : index
              %73 = arith.minsi %72, %c32 : index
              %74 = arith.muli %57, %73 : index
              %75 = arith.muli %74, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %75 : !ascendc.tbuf<a2>, index
              %76 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %77 = arith.divui %57, %c16 : index
              %78 = arith.index_cast %77 : index to i16
              %79 = arith.divui %73, %c16 : index
              %80 = arith.index_cast %79 : index to i64
              %81 = arith.trunci %80 : i64 to i8
              %82 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %81, %78, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %76, %49, %82 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %76 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %83 = arith.muli %73, %59 : index
              %84 = arith.muli %83, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %84 : !ascendc.tbuf<b2>, index
              %85 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %86 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %81, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %85, %50, %86 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %85 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %89 = arith.index_cast %57 : index to i16
              %90 = arith.index_cast %73 : index to i16
              %91 = arith.index_cast %59 : index to i16
              %92 = ascendc.construct !ascendc.mmad_params(%89, %91, %90, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %62, %87, %88, %92 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %87 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %88 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            scf.for %arg8 = %63 to %dim_1 step %c32 {
              %72 = arith.subi %dim_1, %arg8 : index
              %73 = arith.minsi %72, %c32 : index
              %74 = arith.muli %57, %73 : index
              %75 = arith.muli %74, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %75 : !ascendc.tbuf<a2>, index
              %76 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %77 = arith.divui %57, %c16 : index
              %78 = arith.index_cast %77 : index to i16
              %79 = arith.divui %73, %c16 : index
              %80 = arith.index_cast %79 : index to i64
              %81 = arith.trunci %80 : i64 to i8
              %82 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %81, %78, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %76, %49, %82 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %76 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %83 = arith.muli %73, %59 : index
              %84 = arith.muli %83, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %84 : !ascendc.tbuf<b2>, index
              %85 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %86 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %81, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %85, %50, %86 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %85 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %89 = arith.index_cast %57 : index to i16
              %90 = arith.index_cast %73 : index to i16
              %91 = arith.index_cast %59 : index to i16
              %92 = ascendc.construct !ascendc.mmad_params(%89, %91, %90, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %62, %87, %88, %92 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %87 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %88 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %62 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %13, %61 : !ascendc.tbuf<vecin>, index
            %64 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %65 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %66 = ascendc.construct !ascendc.fixpipe_params<f32>()
            ascendc.fixpipe %65, %64, %66 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<f32>
            ascendc.que_bind.enque_tensor %8, %65 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %64 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %12, %61 : !ascendc.tbuf<veccalc>, index
            %67 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %68 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %68, %67, %48, %60 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %68 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %67 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %11, %61 : !ascendc.tbuf<vecin>, index
            %69 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %69, %cst, %60 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %69 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %70 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %54, %70, %71, %60 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %70 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %71 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          scf.for %arg7 = %55 to %26 step %c64 {
            %56 = arith.subi %24, %arg6 : index
            %57 = arith.minsi %56, %c64 : index
            %58 = arith.subi %26, %arg7 : index
            %59 = arith.minsi %58, %c64 : index
            %60 = arith.muli %57, %59 : index
            %61 = arith.muli %60, %c4 : index
            ascendc.pipe.init_buffer %0, %16, %61 : !ascendc.tbuf<co1>, index
            %62 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %63 = affine.apply #map2()[%dim_1]
            scf.for %arg8 = %c0 to %63 step %c32 {
              %72 = arith.subi %dim_1, %arg8 : index
              %73 = arith.minsi %72, %c32 : index
              %74 = arith.muli %57, %73 : index
              %75 = arith.muli %74, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %75 : !ascendc.tbuf<a2>, index
              %76 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %77 = arith.divui %57, %c16 : index
              %78 = arith.index_cast %77 : index to i16
              %79 = arith.divui %73, %c16 : index
              %80 = arith.index_cast %79 : index to i64
              %81 = arith.trunci %80 : i64 to i8
              %82 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %81, %78, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %76, %49, %82 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %76 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %83 = arith.muli %73, %59 : index
              %84 = arith.muli %83, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %84 : !ascendc.tbuf<b2>, index
              %85 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %86 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %81, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %85, %50, %86 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %85 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %89 = arith.index_cast %57 : index to i16
              %90 = arith.index_cast %73 : index to i16
              %91 = arith.index_cast %59 : index to i16
              %92 = ascendc.construct !ascendc.mmad_params(%89, %91, %90, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %62, %87, %88, %92 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %87 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %88 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            scf.for %arg8 = %63 to %dim_1 step %c32 {
              %72 = arith.subi %dim_1, %arg8 : index
              %73 = arith.minsi %72, %c32 : index
              %74 = arith.muli %57, %73 : index
              %75 = arith.muli %74, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %75 : !ascendc.tbuf<a2>, index
              %76 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %77 = arith.divui %57, %c16 : index
              %78 = arith.index_cast %77 : index to i16
              %79 = arith.divui %73, %c16 : index
              %80 = arith.index_cast %79 : index to i64
              %81 = arith.trunci %80 : i64 to i8
              %82 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %81, %78, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %76, %49, %82 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %76 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %83 = arith.muli %73, %59 : index
              %84 = arith.muli %83, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %84 : !ascendc.tbuf<b2>, index
              %85 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %86 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %81, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %85, %50, %86 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %85 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %89 = arith.index_cast %57 : index to i16
              %90 = arith.index_cast %73 : index to i16
              %91 = arith.index_cast %59 : index to i16
              %92 = ascendc.construct !ascendc.mmad_params(%89, %91, %90, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %62, %87, %88, %92 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %87 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %88 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %62 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %13, %61 : !ascendc.tbuf<vecin>, index
            %64 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %65 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %66 = ascendc.construct !ascendc.fixpipe_params<f32>()
            ascendc.fixpipe %65, %64, %66 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<f32>
            ascendc.que_bind.enque_tensor %8, %65 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %64 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %12, %61 : !ascendc.tbuf<veccalc>, index
            %67 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %68 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %68, %67, %48, %60 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %68 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %67 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %11, %61 : !ascendc.tbuf<vecin>, index
            %69 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %69, %cst, %60 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %69 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %70 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %54, %70, %71, %60 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %70 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %71 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          ascendc.que_bind.enque_tensor %4, %54 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        }
        scf.for %arg6 = %51 to %24 step %c64 {
          %54 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          %55 = affine.apply #map1()[%26]
          scf.for %arg7 = %c0 to %55 step %c64 {
            %56 = arith.subi %24, %arg6 : index
            %57 = arith.minsi %56, %c64 : index
            %58 = arith.subi %26, %arg7 : index
            %59 = arith.minsi %58, %c64 : index
            %60 = arith.muli %57, %59 : index
            %61 = arith.muli %60, %c4 : index
            ascendc.pipe.init_buffer %0, %16, %61 : !ascendc.tbuf<co1>, index
            %62 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %63 = affine.apply #map2()[%dim_1]
            scf.for %arg8 = %c0 to %63 step %c32 {
              %72 = arith.subi %dim_1, %arg8 : index
              %73 = arith.minsi %72, %c32 : index
              %74 = arith.muli %57, %73 : index
              %75 = arith.muli %74, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %75 : !ascendc.tbuf<a2>, index
              %76 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %77 = arith.divui %57, %c16 : index
              %78 = arith.index_cast %77 : index to i16
              %79 = arith.divui %73, %c16 : index
              %80 = arith.index_cast %79 : index to i64
              %81 = arith.trunci %80 : i64 to i8
              %82 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %81, %78, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %76, %49, %82 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %76 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %83 = arith.muli %73, %59 : index
              %84 = arith.muli %83, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %84 : !ascendc.tbuf<b2>, index
              %85 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %86 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %81, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %85, %50, %86 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %85 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %89 = arith.index_cast %57 : index to i16
              %90 = arith.index_cast %73 : index to i16
              %91 = arith.index_cast %59 : index to i16
              %92 = ascendc.construct !ascendc.mmad_params(%89, %91, %90, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %62, %87, %88, %92 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %87 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %88 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            scf.for %arg8 = %63 to %dim_1 step %c32 {
              %72 = arith.subi %dim_1, %arg8 : index
              %73 = arith.minsi %72, %c32 : index
              %74 = arith.muli %57, %73 : index
              %75 = arith.muli %74, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %75 : !ascendc.tbuf<a2>, index
              %76 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %77 = arith.divui %57, %c16 : index
              %78 = arith.index_cast %77 : index to i16
              %79 = arith.divui %73, %c16 : index
              %80 = arith.index_cast %79 : index to i64
              %81 = arith.trunci %80 : i64 to i8
              %82 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %81, %78, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %76, %49, %82 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %76 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %83 = arith.muli %73, %59 : index
              %84 = arith.muli %83, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %84 : !ascendc.tbuf<b2>, index
              %85 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %86 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %81, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %85, %50, %86 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %85 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %89 = arith.index_cast %57 : index to i16
              %90 = arith.index_cast %73 : index to i16
              %91 = arith.index_cast %59 : index to i16
              %92 = ascendc.construct !ascendc.mmad_params(%89, %91, %90, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %62, %87, %88, %92 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %87 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %88 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %62 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %13, %61 : !ascendc.tbuf<vecin>, index
            %64 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %65 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %66 = ascendc.construct !ascendc.fixpipe_params<f32>()
            ascendc.fixpipe %65, %64, %66 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<f32>
            ascendc.que_bind.enque_tensor %8, %65 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %64 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %12, %61 : !ascendc.tbuf<veccalc>, index
            %67 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %68 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %68, %67, %48, %60 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %68 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %67 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %11, %61 : !ascendc.tbuf<vecin>, index
            %69 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %69, %cst, %60 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %69 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %70 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %54, %70, %71, %60 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %70 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %71 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          scf.for %arg7 = %55 to %26 step %c64 {
            %56 = arith.subi %24, %arg6 : index
            %57 = arith.minsi %56, %c64 : index
            %58 = arith.subi %26, %arg7 : index
            %59 = arith.minsi %58, %c64 : index
            %60 = arith.muli %57, %59 : index
            %61 = arith.muli %60, %c4 : index
            ascendc.pipe.init_buffer %0, %16, %61 : !ascendc.tbuf<co1>, index
            %62 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %63 = affine.apply #map2()[%dim_1]
            scf.for %arg8 = %c0 to %63 step %c32 {
              %72 = arith.subi %dim_1, %arg8 : index
              %73 = arith.minsi %72, %c32 : index
              %74 = arith.muli %57, %73 : index
              %75 = arith.muli %74, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %75 : !ascendc.tbuf<a2>, index
              %76 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %77 = arith.divui %57, %c16 : index
              %78 = arith.index_cast %77 : index to i16
              %79 = arith.divui %73, %c16 : index
              %80 = arith.index_cast %79 : index to i64
              %81 = arith.trunci %80 : i64 to i8
              %82 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %81, %78, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %76, %49, %82 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %76 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %83 = arith.muli %73, %59 : index
              %84 = arith.muli %83, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %84 : !ascendc.tbuf<b2>, index
              %85 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %86 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %81, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %85, %50, %86 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %85 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %89 = arith.index_cast %57 : index to i16
              %90 = arith.index_cast %73 : index to i16
              %91 = arith.index_cast %59 : index to i16
              %92 = ascendc.construct !ascendc.mmad_params(%89, %91, %90, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %62, %87, %88, %92 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %87 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %88 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            scf.for %arg8 = %63 to %dim_1 step %c32 {
              %72 = arith.subi %dim_1, %arg8 : index
              %73 = arith.minsi %72, %c32 : index
              %74 = arith.muli %57, %73 : index
              %75 = arith.muli %74, %c4 : index
              ascendc.pipe.init_buffer %0, %15, %75 : !ascendc.tbuf<a2>, index
              %76 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %77 = arith.divui %57, %c16 : index
              %78 = arith.index_cast %77 : index to i16
              %79 = arith.divui %73, %c16 : index
              %80 = arith.index_cast %79 : index to i64
              %81 = arith.trunci %80 : i64 to i8
              %82 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %81, %78, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %76, %49, %82 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %76 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %83 = arith.muli %73, %59 : index
              %84 = arith.muli %83, %c4 : index
              ascendc.pipe.init_buffer %0, %14, %84 : !ascendc.tbuf<b2>, index
              %85 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %86 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %81, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %85, %50, %86 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %85 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %87 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %88 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %89 = arith.index_cast %57 : index to i16
              %90 = arith.index_cast %73 : index to i16
              %91 = arith.index_cast %59 : index to i16
              %92 = ascendc.construct !ascendc.mmad_params(%89, %91, %90, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %62, %87, %88, %92 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %87 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %7, %88 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %5, %62 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %13, %61 : !ascendc.tbuf<vecin>, index
            %64 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %65 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %66 = ascendc.construct !ascendc.fixpipe_params<f32>()
            ascendc.fixpipe %65, %64, %66 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<f32>
            ascendc.que_bind.enque_tensor %8, %65 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %64 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %12, %61 : !ascendc.tbuf<veccalc>, index
            %67 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %68 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %68, %67, %48, %60 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %68 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %8, %67 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %11, %61 : !ascendc.tbuf<vecin>, index
            %69 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %69, %cst, %60 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %10, %69 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %70 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %54, %70, %71, %60 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %70 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %10, %71 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          ascendc.que_bind.enque_tensor %4, %54 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        }
        ascendc.que_bind.free_tensor %2, %50 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %1, %49 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %52 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %53 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %53, %subview_3 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        ascendc.data_copy_l2 %53, %52, %44 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %4, %52 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %3, %48 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    %22 = bufferization.clone %arg3 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    return %22 : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }
}

