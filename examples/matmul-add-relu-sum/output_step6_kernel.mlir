#map = affine_map<()[s0, s1, s2] -> (s1, s0 - s2)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module {
  func.func @fc_relu(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg1: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg3: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg4: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>, 22 : i32>) attributes {ascendc.aicore, ascendc.global} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c0_i16 = arith.constant 0 : i16
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %c1_i16 = arith.constant 1 : i16
    %true = arith.constant true
    %0 = emitasc.copy_struct %arg4 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>
    %1 = emitasc.member %0 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>, i64
    %2 = emitasc.member %0 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>, i64
    %3 = emitasc.member %0 "Tb_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>, i64
    %4 = emitasc.member %0 "Tb_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>, i64
    %5 = emitasc.member %0 "t_K" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>, i64
    %6 = ascendc.pipe
    %7 = ascendc.queue : <a1, 1>
    %8 = ascendc.queue : <b1, 1>
    %9 = ascendc.queue : <vecin, 1>
    %10 = ascendc.queue : <vecout, 1>
    %11 = ascendc.queue : <co1, 1>
    %12 = ascendc.queue : <a2, 1>
    %13 = ascendc.queue : <b2, 1>
    %14 = ascendc.queue : <vecin, 1>
    %15 = ascendc.queue : <veccalc, 1>
    %16 = ascendc.queue : <vecin, 1>
    %17 = arith.index_cast %5 : i64 to index
    %18 = arith.index_cast %4 : i64 to index
    %19 = arith.index_cast %3 : i64 to index
    %20 = arith.index_cast %2 : i64 to index
    %21 = arith.index_cast %1 : i64 to index
    %dim = memref.dim %arg3, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %22 = ascendc.tbuf : <vecin>
    %23 = ascendc.tbuf : <veccalc>
    %24 = ascendc.tbuf : <vecin>
    %25 = ascendc.tbuf : <b2>
    %26 = ascendc.tbuf : <a2>
    %27 = ascendc.tbuf : <co1>
    %28 = ascendc.tbuf : <vecout>
    %29 = ascendc.tbuf : <vecin>
    %30 = ascendc.tbuf : <b1>
    %31 = ascendc.tbuf : <a1>
    %32 = ascendc.get_block_idx : index
    %33 = arith.muli %32, %21 : index
    %34 = arith.cmpi ult, %33, %dim : index
    scf.if %34 {
      scf.for %arg5 = %c0 to %dim_0 step %20 {
        %35 = affine.min #map()[%dim, %21, %33]
        %36 = affine.min #map1(%arg5)[%dim_0, %20]
        %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview = memref.subview %arg0[%33, 0] [%35, %dim_1] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %37 = arith.muli %35, %dim_1 : index
        %38 = arith.muli %37, %c4 : index
        ascendc.pipe.init_buffer %6, %31, %38 : !ascendc.tbuf<a1>, index
        %39 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %40 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %40, %subview : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        %41 = arith.divui %dim_1, %c16 : index
        %42 = arith.index_cast %35 : index to i16
        %43 = arith.index_cast %41 : index to i16
        %44 = arith.index_cast %dim_1 : index to i16
        %45 = ascendc.construct !ascendc.nd2nz_params(%42, %43, %42, %44, %42, %c0_i16, %42, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %39, %40, %45 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %7, %39 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %subview_2 = memref.subview %arg1[0, %arg5] [%dim_1, %36] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %46 = arith.muli %dim_1, %36 : index
        %47 = arith.muli %46, %c4 : index
        ascendc.pipe.init_buffer %6, %30, %47 : !ascendc.tbuf<b1>, index
        %48 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %49 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %49, %subview_2 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        %50 = arith.divui %36, %c16 : index
        %51 = arith.index_cast %50 : index to i16
        %52 = arith.index_cast %36 : index to i16
        %53 = ascendc.construct !ascendc.nd2nz_params(%44, %51, %44, %52, %44, %c0_i16, %44, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %48, %49, %53 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %8, %48 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        %subview_3 = memref.subview %arg3[%33, %arg5] [%35, %36] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_4 = memref.subview %arg2[%33, %arg5] [%35, %36] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %54 = arith.muli %35, %36 : index
        %55 = arith.muli %54, %c4 : index
        ascendc.pipe.init_buffer %6, %29, %55 : !ascendc.tbuf<vecin>, index
        %56 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %57 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %57, %subview_4 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        ascendc.data_copy_l2 %56, %57, %54 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %9, %56 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %58 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %6, %28, %55 : !ascendc.tbuf<vecout>, index
        %59 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %60 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        scf.for %arg6 = %c0 to %35 step %19 {
          %63 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          scf.for %arg7 = %c0 to %36 step %18 {
            %64 = affine.min #map1(%arg6)[%35, %19]
            %65 = affine.min #map1(%arg7)[%36, %18]
            %66 = arith.muli %64, %65 : index
            %67 = arith.muli %66, %c4 : index
            ascendc.pipe.init_buffer %6, %27, %67 : !ascendc.tbuf<co1>, index
            %68 = ascendc.que_bind.alloc_tensor %11 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            scf.for %arg8 = %c0 to %dim_1 step %17 {
              %82 = affine.min #map1(%arg8)[%dim_1, %17]
              %83 = arith.muli %64, %82 : index
              %84 = arith.muli %83, %c4 : index
              ascendc.pipe.init_buffer %6, %26, %84 : !ascendc.tbuf<a2>, index
              %85 = ascendc.que_bind.alloc_tensor %12 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %86 = arith.divui %64, %c16 : index
              %87 = arith.index_cast %86 : index to i16
              %88 = arith.divui %82, %c16 : index
              %89 = arith.index_cast %88 : index to i64
              %90 = arith.trunci %89 : i64 to i8
              %91 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %90, %87, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %85, %59, %91 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %12, %85 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %92 = arith.muli %82, %65 : index
              %93 = arith.muli %92, %c4 : index
              ascendc.pipe.init_buffer %6, %25, %93 : !ascendc.tbuf<b2>, index
              %94 = ascendc.que_bind.alloc_tensor %13 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %95 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %90, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %94, %60, %95 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %13, %94 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %96 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %97 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
              %98 = arith.index_cast %64 : index to i16
              %99 = arith.index_cast %82 : index to i16
              %100 = arith.index_cast %65 : index to i16
              %101 = ascendc.construct !ascendc.mmad_params(%98, %100, %99, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %68, %96, %97, %101 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %12, %96 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %13, %97 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }
            ascendc.que_bind.enque_tensor %11, %68 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %6, %24, %67 : !ascendc.tbuf<vecin>, index
            %69 = ascendc.que_bind.deque_tensor %11 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %70 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.construct !ascendc.data_copy_co12dst_params()
            ascendc.data_copy_co12dst %70, %69, %71 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.data_copy_co12dst_params
            ascendc.que_bind.enque_tensor %14, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %11, %69 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %6, %23, %67 : !ascendc.tbuf<veccalc>, index
            %72 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %73 = arith.muli %arg6, %36 : index
            %74 = arith.addi %73, %arg7 : index
            %75 = arith.muli %74, %c4 : index
            %76 = ascendc.tbuf.get_with_offset %29, %67, %75 : !ascendc.tbuf<vecin>, index, index, !ascendc.local_tensor<*xf32>
            %77 = ascendc.que_bind.alloc_tensor %15 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.add_l2 %77, %72, %76, %66 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %15, %77 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %14, %72 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %6, %22, %67 : !ascendc.tbuf<vecin>, index
            %78 = ascendc.que_bind.alloc_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %78, %cst, %66 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.que_bind.enque_tensor %16, %78 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %79 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %80 = ascendc.que_bind.deque_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %81 = ascendc.tbuf.get_with_offset %28, %67, %75 : !ascendc.tbuf<vecout>, index, index, !ascendc.local_tensor<*xf32>
            ascendc.max_l2 %81, %79, %80, %66 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %15, %79 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %16, %80 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }
          ascendc.que_bind.enque_tensor %10, %63 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        }
        ascendc.que_bind.free_tensor %8, %60 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %7, %59 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %61 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %62 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %62, %subview_3 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
        ascendc.data_copy_l2 %62, %61, %54 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %10, %61 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %9, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    return
  }
}

