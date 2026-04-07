module attributes {transform.with_named_sequence} {
  func.func @matmul_add_leakyrelu(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, 22 : i32>, %arg5: memref<?x?xf32, strided<[1, 1], offset: ?>>) attributes {ascendc.aicore, ascendc.global, ascendc.kernel_kind = "mix"} {
    %cst = arith.constant 1.000000e-03 : f32
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c0_i16 = arith.constant 0 : i16
    %c4 = arith.constant 4 : index
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %c1_i16 = arith.constant 1 : i16
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %0 = emitasc.copy_struct %arg4 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>
    %1 = emitasc.member %0 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %2 = emitasc.member %0 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %3 = emitasc.member %0 "Tb_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %4 = emitasc.member %0 "Tb_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %5 = emitasc.member %0 "t_K" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %6 = emitasc.member %0 "dim_arg3_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %7 = emitasc.member %0 "dim_arg3_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %8 = emitasc.member %0 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %9 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %10 = emitasc.member %0 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %11 = emitasc.member %0 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %12 = emitasc.member %0 "dim_arg2_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %13 = emitasc.member %0 "dim_arg2_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %14 = ascendc.pipe
    %15 = ascendc.queue : <a1, 1>
    %16 = ascendc.queue : <b1, 1>
    %17 = ascendc.queue : <vecin, 1>
    %18 = ascendc.queue : <veccalc, 1>
    %19 = ascendc.queue : <co1, 1>
    %20 = ascendc.queue : <a2, 1>
    %21 = ascendc.queue : <b2, 1>
    %22 = ascendc.queue : <vecin, 1>
    %23 = ascendc.queue : <vecout, 1>
    %24 = arith.index_cast %5 : i64 to index
    %25 = arith.index_cast %4 : i64 to index
    %26 = arith.index_cast %3 : i64 to index
    %27 = arith.index_cast %2 : i64 to index
    %28 = arith.index_cast %1 : i64 to index
    %29 = arith.index_cast %6 : i64 to index
    %30 = arith.index_cast %7 : i64 to index
    %31 = ascendc.tbuf : <veccalc>
    %32 = ascendc.tbuf : <veccalc>
    %33 = ascendc.tbuf : <vecout>
    %34 = ascendc.tbuf : <veccalc>
    %35 = ascendc.queue : <vecin, 1>
    %36 = ascendc.tbuf : <vecin>
    %37 = ascendc.tbuf : <co1>
    %38 = ascendc.tbuf : <veccalc>
    %39 = ascendc.tbuf : <vecin>
    %40 = ascendc.tbuf : <b2>
    %41 = ascendc.tbuf : <a2>
    %42 = ascendc.tbuf : <co1>
    %43 = ascendc.tbuf : <veccalc>
    %44 = ascendc.tbuf : <vecin>
    %45 = ascendc.tbuf : <b1>
    %46 = ascendc.tbuf : <a1>
    %47 = ascendc.get_block_idx : index
    %48 = arith.muli %47, %28 : index
    %49 = arith.cmpi ult, %48, %29 : index
    scf.if %49 {
      scf.for %arg6 = %c0 to %30 step %27 {
        %50 = arith.subi %29, %48 : index
        %51 = arith.minsi %28, %50 : index
        %52 = arith.subi %30, %arg6 : index
        %53 = arith.minsi %52, %27 : index
        %54 = arith.index_cast %8 : i64 to index
        %55 = arith.muli %51, %54 : index
        %56 = arith.muli %55, %c2 : index
        ascendc.pipe.init_buffer %14, %46, %56 : !ascendc.tbuf<a1>, index
        ascendc.pipe.init_queue %14, %15, %c1_i32, %56 : !ascendc.queue<a1, 1>, i32, index
        %57 = ascendc.que_bind.alloc_tensor %15 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        %58 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %59 = arith.muli %48, %54 : index
        %60 = arith.index_cast %59 : index to i32
        %61 = emitasc.reinterpret_cast %arg0 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %58, %61, %60 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        %62 = arith.divui %54, %c16 : index
        %63 = arith.index_cast %51 : index to i16
        %64 = arith.index_cast %62 : index to i16
        %65 = arith.index_cast %54 : index to i16
        %66 = ascendc.construct !ascendc.nd2nz_params(%63, %64, %63, %65, %63, %c0_i16, %63, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %57, %58, %66 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %15, %57 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        %67 = arith.muli %54, %53 : index
        %68 = arith.muli %67, %c2 : index
        ascendc.pipe.init_buffer %14, %45, %68 : !ascendc.tbuf<b1>, index
        ascendc.pipe.init_queue %14, %16, %c1_i32, %68 : !ascendc.queue<b1, 1>, i32, index
        %69 = ascendc.que_bind.alloc_tensor %16 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        %70 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %71 = arith.index_cast %arg6 : index to i32
        %72 = emitasc.reinterpret_cast %arg1 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %70, %72, %71 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        %73 = arith.divui %53, %c16 : index
        %74 = arith.index_cast %73 : index to i16
        %75 = arith.index_cast %53 : index to i16
        %76 = ascendc.construct !ascendc.nd2nz_params(%65, %74, %65, %75, %65, %c0_i16, %65, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %69, %70, %76 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %16, %69 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        %77 = arith.muli %53, %c4 : index
        ascendc.pipe.init_buffer %14, %44, %77 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %14, %17, %c1_i32, %77 : !ascendc.queue<vecin, 1>, i32, index
        %78 = ascendc.que_bind.alloc_tensor %17 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %79 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %80 = emitasc.reinterpret_cast %arg2 : memref<?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %79, %80, %71 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %78, %79, %53 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %17, %78 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %81 = ascendc.que_bind.deque_tensor %17 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %82 = arith.muli %51, %53 : index
        %83 = arith.muli %82, %c4 : index
        ascendc.pipe.init_buffer %14, %43, %83 : !ascendc.tbuf<veccalc>, index
        ascendc.pipe.init_queue %14, %18, %c1_i32, %83 : !ascendc.queue<veccalc, 1>, i32, index
        %84 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        %85 = ascendc.que_bind.deque_tensor %16 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        scf.for %arg7 = %c0 to %51 step %26 {
          scf.for %arg8 = %c0 to %53 step %25 {
            %86 = arith.subi %51, %arg7 : index
            %87 = arith.minsi %86, %26 : index
            %88 = arith.subi %53, %arg8 : index
            %89 = arith.minsi %88, %25 : index
            %90 = arith.muli %87, %89 : index
            %91 = arith.muli %90, %c4 : index
            ascendc.pipe.init_buffer %14, %42, %91 : !ascendc.tbuf<co1>, index
            ascendc.pipe.init_queue %14, %19, %c1_i32, %91 : !ascendc.queue<co1, 1>, i32, index
            %92 = ascendc.que_bind.alloc_tensor %19 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf16>
            scf.for %arg9 = %c0 to %54 step %24 {
              %117 = arith.subi %54, %arg9 : index
              %118 = arith.minsi %117, %24 : index
              %119 = arith.muli %87, %118 : index
              %120 = arith.muli %119, %c2 : index
              ascendc.pipe.init_buffer %14, %41, %120 : !ascendc.tbuf<a2>, index
              ascendc.pipe.init_queue %14, %20, %c1_i32, %120 : !ascendc.queue<a2, 1>, i32, index
              %121 = ascendc.que_bind.alloc_tensor %20 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %122 = arith.divui %87, %c16 : index
              %123 = arith.index_cast %122 : index to i16
              %124 = arith.divui %118, %c16 : index
              %125 = arith.index_cast %124 : index to i64
              %126 = arith.trunci %125 : i64 to i8
              %127 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %126, %123, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %121, %84, %127 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %20, %121 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %128 = arith.muli %118, %89 : index
              %129 = arith.muli %128, %c2 : index
              ascendc.pipe.init_buffer %14, %40, %129 : !ascendc.tbuf<b2>, index
              ascendc.pipe.init_queue %14, %21, %c1_i32, %129 : !ascendc.queue<b2, 1>, i32, index
              %130 = ascendc.que_bind.alloc_tensor %21 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %131 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %126, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %130, %85, %131 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %21, %130 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %132 = ascendc.que_bind.deque_tensor %20 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %133 = ascendc.que_bind.deque_tensor %21 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %134 = arith.index_cast %87 : index to i16
              %135 = arith.index_cast %118 : index to i16
              %136 = arith.index_cast %89 : index to i16
              %137 = ascendc.construct !ascendc.mmad_params(%134, %136, %135, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %92, %132, %133, %137 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %20, %132 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              ascendc.que_bind.free_tensor %21, %133 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
            }
            ascendc.que_bind.enque_tensor %19, %92 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf16>
            ascendc.pipe.init_buffer %14, %39, %91 : !ascendc.tbuf<vecin>, index
            ascendc.pipe.init_queue %14, %22, %c1_i32, %91 : !ascendc.queue<vecin, 1>, i32, index
            %93 = ascendc.que_bind.deque_tensor %19 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %94 = ascendc.que_bind.alloc_tensor %22 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %95 = ascendc.construct !ascendc.data_copy_co12dst_params()
            ascendc.data_copy_co12dst %94, %93, %95 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.data_copy_co12dst_params
            ascendc.que_bind.enque_tensor %22, %94 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %19, %93 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %14, %38, %91 : !ascendc.tbuf<veccalc>, index
            %96 = ascendc.tbuf.get_tensor %38 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            %97 = ascendc.tbuf.get_tensor %37 : !ascendc.tbuf<co1>, !ascendc.local_tensor<*xf32>
            %98 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
            %99 = arith.addi %arg8, %arg6 : index
            %100 = arith.index_cast %99 : index to i32
            ascendc.global_tensor.set_global_buffer %98, %80, %100 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
            %101 = arith.muli %89, %c4 : index
            ascendc.pipe.init_buffer %14, %36, %101 : !ascendc.tbuf<vecin>, index
            ascendc.pipe.init_queue %14, %35, %c1_i32, %101 : !ascendc.queue<vecin, 1>, i32, index
            %102 = ascendc.que_bind.alloc_tensor %35 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.data_copy_l2 %102, %98, %89 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %35, %102 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %103 = ascendc.que_bind.deque_tensor %35 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %104 = arith.index_cast %87 : index to i32
            %105 = arith.index_cast %89 : index to i32
            ascendc.pipe.init_buffer %14, %34, %91 : !ascendc.tbuf<veccalc>, index
            %106 = ascendc.tbuf.get_tensor %34 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            ascendc.broadcast_l2 %106, %103, %104, %105, %c1_i32, %105 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32, i32, i32, i32
            ascendc.add_l2 %96, %97, %106, %90 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %18, %96 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %14, %33, %91 : !ascendc.tbuf<vecout>, index
            ascendc.pipe.init_queue %14, %23, %c1_i32, %91 : !ascendc.queue<vecout, 1>, i32, index
            ascendc.pipe.init_buffer %14, %32, %91 : !ascendc.tbuf<veccalc>, index
            %107 = ascendc.tbuf.get_tensor %32 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            %108 = ascendc.que_bind.deque_tensor %18 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %14, %31, %91 : !ascendc.tbuf<veccalc>, index
            %109 = ascendc.tbuf.get_tensor %31 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %109, %cst, %90 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.mul_l2 %107, %108, %109, %90 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.max_l2 %107, %108, %107, %90 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %23, %107 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
            %110 = ascendc.que_bind.deque_tensor %23 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
            %111 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
            %112 = arith.addi %arg7, %48 : index
            %113 = arith.muli %112, %30 : index
            %114 = arith.addi %113, %99 : index
            %115 = arith.index_cast %114 : index to i32
            %116 = emitasc.reinterpret_cast %arg5 : memref<?x?xf32, strided<[1, 1], offset: ?>> to memref<?xf32, 22 : i32>
            ascendc.global_tensor.set_global_buffer %111, %116, %115 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
            ascendc.data_copy_l2 %111, %110, %90 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %23, %110 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          }
        }
        ascendc.que_bind.free_tensor %16, %85 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %15, %84 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %17, %81 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    return
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:5 = transform.func.add_index_args %0, 5 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.matmul"]} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.generic"]} in %transformed : (!transform.any_op) -> !transform.any_op
    %3:2 = transform.split_handle %2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %4 = transform.param.constant true -> !transform.any_param
    %5 = transform.param.constant "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN" -> !transform.any_param
    %6 = transform.param.constant "result:VECOUT->GM" -> !transform.any_param
    %7 = transform.param.constant "lhs:A1->A2,rhs:B1->B2" -> !transform.any_param
    %8 = transform.param.constant "acc:CO1->VECIN" -> !transform.any_param
    %9 = transform.param.constant "AiCore.Cube" -> !transform.any_param
    %10 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %3#1 tile_sizes [%new_args#0, %new_args#1] : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %3#0 into %loops#1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_0, %new_containing_op_1 = transform.structured.fuse_into_containing_op %1 into %loops#1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %11:3 = transform.split_handle %fused_op_0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.annotate %loops#0 "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    transform.annotate %loops#1 "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    transform.annotate %loops#1 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops#1 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_2, %loops_3:2 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#2, %new_args#3] : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %fused_op_4, %new_containing_op_5 = transform.structured.fuse_into_containing_op %fused_op into %loops_3#1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_6, %new_containing_op_7 = transform.structured.fuse_into_containing_op %11#0 into %loops_3#1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %12:3 = transform.split_handle %fused_op_6 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %12#0 tile_sizes [0, 0, %new_args#4] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %13 = transform.structured.match ops{["linalg.matmul"]} in %loops_9 : (!transform.any_op) -> !transform.any_op
    transform.annotate %13 "ascendc.unit" = %9 : !transform.any_op, !transform.any_param
    transform.annotate %fused_op_4 "ascendc.unit" = %10 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_2 "ascendc.unit" = %10 : !transform.any_op, !transform.any_param
    transform.annotate %loops_9 "ascendc.prologue" = %7 : !transform.any_op, !transform.any_param
    transform.annotate %loops_9 "ascendc.epilogue" = %8 : !transform.any_op, !transform.any_param
    transform.yield 
  }
}
