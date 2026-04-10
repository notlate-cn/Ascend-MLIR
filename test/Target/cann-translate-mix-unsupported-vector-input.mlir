// Valid single-chain mix shape with an unsupported vector-region op.
module attributes {transform.with_named_sequence} {
  func.func @matmul_add_leakyrelu(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32, strided<[1, 1], offset: ?>>, %arg5: memref<ui8>, %arg6: !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>) attributes {ascendc.aicore, ascendc.global, ascendc.kernel_kind = "mix", cann.num_inputs = 4 : i32} {
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
    %0 = emitasc.member %arg6 "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %1 = emitasc.member %arg6 "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %2 = emitasc.member %arg6 "Tb_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %3 = emitasc.member %arg6 "Tb_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %4 = emitasc.member %arg6 "t_K" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %5 = emitasc.member %arg6 "dim_arg3_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %6 = emitasc.member %arg6 "dim_arg3_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %7 = emitasc.member %arg6 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %8 = emitasc.member %arg6 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %9 = emitasc.member %arg6 "dim_arg1_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %10 = emitasc.member %arg6 "dim_arg1_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %11 = emitasc.member %arg6 "dim_arg2_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %12 = emitasc.member %arg6 "dim_arg2_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K", "dim_arg3_0", "dim_arg3_1", "dim_arg0_1", "dim_arg0_0", "dim_arg1_0", "dim_arg1_1", "dim_arg2_0", "dim_arg2_1"]>, i64
    %13 = ascendc.pipe
    %14 = ascendc.queue : <a1, 1>
    %15 = ascendc.queue : <b1, 1>
    %16 = ascendc.queue : <vecin, 1>
    %17 = ascendc.queue : <veccalc, 1>
    %18 = ascendc.queue : <co1, 1>
    %19 = ascendc.queue : <a2, 1>
    %20 = ascendc.queue : <b2, 1>
    %21 = ascendc.queue : <vecin, 1>
    %22 = ascendc.queue : <vecout, 1>
    %23 = arith.index_cast %4 : i64 to index
    %24 = arith.index_cast %3 : i64 to index
    %25 = arith.index_cast %2 : i64 to index
    %26 = arith.index_cast %1 : i64 to index
    %27 = arith.index_cast %0 : i64 to index
    %28 = arith.index_cast %5 : i64 to index
    %29 = arith.index_cast %6 : i64 to index
    %30 = ascendc.tbuf : <veccalc>
    %31 = ascendc.tbuf : <veccalc>
    %32 = ascendc.tbuf : <vecout>
    %33 = ascendc.tbuf : <veccalc>
    %34 = ascendc.queue : <vecin, 1>
    %35 = ascendc.tbuf : <vecin>
    %36 = ascendc.tbuf : <co1>
    %37 = ascendc.tbuf : <veccalc>
    %38 = ascendc.tbuf : <vecin>
    %39 = ascendc.tbuf : <b2>
    %40 = ascendc.tbuf : <a2>
    %41 = ascendc.tbuf : <co1>
    %42 = ascendc.tbuf : <veccalc>
    %43 = ascendc.tbuf : <vecin>
    %44 = ascendc.tbuf : <b1>
    %45 = ascendc.tbuf : <a1>
    %46 = ascendc.get_block_idx : index
    %47 = arith.muli %46, %27 : index
    %48 = arith.cmpi ult, %47, %28 : index
    scf.if %48 {
      scf.for %arg7 = %c0 to %29 step %26 {
        %49 = arith.subi %28, %47 : index
        %50 = arith.minsi %27, %49 : index
        %51 = arith.subi %29, %arg7 : index
        %52 = arith.minsi %51, %26 : index
        %53 = arith.index_cast %7 : i64 to index
        %54 = arith.muli %50, %53 : index
        %55 = arith.muli %54, %c2 : index
        ascendc.pipe.init_buffer %13, %45, %55 : !ascendc.tbuf<a1>, index
        ascendc.pipe.init_queue %13, %14, %c1_i32, %55 : !ascendc.queue<a1, 1>, i32, index
        %56 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        %57 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %58 = arith.muli %47, %53 : index
        %59 = arith.index_cast %58 : index to i32
        %60 = emitasc.reinterpret_cast %arg0 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %57, %60, %59 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        %61 = arith.divui %53, %c16 : index
        %62 = arith.index_cast %50 : index to i16
        %63 = arith.index_cast %61 : index to i16
        %64 = arith.index_cast %53 : index to i16
        %65 = ascendc.construct !ascendc.nd2nz_params(%62, %63, %62, %64, %62, %c0_i16, %62, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %56, %57, %65 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %14, %56 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        %66 = arith.muli %53, %52 : index
        %67 = arith.muli %66, %c2 : index
        ascendc.pipe.init_buffer %13, %44, %67 : !ascendc.tbuf<b1>, index
        ascendc.pipe.init_queue %13, %15, %c1_i32, %67 : !ascendc.queue<b1, 1>, i32, index
        %68 = ascendc.que_bind.alloc_tensor %15 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        %69 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        %70 = arith.index_cast %arg7 : index to i32
        %71 = emitasc.reinterpret_cast %arg1 : memref<?x?xf16> to memref<?xf16, 22 : i32>
        ascendc.global_tensor.set_global_buffer %69, %71, %70 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
        %72 = arith.divui %52, %c16 : index
        %73 = arith.index_cast %72 : index to i16
        %74 = arith.index_cast %52 : index to i16
        %75 = ascendc.construct !ascendc.nd2nz_params(%64, %73, %64, %74, %64, %c0_i16, %64, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %68, %69, %75 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %15, %68 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        %76 = arith.muli %52, %c4 : index
        ascendc.pipe.init_buffer %13, %43, %76 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %13, %16, %c1_i32, %76 : !ascendc.queue<vecin, 1>, i32, index
        %77 = ascendc.que_bind.alloc_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %78 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %79 = emitasc.reinterpret_cast %arg2 : memref<?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %78, %79, %70 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %77, %78, %52 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %16, %77 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %80 = ascendc.que_bind.deque_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %81 = arith.muli %50, %52 : index
        %82 = arith.muli %81, %c4 : index
        ascendc.pipe.init_buffer %13, %42, %82 : !ascendc.tbuf<veccalc>, index
        ascendc.pipe.init_queue %13, %17, %c1_i32, %82 : !ascendc.queue<veccalc, 1>, i32, index
        %83 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        %84 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        scf.for %arg8 = %c0 to %50 step %25 {
          scf.for %arg9 = %c0 to %52 step %24 {
            %85 = arith.subi %50, %arg8 : index
            %86 = arith.minsi %85, %25 : index
            %87 = arith.subi %52, %arg9 : index
            %88 = arith.minsi %87, %24 : index
            %89 = arith.muli %86, %88 : index
            %90 = arith.muli %89, %c4 : index
            ascendc.pipe.init_buffer %13, %41, %90 : !ascendc.tbuf<co1>, index
            ascendc.pipe.init_queue %13, %18, %c1_i32, %90 : !ascendc.queue<co1, 1>, i32, index
            %91 = ascendc.que_bind.alloc_tensor %18 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf16>
            scf.for %arg10 = %c0 to %53 step %23 {
              %116 = arith.subi %53, %arg10 : index
              %117 = arith.minsi %116, %23 : index
              %118 = arith.muli %86, %117 : index
              %119 = arith.muli %118, %c2 : index
              ascendc.pipe.init_buffer %13, %40, %119 : !ascendc.tbuf<a2>, index
              ascendc.pipe.init_queue %13, %19, %c1_i32, %119 : !ascendc.queue<a2, 1>, i32, index
              %120 = ascendc.que_bind.alloc_tensor %19 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %121 = arith.divui %86, %c16 : index
              %122 = arith.index_cast %121 : index to i16
              %123 = arith.divui %117, %c16 : index
              %124 = arith.index_cast %123 : index to i64
              %125 = arith.trunci %124 : i64 to i8
              %126 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %125, %122, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %120, %83, %126 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %19, %120 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %127 = arith.muli %117, %88 : index
              %128 = arith.muli %127, %c2 : index
              ascendc.pipe.init_buffer %13, %39, %128 : !ascendc.tbuf<b2>, index
              ascendc.pipe.init_queue %13, %20, %c1_i32, %128 : !ascendc.queue<b2, 1>, i32, index
              %129 = ascendc.que_bind.alloc_tensor %20 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %130 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %125, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %129, %84, %130 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %20, %129 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %131 = ascendc.que_bind.deque_tensor %19 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %132 = ascendc.que_bind.deque_tensor %20 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %133 = arith.index_cast %86 : index to i16
              %134 = arith.index_cast %117 : index to i16
              %135 = arith.index_cast %88 : index to i16
              %136 = ascendc.construct !ascendc.mmad_params(%133, %135, %134, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %91, %131, %132, %136 {ascendc.unit = "AiCore.Cube"} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %19, %131 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              ascendc.que_bind.free_tensor %20, %132 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
            }
            ascendc.que_bind.enque_tensor %18, %91 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf16>
            ascendc.pipe.init_buffer %13, %38, %90 : !ascendc.tbuf<vecin>, index
            ascendc.pipe.init_queue %13, %21, %c1_i32, %90 : !ascendc.queue<vecin, 1>, i32, index
            %92 = ascendc.que_bind.deque_tensor %18 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %93 = ascendc.que_bind.alloc_tensor %21 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %94 = ascendc.construct !ascendc.data_copy_co12dst_params()
            ascendc.data_copy_co12dst %93, %92, %94 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.data_copy_co12dst_params
            ascendc.que_bind.enque_tensor %21, %93 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %18, %92 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %13, %37, %90 : !ascendc.tbuf<veccalc>, index
            %95 = ascendc.tbuf.get_tensor %37 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            %96 = ascendc.tbuf.get_tensor %36 : !ascendc.tbuf<co1>, !ascendc.local_tensor<*xf32>
            %97 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
            %98 = arith.addi %arg9, %arg7 : index
            %99 = arith.index_cast %98 : index to i32
            ascendc.global_tensor.set_global_buffer %97, %79, %99 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
            %100 = arith.muli %88, %c4 : index
            ascendc.pipe.init_buffer %13, %35, %100 : !ascendc.tbuf<vecin>, index
            ascendc.pipe.init_queue %13, %34, %c1_i32, %100 : !ascendc.queue<vecin, 1>, i32, index
            %101 = ascendc.que_bind.alloc_tensor %34 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.data_copy_l2 %101, %97, %88 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %34, %101 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %102 = ascendc.que_bind.deque_tensor %34 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %103 = arith.index_cast %86 : index to i32
            %104 = arith.index_cast %88 : index to i32
            ascendc.pipe.init_buffer %13, %33, %90 : !ascendc.tbuf<veccalc>, index
            %105 = ascendc.tbuf.get_tensor %33 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            ascendc.broadcast_l2 %105, %102, %103, %104, %c1_i32, %104 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32, i32, i32, i32
            ascendc.sub_l2 %95, %96, %105, %89 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %17, %95 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %13, %32, %90 : !ascendc.tbuf<vecout>, index
            ascendc.pipe.init_queue %13, %22, %c1_i32, %90 : !ascendc.queue<vecout, 1>, i32, index
            ascendc.pipe.init_buffer %13, %31, %90 : !ascendc.tbuf<veccalc>, index
            %106 = ascendc.tbuf.get_tensor %31 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            %107 = ascendc.que_bind.deque_tensor %17 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %13, %30, %90 : !ascendc.tbuf<veccalc>, index
            %108 = ascendc.tbuf.get_tensor %30 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %108, %cst, %89 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.mul_l2 %106, %107, %108, %89 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.max_l2 %106, %107, %106, %89 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %22, %106 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
            %109 = ascendc.que_bind.deque_tensor %22 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
            %110 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
            %111 = arith.addi %arg8, %47 : index
            %112 = arith.muli %111, %29 : index
            %113 = arith.addi %112, %98 : index
            %114 = arith.index_cast %113 : index to i32
            %115 = emitasc.reinterpret_cast %arg4 : memref<?x?xf32, strided<[1, 1], offset: ?>> to memref<?xf32, 22 : i32>
            ascendc.global_tensor.set_global_buffer %110, %115, %114 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
            ascendc.data_copy_l2 %110, %109, %89 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %22, %109 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          }
        }
        ascendc.que_bind.free_tensor %15, %84 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %14, %83 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %16, %80 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    return
  }
}
