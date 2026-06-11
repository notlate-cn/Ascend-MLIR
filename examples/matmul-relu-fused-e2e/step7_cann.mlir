module attributes {auto_fuse.tiling_infos = [{args = [{call_arg_index = 0 : i32, mlir_index = 0 : i32, role = "input"}, {call_arg_index = 1 : i32, mlir_index = 1 : i32, role = "input"}, {call_arg_index = 2 : i32, mlir_index = 2 : i32, role = "input"}, {mlir_index = 3 : i32, name = "XBLOCK_M", role = "tile_param"}, {mlir_index = 4 : i32, name = "M_INNER", role = "tile_param"}, {mlir_index = 5 : i32, name = "XBLOCK_N", role = "tile_param"}, {mlir_index = 6 : i32, name = "N_INNER", role = "tile_param"}, {mlir_index = 7 : i32, name = "K_INNER", role = "tile_param"}, {mlir_index = 8 : i32, result_index = 0 : i32, role = "output", shape_expr = ["32", "64"]}], fields = [{arg_index = 3 : i32, axis_size = -1 : i64, default_value = 128 : i64, kind = "tunable", name = "XBLOCK_M"}, {arg_index = 4 : i32, axis_size = -1 : i64, default_value = 32 : i64, kind = "tunable", name = "M_INNER"}, {arg_index = 5 : i32, axis_size = -1 : i64, default_value = 128 : i64, kind = "tunable", name = "XBLOCK_N"}, {arg_index = 6 : i32, axis_size = -1 : i64, default_value = 32 : i64, kind = "tunable", name = "N_INNER"}, {arg_index = 7 : i32, axis_size = -1 : i64, default_value = 16 : i64, kind = "tunable", name = "K_INNER"}], kernel_id = "mm_relu__v0", schema_version = 2 : i32}]} {
  func.func @mm_relu__v0(%arg0: memref<32x16xf16>, %arg1: memref<16x64xf16>, %arg2: memref<32x64xf32>, %arg3: memref<32x64xf32, strided<[?, 1], offset: ?>>, %arg4: memref<ui8>, %arg5: !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK_M", "M_INNER", "XBLOCK_N", "N_INNER", "K_INNER"]>) attributes {abi_matmul_epilogue_kind = "Relu", abi_matmul_has_bias = false, abi_matmul_layout_a = "ND", abi_matmul_layout_b = "ND", abi_matmul_layout_c = "ND", abi_matmul_op_kind = "matmul", abi_matmul_trans_a = false, abi_matmul_trans_b = false, afir.cube_kind = "MatmulVecFuse", ascendc.aicore, ascendc.global, ascendc.kernel_kind = "mix", cann.num_inputs = 3 : i32} {
    %c0_i32 = arith.constant 0 : i32
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i16 = arith.constant 0 : i16
    %c4 = arith.constant 4 : index
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %c1_i16 = arith.constant 1 : i16
    %c1024 = arith.constant 1024 : index
    %c32_i16 = arith.constant 32 : i16
    %c16_i16 = arith.constant 16 : i16
    %c2048 = arith.constant 2048 : index
    %c4_i16 = arith.constant 4 : i16
    %c64_i16 = arith.constant 64 : i16
    %0 = emitasc.member %arg5 "XBLOCK_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK_M", "M_INNER", "XBLOCK_N", "N_INNER", "K_INNER"]>, i64
    %1 = emitasc.member %arg5 "M_INNER" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK_M", "M_INNER", "XBLOCK_N", "N_INNER", "K_INNER"]>, i64
    %2 = emitasc.member %arg5 "XBLOCK_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK_M", "M_INNER", "XBLOCK_N", "N_INNER", "K_INNER"]>, i64
    %3 = emitasc.member %arg5 "N_INNER" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK_M", "M_INNER", "XBLOCK_N", "N_INNER", "K_INNER"]>, i64
    %4 = emitasc.member %arg5 "K_INNER" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK_M", "M_INNER", "XBLOCK_N", "N_INNER", "K_INNER"]>, i64
    %5 = arith.index_cast %0 : i64 to index
    %6 = arith.index_cast %1 : i64 to index
    %7 = arith.index_cast %2 : i64 to index
    %8 = arith.index_cast %3 : i64 to index
    %9 = arith.index_cast %4 : i64 to index
    %10 = ascendc.pipe
    %11 = ascendc.queue : <a1, 1>
    %12 = ascendc.queue : <b1, 1>
    %13 = ascendc.queue : <co1, 1>
    %14 = ascendc.queue : <a2, 1>
    %15 = ascendc.queue : <b2, 1>
    %16 = ascendc.queue : <vecin, 1>
    %17 = ascendc.queue : <vecout, 1>
    %cast = memref.cast %arg3 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
    %18 = ascendc.construct !ascendc.nd2nz_params(%c32_i16, %c1_i16, %c32_i16, %c16_i16, %c32_i16, %c0_i16, %c32_i16, %c0_i16) [ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16] : i16, i16, i16, i16, i16, i16, i16, i16
    %19 = ascendc.construct !ascendc.nd2nz_params(%c16_i16, %c4_i16, %c16_i16, %c64_i16, %c16_i16, %c0_i16, %c16_i16, %c0_i16) [ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16] : i16, i16, i16, i16, i16, i16, i16, i16
    %20 = ascendc.construct !ascendc.data_copy_co12dst_params()
    %21 = ascendc.tbuf : <veccalc>
    %22 = ascendc.tbuf : <veccalc>
    %23 = ascendc.tbuf : <vecout>
    %24 = ascendc.tbuf : <vecin>
    %25 = ascendc.tbuf : <b2>
    %26 = ascendc.tbuf : <a2>
    %27 = ascendc.tbuf : <co1>
    %28 = ascendc.tbuf : <b1>
    ascendc.pipe.init_buffer %10, %28, %c2048 : !ascendc.tbuf<b1>, index
    ascendc.pipe.init_queue %10, %11, %c1_i32, %c1024 : !ascendc.queue<a1, 1>, i32, index
    %29 = ascendc.tbuf : <a1>
    ascendc.pipe.init_buffer %10, %29, %c1024 : !ascendc.tbuf<a1>, index
    ascendc.pipe.init_queue %10, %12, %c1_i32, %c2048 : !ascendc.queue<b1, 1>, i32, index
    %30 = ascendc.get_block_idx : index
    %31 = arith.muli %30, %5 : index
    %32 = arith.cmpi ult, %31, %c32 : index
    scf.if %32 {
      %33 = ascendc.que_bind.alloc_tensor %11 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
      %34 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
      %35 = emitasc.reinterpret_cast %arg0 : memref<32x16xf16> to memref<?xf16, 22 : i32>
      ascendc.global_tensor.set_global_buffer %34, %35, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
      ascendc.data_copy_nd2nz %33, %34, %18 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, !ascendc.nd2nz_params
      ascendc.que_bind.enque_tensor %11, %33 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
      %36 = ascendc.que_bind.alloc_tensor %12 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
      %37 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
      %38 = emitasc.reinterpret_cast %arg1 : memref<16x64xf16> to memref<?xf16, 22 : i32>
      ascendc.global_tensor.set_global_buffer %37, %38, %c0_i32 : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32
      ascendc.data_copy_nd2nz %36, %37, %19 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, !ascendc.nd2nz_params
      ascendc.que_bind.enque_tensor %12, %36 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
      %39 = ascendc.que_bind.deque_tensor %11 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
      %40 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
      %41 = arith.subi %c32, %31 : index
      %42 = arith.minsi %41, %5 : index
      scf.for %arg6 = %c0 to %c64 step %7 {
        %43 = arith.subi %c64, %arg6 : index
        %44 = arith.minsi %43, %7 : index
        %subview = memref.subview %cast[%31, %arg6] [%42, %44] [1, 1] : memref<32x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        scf.for %arg7 = %c0 to %42 step %6 {
          %45 = arith.subi %42, %arg7 : index
          %46 = arith.minsi %45, %6 : index
          scf.for %arg8 = %c0 to %44 step %8 {
            %47 = arith.subi %44, %arg8 : index
            %48 = arith.minsi %47, %8 : index
            %49 = arith.muli %46, %48 : index
            %50 = arith.muli %49, %c4 : index
            ascendc.pipe.init_buffer %10, %27, %50 : !ascendc.tbuf<co1>, index
            ascendc.pipe.init_queue %10, %13, %c1_i32, %50 : !ascendc.queue<co1, 1>, i32, index
            %51 = ascendc.que_bind.alloc_tensor %13 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            scf.for %arg9 = %c0 to %c16 step %9 {
              %59 = arith.subi %c16, %arg9 : index
              %60 = arith.minsi %59, %9 : index
              %61 = arith.muli %46, %60 : index
              %62 = arith.muli %61, %c2 : index
              ascendc.pipe.init_buffer %10, %26, %62 : !ascendc.tbuf<a2>, index
              ascendc.pipe.init_queue %10, %14, %c1_i32, %62 : !ascendc.queue<a2, 1>, i32, index
              %63 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %64 = arith.divui %46, %c16 : index
              %65 = arith.index_cast %64 : index to i16
              %66 = arith.divui %60, %c16 : index
              %67 = arith.index_cast %66 : index to i64
              %68 = arith.trunci %67 : i64 to i8
              %69 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %68, %65, %c0_i16, %c0_i16, %false, %c0_i8) [ui16, ui8, ui16, ui8, ui16, i1, ui8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %63, %39, %69 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %14, %63 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %70 = arith.muli %60, %48 : index
              %71 = arith.muli %70, %c2 : index
              ascendc.pipe.init_buffer %10, %25, %71 : !ascendc.tbuf<b2>, index
              ascendc.pipe.init_queue %10, %15, %c1_i32, %71 : !ascendc.queue<b2, 1>, i32, index
              %72 = ascendc.que_bind.alloc_tensor %15 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %73 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %68, %c1_i16, %c0_i16, %c0_i16, %c0_i8) [ui16, ui8, ui16, ui16, ui16, ui8] : i16, i8, i16, i16, i16, i8
              ascendc.load_data_with_transpose %72, %40, %73 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %15, %72 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %74 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %75 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %76 = arith.index_cast %46 : index to i16
              %77 = arith.index_cast %60 : index to i16
              %78 = arith.index_cast %48 : index to i16
              %79 = ascendc.construct !ascendc.mmad_params(%76, %78, %77, %c0_i8, %c0_i8, %c0_i8) [ui16, ui16, ui16, ui8, ui8, ui8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %51, %74, %75, %79 {ascendc.unit = "AiCore.Cube"} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %14, %74 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              ascendc.que_bind.free_tensor %15, %75 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
            }
            ascendc.que_bind.enque_tensor %13, %51 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %10, %24, %50 : !ascendc.tbuf<vecin>, index
            ascendc.pipe.init_queue %10, %16, %c1_i32, %50 : !ascendc.queue<vecin, 1>, i32, index
            %52 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %53 = ascendc.que_bind.alloc_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.data_copy_co12dst %53, %52, %20 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.data_copy_co12dst_params
            ascendc.que_bind.enque_tensor %16, %53 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %13, %52 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %subview_0 = memref.subview %subview[%arg7, %arg8] [%46, %48] [1, 1] : memref<?x?xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
            ascendc.pipe.init_buffer %10, %23, %50 : !ascendc.tbuf<vecout>, index
            ascendc.pipe.init_queue %10, %17, %c1_i32, %50 : !ascendc.queue<vecout, 1>, i32, index
            ascendc.pipe.init_buffer %10, %22, %50 : !ascendc.tbuf<veccalc>, index
            %54 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            %55 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %10, %21, %50 : !ascendc.tbuf<veccalc>, index
            %56 = ascendc.tbuf.get_tensor %21 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %56, %cst, %49 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.max_l2 %54, %55, %56, %49 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %17, %54 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
            %57 = ascendc.que_bind.deque_tensor %17 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
            %58 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
            ascendc.global_tensor.set_global_buffer %58, %subview_0 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[64, 1], offset: ?>>
            emitasc.verbatim "{\0A  for (uint32_t _afir_i = 0; _afir_i < (uint32_t)$2; _afir_i++) {\0A    AscendC::GlobalTensor<float> _afir_gt;\0A    _afir_gt.SetGlobalBuffer($0.GetPhyAddr(_afir_i * (uint32_t)$4));\0A    AscendC::DataCopy(_afir_gt, $1[_afir_i * (uint32_t)$3], (uint32_t)$3);\0A  }\0A}" %58, %57, %46, %48, %c64 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index, index, index
            ascendc.que_bind.free_tensor %17, %57 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          }
        }
      }
      ascendc.que_bind.free_tensor %12, %40 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
      ascendc.que_bind.free_tensor %11, %39 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
    }
    return
  }
}

