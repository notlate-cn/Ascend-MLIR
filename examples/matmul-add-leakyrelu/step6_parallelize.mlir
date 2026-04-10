#map = affine_map<()[s0, s1, s2] -> (s1, s0 - s2)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module attributes {transform.with_named_sequence} {
  func.func @matmul_add_leakyrelu(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?xf32>, %arg3: memref<?x?xf32>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) -> memref<?x?xf32> attributes {ascendc.kernel_kind = "mix"} {
    %c1_i16 = arith.constant 1 : i16
    %c0_i8 = arith.constant 0 : i8
    %false = arith.constant false
    %c4 = arith.constant 4 : index
    %c0_i16 = arith.constant 0 : i16
    %c16 = arith.constant 16 : index
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e-03 : f32
    %0 = ascendc.pipe
    %1 = ascendc.queue : <a1, 1>
    %2 = ascendc.queue : <b1, 1>
    %3 = ascendc.queue : <vecin, 1>
    %4 = ascendc.queue : <veccalc, 1>
    %5 = ascendc.queue : <co1, 1>
    %6 = ascendc.queue : <a2, 1>
    %7 = ascendc.queue : <b2, 1>
    %8 = ascendc.queue : <vecin, 1>
    %9 = ascendc.queue : <vecout, 1>
    %10 = arith.index_cast %arg8 : i64 to index
    %11 = arith.index_cast %arg7 : i64 to index
    %12 = arith.index_cast %arg6 : i64 to index
    %13 = arith.index_cast %arg5 : i64 to index
    %14 = arith.index_cast %arg4 : i64 to index
    %dim = memref.dim %arg3, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf32>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
    %15 = ascendc.tbuf : <veccalc>
    %16 = ascendc.tbuf : <veccalc>
    %17 = ascendc.tbuf : <vecout>
    %18 = ascendc.tbuf : <veccalc>
    %19 = ascendc.queue : <vecin, 1>
    %20 = ascendc.tbuf : <vecin>
    %21 = ascendc.tbuf : <co1>
    %22 = ascendc.tbuf : <veccalc>
    %23 = ascendc.tbuf : <vecin>
    %24 = ascendc.tbuf : <b2>
    %25 = ascendc.tbuf : <a2>
    %26 = ascendc.tbuf : <co1>
    %27 = ascendc.tbuf : <veccalc>
    %28 = ascendc.tbuf : <vecin>
    %29 = ascendc.tbuf : <b1>
    %30 = ascendc.tbuf : <a1>
    %31 = ascendc.get_block_idx : index
    %32 = arith.muli %31, %14 : index
    %33 = arith.cmpi ult, %32, %dim : index
    scf.if %33 {
      scf.for %arg9 = %c0 to %dim_0 step %13 {
        %34 = affine.min #map()[%dim, %14, %32]
        %35 = affine.min #map1(%arg9)[%dim_0, %13]
        %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf16>
        %subview = memref.subview %arg0[%32, 0] [%34, %dim_1] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %36 = arith.muli %34, %dim_1 : index
        %37 = arith.muli %36, %c2 : index
        ascendc.pipe.init_buffer %0, %30, %37 : !ascendc.tbuf<a1>, index
        ascendc.pipe.init_queue %0, %1, %c1_i32, %37 : !ascendc.queue<a1, 1>, i32, index
        %38 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        %39 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %39, %subview : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        %40 = arith.divui %dim_1, %c16 : index
        %41 = arith.index_cast %34 : index to i16
        %42 = arith.index_cast %40 : index to i16
        %43 = arith.index_cast %dim_1 : index to i16
        %44 = ascendc.construct !ascendc.nd2nz_params(%41, %42, %41, %43, %41, %c0_i16, %41, %c0_i16) [ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %38, %39, %44 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %1, %38 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        %subview_2 = memref.subview %arg1[0, %arg9] [%dim_1, %35] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %45 = arith.muli %dim_1, %35 : index
        %46 = arith.muli %45, %c2 : index
        ascendc.pipe.init_buffer %0, %29, %46 : !ascendc.tbuf<b1>, index
        ascendc.pipe.init_queue %0, %2, %c1_i32, %46 : !ascendc.queue<b1, 1>, i32, index
        %47 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        %48 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %48, %subview_2 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        %49 = arith.divui %35, %c16 : index
        %50 = arith.index_cast %49 : index to i16
        %51 = arith.index_cast %35 : index to i16
        %52 = ascendc.construct !ascendc.nd2nz_params(%43, %50, %43, %51, %43, %c0_i16, %43, %c0_i16) [ui16, ui16, ui16, ui16, ui16, ui16, ui16, ui16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %47, %48, %52 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %2, %47 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        %subview_3 = memref.subview %arg2[%arg9] [%35] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
        %53 = arith.muli %35, %c4 : index
        ascendc.pipe.init_buffer %0, %28, %53 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %3, %c1_i32, %53 : !ascendc.queue<vecin, 1>, i32, index
        %54 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %55 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %55, %subview_3 : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %54, %55, %35 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %3, %54 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %56 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %57 = arith.muli %34, %35 : index
        %58 = arith.muli %57, %c4 : index
        ascendc.pipe.init_buffer %0, %27, %58 : !ascendc.tbuf<veccalc>, index
        ascendc.pipe.init_queue %0, %4, %c1_i32, %58 : !ascendc.queue<veccalc, 1>, i32, index
        %subview_4 = memref.subview %alloc[%32, %arg9] [%34, %35] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %59 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        %60 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        scf.for %arg10 = %c0 to %34 step %12 {
          scf.for %arg11 = %c0 to %35 step %11 {
            %61 = affine.min #map1(%arg10)[%34, %12]
            %62 = affine.min #map1(%arg11)[%35, %11]
            %63 = arith.muli %61, %62 : index
            %64 = arith.muli %63, %c4 : index
            ascendc.pipe.init_buffer %0, %26, %64 : !ascendc.tbuf<co1>, index
            ascendc.pipe.init_queue %0, %5, %c1_i32, %64 : !ascendc.queue<co1, 1>, i32, index
            %65 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            scf.for %arg12 = %c0 to %dim_1 step %10 {
              %83 = affine.min #map1(%arg12)[%dim_1, %10]
              %84 = arith.muli %61, %83 : index
              %85 = arith.muli %84, %c2 : index
              ascendc.pipe.init_buffer %0, %25, %85 : !ascendc.tbuf<a2>, index
              ascendc.pipe.init_queue %0, %6, %c1_i32, %85 : !ascendc.queue<a2, 1>, i32, index
              %86 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %87 = arith.divui %61, %c16 : index
              %88 = arith.index_cast %87 : index to i16
              %89 = arith.divui %83, %c16 : index
              %90 = arith.index_cast %89 : index to i64
              %91 = arith.trunci %90 : i64 to i8
              %92 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %91, %88, %c0_i16, %c0_i16, %false, %c0_i8) [ui16, ui8, ui16, ui8, ui16, i1, ui8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %86, %59, %92 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %86 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %93 = arith.muli %83, %62 : index
              %94 = arith.muli %93, %c2 : index
              ascendc.pipe.init_buffer %0, %24, %94 : !ascendc.tbuf<b2>, index
              ascendc.pipe.init_queue %0, %7, %c1_i32, %94 : !ascendc.queue<b2, 1>, i32, index
              %95 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %96 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %91, %c1_i16, %c0_i16, %c0_i16, %c0_i8) [ui16, ui8, ui16, ui16, ui16, ui8] : i16, i8, i16, i16, i16, i8
              ascendc.load_data_with_transpose %95, %60, %96 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %95 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %97 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %98 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %99 = arith.index_cast %61 : index to i16
              %100 = arith.index_cast %83 : index to i16
              %101 = arith.index_cast %62 : index to i16
              %102 = ascendc.construct !ascendc.mmad_params(%99, %101, %100, %c0_i8, %c0_i8, %c0_i8) [ui16, ui16, ui16, ui8, ui8, ui8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %65, %97, %98, %102 {ascendc.unit = "AiCore.Cube"} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %97 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              ascendc.que_bind.free_tensor %7, %98 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
            }
            ascendc.que_bind.enque_tensor %5, %65 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %23, %64 : !ascendc.tbuf<vecin>, index
            ascendc.pipe.init_queue %0, %8, %c1_i32, %64 : !ascendc.queue<vecin, 1>, i32, index
            %66 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %67 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %68 = ascendc.construct !ascendc.data_copy_co12dst_params()
            ascendc.data_copy_co12dst %67, %66, %68 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.data_copy_co12dst_params
            ascendc.que_bind.enque_tensor %8, %67 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %66 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %subview_5 = memref.subview %subview_3[%arg11] [%62] [1] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
            ascendc.pipe.init_buffer %0, %22, %64 : !ascendc.tbuf<veccalc>, index
            %69 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            %70 = ascendc.tbuf.get_tensor %21 : !ascendc.tbuf<co1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
            ascendc.global_tensor.set_global_buffer %71, %subview_5 : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
            %72 = arith.muli %62, %c4 : index
            ascendc.pipe.init_buffer %0, %20, %72 : !ascendc.tbuf<vecin>, index
            ascendc.pipe.init_queue %0, %19, %c1_i32, %72 : !ascendc.queue<vecin, 1>, i32, index
            %73 = ascendc.que_bind.alloc_tensor %19 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.data_copy_l2 %73, %71, %62 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %19, %73 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %74 = ascendc.que_bind.deque_tensor %19 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %75 = arith.index_cast %61 : index to i32
            %76 = arith.index_cast %62 : index to i32
            ascendc.pipe.init_buffer %0, %18, %64 : !ascendc.tbuf<veccalc>, index
            %77 = ascendc.tbuf.get_tensor %18 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            ascendc.broadcast_l2 %77, %74, %75, %76, %c1_i32, %76 {ascendc.unit = "AiCore.Vector", constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32, i32, i32, i32
            ascendc.add_l2 %69, %70, %77, %63 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %4, %69 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %subview_6 = memref.subview %subview_4[%arg10, %arg11] [%61, %62] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            ascendc.pipe.init_buffer %0, %17, %64 : !ascendc.tbuf<vecout>, index
            ascendc.pipe.init_queue %0, %9, %c1_i32, %64 : !ascendc.queue<vecout, 1>, i32, index
            ascendc.pipe.init_buffer %0, %16, %64 : !ascendc.tbuf<veccalc>, index
            %78 = ascendc.tbuf.get_tensor %16 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            %79 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %15, %64 : !ascendc.tbuf<veccalc>, index
            %80 = ascendc.tbuf.get_tensor %15 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %80, %cst, %63 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.mul_l2 %78, %79, %80, %63 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.max_l2 %78, %79, %78, %63 {ascendc.unit = "AiCore.Vector"} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %78 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
            %81 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
            %82 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
            ascendc.global_tensor.set_global_buffer %82, %subview_6 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
            ascendc.data_copy_l2 %82, %81, %63 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %81 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          }
        }
        ascendc.que_bind.free_tensor %2, %60 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %59 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %3, %56 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    return %alloc : memref<?x?xf32>
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

