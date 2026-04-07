#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module attributes {transform.with_named_sequence} {
  func.func @matmul_add_leakyrelu(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?xf32>, %arg3: memref<?x?xf32>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) -> memref<?x?xf32> attributes {ascendc.kernel_kind = "mix"} {
    %true = arith.constant true
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
    scf.for %arg9 = %c0 to %dim step %14 {
      scf.for %arg10 = %c0 to %dim_0 step %13 {
        %31 = affine.min #map(%arg9)[%dim, %14]
        %32 = affine.min #map(%arg10)[%dim_0, %13]
        %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf16>
        %subview = memref.subview %arg0[%arg9, 0] [%31, %dim_1] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %33 = arith.muli %31, %dim_1 : index
        %34 = arith.muli %33, %c2 : index
        ascendc.pipe.init_buffer %0, %30, %34 : !ascendc.tbuf<a1>, index
        ascendc.pipe.init_queue %0, %1, %c1_i32, %34 : !ascendc.queue<a1, 1>, i32, index
        %35 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        %36 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %36, %subview : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        %37 = arith.divui %dim_1, %c16 : index
        %38 = arith.index_cast %31 : index to i16
        %39 = arith.index_cast %37 : index to i16
        %40 = arith.index_cast %dim_1 : index to i16
        %41 = ascendc.construct !ascendc.nd2nz_params(%38, %39, %38, %40, %38, %c0_i16, %38, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %35, %36, %41 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %1, %35 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        %subview_2 = memref.subview %arg1[0, %arg10] [%dim_1, %32] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %42 = arith.muli %dim_1, %32 : index
        %43 = arith.muli %42, %c2 : index
        ascendc.pipe.init_buffer %0, %29, %43 : !ascendc.tbuf<b1>, index
        ascendc.pipe.init_queue %0, %2, %c1_i32, %43 : !ascendc.queue<b1, 1>, i32, index
        %44 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        %45 = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %45, %subview_2 : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
        %46 = arith.divui %32, %c16 : index
        %47 = arith.index_cast %46 : index to i16
        %48 = arith.index_cast %32 : index to i16
        %49 = ascendc.construct !ascendc.nd2nz_params(%40, %47, %40, %48, %40, %c0_i16, %40, %c0_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16
        ascendc.data_copy_nd2nz %44, %45, %49 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, !ascendc.nd2nz_params
        ascendc.que_bind.enque_tensor %2, %44 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        %subview_3 = memref.subview %arg2[%arg10] [%32] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
        %50 = arith.muli %32, %c4 : index
        ascendc.pipe.init_buffer %0, %28, %50 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %0, %3, %c1_i32, %50 : !ascendc.queue<vecin, 1>, i32, index
        %51 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %52 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %52, %subview_3 : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
        ascendc.data_copy_l2 %51, %52, %32 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %3, %51 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %53 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %54 = arith.muli %31, %32 : index
        %55 = arith.muli %54, %c4 : index
        ascendc.pipe.init_buffer %0, %27, %55 : !ascendc.tbuf<veccalc>, index
        ascendc.pipe.init_queue %0, %4, %c1_i32, %55 : !ascendc.queue<veccalc, 1>, i32, index
        %subview_4 = memref.subview %alloc[%arg9, %arg10] [%31, %32] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %56 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        %57 = ascendc.que_bind.deque_tensor %2 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        scf.for %arg11 = %c0 to %31 step %12 {
          scf.for %arg12 = %c0 to %32 step %11 {
            %58 = affine.min #map(%arg11)[%31, %12]
            %59 = affine.min #map(%arg12)[%32, %11]
            %60 = arith.muli %58, %59 : index
            %61 = arith.muli %60, %c4 : index
            ascendc.pipe.init_buffer %0, %26, %61 : !ascendc.tbuf<co1>, index
            ascendc.pipe.init_queue %0, %5, %c1_i32, %61 : !ascendc.queue<co1, 1>, i32, index
            %62 = ascendc.que_bind.alloc_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf16>
            scf.for %arg13 = %c0 to %dim_1 step %10 {
              %80 = affine.min #map(%arg13)[%dim_1, %10]
              %81 = arith.muli %58, %80 : index
              %82 = arith.muli %81, %c2 : index
              ascendc.pipe.init_buffer %0, %25, %82 : !ascendc.tbuf<a2>, index
              ascendc.pipe.init_queue %0, %6, %c1_i32, %82 : !ascendc.queue<a2, 1>, i32, index
              %83 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %84 = arith.divui %58, %c16 : index
              %85 = arith.index_cast %84 : index to i16
              %86 = arith.divui %80, %c16 : index
              %87 = arith.index_cast %86 : index to i64
              %88 = arith.trunci %87 : i64 to i8
              %89 = ascendc.construct !ascendc.load_data_2d_params(%c0_i16, %88, %85, %c0_i16, %c0_i16, %false, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_l0 %83, %56, %89 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.load_data_2d_params
              ascendc.que_bind.enque_tensor %6, %83 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %90 = arith.muli %80, %59 : index
              %91 = arith.muli %90, %c2 : index
              ascendc.pipe.init_buffer %0, %24, %91 : !ascendc.tbuf<b2>, index
              ascendc.pipe.init_queue %0, %7, %c1_i32, %91 : !ascendc.queue<b2, 1>, i32, index
              %92 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %93 = ascendc.construct !ascendc.load_data_2d_transpose_params(%c0_i16, %88, %c1_i16, %c0_i16, %c0_i16, %true, %c0_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8
              ascendc.load_data_with_transpose %92, %57, %93 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.load_data_2d_transpose_params
              ascendc.que_bind.enque_tensor %7, %92 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %94 = ascendc.que_bind.deque_tensor %6 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              %95 = ascendc.que_bind.deque_tensor %7 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
              %96 = arith.index_cast %58 : index to i16
              %97 = arith.index_cast %80 : index to i16
              %98 = arith.index_cast %59 : index to i16
              %99 = ascendc.construct !ascendc.mmad_params(%96, %98, %97, %c0_i8, %c0_i8, %c0_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8
              ascendc.mmad %62, %94, %95, %99 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
              ascendc.que_bind.free_tensor %6, %94 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf16>
              ascendc.que_bind.free_tensor %7, %95 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf16>
            }
            ascendc.que_bind.enque_tensor %5, %62 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf16>
            ascendc.pipe.init_buffer %0, %23, %61 : !ascendc.tbuf<vecin>, index
            ascendc.pipe.init_queue %0, %8, %c1_i32, %61 : !ascendc.queue<vecin, 1>, i32, index
            %63 = ascendc.que_bind.deque_tensor %5 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %64 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %65 = ascendc.construct !ascendc.data_copy_co12dst_params()
            ascendc.data_copy_co12dst %64, %63, %65 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.data_copy_co12dst_params
            ascendc.que_bind.enque_tensor %8, %64 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %5, %63 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>
            %subview_5 = memref.subview %subview_3[%arg12] [%59] [1] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
            ascendc.pipe.init_buffer %0, %22, %61 : !ascendc.tbuf<veccalc>, index
            %66 = ascendc.tbuf.get_tensor %22 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            %67 = ascendc.tbuf.get_tensor %21 : !ascendc.tbuf<co1>, !ascendc.local_tensor<*xf32>
            %68 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
            ascendc.global_tensor.set_global_buffer %68, %subview_5 : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
            %69 = arith.muli %59, %c4 : index
            ascendc.pipe.init_buffer %0, %20, %69 : !ascendc.tbuf<vecin>, index
            ascendc.pipe.init_queue %0, %19, %c1_i32, %69 : !ascendc.queue<vecin, 1>, i32, index
            %70 = ascendc.que_bind.alloc_tensor %19 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            ascendc.data_copy_l2 %70, %68, %59 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %19, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %71 = ascendc.que_bind.deque_tensor %19 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
            %72 = arith.index_cast %58 : index to i32
            %73 = arith.index_cast %59 : index to i32
            ascendc.pipe.init_buffer %0, %18, %61 : !ascendc.tbuf<veccalc>, index
            %74 = ascendc.tbuf.get_tensor %18 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            ascendc.broadcast_l2 %74, %71, %72, %73, %c1_i32, %73 {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32, i32, i32, i32
            ascendc.add_l2 %66, %67, %74, %60 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %4, %66 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %subview_6 = memref.subview %subview_4[%arg11, %arg12] [%58, %59] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            ascendc.pipe.init_buffer %0, %17, %61 : !ascendc.tbuf<vecout>, index
            ascendc.pipe.init_queue %0, %9, %c1_i32, %61 : !ascendc.queue<vecout, 1>, i32, index
            ascendc.pipe.init_buffer %0, %16, %61 : !ascendc.tbuf<veccalc>, index
            %75 = ascendc.tbuf.get_tensor %16 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            %76 = ascendc.que_bind.deque_tensor %4 : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.pipe.init_buffer %0, %15, %61 : !ascendc.tbuf<veccalc>, index
            %77 = ascendc.tbuf.get_tensor %15 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
            ascendc.duplicate_l2 %77, %cst, %60 : !ascendc.local_tensor<*xf32>, f32, index
            ascendc.mul_l2 %75, %76, %77, %60 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.max_l2 %75, %76, %75, %60 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.enque_tensor %9, %75 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
            %78 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
            %79 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
            ascendc.global_tensor.set_global_buffer %79, %subview_6 : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
            ascendc.data_copy_l2 %79, %78, %60 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
            ascendc.que_bind.free_tensor %9, %78 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
          }
        }
        ascendc.que_bind.free_tensor %2, %57 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %1, %56 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %3, %53 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
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
