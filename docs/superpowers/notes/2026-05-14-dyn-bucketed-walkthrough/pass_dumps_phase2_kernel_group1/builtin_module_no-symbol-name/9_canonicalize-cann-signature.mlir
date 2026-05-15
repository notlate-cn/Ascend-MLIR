// -----// IR Dump After CanonicalizeCannSignaturePass (canonicalize-cann-signature) //----- //
module attributes {vector_plan.tiling_infos = [{block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", fields = [{abi_index = 0 : i32, arg_index = 4 : i32, axis_size = -1 : i64, default_value = 128 : i64, kind = "tunable", name = "XBLOCK"}, {abi_index = 1 : i32, arg_index = 5 : i32, axis_size = -1 : i64, default_value = 16 : i64, kind = "tunable", name = "XBLOCK_SUB"}], kernel_id = "kernel_group1__v0"}]} {
  func.func private @kernel_group1__v0(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg5: memref<ui8>, %arg6: !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>) attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}], ascendc.aicore, ascendc.global, cann.num_inputs = 4 : i32} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = emitasc.member %arg6 "XBLOCK" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %1 = emitasc.member %arg6 "XBLOCK_SUB" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %2 = emitasc.member %arg6 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %3 = emitasc.member %arg6 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %4 = emitasc.member %arg6 "dim_arg0_2" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %5 = arith.index_cast %0 : i64 to index
    %6 = arith.index_cast %1 : i64 to index
    %7 = ascendc.pipe
    %8 = ascendc.queue : <vecin, 1>
    %9 = ascendc.queue : <vecin, 1>
    %10 = ascendc.queue : <vecin, 1>
    %11 = ascendc.queue : <vecout, 1>
    %12 = ascendc.queue : <vecin, 1>
    %13 = ascendc.queue : <vecin, 1>
    %14 = ascendc.queue : <vecin, 1>
    %15 = ascendc.queue : <vecout, 1>
    %16 = arith.index_cast %2 : i64 to index
    %17 = arith.index_cast %3 : i64 to index
    %18 = arith.muli %16, %17 : index
    %19 = arith.index_cast %4 : i64 to index
    %20 = arith.muli %6, %19 : index
    %21 = arith.muli %20, %c4 : index
    %22 = arith.muli %6, %c4 : index
    %23 = ascendc.tbuf : <veccalc>
    %24 = ascendc.tbuf : <veccalc>
    %25 = ascendc.tbuf : <veccalc>
    %26 = ascendc.tbuf : <vecout>
    %27 = ascendc.tbuf : <vecin>
    %28 = ascendc.tbuf : <vecin>
    %29 = ascendc.tbuf : <vecin>
    %30 = ascendc.tbuf : <veccalc>
    %31 = ascendc.tbuf : <veccalc>
    %32 = ascendc.tbuf : <veccalc>
    %33 = ascendc.tbuf : <vecout>
    ascendc.pipe.init_buffer %7, %33, %22 : !ascendc.tbuf<vecout>, index
    %34 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %7, %34, %21 : !ascendc.tbuf<vecin>, index
    %35 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %7, %35, %21 : !ascendc.tbuf<vecin>, index
    ascendc.pipe.init_queue %7, %8, %c1_i32, %21 : !ascendc.queue<vecin, 1>, i32, index
    %36 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %7, %36, %21 : !ascendc.tbuf<vecin>, index
    ascendc.pipe.init_queue %7, %11, %c1_i32, %22 : !ascendc.queue<vecout, 1>, i32, index
    ascendc.pipe.init_queue %7, %10, %c1_i32, %21 : !ascendc.queue<vecin, 1>, i32, index
    ascendc.pipe.init_queue %7, %9, %c1_i32, %21 : !ascendc.queue<vecin, 1>, i32, index
    %37 = ascendc.get_block_idx : index
    %38 = arith.muli %37, %5 : index
    %39 = arith.cmpi ult, %38, %18 : index
    scf.if %39 {
      %40 = arith.subi %18, %38 : index
      %41 = arith.minsi %5, %40 : index
      %42 = arith.divsi %41, %6 : index
      %43 = arith.muli %42, %6 : index
      ascendc.pipe.init_buffer %7, %30, %21 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %7, %31, %21 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %7, %32, %21 : !ascendc.tbuf<veccalc>, index
      scf.for %arg7 = %c0 to %43 step %6 {
        %45 = arith.addi %38, %arg7 : index
        %46 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %47 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %48 = arith.muli %45, %19 : index
        %49 = arith.index_cast %48 : index to i32
        %50 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %47, %50, %49 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %46, %47, %20 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %8, %46 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %51 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %52 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %53 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %54 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %53, %54, %49 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %52, %53, %20 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %9, %52 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %55 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %56 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %57 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %58 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %57, %58, %49 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %56, %57, %20 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %10, %56 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %59 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %60 = ascendc.tbuf.get_tensor %32 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.duplicate_l2 %60, %cst, %20 : !ascendc.local_tensor<*xf32>, f32, index
        %61 = ascendc.tbuf.get_tensor %31 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %61, %51, %55, %20 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %62 = ascendc.tbuf.get_tensor %30 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.mul_l2 %62, %61, %59, %20 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.add_l2 %60, %60, %62, %20 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %63 = ascendc.que_bind.alloc_tensor %11 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.reduce_sum_2d_l2 %63, %60 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.enque_tensor %11, %63 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %64 = ascendc.que_bind.deque_tensor %11 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %65 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %66 = arith.index_cast %45 : index to i32
        %67 = emitasc.reinterpret_cast %arg4 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %65, %67, %66 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %65, %64, %6 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %11, %64 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %8, %51 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %9, %55 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %10, %59 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
      %44 = arith.cmpi slt, %43, %41 : index
      scf.if %44 {
        %45 = arith.subi %18, %6 : index
        ascendc.pipe.init_buffer %7, %29, %21 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %7, %12, %c1_i32, %21 : !ascendc.queue<vecin, 1>, i32, index
        %46 = ascendc.que_bind.alloc_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %47 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %48 = arith.muli %45, %19 : index
        %49 = arith.index_cast %48 : index to i32
        %50 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %47, %50, %49 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %46, %47, %20 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %12, %46 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %51 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %7, %28, %21 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %7, %13, %c1_i32, %21 : !ascendc.queue<vecin, 1>, i32, index
        %52 = ascendc.que_bind.alloc_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %53 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %54 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %53, %54, %49 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %52, %53, %20 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %13, %52 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %55 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %7, %27, %21 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %7, %14, %c1_i32, %21 : !ascendc.queue<vecin, 1>, i32, index
        %56 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %57 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %58 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %57, %58, %49 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %56, %57, %20 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %14, %56 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %59 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %7, %26, %22 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %7, %15, %c1_i32, %22 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %7, %25, %21 : !ascendc.tbuf<veccalc>, index
        %60 = ascendc.tbuf.get_tensor %25 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.duplicate_l2 %60, %cst, %20 : !ascendc.local_tensor<*xf32>, f32, index
        ascendc.pipe.init_buffer %7, %24, %21 : !ascendc.tbuf<veccalc>, index
        %61 = ascendc.tbuf.get_tensor %24 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %61, %51, %55, %20 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.pipe.init_buffer %7, %23, %21 : !ascendc.tbuf<veccalc>, index
        %62 = ascendc.tbuf.get_tensor %23 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.mul_l2 %62, %61, %59, %20 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.add_l2 %60, %60, %62, %20 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %63 = ascendc.que_bind.alloc_tensor %15 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.reduce_sum_2d_l2 %63, %60 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.enque_tensor %15, %63 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %64 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %65 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %66 = arith.index_cast %45 : index to i32
        %67 = emitasc.reinterpret_cast %arg4 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %65, %67, %66 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %65, %64, %6 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %15, %64 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %12, %51 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %13, %55 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %14, %59 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    return
  }
}


