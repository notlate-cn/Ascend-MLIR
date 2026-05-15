// -----// IR Dump After CSE (cse) //----- //
module attributes {vector_plan.tiling_infos = [{block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", fields = [{abi_index = 0 : i32, arg_index = 4 : i32, axis_size = -1 : i64, default_value = 128 : i64, kind = "tunable", name = "XBLOCK"}, {abi_index = 1 : i32, arg_index = 5 : i32, axis_size = -1 : i64, default_value = 16 : i64, kind = "tunable", name = "XBLOCK_SUB"}], kernel_id = "kernel_group1__v0"}]} {
  func.func private @kernel_group1__v0(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg5: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, 22 : i32>) attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}], ascendc.aicore, ascendc.global} {
    %0 = emitasc.copy_struct %arg5 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>
    %1 = emitasc.member %0 "XBLOCK" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %2 = emitasc.member %0 "XBLOCK_SUB" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %3 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %4 = emitasc.member %0 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %5 = emitasc.member %0 "dim_arg0_2" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %6 = arith.index_cast %1 : i64 to index
    %7 = arith.index_cast %2 : i64 to index
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %8 = ascendc.pipe
    %9 = ascendc.queue : <vecin, 1>
    %10 = ascendc.queue : <vecin, 1>
    %11 = ascendc.queue : <vecin, 1>
    %12 = ascendc.queue : <vecout, 1>
    %13 = ascendc.queue : <vecin, 1>
    %14 = ascendc.queue : <vecin, 1>
    %15 = ascendc.queue : <vecin, 1>
    %16 = ascendc.queue : <vecout, 1>
    %17 = arith.index_cast %3 : i64 to index
    %18 = arith.index_cast %4 : i64 to index
    %cast = memref.cast %arg4 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32>
    %collapse_shape = memref.collapse_shape %cast [[0, 1]] : memref<?x?xf32> into memref<?xf32>
    %19 = arith.muli %17, %18 : index
    %20 = arith.index_cast %5 : i64 to index
    %21 = arith.muli %7, %20 : index
    %22 = arith.muli %21, %c4 : index
    %23 = arith.muli %7, %c4 : index
    %24 = ascendc.tbuf : <veccalc>
    %25 = ascendc.tbuf : <veccalc>
    %26 = ascendc.tbuf : <veccalc>
    %27 = ascendc.tbuf : <vecout>
    %28 = ascendc.tbuf : <vecin>
    %29 = ascendc.tbuf : <vecin>
    %30 = ascendc.tbuf : <vecin>
    %31 = ascendc.tbuf : <veccalc>
    %32 = ascendc.tbuf : <veccalc>
    %33 = ascendc.tbuf : <veccalc>
    %34 = ascendc.tbuf : <vecout>
    ascendc.pipe.init_buffer %8, %34, %23 : !ascendc.tbuf<vecout>, index
    %35 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %8, %35, %22 : !ascendc.tbuf<vecin>, index
    %36 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %8, %36, %22 : !ascendc.tbuf<vecin>, index
    ascendc.pipe.init_queue %8, %9, %c1_i32, %22 : !ascendc.queue<vecin, 1>, i32, index
    %37 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %8, %37, %22 : !ascendc.tbuf<vecin>, index
    ascendc.pipe.init_queue %8, %12, %c1_i32, %23 : !ascendc.queue<vecout, 1>, i32, index
    ascendc.pipe.init_queue %8, %11, %c1_i32, %22 : !ascendc.queue<vecin, 1>, i32, index
    ascendc.pipe.init_queue %8, %10, %c1_i32, %22 : !ascendc.queue<vecin, 1>, i32, index
    %38 = ascendc.get_block_idx : index
    %39 = arith.muli %38, %6 : index
    %40 = arith.cmpi ult, %39, %19 : index
    scf.if %40 {
      %41 = arith.subi %19, %39 : index
      %42 = arith.minsi %6, %41 : index
      %43 = arith.divsi %42, %7 : index
      %44 = arith.muli %43, %7 : index
      ascendc.pipe.init_buffer %8, %31, %22 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %8, %32, %22 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %8, %33, %22 : !ascendc.tbuf<veccalc>, index
      scf.for %arg6 = %c0 to %44 step %7 {
        %46 = arith.addi %39, %arg6 : index
        %47 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %48 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %c1 = arith.constant 1 : index
        %49 = arith.muli %c1, %20 : index
        %50 = arith.muli %46, %49 : index
        %51 = arith.addi %c0, %50 : index
        %52 = arith.muli %c0, %c1 : index
        %53 = arith.addi %51, %52 : index
        %54 = arith.addi %c0, %53 : index
        %55 = arith.index_cast %54 : index to i32
        %56 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %48, %56, %55 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %47, %48, %21 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %9, %47 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %57 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %58 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %59 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %60 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %59, %60, %55 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %58, %59, %21 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %10, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %61 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %62 = ascendc.que_bind.alloc_tensor %11 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %63 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %64 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %63, %64, %55 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %62, %63, %21 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %11, %62 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %65 = ascendc.que_bind.deque_tensor %11 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %66 = ascendc.tbuf.get_tensor %33 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.duplicate_l2 %66, %cst, %21 : !ascendc.local_tensor<*xf32>, f32, index
        %67 = ascendc.tbuf.get_tensor %32 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %67, %57, %61, %21 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %68 = ascendc.tbuf.get_tensor %31 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.mul_l2 %68, %67, %65, %21 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.add_l2 %66, %66, %68, %21 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %69 = ascendc.que_bind.alloc_tensor %12 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.reduce_sum_2d_l2 %69, %66 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.enque_tensor %12, %69 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %70 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %71 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %72 = arith.addi %c0, %46 : index
        %73 = arith.index_cast %72 : index to i32
        %74 = emitasc.reinterpret_cast %arg4 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %71, %74, %73 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %71, %70, %7 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %12, %70 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %9, %57 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %10, %61 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %11, %65 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
      %45 = arith.cmpi slt, %44, %42 : index
      scf.if %45 {
        %46 = arith.subi %19, %7 : index
        ascendc.pipe.init_buffer %8, %30, %22 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %8, %13, %c1_i32, %22 : !ascendc.queue<vecin, 1>, i32, index
        %47 = ascendc.que_bind.alloc_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %48 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %c1 = arith.constant 1 : index
        %49 = arith.muli %c1, %20 : index
        %50 = arith.muli %46, %49 : index
        %51 = arith.addi %c0, %50 : index
        %52 = arith.muli %c0, %c1 : index
        %53 = arith.addi %51, %52 : index
        %54 = arith.addi %c0, %53 : index
        %55 = arith.index_cast %54 : index to i32
        %56 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %48, %56, %55 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %47, %48, %21 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %13, %47 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %57 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %8, %29, %22 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %8, %14, %c1_i32, %22 : !ascendc.queue<vecin, 1>, i32, index
        %58 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %59 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %60 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %59, %60, %55 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %58, %59, %21 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %14, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %61 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %8, %28, %22 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %8, %15, %c1_i32, %22 : !ascendc.queue<vecin, 1>, i32, index
        %62 = ascendc.que_bind.alloc_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %63 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %64 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %63, %64, %55 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %62, %63, %21 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %15, %62 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %65 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %8, %27, %23 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %8, %16, %c1_i32, %23 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %8, %26, %22 : !ascendc.tbuf<veccalc>, index
        %66 = ascendc.tbuf.get_tensor %26 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.duplicate_l2 %66, %cst, %21 : !ascendc.local_tensor<*xf32>, f32, index
        ascendc.pipe.init_buffer %8, %25, %22 : !ascendc.tbuf<veccalc>, index
        %67 = ascendc.tbuf.get_tensor %25 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %67, %57, %61, %21 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.pipe.init_buffer %8, %24, %22 : !ascendc.tbuf<veccalc>, index
        %68 = ascendc.tbuf.get_tensor %24 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.mul_l2 %68, %67, %65, %21 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.add_l2 %66, %66, %68, %21 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %69 = ascendc.que_bind.alloc_tensor %16 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.reduce_sum_2d_l2 %69, %66 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.enque_tensor %16, %69 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %70 = ascendc.que_bind.deque_tensor %16 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %71 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %72 = arith.addi %c0, %46 : index
        %73 = arith.index_cast %72 : index to i32
        %74 = emitasc.reinterpret_cast %arg4 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %71, %74, %73 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %71, %70, %7 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %16, %70 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %13, %57 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %14, %61 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %15, %65 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    return
  }
}


