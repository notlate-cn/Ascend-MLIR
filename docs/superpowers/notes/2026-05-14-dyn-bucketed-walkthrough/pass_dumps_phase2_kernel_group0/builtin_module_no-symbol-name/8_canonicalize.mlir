// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module attributes {vector_plan.tiling_infos = [{block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", fields = [{abi_index = 0 : i32, arg_index = 5 : i32, axis_size = -1 : i64, default_value = 128 : i64, kind = "tunable", name = "XBLOCK"}, {abi_index = 1 : i32, arg_index = 6 : i32, axis_size = -1 : i64, default_value = 16 : i64, kind = "tunable", name = "XBLOCK_SUB"}], kernel_id = "kernel_group0__v0"}]} {
  func.func private @kernel_group0__v0(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>, %arg3: memref<?x?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg6: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, 22 : i32>) attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}], ascendc.aicore, ascendc.global} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = emitasc.copy_struct %arg6 : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>
    %1 = emitasc.member %0 "XBLOCK" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %2 = emitasc.member %0 "XBLOCK_SUB" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %3 = emitasc.member %0 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %4 = emitasc.member %0 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %5 = emitasc.member %0 "dim_arg0_2" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %6 = arith.index_cast %1 : i64 to index
    %7 = arith.index_cast %2 : i64 to index
    %8 = ascendc.pipe
    %9 = ascendc.queue : <vecin, 1>
    %10 = ascendc.queue : <vecin, 1>
    %11 = ascendc.queue : <vecin, 1>
    %12 = ascendc.queue : <vecin, 1>
    %13 = ascendc.queue : <vecout, 1>
    %14 = ascendc.queue : <vecin, 1>
    %15 = ascendc.queue : <vecin, 1>
    %16 = ascendc.queue : <vecin, 1>
    %17 = ascendc.queue : <vecin, 1>
    %18 = ascendc.queue : <vecout, 1>
    %19 = arith.index_cast %3 : i64 to index
    %20 = arith.index_cast %4 : i64 to index
    %21 = arith.muli %19, %20 : index
    %22 = arith.index_cast %5 : i64 to index
    %23 = arith.muli %7, %22 : index
    %24 = arith.muli %23, %c4 : index
    %25 = arith.muli %7, %c4 : index
    %26 = ascendc.tbuf : <veccalc>
    %27 = ascendc.tbuf : <veccalc>
    %28 = ascendc.tbuf : <veccalc>
    %29 = ascendc.tbuf : <veccalc>
    %30 = ascendc.tbuf : <vecout>
    %31 = ascendc.tbuf : <vecin>
    %32 = ascendc.tbuf : <vecin>
    %33 = ascendc.tbuf : <vecin>
    %34 = ascendc.tbuf : <vecin>
    %35 = ascendc.tbuf : <veccalc>
    %36 = ascendc.tbuf : <veccalc>
    %37 = ascendc.tbuf : <veccalc>
    %38 = ascendc.tbuf : <veccalc>
    %39 = ascendc.tbuf : <vecout>
    ascendc.pipe.init_buffer %8, %39, %25 : !ascendc.tbuf<vecout>, index
    %40 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %8, %40, %24 : !ascendc.tbuf<vecin>, index
    %41 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %8, %41, %24 : !ascendc.tbuf<vecin>, index
    %42 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %8, %42, %24 : !ascendc.tbuf<vecin>, index
    ascendc.pipe.init_queue %8, %9, %c1_i32, %24 : !ascendc.queue<vecin, 1>, i32, index
    %43 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %8, %43, %24 : !ascendc.tbuf<vecin>, index
    ascendc.pipe.init_queue %8, %13, %c1_i32, %25 : !ascendc.queue<vecout, 1>, i32, index
    ascendc.pipe.init_queue %8, %12, %c1_i32, %24 : !ascendc.queue<vecin, 1>, i32, index
    ascendc.pipe.init_queue %8, %11, %c1_i32, %24 : !ascendc.queue<vecin, 1>, i32, index
    ascendc.pipe.init_queue %8, %10, %c1_i32, %24 : !ascendc.queue<vecin, 1>, i32, index
    %44 = ascendc.get_block_idx : index
    %45 = arith.muli %44, %6 : index
    %46 = arith.cmpi ult, %45, %21 : index
    scf.if %46 {
      %47 = arith.subi %21, %45 : index
      %48 = arith.minsi %6, %47 : index
      %49 = arith.divsi %48, %7 : index
      %50 = arith.muli %49, %7 : index
      ascendc.pipe.init_buffer %8, %35, %24 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %8, %36, %24 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %8, %37, %24 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %8, %38, %24 : !ascendc.tbuf<veccalc>, index
      scf.for %arg7 = %c0 to %50 step %7 {
        %52 = arith.addi %45, %arg7 : index
        %53 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %55 = arith.muli %52, %22 : index
        %56 = arith.index_cast %55 : index to i32
        %57 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %54, %57, %56 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %53, %54, %23 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %9, %53 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %58 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %59 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %60 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %61 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %60, %61, %56 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %59, %60, %23 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %10, %59 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %62 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %63 = ascendc.que_bind.alloc_tensor %11 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %64 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %65 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %64, %65, %56 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %63, %64, %23 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %11, %63 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %66 = ascendc.que_bind.deque_tensor %11 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %67 = ascendc.que_bind.alloc_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %68 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %69 = emitasc.reinterpret_cast %arg3 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %68, %69, %56 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %67, %68, %23 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %12, %67 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %70 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %71 = ascendc.tbuf.get_tensor %38 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.duplicate_l2 %71, %cst, %23 : !ascendc.local_tensor<*xf32>, f32, index
        %72 = ascendc.tbuf.get_tensor %37 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %72, %58, %62, %23 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %73 = ascendc.tbuf.get_tensor %36 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.mul_l2 %73, %72, %66, %23 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %74 = ascendc.tbuf.get_tensor %35 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %74, %73, %70, %23 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.add_l2 %71, %71, %74, %23 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %75 = ascendc.que_bind.alloc_tensor %13 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.reduce_sum_2d_l2 %75, %71 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.enque_tensor %13, %75 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %76 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %77 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %78 = arith.index_cast %52 : index to i32
        %79 = emitasc.reinterpret_cast %arg5 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %77, %79, %78 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %77, %76, %7 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %13, %76 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %9, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %10, %62 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %11, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %12, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
      %51 = arith.cmpi slt, %50, %48 : index
      scf.if %51 {
        %52 = arith.subi %21, %7 : index
        ascendc.pipe.init_buffer %8, %34, %24 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %8, %14, %c1_i32, %24 : !ascendc.queue<vecin, 1>, i32, index
        %53 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %54 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %55 = arith.muli %52, %22 : index
        %56 = arith.index_cast %55 : index to i32
        %57 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %54, %57, %56 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %53, %54, %23 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %14, %53 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %58 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %8, %33, %24 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %8, %15, %c1_i32, %24 : !ascendc.queue<vecin, 1>, i32, index
        %59 = ascendc.que_bind.alloc_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %60 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %61 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %60, %61, %56 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %59, %60, %23 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %15, %59 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %62 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %8, %32, %24 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %8, %16, %c1_i32, %24 : !ascendc.queue<vecin, 1>, i32, index
        %63 = ascendc.que_bind.alloc_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %64 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %65 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %64, %65, %56 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %63, %64, %23 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %16, %63 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %66 = ascendc.que_bind.deque_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %8, %31, %24 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %8, %17, %c1_i32, %24 : !ascendc.queue<vecin, 1>, i32, index
        %67 = ascendc.que_bind.alloc_tensor %17 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %68 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %69 = emitasc.reinterpret_cast %arg3 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %68, %69, %56 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %67, %68, %23 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %17, %67 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %70 = ascendc.que_bind.deque_tensor %17 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %8, %30, %25 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %8, %18, %c1_i32, %25 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %8, %29, %24 : !ascendc.tbuf<veccalc>, index
        %71 = ascendc.tbuf.get_tensor %29 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.duplicate_l2 %71, %cst, %23 : !ascendc.local_tensor<*xf32>, f32, index
        ascendc.pipe.init_buffer %8, %28, %24 : !ascendc.tbuf<veccalc>, index
        %72 = ascendc.tbuf.get_tensor %28 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %72, %58, %62, %23 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.pipe.init_buffer %8, %27, %24 : !ascendc.tbuf<veccalc>, index
        %73 = ascendc.tbuf.get_tensor %27 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.mul_l2 %73, %72, %66, %23 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.pipe.init_buffer %8, %26, %24 : !ascendc.tbuf<veccalc>, index
        %74 = ascendc.tbuf.get_tensor %26 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %74, %73, %70, %23 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.add_l2 %71, %71, %74, %23 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %75 = ascendc.que_bind.alloc_tensor %18 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.reduce_sum_2d_l2 %75, %71 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.enque_tensor %18, %75 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %76 = ascendc.que_bind.deque_tensor %18 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %77 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %78 = arith.index_cast %52 : index to i32
        %79 = emitasc.reinterpret_cast %arg5 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %77, %79, %78 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %77, %76, %7 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %18, %76 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %14, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %15, %62 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %16, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %17, %70 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    return
  }
}


