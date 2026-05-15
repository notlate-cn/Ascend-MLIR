// -----// IR Dump After CanonicalizeCannSignaturePass (canonicalize-cann-signature) //----- //
module attributes {vector_plan.tiling_infos = [{block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", fields = [{abi_index = 0 : i32, arg_index = 5 : i32, axis_size = -1 : i64, default_value = 128 : i64, kind = "tunable", name = "XBLOCK"}, {abi_index = 1 : i32, arg_index = 6 : i32, axis_size = -1 : i64, default_value = 16 : i64, kind = "tunable", name = "XBLOCK_SUB"}], kernel_id = "kernel_group0__v0"}]} {
  func.func private @kernel_group0__v0(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>, %arg3: memref<?x?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg6: memref<ui8>, %arg7: !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>) attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}], ascendc.aicore, ascendc.global, cann.num_inputs = 5 : i32} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = emitasc.member %arg7 "XBLOCK" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %1 = emitasc.member %arg7 "XBLOCK_SUB" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %2 = emitasc.member %arg7 "dim_arg0_0" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %3 = emitasc.member %arg7 "dim_arg0_1" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %4 = emitasc.member %arg7 "dim_arg0_2" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["XBLOCK", "XBLOCK_SUB", "dim_arg0_0", "dim_arg0_1", "dim_arg0_2"]>, i64
    %5 = arith.index_cast %0 : i64 to index
    %6 = arith.index_cast %1 : i64 to index
    %7 = ascendc.pipe
    %8 = ascendc.queue : <vecin, 1>
    %9 = ascendc.queue : <vecin, 1>
    %10 = ascendc.queue : <vecin, 1>
    %11 = ascendc.queue : <vecin, 1>
    %12 = ascendc.queue : <vecout, 1>
    %13 = ascendc.queue : <vecin, 1>
    %14 = ascendc.queue : <vecin, 1>
    %15 = ascendc.queue : <vecin, 1>
    %16 = ascendc.queue : <vecin, 1>
    %17 = ascendc.queue : <vecout, 1>
    %18 = arith.index_cast %2 : i64 to index
    %19 = arith.index_cast %3 : i64 to index
    %20 = arith.muli %18, %19 : index
    %21 = arith.index_cast %4 : i64 to index
    %22 = arith.muli %6, %21 : index
    %23 = arith.muli %22, %c4 : index
    %24 = arith.muli %6, %c4 : index
    %25 = ascendc.tbuf : <veccalc>
    %26 = ascendc.tbuf : <veccalc>
    %27 = ascendc.tbuf : <veccalc>
    %28 = ascendc.tbuf : <veccalc>
    %29 = ascendc.tbuf : <vecout>
    %30 = ascendc.tbuf : <vecin>
    %31 = ascendc.tbuf : <vecin>
    %32 = ascendc.tbuf : <vecin>
    %33 = ascendc.tbuf : <vecin>
    %34 = ascendc.tbuf : <veccalc>
    %35 = ascendc.tbuf : <veccalc>
    %36 = ascendc.tbuf : <veccalc>
    %37 = ascendc.tbuf : <veccalc>
    %38 = ascendc.tbuf : <vecout>
    ascendc.pipe.init_buffer %7, %38, %24 : !ascendc.tbuf<vecout>, index
    %39 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %7, %39, %23 : !ascendc.tbuf<vecin>, index
    %40 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %7, %40, %23 : !ascendc.tbuf<vecin>, index
    %41 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %7, %41, %23 : !ascendc.tbuf<vecin>, index
    ascendc.pipe.init_queue %7, %8, %c1_i32, %23 : !ascendc.queue<vecin, 1>, i32, index
    %42 = ascendc.tbuf : <vecin>
    ascendc.pipe.init_buffer %7, %42, %23 : !ascendc.tbuf<vecin>, index
    ascendc.pipe.init_queue %7, %12, %c1_i32, %24 : !ascendc.queue<vecout, 1>, i32, index
    ascendc.pipe.init_queue %7, %11, %c1_i32, %23 : !ascendc.queue<vecin, 1>, i32, index
    ascendc.pipe.init_queue %7, %10, %c1_i32, %23 : !ascendc.queue<vecin, 1>, i32, index
    ascendc.pipe.init_queue %7, %9, %c1_i32, %23 : !ascendc.queue<vecin, 1>, i32, index
    %43 = ascendc.get_block_idx : index
    %44 = arith.muli %43, %5 : index
    %45 = arith.cmpi ult, %44, %20 : index
    scf.if %45 {
      %46 = arith.subi %20, %44 : index
      %47 = arith.minsi %5, %46 : index
      %48 = arith.divsi %47, %6 : index
      %49 = arith.muli %48, %6 : index
      ascendc.pipe.init_buffer %7, %34, %23 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %7, %35, %23 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %7, %36, %23 : !ascendc.tbuf<veccalc>, index
      ascendc.pipe.init_buffer %7, %37, %23 : !ascendc.tbuf<veccalc>, index
      scf.for %arg8 = %c0 to %49 step %6 {
        %51 = arith.addi %44, %arg8 : index
        %52 = ascendc.que_bind.alloc_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %53 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %54 = arith.muli %51, %21 : index
        %55 = arith.index_cast %54 : index to i32
        %56 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %53, %56, %55 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %52, %53, %22 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %8, %52 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %57 = ascendc.que_bind.deque_tensor %8 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %58 = ascendc.que_bind.alloc_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %59 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %60 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %59, %60, %55 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %58, %59, %22 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %9, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %61 = ascendc.que_bind.deque_tensor %9 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %62 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %63 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %64 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %63, %64, %55 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %62, %63, %22 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %10, %62 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %65 = ascendc.que_bind.deque_tensor %10 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %66 = ascendc.que_bind.alloc_tensor %11 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %67 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %68 = emitasc.reinterpret_cast %arg3 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %67, %68, %55 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %66, %67, %22 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %11, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %69 = ascendc.que_bind.deque_tensor %11 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %70 = ascendc.tbuf.get_tensor %37 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.duplicate_l2 %70, %cst, %22 : !ascendc.local_tensor<*xf32>, f32, index
        %71 = ascendc.tbuf.get_tensor %36 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %71, %57, %61, %22 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %72 = ascendc.tbuf.get_tensor %35 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.mul_l2 %72, %71, %65, %22 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %73 = ascendc.tbuf.get_tensor %34 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %73, %72, %69, %22 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.add_l2 %70, %70, %73, %22 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %74 = ascendc.que_bind.alloc_tensor %12 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.reduce_sum_2d_l2 %74, %70 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.enque_tensor %12, %74 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %75 = ascendc.que_bind.deque_tensor %12 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %76 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %77 = arith.index_cast %51 : index to i32
        %78 = emitasc.reinterpret_cast %arg5 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %76, %78, %77 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %76, %75, %6 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %12, %75 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %8, %57 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %9, %61 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %10, %65 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %11, %69 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
      %50 = arith.cmpi slt, %49, %47 : index
      scf.if %50 {
        %51 = arith.subi %20, %6 : index
        ascendc.pipe.init_buffer %7, %33, %23 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %7, %13, %c1_i32, %23 : !ascendc.queue<vecin, 1>, i32, index
        %52 = ascendc.que_bind.alloc_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %53 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %54 = arith.muli %51, %21 : index
        %55 = arith.index_cast %54 : index to i32
        %56 = emitasc.reinterpret_cast %arg0 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %53, %56, %55 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %52, %53, %22 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %13, %52 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %57 = ascendc.que_bind.deque_tensor %13 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %7, %32, %23 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %7, %14, %c1_i32, %23 : !ascendc.queue<vecin, 1>, i32, index
        %58 = ascendc.que_bind.alloc_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %59 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %60 = emitasc.reinterpret_cast %arg1 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %59, %60, %55 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %58, %59, %22 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %14, %58 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %61 = ascendc.que_bind.deque_tensor %14 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %7, %31, %23 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %7, %15, %c1_i32, %23 : !ascendc.queue<vecin, 1>, i32, index
        %62 = ascendc.que_bind.alloc_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %63 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %64 = emitasc.reinterpret_cast %arg2 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %63, %64, %55 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %62, %63, %22 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %15, %62 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %65 = ascendc.que_bind.deque_tensor %15 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %7, %30, %23 : !ascendc.tbuf<vecin>, index
        ascendc.pipe.init_queue %7, %16, %c1_i32, %23 : !ascendc.queue<vecin, 1>, i32, index
        %66 = ascendc.que_bind.alloc_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %67 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %68 = emitasc.reinterpret_cast %arg3 : memref<?x?x?xf32> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %67, %68, %55 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %66, %67, %22 : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
        ascendc.que_bind.enque_tensor %16, %66 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        %69 = ascendc.que_bind.deque_tensor %16 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.pipe.init_buffer %7, %29, %24 : !ascendc.tbuf<vecout>, index
        ascendc.pipe.init_queue %7, %17, %c1_i32, %24 : !ascendc.queue<vecout, 1>, i32, index
        ascendc.pipe.init_buffer %7, %28, %23 : !ascendc.tbuf<veccalc>, index
        %70 = ascendc.tbuf.get_tensor %28 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.duplicate_l2 %70, %cst, %22 : !ascendc.local_tensor<*xf32>, f32, index
        ascendc.pipe.init_buffer %7, %27, %23 : !ascendc.tbuf<veccalc>, index
        %71 = ascendc.tbuf.get_tensor %27 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %71, %57, %61, %22 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.pipe.init_buffer %7, %26, %23 : !ascendc.tbuf<veccalc>, index
        %72 = ascendc.tbuf.get_tensor %26 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.mul_l2 %72, %71, %65, %22 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.pipe.init_buffer %7, %25, %23 : !ascendc.tbuf<veccalc>, index
        %73 = ascendc.tbuf.get_tensor %25 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf32>
        ascendc.add_l2 %73, %72, %69, %22 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.add_l2 %70, %70, %73, %22 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        %74 = ascendc.que_bind.alloc_tensor %17 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.reduce_sum_2d_l2 %74, %70 {layout = 0 : i32} : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.enque_tensor %17, %74 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %75 = ascendc.que_bind.deque_tensor %17 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        %76 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        %77 = arith.index_cast %51 : index to i32
        %78 = emitasc.reinterpret_cast %arg5 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, 22 : i32>
        ascendc.global_tensor.set_global_buffer %76, %78, %77 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22 : i32>, i32
        ascendc.data_copy_l2 %76, %75, %6 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
        ascendc.que_bind.free_tensor %17, %75 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %13, %57 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %14, %61 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %15, %65 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %16, %69 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }
    return
  }
}


