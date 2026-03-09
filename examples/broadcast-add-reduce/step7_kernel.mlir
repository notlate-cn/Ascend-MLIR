// ============================================================
// STAGE 7: Kernel IR - 内核级中间表示
//
// 本阶段是接近最终代码生成的 IR 表示：
//   - 引入 emitasc 方言处理结构体和类型转换
//   - 使用 TilingData 结构体传递分块参数
//   - 使用 reinterpret_cast 进行内存类型转换
//   - 添加 ascendc.aicore 和 ascendc.global 属性标记内核函数
//
// 关键特性：
//   - 函数标记为 AiCore 内核函数
//   - TilingData 通过 GM 传递，包含 TB_M, TB_N 等参数
//   - 使用 emitasc.member 访问结构体成员
//   - 内存空间 22 表示 GM (Global Memory)
//
// 数据流：
//   1. 从 GM 复制 TilingData 到本地
//   2. 解包 TilingData 获取分块参数
//   3. 计算本核的数据范围
//   4. 执行核心计算循环
// ============================================================

module attributes {transform.with_named_sequence} {

  // 主内核函数：广播加法归约
  // 标记为 AiCore 内核函数，运行在 Global Memory 上
  func.func @broadcast_add_reducesum(
      // 输入 A [M] - GM (memory_space = 22)
      %input_a: memref<?xf16>,
      // 输入 B [M,N] - GM
      %input_b: memref<?x?xf16>,
      // TilingData 结构体 - GM
      // 包含: TB_M, TB_N, dim_arg0_0 (M), dim_arg1_1 (N)
      %tiling_data: memref<?x!emitasc.py_struct<"TilingData",
          [i64, i64, i64, i64],
          ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, 22 : i32>,
      // 输出 - GM
      %output: memref<?xf16, strided<[1], offset: ?>>
  ) attributes {ascendc.aicore, ascendc.global} {

    // 常量定义
    %idx_0 = arith.constant 0 : index
    %const_2 = arith.constant 2 : index
    %const_1_i32 = arith.constant 1 : i32

    // ---- 复制 TilingData 到本地 ----
    %local_tiling = emitasc.copy_struct %tiling_data
        : memref<?x!emitasc.py_struct<"TilingData",
            [i64, i64, i64, i64],
            ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, 22 : i32>,
          !emitasc.py_struct<"TilingData",
            [i64, i64, i64, i64],
            ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>

    // ---- 解包 TilingData ----
    %tb_m_val = emitasc.member %local_tiling "TB_M"
        : !emitasc.py_struct<"TilingData",
            [i64, i64, i64, i64],
            ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, i64
    %tb_n_val = emitasc.member %local_tiling "TB_N"
        : !emitasc.py_struct<"TilingData",
            [i64, i64, i64, i64],
            ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, i64
    %dim_m_val = emitasc.member %local_tiling "dim_arg0_0"
        : !emitasc.py_struct<"TilingData",
            [i64, i64, i64, i64],
            ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, i64
    %dim_n_val = emitasc.member %local_tiling "dim_arg1_1"
        : !emitasc.py_struct<"TilingData",
            [i64, i64, i64, i64],
            ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, i64

    // ---- 初始化 AscendC 运行时 ----
    %pipe = ascendc.pipe
    %queue_in = ascendc.queue : <vecin, 1>
    %queue_out = ascendc.queue : <vecout, 1>

    // 类型转换
    %tb_n = arith.index_cast %tb_n_val : i64 to index
    %tb_m = arith.index_cast %tb_m_val : i64 to index
    %dim_m = arith.index_cast %dim_m_val : i64 to index
    %dim_n = arith.index_cast %dim_n_val : i64 to index

    // ---- 分配 TBUF 缓冲区 ----
    %tbuf_calc_0 = ascendc.tbuf : <veccalc>
    %tbuf_calc_1 = ascendc.tbuf : <veccalc>
    %tbuf_calc_2 = ascendc.tbuf : <veccalc>
    %tbuf_out = ascendc.tbuf : <vecout>
    %tbuf_in = ascendc.tbuf : <vecin>

    // ---- 获取当前核ID ----
    %block_idx = ascendc.get_block_idx : index

    // ---- 计算本核的起始偏移 ----
    %block_offset = arith.muli %block_idx, %tb_m : index

    // ---- 边界检查 ----
    %is_in_bounds = arith.cmpi ult, %block_offset, %dim_m : index

    scf.if %is_in_bounds {

      // ---- 计算本核实际处理的行数 ----
      %remaining_rows = arith.subi %dim_m, %block_offset : index
      %actual_rows = arith.minsi %tb_m, %remaining_rows : index

      // ---- 内层循环：沿 N 维度分块 ----
      scf.for %inner_iv = %idx_0 to %actual_rows step %tb_n {

        // 计算本批次实际处理的行数
        %remaining_inner = arith.subi %actual_rows, %inner_iv : index
        %inner_size = arith.minsi %remaining_inner, %tb_n : index

        // ---- 计算缓冲区大小 ----
        %buffer_size_in = arith.muli %inner_size, %const_2 : index

        // ---- 初始化 VECIN 缓冲区 ----
        ascendc.pipe.init_buffer %pipe, %tbuf_in, %buffer_size_in
            : !ascendc.tbuf<vecin>, index

        // ---- 分配本地张量 ----
        %local_tensor_in = ascendc.que_bind.alloc_tensor %queue_in
            : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>

        // ---- 设置全局张量 ----
        %global_tensor_a = ascendc.global_tensor : !ascendc.global_tensor<*xf16>

        // 计算全局偏移
        %global_offset = arith.addi %inner_iv, %block_offset : index
        %global_offset_i32 = arith.index_cast %global_offset : index to i32

        // 类型转换：memref → GM memref
        %input_a_gm = emitasc.reinterpret_cast %input_a
            : memref<?xf16> to memref<?xf16, 22 : i32>

        ascendc.global_tensor.set_global_buffer %global_tensor_a, %input_a_gm, %global_offset_i32
            : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32

        // ---- 数据搬运：GM → VECIN ----
        ascendc.data_copy_l2 %local_tensor_in, %global_tensor_a, %inner_size
            : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index

        // ---- 入队/出队 ----
        ascendc.que_bind.enque_tensor %queue_in, %local_tensor_in
            : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %dequeued_in = ascendc.que_bind.deque_tensor %queue_in
            : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>

        // ---- 初始化 VECOUT 缓冲区 ----
        ascendc.pipe.init_buffer %pipe, %tbuf_out, %buffer_size_in
            : !ascendc.tbuf<vecout>, index

        // ---- 计算 B 的缓冲区大小 ----
        %b_total_size = arith.muli %inner_size, %dim_n : index
        %buffer_size_b = arith.muli %b_total_size, %const_2 : index

        // ---- 初始化 VECCALC 缓冲区 ----
        ascendc.pipe.init_buffer %pipe, %tbuf_calc_2, %buffer_size_b
            : !ascendc.tbuf<veccalc>, index
        %local_calc_2 = ascendc.tbuf.get_tensor %tbuf_calc_2
            : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>

        // 转换索引为 i32
        %inner_size_i32 = arith.index_cast %inner_size : index to i32
        %dim_n_i32 = arith.index_cast %dim_n : index to i32

        // ---- 初始化更多 VECCALC 缓冲区 ----
        ascendc.pipe.init_buffer %pipe, %tbuf_calc_1, %buffer_size_b
            : !ascendc.tbuf<veccalc>, index
        %local_calc_1 = ascendc.tbuf.get_tensor %tbuf_calc_1
            : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>

        // ---- 广播 ----
        ascendc.broadcast_l2 %local_calc_1, %dequeued_in,
            %inner_size_i32, %dim_n_i32, %inner_size_i32, %const_1_i32
            {constRank = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 2>}
            : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>,
              i32, i32, i32, i32

        // ---- 初始化最后一个 VECCALC 缓冲区 ----
        ascendc.pipe.init_buffer %pipe, %tbuf_calc_0, %buffer_size_b
            : !ascendc.tbuf<veccalc>, index
        %local_calc_0 = ascendc.tbuf.get_tensor %tbuf_calc_0
            : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>

        // ---- 设置 B 的全局张量 ----
        %global_tensor_b = ascendc.global_tensor : !ascendc.global_tensor<*xf16>

        // 计算 B 的全局偏移
        %b_offset = arith.muli %global_offset, %dim_n : index
        %b_offset_i32 = arith.index_cast %b_offset : index to i32

        // 类型转换
        %input_b_gm = emitasc.reinterpret_cast %input_b
            : memref<?x?xf16> to memref<?xf16, 22 : i32>

        ascendc.global_tensor.set_global_buffer %global_tensor_b, %input_b_gm, %b_offset_i32
            : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32

        // ---- 数据搬运：B 从 GM → VECCALC ----
        ascendc.data_copy_l2 %local_calc_0, %global_tensor_b, %b_total_size
            : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index

        // ---- 向量加法 ----
        ascendc.add_l2 %local_calc_2, %local_calc_1, %local_calc_0, %b_total_size
            : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>,
              !ascendc.local_tensor<*xf16>, index

        // ---- 累加 ----
        ascendc.add_l2 %local_calc_2, %local_calc_2, %local_calc_2, %b_total_size
            : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>,
              !ascendc.local_tensor<*xf16>, index

        // ---- 分配输出本地张量 ----
        %local_out = ascendc.que_bind.alloc_tensor %queue_out
            : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>

        // ---- 二维归约求和 ----
        ascendc.reduce_sum_2d_l2 %local_out, %local_calc_2 {layout = 0 : i32}
            : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>

        // ---- 入队/出队 ----
        ascendc.que_bind.enque_tensor %queue_out, %local_out
            : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        %dequeued_out = ascendc.que_bind.deque_tensor %queue_out
            : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>

        // ---- 设置输出全局张量 ----
        %global_tensor_out = ascendc.global_tensor : !ascendc.global_tensor<*xf16>

        // 类型转换
        %output_gm = emitasc.reinterpret_cast %output
            : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, 22 : i32>

        ascendc.global_tensor.set_global_buffer %global_tensor_out, %output_gm, %global_offset_i32
            : !ascendc.global_tensor<*xf16>, memref<?xf16, 22 : i32>, i32

        // ---- 数据写回：VECOUT → GM ----
        ascendc.data_copy_l2 %global_tensor_out, %dequeued_out, %inner_size
            : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index

        // ---- 释放张量 ----
        ascendc.que_bind.free_tensor %queue_out, %dequeued_out
            : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %queue_in, %dequeued_in
            : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>

      } // 内层循环结束
    } // scf.if 结束

    return
  }

  // Transform 调度脚本
  transform.named_sequence @__transform_main(
      %root_op: !transform.any_op {transform.readonly}
  ) {
    %matched_func = transform.structured.match ops{["func.func"]} in %root_op
        : (!transform.any_op) -> !transform.any_op
    %transformed_func, %new_arg_0, %new_arg_1 =
        transform.func.add_index_args %matched_func, 2
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %matched_generic = transform.structured.match ops{["linalg.generic"]} in %transformed_func
        : (!transform.any_op) -> !transform.any_op
    %tiled_op, %outer_loop = transform.structured.tile_using_for %matched_generic
        tile_sizes [%new_arg_0, 0]
        : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    %true_val = transform.param.constant true -> !transform.any_param
    transform.annotate %outer_loop "ascendc.parallel" = %true_val
        : !transform.any_op, !transform.any_param
    %tiled_inner_op, %inner_loop = transform.structured.tile_using_for %tiled_op
        tile_sizes [%new_arg_1, 0]
        : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    %prologue_val = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %epilogue_val = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    transform.annotate %inner_loop "ascendc.prologue" = %prologue_val
        : !transform.any_op, !transform.any_param
    transform.annotate %inner_loop "ascendc.epilogue" = %epilogue_val
        : !transform.any_op, !transform.any_param
    %unit_val = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.annotate %tiled_inner_op "ascendc.unit" = %unit_val
        : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %inner_loop : !transform.any_op
    transform.yield
  }
}
