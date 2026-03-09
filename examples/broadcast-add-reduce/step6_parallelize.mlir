// ============================================================
// STAGE 6: Parallelization - 多核并行化
//
// 本阶段引入多核并行执行：
//   - 使用 ascendc.get_block_idx 获取当前核ID
//   - 根据核ID计算数据偏移，实现数据并行
//   - 使用 scf.if 进行边界检查，防止越界访问
//
// 并行策略：
//   - 每个 AiCore 处理 TB_M 行数据
//   - 核ID × TB_M = 当前核的起始行号
//   - 最后一核可能处理少于 TB_M 行（边界处理）
//
// 执行流程：
//   1. 获取当前核ID (GetBlockIdx)
//   2. 计算本核负责的数据范围
//   3. 检查是否在有效范围内
//   4. 执行内层循环计算
// ============================================================

// 边界计算映射
#boundary_map = affine_map<()[s0, s1, s2] -> (s1, s0 - s2)>
// 动态边界计算
#dynamic_bound = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>

module attributes {transform.with_named_sequence} {

  // 主函数：多核并行版本
  func.func @broadcast_add_reducesum(
      %input_a: memref<?xf16>,       // 输入 A [M] - GM
      %input_b: memref<?x?xf16>,     // 输入 B [M,N] - GM
      %tb_m_param: i64,              // TB_M
      %tb_inner_m_param: i64         // Tb_M
  ) -> memref<?xf16> {

    // 常量定义
    %const_1_i32 = arith.constant 1 : i32
    %const_2 = arith.constant 2 : index
    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index

    // ---- 初始化 AscendC 运行时 ----
    %pipe = ascendc.pipe
    %queue_in = ascendc.queue : <vecin, 1>
    %queue_out = ascendc.queue : <vecout, 1>

    // 类型转换
    %tb_inner_m = arith.index_cast %tb_inner_m_param : i64 to index
    %tb_m = arith.index_cast %tb_m_param : i64 to index

    // 获取维度
    %dim_m = memref.dim %input_a, %idx_0 : memref<?xf16>

    // 分配输出缓冲区
    %output_buffer = memref.alloc(%dim_m) {alignment = 64 : i64} : memref<?xf16>
    %dim_n = memref.dim %input_b, %idx_1 : memref<?x?xf16>

    // ---- 分配 TBUF 缓冲区 ----
    %tbuf_calc_0 = ascendc.tbuf : <veccalc>
    %tbuf_calc_1 = ascendc.tbuf : <veccalc>
    %tbuf_calc_2 = ascendc.tbuf : <veccalc>
    %tbuf_out = ascendc.tbuf : <vecout>
    %tbuf_in = ascendc.tbuf : <vecin>

    // ---- 获取当前核ID ----
    %block_idx = ascendc.get_block_idx : index

    // ---- 计算本核的起始偏移 ----
    // offset = block_idx × TB_M
    %block_offset = arith.muli %block_idx, %tb_m : index

    // ---- 边界检查：offset < dim_m ----
    %is_in_bounds = arith.cmpi ult, %block_offset, %dim_m : index

    // 只在有效范围内执行
    scf.if %is_in_bounds {

      // ---- 计算本核实际处理的行数 ----
      // actual_rows = min(TB_M, dim_m - offset)
      %actual_rows = affine.min #boundary_map()[%dim_m, %tb_m, %block_offset]

      // 创建子视图（基于本核的数据范围）
      %subview_a = memref.subview %input_a[%block_offset] [%actual_rows] [1]
                   : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_b = memref.subview %input_b[%block_offset, 0] [%actual_rows, %dim_n] [1, 1]
                   : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_out = memref.subview %output_buffer[%block_offset] [%actual_rows] [1]
                     : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>

      // ---- 内层循环：Tb 层 ----
      scf.for %inner_iv = %idx_0 to %actual_rows step %tb_inner_m {

        %inner_size = affine.min #dynamic_bound(%inner_iv)[%actual_rows, %tb_inner_m]

        // 创建子视图
        %inner_subview_a = memref.subview %subview_a[%inner_iv] [%inner_size] [1]
                           : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>

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
        ascendc.global_tensor.set_global_buffer %global_tensor_a, %inner_subview_a
            : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>

        // ---- 数据搬运：GM → VECIN ----
        ascendc.data_copy_l2 %local_tensor_in, %global_tensor_a, %inner_size
            : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index

        // ---- 入队/出队 ----
        ascendc.que_bind.enque_tensor %queue_in, %local_tensor_in
            : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
        %dequeued_in = ascendc.que_bind.deque_tensor %queue_in
            : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>

        // 创建 B 和输出的子视图
        %inner_subview_b = memref.subview %subview_b[%inner_iv, 0] [%inner_size, %dim_n] [1, 1]
                           : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %inner_subview_out = memref.subview %subview_out[%inner_iv] [%inner_size] [1]
                             : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>

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
        ascendc.global_tensor.set_global_buffer %global_tensor_b, %inner_subview_b
            : !ascendc.global_tensor<*xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>

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
        ascendc.global_tensor.set_global_buffer %global_tensor_out, %inner_subview_out
            : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>

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

    return %output_buffer : memref<?xf16>
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
