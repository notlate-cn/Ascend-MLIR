// ============================================================
// STAGE 5: AscendC Dialect Lowering - 昇腾C方言降级
//
// 本阶段将 linalg.generic 转换为 AscendC 专用指令：
//   - 使用 ascendc.pipe 管理内存管道
//   - 使用 ascendc.queue 进行异步数据搬运
//   - 使用 ascendc.tbuf 分配片上缓冲区
//   - 使用 ascendc.data_copy_l2 进行 L2 级数据拷贝
//   - 使用 ascendc.add_l2 执行向量加法
//   - 使用 ascendc.reduce_sum_2d_l2 执行二维归约
//
// AscendC 内存层级：
//   - GM (Global Memory): 全局内存
//   - VECIN (9): 向量输入缓冲区
//   - VECOUT (10): 向量输出缓冲区
//   - VECCALC (11): 向量计算缓冲区
//
// 执行流程：
//   1. 初始化 Pipe 和 Queue
//   2. 分配 TBUF 缓冲区
//   3. 数据搬运: GM → VECIN
//   4. 广播: VECIN → VECCALC
//   5. 计算: Add + ReduceSum
//   6. 数据写回: VECOUT → GM
// ============================================================

// 动态维度边界计算
#dynamic_bound = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
// 广播映射
#broadcast_map = affine_map<(d0, d1) -> (d0)>
// 完整访问映射
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  // 主函数：AscendC 方言实现
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
    // 创建管道 (Pipe) 管理内存流
    %pipe = ascendc.pipe
    // 创建输入队列 (VECIN, 深度1)
    %queue_in = ascendc.queue : <vecin, 1>
    // 创建输出队列 (VECOUT, 深度1)
    %queue_out = ascendc.queue : <vecout, 1>

    // 类型转换
    %tb_inner_m = arith.index_cast %tb_inner_m_param : i64 to index
    %tb_m = arith.index_cast %tb_m_param : i64 to index

    // 获取维度
    %dim_m = memref.dim %input_a, %idx_0 : memref<?xf16>

    // 分配输出缓冲区 (GM)
    %output_buffer = memref.alloc(%dim_m) {alignment = 64 : i64} : memref<?xf16>
    %dim_n = memref.dim %input_b, %idx_1 : memref<?x?xf16>

    // ---- 分配 TBUF 缓冲区 ----
    // VECCALC (11): 计算缓冲区
    %tbuf_calc_0 = ascendc.tbuf : <veccalc>
    %tbuf_calc_1 = ascendc.tbuf : <veccalc>
    %tbuf_calc_2 = ascendc.tbuf : <veccalc>
    // VECOUT (10): 输出缓冲区
    %tbuf_out = ascendc.tbuf : <vecout>
    // VECIN (9): 输入缓冲区
    %tbuf_in = ascendc.tbuf : <vecin>

    // ---- 外层循环: TB 层 ----
    scf.for %outer_iv = %idx_0 to %dim_m step %tb_m {

      // 计算实际分块大小
      %outer_size = affine.min #dynamic_bound(%outer_iv)[%dim_m, %tb_m]

      // 创建 GM 子视图
      %subview_a = memref.subview %input_a[%outer_iv] [%outer_size] [1]
                   : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_b = memref.subview %input_b[%outer_iv, 0] [%outer_size, %dim_n] [1, 1]
                   : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_out = memref.subview %output_buffer[%outer_iv] [%outer_size] [1]
                     : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>

      // ---- 内层循环: Tb 层 ----
      scf.for %inner_iv = %idx_0 to %outer_size step %tb_inner_m {

        %inner_size = affine.min #dynamic_bound(%inner_iv)[%outer_size, %tb_inner_m]

        // 创建子视图
        %inner_subview_a = memref.subview %subview_a[%inner_iv] [%inner_size] [1]
                           : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>

        // ---- 计算缓冲区大小 ----
        // 输入缓冲区大小 = inner_size * 2 (f16 = 2 bytes)
        %buffer_size_in = arith.muli %inner_size, %const_2 : index

        // ---- 初始化 VECIN 缓冲区 ----
        ascendc.pipe.init_buffer %pipe, %tbuf_in, %buffer_size_in
            : !ascendc.tbuf<vecin>, index

        // ---- 分配本地张量 (VECIN) ----
        %local_tensor_in = ascendc.que_bind.alloc_tensor %queue_in
            : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>

        // ---- 设置全局张量 (GM) ----
        %global_tensor_a = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %global_tensor_a, %inner_subview_a
            : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>

        // ---- 数据搬运: GM → VECIN ----
        ascendc.data_copy_l2 %local_tensor_in, %global_tensor_a, %inner_size
            : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index

        // ---- 入队 VECIN ----
        ascendc.que_bind.enque_tensor %queue_in, %local_tensor_in
            : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>

        // ---- 出队 VECIN ----
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

        // ---- 计算缓冲区大小 (B矩阵) ----
        %b_total_size = arith.muli %inner_size, %dim_n : index
        %buffer_size_b = arith.muli %b_total_size, %const_2 : index

        // ---- 初始化 VECCALC 缓冲区 ----
        ascendc.pipe.init_buffer %pipe, %tbuf_calc_2, %buffer_size_b
            : !ascendc.tbuf<veccalc>, index
        %local_calc_2 = ascendc.tbuf.get_tensor %tbuf_calc_2
            : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>

        // 转换索引为 i32 (用于广播指令)
        %inner_size_i32 = arith.index_cast %inner_size : index to i32
        %dim_n_i32 = arith.index_cast %dim_n : index to i32

        // ---- 初始化更多 VECCALC 缓冲区 ----
        ascendc.pipe.init_buffer %pipe, %tbuf_calc_1, %buffer_size_b
            : !ascendc.tbuf<veccalc>, index
        %local_calc_1 = ascendc.tbuf.get_tensor %tbuf_calc_1
            : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<*xf16>

        // ---- 广播: VECIN → VECCALC ----
        // 将输入 A 广播到二维形状 [inner_size, dim_n]
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

        // ---- 数据搬运: B 从 GM → VECCALC ----
        ascendc.data_copy_l2 %local_calc_0, %global_tensor_b, %b_total_size
            : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index

        // ---- 向量加法: A + B ----
        ascendc.add_l2 %local_calc_2, %local_calc_1, %local_calc_0, %b_total_size
            : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>,
              !ascendc.local_tensor<*xf16>, index

        // ---- 累加 (用于归约) ----
        ascendc.add_l2 %local_calc_2, %local_calc_2, %local_calc_2, %b_total_size
            : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>,
              !ascendc.local_tensor<*xf16>, index

        // ---- 分配输出本地张量 ----
        %local_out = ascendc.que_bind.alloc_tensor %queue_out
            : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>

        // ---- 二维归约求和 ----
        ascendc.reduce_sum_2d_l2 %local_out, %local_calc_2 {layout = 0 : i32}
            : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>

        // ---- 入队 VECOUT ----
        ascendc.que_bind.enque_tensor %queue_out, %local_out
            : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>

        // ---- 出队 VECOUT ----
        %dequeued_out = ascendc.que_bind.deque_tensor %queue_out
            : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>

        // ---- 设置输出全局张量 ----
        %global_tensor_out = ascendc.global_tensor : !ascendc.global_tensor<*xf16>
        ascendc.global_tensor.set_global_buffer %global_tensor_out, %inner_subview_out
            : !ascendc.global_tensor<*xf16>, memref<?xf16, strided<[1], offset: ?>>

        // ---- 数据写回: VECOUT → GM ----
        ascendc.data_copy_l2 %global_tensor_out, %dequeued_out, %inner_size
            : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index

        // ---- 释放张量 ----
        ascendc.que_bind.free_tensor %queue_out, %dequeued_out
            : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf16>
        ascendc.que_bind.free_tensor %queue_in, %dequeued_in
            : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>

      } // 内层循环结束
    } // 外层循环结束

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
