// ============================================
// 最终 AscendC 内核 IR - 完整的 AiCore 执行代码
// ============================================
// 这是编译流水线的最终阶段 (Step 6)，生成可直接在华为昇腾 NPU 上执行的
// 完整内核代码。包含完整的内存管理、数据搬运、计算流水线。
//
// 本阶段特点：
// - 完整的 AscendC 运行时初始化 (pipe, queue, tbuf)
// - 显式的内存分配和释放 (que_bind.alloc_tensor/free_tensor)
// - 完整的 3 级嵌套循环结构 (TB_M/TB_N → Tb_M/Tb_N → t_K)
// - 数据搬运与计算的双缓冲流水线
// ============================================

// 仿射映射定义
// 计算边界：min(实际大小, 分块大小)
#min_bound_map = affine_map<()[s0, s1, s2] -> (s1, s0 - s2)>
#min_bound_map_inner = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>

module {
  // ============================================
  // 主内核函数：fc_relu
  // ============================================
  // 执行完整的矩阵乘法 + 偏置加法 + ReLU 激活
  //
  // 参数说明：
  // - %input_a: 输入矩阵 A (M×K)，位于全局内存 GM
  // - %input_b: 输入矩阵 B (K×N)，位于全局内存 GM
  // - %bias: 偏置矩阵 (M×N)，位于全局内存 GM
  // - %output: 输出矩阵 (M×N)，位于全局内存 GM
  // - %tiling_data: 分块参数结构体，包含 TB_M, TB_N, Tb_M, Tb_N, t_K
  //
  // 属性说明：
  // - ascendc.aicore: 表示这是 AiCore 内核函数
  // - ascendc.global: 表示这是全局可见的内核入口
  // ============================================
  func.func @fc_relu(
    %input_a: memref<?x?xf32, strided<[?, ?], offset: ?>>,
    %input_b: memref<?x?xf32, strided<[?, ?], offset: ?>>,
    %bias: memref<?x?xf32, strided<[?, ?], offset: ?>>,
    %output: memref<?x?xf32, strided<[?, ?], offset: ?>>,
    %tiling_data: memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>, 22 : i32>
  ) attributes {ascendc.aicore, ascendc.global} {

    // ============================================
    // 常量定义
    // ============================================
    %cst_one = arith.constant 1 : index           // 常量 1
    %cst_zero = arith.constant 0 : index          // 常量 0
    %cst_zero_f32 = arith.constant 0.000000e+00 : f32  // 浮点零值 (用于 ReLU)
    %cst_four = arith.constant 4 : index          // 常量 4 (float32 字节数)
    %cst_sixteen = arith.constant 16 : index      // 常量 16 (数据对齐粒度)
    %cst_zero_i16 = arith.constant 0 : i16        // i16 零值
    %cst_false = arith.constant false             // 布尔假
    %cst_zero_i8 = arith.constant 0 : i8          // i8 零值
    %cst_one_i16 = arith.constant 1 : i16         // i16 一值
    %cst_true = arith.constant true               // 布尔真

    // ============================================
    // 解析分块参数 (TilingData)
    // ============================================
    // 从 Host 传递的结构体中提取各级分块大小
    %tiling_struct = emitasc.copy_struct %tiling_data : memref<?x!emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>, 22 : i32>, !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>

    // 提取各级分块参数
    %tb_m_val = emitasc.member %tiling_struct "TB_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>, i64
    %tb_n_val = emitasc.member %tiling_struct "TB_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>, i64
    %tb_m_inner_val = emitasc.member %tiling_struct "Tb_M" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>, i64
    %tb_n_inner_val = emitasc.member %tiling_struct "Tb_N" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>, i64
    %t_k_val = emitasc.member %tiling_struct "t_K" : !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, i64], ["TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"]>, i64

    // ============================================
    // 初始化 AscendC 运行时
    // ============================================
    // 创建 Pipe 用于内存管理同步
    %pipe = ascendc.pipe

    // 创建各级队列 (Queue) - 用于数据流同步
    // A1/B1 队列：L1 缓冲区队列
    %queue_a1 = ascendc.queue : <a1, 1>
    %queue_b1 = ascendc.queue : <b1, 1>

    // VECIN/VECOUT 队列：UB 向量缓冲区队列
    %queue_vecin_1 = ascendc.queue : <vecin, 1>
    %queue_vecout = ascendc.queue : <vecout, 1>

    // CO1 队列：L0C 输出队列
    %queue_co1 = ascendc.queue : <co1, 1>

    // A2/B2 队列：L0A/L0B 输入队列
    %queue_a2 = ascendc.queue : <a2, 1>
    %queue_b2 = ascendc.queue : <b2, 1>

    // 额外的 VECIN/VECCALC 队列 (用于偏置加法和 ReLU)
    %queue_vecin_2 = ascendc.queue : <vecin, 1>
    %queue_veccalc = ascendc.queue : <veccalc, 1>
    %queue_vecin_3 = ascendc.queue : <vecin, 1>

    // ============================================
    // 类型转换：i64 → index
    // ============================================
    %tile_k_size = arith.index_cast %t_k_val : i64 to index
    %tile_n_inner = arith.index_cast %tb_n_inner_val : i64 to index
    %tile_m_inner = arith.index_cast %tb_m_inner_val : i64 to index
    %tile_n_outer = arith.index_cast %tb_n_val : i64 to index
    %tile_m_outer = arith.index_cast %tb_m_val : i64 to index

    // ============================================
    // 获取输出矩阵维度
    // ============================================
    %dim_m = memref.dim %output, %cst_zero : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_n = memref.dim %output, %cst_one : memref<?x?xf32, strided<[?, ?], offset: ?>>

    // ============================================
    // 分配各级缓冲区 (tbuf)
    // ============================================
    // VECIN 缓冲区：用于偏置数据
    %tbuf_vecin_bias = ascendc.tbuf : <vecin>
    %tbuf_veccalc = ascendc.tbuf : <veccalc>
    %tbuf_vecin_zero = ascendc.tbuf : <vecin>

    // B2/A2/CO1 缓冲区：用于矩阵乘法
    %tbuf_b2 = ascendc.tbuf : <b2>
    %tbuf_a2 = ascendc.tbuf : <a2>
    %tbuf_co1 = ascendc.tbuf : <co1>

    // VECOUT 缓冲区：用于向量计算输出
    %tbuf_vecout = ascendc.tbuf : <vecout>
    %tbuf_vecin_result = ascendc.tbuf : <vecin>

    // B1/A1 缓冲区：用于 L1 数据缓存
    %tbuf_b1 = ascendc.tbuf : <b1>
    %tbuf_a1 = ascendc.tbuf : <a1>

    // ============================================
    // 多核并行：获取当前核 ID
    // ============================================
    %block_idx = ascendc.get_block_idx : index

    // 计算本核负责的 M 维度起始偏移
    // offset = block_idx × TB_M
    %block_offset_m = arith.muli %block_idx, %tile_m_outer : index

    // 边界检查：offset < dim_m
    %is_in_bounds = arith.cmpi ult, %block_offset_m, %dim_m : index

    // ============================================
    // 主执行逻辑 (条件执行)
    // ============================================
    scf.if %is_in_bounds {

      // ============================================
      // 外层循环：遍历 N 维度 (TB_N 级别)
      // ============================================
      scf.for %n_outer_idx = %cst_zero to %dim_n step %tile_n_outer {

        // 计算当前 TB_M 块的实际大小 (处理边界)
        %tb_m_actual = affine.min #min_bound_map()[%dim_m, %tile_m_outer, %block_offset_m]

        // 计算当前 TB_N 块的实际大小 (处理边界)
        %tb_n_actual = affine.min #min_bound_map_inner(%n_outer_idx)[%dim_n, %tile_n_outer]

        // ============================================
        // 获取输入矩阵 A 的 K 维度大小
        // ============================================
        %dim_k = memref.dim %input_a, %cst_one : memref<?x?xf32, strided<[?, ?], offset: ?>>

        // ============================================
        // 准备输入 A 的数据搬运 (GM → A1)
        // ============================================
        // 创建 A 矩阵的子视图 [offset_m:offset_m+TB_M, 0:K]
        %subview_a = memref.subview %input_a[%block_offset_m, 0] [%tb_m_actual, %dim_k] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 计算 A1 缓冲区所需大小
        %a_total_elems = arith.muli %tb_m_actual, %dim_k : index
        %a_buffer_size = arith.muli %a_total_elems, %cst_four : index

        // 初始化 A1 缓冲区
        ascendc.pipe.init_buffer %pipe, %tbuf_a1, %a_buffer_size : !ascendc.tbuf<a1>, index

        // 分配 A1 本地张量
        %tensor_a1 = ascendc.que_bind.alloc_tensor %queue_a1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>

        // 创建全局张量包装
        %global_a = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %global_a, %subview_a : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 计算 ND2NZ 数据搬运参数
        %a_k_blocks = arith.divui %dim_k, %cst_sixteen : index
        %a_m_i16 = arith.index_cast %tb_m_actual : index to i16
        %a_k_blocks_i16 = arith.index_cast %a_k_blocks : index to i16
        %a_k_i16 = arith.index_cast %dim_k : index to i16

        // 构建 ND2NZ 参数
        %nd2nz_params_a = ascendc.construct !ascendc.nd2nz_params(%a_m_i16, %a_k_blocks_i16, %a_m_i16, %a_k_i16, %a_m_i16, %cst_zero_i16, %a_m_i16, %cst_zero_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16

        // 执行 GM → A1 数据搬运 (ND2NZ 格式转换)
        ascendc.data_copy_nd2nz %tensor_a1, %global_a, %nd2nz_params_a : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params

        // 将 A1 数据入队
        ascendc.que_bind.enque_tensor %queue_a1, %tensor_a1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>

        // ============================================
        // 准备输入 B 的数据搬运 (GM → B1)
        // ============================================
        // 创建 B 矩阵的子视图 [0:K, n_outer:n_outer+TB_N]
        %subview_b = memref.subview %input_b[0, %n_outer_idx] [%dim_k, %tb_n_actual] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 计算 B1 缓冲区所需大小
        %b_total_elems = arith.muli %dim_k, %tb_n_actual : index
        %b_buffer_size = arith.muli %b_total_elems, %cst_four : index

        // 初始化 B1 缓冲区
        ascendc.pipe.init_buffer %pipe, %tbuf_b1, %b_buffer_size : !ascendc.tbuf<b1>, index

        // 分配 B1 本地张量
        %tensor_b1 = ascendc.que_bind.alloc_tensor %queue_b1 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>

        // 创建全局张量包装
        %global_b = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %global_b, %subview_b : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 计算 ND2NZ 数据搬运参数
        %b_n_blocks = arith.divui %tb_n_actual, %cst_sixteen : index
        %b_n_blocks_i16 = arith.index_cast %b_n_blocks : index to i16
        %b_n_i16 = arith.index_cast %tb_n_actual : index to i16

        // 构建 ND2NZ 参数
        %nd2nz_params_b = ascendc.construct !ascendc.nd2nz_params(%a_k_i16, %b_n_blocks_i16, %a_k_i16, %b_n_i16, %a_k_i16, %cst_zero_i16, %a_k_i16, %cst_zero_i16) [i16, i16, i16, i16, i16, i16, i16, i16] : i16, i16, i16, i16, i16, i16, i16, i16

        // 执行 GM → B1 数据搬运 (ND2NZ 格式转换)
        ascendc.data_copy_nd2nz %tensor_b1, %global_b, %nd2nz_params_b : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.nd2nz_params

        // 将 B1 数据入队
        ascendc.que_bind.enque_tensor %queue_b1, %tensor_b1 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>

        // ============================================
        // 准备偏置数据搬运 (GM → VECIN)
        // ============================================
        // 创建输出矩阵的子视图 (用于后续写回)
        %subview_output = memref.subview %output[%block_offset_m, %n_outer_idx] [%tb_m_actual, %tb_n_actual] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 创建偏置矩阵的子视图
        %subview_bias = memref.subview %bias[%block_offset_m, %n_outer_idx] [%tb_m_actual, %tb_n_actual] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 计算 VECIN 缓冲区所需大小
        %bias_total_elems = arith.muli %tb_m_actual, %tb_n_actual : index
        %bias_buffer_size = arith.muli %bias_total_elems, %cst_four : index

        // 初始化 VECIN 缓冲区
        ascendc.pipe.init_buffer %pipe, %tbuf_vecin_bias, %bias_buffer_size : !ascendc.tbuf<vecin>, index

        // 分配 VECIN 本地张量
        %tensor_vecin_bias = ascendc.que_bind.alloc_tensor %queue_vecin_1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>

        // 创建全局张量包装
        %global_bias = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %global_bias, %subview_bias : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 执行 GM → VECIN 数据搬运 (L2 级别)
        ascendc.data_copy_l2 %tensor_vecin_bias, %global_bias, %bias_total_elems : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index

        // 将偏置数据入队
        ascendc.que_bind.enque_tensor %queue_vecin_1, %tensor_vecin_bias : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>

        // ============================================
        // 出队偏置数据 (准备后续使用)
        // ============================================
        %tensor_bias_dequeued = ascendc.que_bind.deque_tensor %queue_vecin_1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>

        // 初始化 VECOUT 缓冲区
        ascendc.pipe.init_buffer %pipe, %tbuf_vecout, %bias_buffer_size : !ascendc.tbuf<vecout>, index

        // ============================================
        // 出队 A1/B1 数据 (准备矩阵乘法)
        // ============================================
        %tensor_a1_dequeued = ascendc.que_bind.deque_tensor %queue_a1 : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>
        %tensor_b1_dequeued = ascendc.que_bind.deque_tensor %queue_b1 : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>

        // ============================================
        // 中层循环：遍历 M 维度 (Tb_M 级别)
        // ============================================
        scf.for %m_inner_idx = %cst_zero to %tb_m_actual step %tile_m_inner {

          // 分配 VECOUT 张量
          %tensor_vecout = ascendc.que_bind.alloc_tensor %queue_vecout : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>

          // ============================================
          // 内层循环：遍历 N 维度 (Tb_N 级别)
          // ============================================
          scf.for %n_inner_idx = %cst_zero to %tb_n_actual step %tile_n_inner {

            // 计算当前 Tb_M 块的实际大小
            %tb_m_inner_actual = affine.min #min_bound_map_inner(%m_inner_idx)[%tb_m_actual, %tile_m_inner]

            // 计算当前 Tb_N 块的实际大小
            %tb_n_inner_actual = affine.min #min_bound_map_inner(%n_inner_idx)[%tb_n_actual, %tile_n_inner]

            // 计算当前块的总元素数
            %block_elems = arith.muli %tb_m_inner_actual, %tb_n_inner_actual : index
            %block_buffer_size = arith.muli %block_elems, %cst_four : index

            // 初始化 CO1 缓冲区
            ascendc.pipe.init_buffer %pipe, %tbuf_co1, %block_buffer_size : !ascendc.tbuf<co1>, index

            // 分配 CO1 张量 (矩阵乘法累加缓冲区)
            %tensor_co1 = ascendc.que_bind.alloc_tensor %queue_co1 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>

            // ============================================
            // 最内层循环：遍历 K 维度 (t_K 级别)
            // ============================================
            scf.for %k_idx = %cst_zero to %dim_k step %tile_k_size {

              // 计算当前 t_K 块的实际大小
              %t_k_actual = affine.min #min_bound_map_inner(%k_idx)[%dim_k, %tile_k_size]

              // 计算 A2 缓冲区大小
              %a2_elems = arith.muli %tb_m_inner_actual, %t_k_actual : index
              %a2_buffer_size = arith.muli %a2_elems, %cst_four : index

              // 初始化 A2 缓冲区
              ascendc.pipe.init_buffer %pipe, %tbuf_a2, %a2_buffer_size : !ascendc.tbuf<a2>, index

              // 分配 A2 张量
              %tensor_a2 = ascendc.que_bind.alloc_tensor %queue_a2 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>

              // 计算 L0 数据加载参数
              %a2_m_blocks = arith.divui %tb_m_inner_actual, %cst_sixteen : index
              %a2_m_blocks_i16 = arith.index_cast %a2_m_blocks : index to i16
              %a2_k_blocks = arith.divui %t_k_actual, %cst_sixteen : index
              %a2_k_blocks_i64 = arith.index_cast %a2_k_blocks : index to i64
              %a2_k_blocks_i8 = arith.trunci %a2_k_blocks_i64 : i64 to i8

              // 构建 A2 加载参数
              %load_a2_params = ascendc.construct !ascendc.load_data_2d_params(%cst_zero_i16, %a2_k_blocks_i8, %a2_m_blocks_i16, %cst_zero_i16, %cst_zero_i16, %cst_false, %cst_zero_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8

              // 从 A1 加载数据到 A2
              ascendc.load_data_l0 %tensor_a2, %tensor_a1_dequeued, %load_a2_params : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_params

              // 将 A2 数据入队
              ascendc.que_bind.enque_tensor %queue_a2, %tensor_a2 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>

              // 计算 B2 缓冲区大小
              %b2_elems = arith.muli %t_k_actual, %tb_n_inner_actual : index
              %b2_buffer_size = arith.muli %b2_elems, %cst_four : index

              // 初始化 B2 缓冲区
              ascendc.pipe.init_buffer %pipe, %tbuf_b2, %b2_buffer_size : !ascendc.tbuf<b2>, index

              // 分配 B2 张量
              %tensor_b2 = ascendc.que_bind.alloc_tensor %queue_b2 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>

              // 构建 B2 加载参数 (带转置)
              %load_b2_params = ascendc.construct !ascendc.load_data_2d_transpose_params(%cst_zero_i16, %a2_k_blocks_i8, %cst_one_i16, %cst_zero_i16, %cst_zero_i16, %cst_true, %cst_zero_i8) [i16, i8, i16, i16, i16, i1, i8] : i16, i8, i16, i16, i16, i1, i8

              // 从 B1 加载数据到 B2 (带转置)
              ascendc.load_data_with_transpose %tensor_b2, %tensor_b1_dequeued, %load_b2_params : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.load_data_2d_transpose_params

              // 将 B2 数据入队
              ascendc.que_bind.enque_tensor %queue_b2, %tensor_b2 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>

              // ============================================
              // 出队 A2/B2 数据 (准备矩阵乘法)
              // ============================================
              %tensor_a2_dequeued = ascendc.que_bind.deque_tensor %queue_a2 : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              %tensor_b2_dequeued = ascendc.que_bind.deque_tensor %queue_b2 : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>

              // 构建 MMA 参数
              %mm_m = arith.index_cast %tb_m_inner_actual : index to i16
              %mm_k = arith.index_cast %t_k_actual : index to i16
              %mm_n = arith.index_cast %tb_n_inner_actual : index to i16
              %mmad_params = ascendc.construct !ascendc.mmad_params(%mm_m, %mm_k, %mm_n, %cst_zero_i8, %cst_zero_i8, %cst_zero_i8) [i16, i16, i16, i8, i8, i8] : i16, i16, i16, i8, i8, i8

              // 执行矩阵乘法：CO1 += A2 × B2
              ascendc.mmad %tensor_co1, %tensor_a2_dequeued, %tensor_b2_dequeued, %mmad_params : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.mmad_params

              // 释放 A2/B2 张量
              ascendc.que_bind.free_tensor %queue_a2, %tensor_a2_dequeued : !ascendc.queue<a2, 1>, !ascendc.local_tensor<*xf32>
              ascendc.que_bind.free_tensor %queue_b2, %tensor_b2_dequeued : !ascendc.queue<b2, 1>, !ascendc.local_tensor<*xf32>
            }

            // ============================================
            // 矩阵乘法完成，将结果从 CO1 搬运到 VECIN
            // ============================================
            ascendc.que_bind.enque_tensor %queue_co1, %tensor_co1 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>

            // 初始化 VECIN 缓冲区
            ascendc.pipe.init_buffer %pipe, %tbuf_vecin_result, %block_buffer_size : !ascendc.tbuf<vecin>, index

            // 出队 CO1 数据
            %tensor_co1_dequeued = ascendc.que_bind.deque_tensor %queue_co1 : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>

            // 分配 VECIN 张量
            %tensor_vecin_result = ascendc.que_bind.alloc_tensor %queue_vecin_2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>

            // 构建 CO1→VECIN 搬运参数
            %co12dst_params = ascendc.construct !ascendc.data_copy_co12dst_params()

            // 将数据从 CO1 搬运到 VECIN
            ascendc.data_copy_co12dst %tensor_vecin_result, %tensor_co1_dequeued, %co12dst_params : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.data_copy_co12dst_params

            // 将结果入队
            ascendc.que_bind.enque_tensor %queue_vecin_2, %tensor_vecin_result : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>

            // 释放 CO1 张量
            ascendc.que_bind.free_tensor %queue_co1, %tensor_co1_dequeued : !ascendc.queue<co1, 1>, !ascendc.local_tensor<*xf32>

            // ============================================
            // 初始化 VECCALC 缓冲区 (用于加法计算)
            // ============================================
            ascendc.pipe.init_buffer %pipe, %tbuf_veccalc, %block_buffer_size : !ascendc.tbuf<veccalc>, index

            // 出队 VECIN 数据
            %tensor_vecin_dequeued = ascendc.que_bind.deque_tensor %queue_vecin_2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>

            // 计算偏置偏移量
            %bias_offset_m = arith.muli %m_inner_idx, %tb_n_actual : index
            %bias_offset_total = arith.addi %bias_offset_m, %n_inner_idx : index
            %bias_offset_bytes = arith.muli %bias_offset_total, %cst_four : index

            // 获取对应位置的偏置数据
            %tensor_bias_slice = ascendc.tbuf.get_with_offset %tbuf_vecin_bias, %block_buffer_size, %bias_offset_bytes : !ascendc.tbuf<vecin>, index, index, !ascendc.local_tensor<*xf32>

            // 分配 VECCALC 张量
            %tensor_veccalc = ascendc.que_bind.alloc_tensor %queue_veccalc : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>

            // 执行加法：result = matmul_result + bias
            ascendc.add_l2 %tensor_veccalc, %tensor_vecin_dequeued, %tensor_bias_slice, %block_elems : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index

            // 将加法结果入队
            ascendc.que_bind.enque_tensor %queue_veccalc, %tensor_veccalc : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>

            // 释放 VECIN 张量
            ascendc.que_bind.free_tensor %queue_vecin_2, %tensor_vecin_dequeued : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>

            // ============================================
            // 准备 ReLU 激活 (max(x, 0))
            // ============================================
            // 初始化 VECIN 缓冲区 (用于零值填充)
            ascendc.pipe.init_buffer %pipe, %tbuf_vecin_zero, %block_buffer_size : !ascendc.tbuf<vecin>, index

            // 分配零值张量
            %tensor_zero = ascendc.que_bind.alloc_tensor %queue_vecin_3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>

            // 填充零值
            ascendc.duplicate_l2 %tensor_zero, %cst_zero_f32, %block_elems : !ascendc.local_tensor<*xf32>, f32, index

            // 将零值入队
            ascendc.que_bind.enque_tensor %queue_vecin_3, %tensor_zero : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>

            // 出队 VECCALC 和零值数据
            %tensor_veccalc_dequeued = ascendc.que_bind.deque_tensor %queue_veccalc : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            %tensor_zero_dequeued = ascendc.que_bind.deque_tensor %queue_vecin_3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>

            // 获取 VECOUT 缓冲区中的输出位置
            %tensor_output_slice = ascendc.tbuf.get_with_offset %tbuf_vecout, %block_buffer_size, %bias_offset_bytes : !ascendc.tbuf<vecout>, index, index, !ascendc.local_tensor<*xf32>

            // 执行 ReLU：output = max(add_result, 0)
            ascendc.max_l2 %tensor_output_slice, %tensor_veccalc_dequeued, %tensor_zero_dequeued, %block_elems : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index

            // 释放 VECCALC 和零值张量
            ascendc.que_bind.free_tensor %queue_veccalc, %tensor_veccalc_dequeued : !ascendc.queue<veccalc, 1>, !ascendc.local_tensor<*xf32>
            ascendc.que_bind.free_tensor %queue_vecin_3, %tensor_zero_dequeued : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
          }

          // ============================================
          // 将 VECOUT 结果入队 (准备写回 GM)
          // ============================================
          ascendc.que_bind.enque_tensor %queue_vecout, %tensor_vecout : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        }

        // ============================================
        // 释放 A1/B1 张量
        // ============================================
        ascendc.que_bind.free_tensor %queue_b1, %tensor_b1_dequeued : !ascendc.queue<b1, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %queue_a1, %tensor_a1_dequeued : !ascendc.queue<a1, 1>, !ascendc.local_tensor<*xf32>

        // ============================================
        // 出队 VECOUT 数据 (准备写回 GM)
        // ============================================
        %tensor_vecout_dequeued = ascendc.que_bind.deque_tensor %queue_vecout : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>

        // 创建输出全局张量
        %global_output = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
        ascendc.global_tensor.set_global_buffer %global_output, %subview_output : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 将结果从 VECOUT 写回 GM
        ascendc.data_copy_l2 %global_output, %tensor_vecout_dequeued, %bias_total_elems : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index

        // 释放 VECOUT 张量
        ascendc.que_bind.free_tensor %queue_vecout, %tensor_vecout_dequeued : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
        ascendc.que_bind.free_tensor %queue_vecin_1, %tensor_bias_dequeued : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
      }
    }

    // 函数返回
    return
  }
}
