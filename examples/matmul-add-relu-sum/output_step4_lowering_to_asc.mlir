// ============================================================
// output_step4_lowering_to_asc.mlir - AscendC 方言降级
//
// 本阶段将 linalg 算子转换为 AscendC 专用指令：
//   - linalg.matmul → ascendc.matmul
//   - linalg.elementwise(add) → ascendc.add_l2
//   - linalg.elementwise(max) → ascendc.max_l2
//
// 计算图: output = ReLU(matmul(A, B) + bias)
//
// AscendC 内存层级：
//   - GM (Global Memory): 全局内存 (memory_space = 22)
//   - A1 (1): L1 输入缓冲区 A
//   - A2 (2): L0A 输入缓冲区 A
//   - B1 (3): L1 输入缓冲区 B
//   - B2 (4): L0B 输入缓冲区 B
//   - CO1 (7): L0C 输出缓冲区
//   - VECIN (9): 向量输入缓冲区
//   - VECOUT (10): 向量输出缓冲区
//
// 执行流程：
//   1. 初始化 Pipe 和 Queue
//   2. 分配 TBUF 缓冲区
//   3. 数据搬运: GM → A1 → A2, GM → B1 → B2
//   4. Cube 计算: matmul
//   5. 数据搬运: CO1 → VECIN
//   6. Vector 计算: add + ReLU
//   7. 数据写回: VECOUT → GM
// ============================================================

// 动态边界计算映射
#dynamic_bound = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>

module {
  // 主函数：AscendC 方言实现
  func.func @fc_relu(
      %input_a: memref<?x?xf32, strided<[?, ?], offset: ?>>,      // 输入 A [M, K] - GM
      %input_b: memref<?x?xf32, strided<[?, ?], offset: ?>>,      // 输入 B [K, N] - GM
      %bias: memref<?x?xf32, strided<[?, ?], offset: ?>>,          // 偏置 [M, N] - GM
      %output: memref<?x?xf32, strided<[?, ?], offset: ?>>,       // 输出 [M, N] - GM
      %tile_m_outer_i64: i64,                                      // TB_M
      %tile_n_outer_i64: i64,                                      // TB_N
      %tile_m_inner_i64: i64,                                      // Tb_M
      %tile_n_inner_i64: i64,                                      // Tb_N
      %tile_k_i64: i64                                             // t_K
  ) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {

    // 常量定义
    %idx_1 = arith.constant 1 : index
    %idx_0 = arith.constant 0 : index
    %zero_f32 = arith.constant 0.000000e+00 : f32
    %const_4 = arith.constant 4 : index  // f32 = 4 bytes

    // 初始化 AscendC 运行时
    %pipe = ascendc.pipe
    %queue_vecin = ascendc.queue : <vecin, 1>
    %queue_vecout = ascendc.queue : <vecout, 1>

    // 类型转换
    %tile_k = arith.index_cast %tile_k_i64 : i64 to index
    %tile_n_inner = arith.index_cast %tile_n_inner_i64 : i64 to index
    %tile_m_inner = arith.index_cast %tile_m_inner_i64 : i64 to index
    %tile_n_outer = arith.index_cast %tile_n_outer_i64 : i64 to index
    %tile_m_outer = arith.index_cast %tile_m_outer_i64 : i64 to index

    // 获取维度
    %dim_m = memref.dim %output, %idx_0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_n = memref.dim %output, %idx_1 : memref<?x?xf32, strided<[?, ?], offset: ?>>

    // 在 UB 分配零张量缓冲区
    %zero_buffer = memref.alloc(%dim_m, %dim_n) {alignment = 64 : i64} : memref<?x?xf32, 10 : i32>
    linalg.fill ins(%zero_f32 : f32) outs(%zero_buffer : memref<?x?xf32, 10 : i32>)

    // 分配 TBUF 缓冲区
    %tbuf_a1 = ascendc.tbuf : <a1>
    %tbuf_a2 = ascendc.tbuf : <a2>
    %tbuf_b1 = ascendc.tbuf : <b1>
    %tbuf_b2 = ascendc.tbuf : <b2>
    %tbuf_co1 = ascendc.tbuf : <co1>
    %tbuf_vecin = ascendc.tbuf : <vecin>
    %tbuf_vecout = ascendc.tbuf : <vecout>

    // 外层循环：TB 层 (核间分块)
    scf.for %m_outer = %idx_0 to %dim_m step %tile_m_outer {
      scf.for %n_outer = %idx_0 to %dim_n step %tile_n_outer {

        // 计算实际 tile 大小
        %m_outer_size = affine.min #dynamic_bound(%m_outer)[%dim_m, %tile_m_outer]
        %n_outer_size = affine.min #dynamic_bound(%n_outer)[%dim_n, %tile_n_outer]

        %dim_k = memref.dim %input_a, %idx_1 : memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 创建子视图
        %subview_a = memref.subview %input_a[%m_outer, 0] [%m_outer_size, %dim_k] [1, 1]
                     : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_b = memref.subview %input_b[0, %n_outer] [%dim_k, %n_outer_size] [1, 1]
                     : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_out = memref.subview %output[%m_outer, %n_outer] [%m_outer_size, %n_outer_size] [1, 1]
                       : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_bias = memref.subview %bias[%m_outer, %n_outer] [%m_outer_size, %n_outer_size] [1, 1]
                        : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_zero = memref.subview %zero_buffer[%m_outer, %n_outer] [%m_outer_size, %n_outer_size] [1, 1]
                        : memref<?x?xf32, 10 : i32> to memref<?x?xf32, strided<[?, 1], offset: ?>, 10 : i32>

        // 在 VECOUT 分配输出累加缓冲区
        %out_acc_buffer = memref.alloc(%m_outer_size, %n_outer_size) {alignment = 64 : i64} : memref<?x?xf32, 10 : i32>
        memref.copy %subview_out, %out_acc_buffer : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, 10 : i32>

        // 内层循环：Tb 层 (核内分块)
        scf.for %m_inner = %idx_0 to %m_outer_size step %tile_m_inner {
          scf.for %n_inner = %idx_0 to %n_outer_size step %tile_n_inner {

            %m_inner_size = affine.min #dynamic_bound(%m_inner)[%m_outer_size, %tile_m_inner]
            %n_inner_size = affine.min #dynamic_bound(%n_inner)[%n_outer_size, %tile_n_inner]

            // 创建内层子视图
            %inner_subview_a = memref.subview %subview_a[%m_inner, 0] [%m_inner_size, %dim_k] [1, 1]
                               : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            %inner_subview_b = memref.subview %subview_b[0, %n_inner] [%dim_k, %n_inner_size] [1, 1]
                               : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            %inner_subview_out = memref.subview %subview_out[%m_inner, %n_inner] [%m_inner_size, %n_inner_size] [1, 1]
                                 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>

            // 在 L0C 分配矩阵乘法结果缓冲区
            %matmul_buffer = memref.alloc(%m_inner_size, %n_inner_size) {alignment = 64 : i64} : memref<?x?xf32, 7 : i32>
            memref.copy %inner_subview_out, %matmul_buffer : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, 7 : i32>

            // K 维度循环
            scf.for %k = %idx_0 to %dim_k step %tile_k {
              %k_tile_size = affine.min #dynamic_bound(%k)[%dim_k, %tile_k]

              %k_subview_a = memref.subview %inner_subview_a[0, %k] [%m_inner_size, %k_tile_size] [1, 1]
                             : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %k_subview_b = memref.subview %inner_subview_b[%k, 0] [%k_tile_size, %n_inner_size] [1, 1]
                             : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %k_subview_out = memref.subview %matmul_buffer[0, 0] [%m_inner_size, %n_inner_size] [1, 1]
                               : memref<?x?xf32, 7 : i32> to memref<?x?xf32, strided<[?, 1]>, 7 : i32>

              // 计算缓冲区大小
              %a_buffer_size = arith.muli %m_inner_size, %k_tile_size : index
              %a_buffer_bytes = arith.muli %a_buffer_size, %const_4 : index
              %b_buffer_size = arith.muli %k_tile_size, %n_inner_size : index
              %b_buffer_bytes = arith.muli %b_buffer_size, %const_4 : index

              // 初始化 TBUF 缓冲区
              ascendc.pipe.init_buffer %pipe, %tbuf_a1, %a_buffer_bytes : !ascendc.tbuf<a1>, index
              ascendc.pipe.init_buffer %pipe, %tbuf_b1, %b_buffer_bytes : !ascendc.tbuf<b1>, index
              ascendc.pipe.init_buffer %pipe, %tbuf_a2, %a_buffer_bytes : !ascendc.tbuf<a2>, index
              ascendc.pipe.init_buffer %pipe, %tbuf_b2, %b_buffer_bytes : !ascendc.tbuf<b2>, index

              // 分配本地张量
              %local_a1 = ascendc.tbuf.get_tensor %tbuf_a1 : !ascendc.tbuf<a1>, !ascendc.local_tensor<*xf32>
              %local_b1 = ascendc.tbuf.get_tensor %tbuf_b1 : !ascendc.tbuf<b1>, !ascendc.local_tensor<*xf32>
              %local_a2 = ascendc.tbuf.get_tensor %tbuf_a2 : !ascendc.tbuf<a2>, !ascendc.local_tensor<*xf32>
              %local_b2 = ascendc.tbuf.get_tensor %tbuf_b2 : !ascendc.tbuf<b2>, !ascendc.local_tensor<*xf32>

              // 设置全局张量
              %global_a = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
              %global_b = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
              ascendc.global_tensor.set_global_buffer %global_a, %k_subview_a : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>
              ascendc.global_tensor.set_global_buffer %global_b, %k_subview_b : !ascendc.global_tensor<*xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>

              // 数据搬运: GM → A1/B1
              ascendc.data_copy_l2 %local_a1, %global_a, %a_buffer_size : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index
              ascendc.data_copy_l2 %local_b1, %global_b, %b_buffer_size : !ascendc.local_tensor<*xf32>, !ascendc.global_tensor<*xf32>, index

              // 数据搬运: A1 → A2, B1 → B2
              ascendc.data_copy_l2 %local_a2, %local_a1, %a_buffer_size : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
              ascendc.data_copy_l2 %local_b2, %local_b1, %b_buffer_size : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index

              // 矩阵乘法 (Cube)
              ascendc.matmul %local_a2, %local_b2, %k_subview_out : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, memref<?x?xf32, strided<[?, 1]>, 7 : i32>
            } {ascendc.epilogue = "acc:CO1->VECIN", ascendc.prologue = "lhs:A1->A2,rhs:B1->B2"}

            // 获取偏置子视图
            %bias_inner = memref.subview %subview_bias[%m_inner, %n_inner] [%m_inner_size, %n_inner_size] [1, 1]
                          : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>

            // 在 VECIN 分配偏置缓冲区
            %bias_buffer = memref.alloc(%m_inner_size, %n_inner_size) {alignment = 64 : i64} : memref<?x?xf32, 9 : i32>
            memref.copy %bias_inner, %bias_buffer : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, 9 : i32>

            // 在 VECOUT 分配加法结果缓冲区
            %add_buffer = memref.alloc(%m_inner_size, %n_inner_size) {alignment = 64 : i64} : memref<?x?xf32, 10 : i32>

            // 计算元素数量
            %elem_count = arith.muli %m_inner_size, %n_inner_size : index

            // 加法: matmul + bias (Vector)
            ascendc.add_l2 %add_buffer, %matmul_buffer, %bias_buffer, %elem_count
                : memref<?x?xf32, 10 : i32>, memref<?x?xf32, 7 : i32>, memref<?x?xf32, 9 : i32>, index

            // 获取零张量子视图
            %zero_inner = memref.subview %subview_zero[%m_inner, %n_inner] [%m_inner_size, %n_inner_size] [1, 1]
                          : memref<?x?xf32, strided<[?, 1], offset: ?>, 10 : i32> to memref<?x?xf32, strided<[?, 1], offset: ?>, 10 : i32>
            %out_inner = memref.subview %out_acc_buffer[%m_inner, %n_inner] [%m_inner_size, %n_inner_size] [1, 1]
                         : memref<?x?xf32, 10 : i32> to memref<?x?xf32, strided<[?, 1], offset: ?>, 10 : i32>

            // ReLU: max(add_result, 0) (Vector)
            ascendc.max_l2 %out_inner, %add_buffer, %zero_inner, %elem_count
                : memref<?x?xf32, strided<[?, 1], offset: ?>, 10 : i32>, memref<?x?xf32, 10 : i32>, memref<?x?xf32, strided<[?, 1], offset: ?>, 10 : i32>, index

            // 释放临时缓冲区
            memref.dealloc %matmul_buffer : memref<?x?xf32, 7 : i32>
            memref.dealloc %bias_buffer : memref<?x?xf32, 9 : i32>
            memref.dealloc %add_buffer : memref<?x?xf32, 10 : i32>
          }
        }

        // 写回结果: VECOUT → GM
        memref.copy %out_acc_buffer, %subview_out : memref<?x?xf32, 10 : i32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        memref.dealloc %out_acc_buffer : memref<?x?xf32, 10 : i32>
      } {ascendc.epilogue = "result:VECOUT->GM", ascendc.parallel = true, ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"}
    } {ascendc.parallel = true}

    // 克隆输出作为返回值
    %result = bufferization.clone %output : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    memref.dealloc %zero_buffer : memref<?x?xf32, 10 : i32>
    return %result : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }
}
