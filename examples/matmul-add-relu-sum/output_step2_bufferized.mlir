// ============================================================
// output_step2_bufferized.mlir - 缓冲区化后的 Memref IR
//
// 本阶段将 tensor 方言转换为 memref 方言：
//   - tensor.extract_slice → memref.subview
//   - tensor.insert_slice → memref.copy
//   - 所有张量操作变为内存引用操作
//
// 计算图: output = ReLU(matmul(A, B) + bias)
//
// 关键变化：
//   - 函数参数从 tensor 变为 memref
//   - 使用 memref.subview 进行零拷贝切片
//   - 使用 memref.alloc 分配临时缓冲区
//   - 使用 memref.copy 进行数据搬运
//
// 内存层级：
//   - GM (Global Memory): 主内存，通过 memref 访问
//   - 后续阶段将引入 A1/B1/CO1 等片上内存
// ============================================================

// 动态边界计算映射
#dynamic_bound = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>

module {
  // 主函数：缓冲区化后的全连接 + ReLU
  func.func @fc_relu(
      %input_a: memref<?x?xf32, strided<[?, ?], offset: ?>>,
      %input_b: memref<?x?xf32, strided<[?, ?], offset: ?>>,
      %bias: memref<?x?xf32, strided<[?, ?], offset: ?>>,
      %output: memref<?x?xf32, strided<[?, ?], offset: ?>>,
      %tile_m_outer_i64: i64,
      %tile_n_outer_i64: i64,
      %tile_m_inner_i64: i64,
      %tile_n_inner_i64: i64,
      %tile_k_i64: i64
  ) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {

    // 常量定义
    %idx_1 = arith.constant 1 : index
    %idx_0 = arith.constant 0 : index
    %zero_f32 = arith.constant 0.000000e+00 : f32

    // 类型转换
    %tile_k = arith.index_cast %tile_k_i64 : i64 to index
    %tile_n_inner = arith.index_cast %tile_n_inner_i64 : i64 to index
    %tile_m_inner = arith.index_cast %tile_m_inner_i64 : i64 to index
    %tile_n_outer = arith.index_cast %tile_n_outer_i64 : i64 to index
    %tile_m_outer = arith.index_cast %tile_m_outer_i64 : i64 to index

    // 获取维度
    %dim_m = memref.dim %output, %idx_0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_n = memref.dim %output, %idx_1 : memref<?x?xf32, strided<[?, ?], offset: ?>>

    // 分配零张量缓冲区
    %zero_buffer = memref.alloc(%dim_m, %dim_n) {alignment = 64 : i64} : memref<?x?xf32>
    linalg.fill ins(%zero_f32 : f32) outs(%zero_buffer : memref<?x?xf32>)

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
                        : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>

        // 分配输出累加缓冲区
        %out_acc_buffer = memref.alloc(%m_outer_size, %n_outer_size) {alignment = 64 : i64} : memref<?x?xf32>
        memref.copy %subview_out, %out_acc_buffer : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>

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

            // 分配矩阵乘法结果缓冲区
            %matmul_buffer = memref.alloc(%m_inner_size, %n_inner_size) {alignment = 64 : i64} : memref<?x?xf32>
            memref.copy %inner_subview_out, %matmul_buffer : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>

            // K 维度循环
            scf.for %k = %idx_0 to %dim_k step %tile_k {
              %k_tile_size = affine.min #dynamic_bound(%k)[%dim_k, %tile_k]

              %k_subview_a = memref.subview %inner_subview_a[0, %k] [%m_inner_size, %k_tile_size] [1, 1]
                             : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %k_subview_b = memref.subview %inner_subview_b[%k, 0] [%k_tile_size, %n_inner_size] [1, 1]
                             : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %k_subview_out = memref.subview %matmul_buffer[0, 0] [%m_inner_size, %n_inner_size] [1, 1]
                               : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>

              // 矩阵乘法 (Cube 单元)
              linalg.matmul {ascendc.unit = "AiCore.Cube"}
                ins(%k_subview_a, %k_subview_b : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>)
                outs(%k_subview_out : memref<?x?xf32, strided<[?, 1]>>)
            } {ascendc.epilogue = "acc:CO1->VECIN", ascendc.prologue = "lhs:A1->A2,rhs:B1->B2"}

            // 获取偏置子视图
            %bias_inner = memref.subview %subview_bias[%m_inner, %n_inner] [%m_inner_size, %n_inner_size] [1, 1]
                          : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>

            // 分配加法结果缓冲区
            %add_buffer = memref.alloc(%m_inner_size, %n_inner_size) {alignment = 64 : i64} : memref<?x?xf32>
            linalg.elementwise kind=#linalg.elementwise_kind<add> {ascendc.unit = "AiCore.Vector"}
              ins(%matmul_buffer, %bias_inner : memref<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>)
              outs(%add_buffer : memref<?x?xf32>)

            // 获取零张量子视图
            %zero_inner = memref.subview %subview_zero[%m_inner, %n_inner] [%m_inner_size, %n_inner_size] [1, 1]
                          : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %out_inner = memref.subview %out_acc_buffer[%m_inner, %n_inner] [%m_inner_size, %n_inner_size] [1, 1]
                         : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>

            // ReLU: max(add_result, 0)
            linalg.elementwise kind=#linalg.elementwise_kind<max_signed> {ascendc.unit = "AiCore.Vector"}
              ins(%add_buffer, %zero_inner : memref<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>)
              outs(%out_inner : memref<?x?xf32, strided<[?, 1], offset: ?>>)

            // 释放临时缓冲区
            memref.dealloc %matmul_buffer : memref<?x?xf32>
            memref.dealloc %add_buffer : memref<?x?xf32>
          }
        }

        // 写回结果
        memref.copy %out_acc_buffer, %subview_out : memref<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        memref.dealloc %out_acc_buffer : memref<?x?xf32>
      } {ascendc.epilogue = "result:VECOUT->GM", ascendc.parallel = true, ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"}
    } {ascendc.parallel = true}

    // 克隆输出作为返回值
    %result = bufferization.clone %output : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    memref.dealloc %zero_buffer : memref<?x?xf32>
    return %result : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }
}
