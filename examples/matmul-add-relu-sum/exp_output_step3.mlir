// 用于计算tile大小的仿射映射，处理边界情况
// 计算 min(-d0 + s0, s1) = min(剩余元素数量, tile大小)
#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>

module {
  // FC+ReLU融合算子: output = max(input_a @ input_b + bias, 0)
  // 这是buffer placement后的版本，优化了内存分配和拷贝
  func.func @fc_relu(
    %arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>,  // 输入矩阵A (M x K)
    %arg1: memref<?x?xf32, strided<[?, ?], offset: ?>>,  // 权重矩阵B (K x N)
    %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>,  // 偏置矩阵 (M x N)
    %arg3: memref<?x?xf32, strided<[?, ?], offset: ?>>,  // 输出矩阵 (M x N)
    %arg4: i64,  // block大小 (外层分块)
    %arg5: i64,  // tile大小 (内层分块)
    %arg6: i64,  // K维度的tile大小
    %arg7: i64,  // tile大小
    %arg8: i64   // K维度的block大小
  ) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32  // ReLU用的零值

    // 将参数从i64转换为index类型
    %0 = arith.index_cast %arg8 : i64 to index
    %1 = arith.index_cast %arg7 : i64 to index
    %2 = arith.index_cast %arg6 : i64 to index
    %3 = arith.index_cast %arg5 : i64 to index
    %4 = arith.index_cast %arg4 : i64 to index

    // 获取输出矩阵的维度
    %dim = memref.dim %arg3, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>

    // 外层循环: 沿M和N维度对输出矩阵进行分块
    scf.for %arg9 = %c0 to %dim step %4 {
      scf.for %arg10 = %c0 to %dim_0 step %3 {
        // 计算实际的tile大小 (处理边界情况)
        %6 = affine.min #map(%arg9)[%dim, %4]
        %7 = affine.min #map(%arg10)[%dim_0, %3]

        // 获取K维度 (归约维度)
        %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 为输入A创建子视图
        %subview = memref.subview %arg0[%arg9, 0] [%6, %dim_1] [1, 1]
          : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 在L1缓冲区(A1 - memory space 1)分配空间并拷贝输入A的tile
        %alloc = memref.alloc(%6, %dim_1) : memref<?x?xf32, 1 : i32>
        memref.copy %subview, %alloc : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, 1 : i32>

        // 为权重B创建子视图
        %subview_2 = memref.subview %arg1[0, %arg10] [%dim_1, %7] [1, 1]
          : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 在L1缓冲区(B1 - memory space 3)分配空间并拷贝权重B的tile
        %alloc_3 = memref.alloc(%dim_1, %7) : memref<?x?xf32, 3 : i32>
        memref.copy %subview_2, %alloc_3 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, 3 : i32>

        // 为输出创建子视图
        %subview_4 = memref.subview %arg3[%arg9, %arg10] [%6, %7] [1, 1]
          : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 为偏置创建子视图
        %subview_5 = memref.subview %arg2[%arg9, %arg10] [%6, %7] [1, 1]
          : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 在向量输入缓冲区(VECIN - memory space 9)分配空间并拷贝偏置
        %alloc_6 = memref.alloc(%6, %7) : memref<?x?xf32, 9 : i32>
        memref.copy %subview_5, %alloc_6 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, 9 : i32>

        // 在向量输出缓冲区(VECOUT - memory space 10)分配空间并初始化为0
        // 用于存储最终的ReLU输出结果
        %alloc_7 = memref.alloc(%6, %7) : memref<?x?xf32, 10 : i32>
        linalg.fill ins(%cst : f32) outs(%alloc_7 : memref<?x?xf32, 10 : i32>)

        // 内层循环: 进一步对计算进行分块
        scf.for %arg11 = %c0 to %6 step %2 {
          scf.for %arg12 = %c0 to %7 step %1 {
            // 计算实际的内层tile大小
            %8 = affine.min #map(%arg11)[%6, %2]
            %9 = affine.min #map(%arg12)[%7, %1]

            // 在立方体输出缓冲区(CO1 - memory space 7)分配矩阵乘法结果
            // 初始化为0，用于累加矩阵乘法的部分结果
            %alloc_8 = memref.alloc(%8, %9) : memref<?x?xf32, 7 : i32>
            linalg.fill ins(%cst : f32) outs(%alloc_8 : memref<?x?xf32, 7 : i32>)

            // K维度的归约循环，用于矩阵乘法
            scf.for %arg13 = %c0 to %dim_1 step %0 {
              // 计算实际的K tile大小
              %10 = affine.min #map(%arg13)[%dim_1, %0]

              // 从L1缓冲区获取输入A的子视图
              %subview_14 = memref.subview %alloc[0, %arg13] [%8, %10] [1, 1]
                : memref<?x?xf32, 1 : i32> to memref<?x?xf32, strided<[?, 1], offset: ?>, 1 : i32>

              // 在L0缓冲区(A2 - memory space 2)分配空间并拷贝输入A的K tile
              %alloc_15 = memref.alloc(%8, %10) : memref<?x?xf32, 2 : i32>
              memref.copy %subview_14, %alloc_15 : memref<?x?xf32, strided<[?, 1], offset: ?>, 1 : i32> to memref<?x?xf32, 2 : i32>

              // 从L1缓冲区获取权重B的子视图
              %subview_16 = memref.subview %alloc_3[%arg13, 0] [%10, %9] [1, 1]
                : memref<?x?xf32, 3 : i32> to memref<?x?xf32, strided<[?, 1], offset: ?>, 3 : i32>

              // 在L0缓冲区(B2 - memory space 4)分配空间并拷贝权重B的K tile
              %alloc_17 = memref.alloc(%10, %9) : memref<?x?xf32, 4 : i32>
              memref.copy %subview_16, %alloc_17 : memref<?x?xf32, strided<[?, 1], offset: ?>, 3 : i32> to memref<?x?xf32, 4 : i32>

              // 获取矩阵乘法结果的子视图，用于累加
              %subview_18 = memref.subview %alloc_8[0, 0] [%8, %9] [1, 1]
                : memref<?x?xf32, 7 : i32> to memref<?x?xf32, strided<[?, 1]>, 7 : i32>

              // 矩阵乘法: C += A @ B
              // 在CO1缓冲区累加部分结果
              linalg.matmul
                ins(%alloc_15, %alloc_17 : memref<?x?xf32, 2 : i32>, memref<?x?xf32, 4 : i32>)
                outs(%subview_18 : memref<?x?xf32, strided<[?, 1]>, 7 : i32>)

              // 释放L0缓冲区
              memref.dealloc %alloc_15 : memref<?x?xf32, 2 : i32>
              memref.dealloc %alloc_17 : memref<?x?xf32, 4 : i32>
            }

            // 将矩阵乘法结果从CO1拷贝到向量输入缓冲区(VECIN - memory space 9)
            // 准备进行向量运算（加法和ReLU）
            %alloc_9 = memref.alloc(%8, %9) : memref<?x?xf32, 9 : i32>
            memref.copy %alloc_8, %alloc_9 : memref<?x?xf32, 7 : i32> to memref<?x?xf32, 9 : i32>

            // 在向量计算缓冲区(VECCALC - memory space 11)分配加法结果
            %alloc_10 = memref.alloc(%8, %9) : memref<?x?xf32, 11 : i32>

            // 获取偏置的子视图
            %subview_11 = memref.subview %alloc_6[%arg11, %arg12] [%8, %9] [1, 1]
              : memref<?x?xf32, 9 : i32> to memref<?x?xf32, strided<[?, 1], offset: ?>, 9 : i32>

            // 逐元素加法: result = matmul_result + bias
            linalg.elementwise kind=#linalg.elementwise_kind<add>
              ins(%alloc_9, %subview_11 : memref<?x?xf32, 9 : i32>, memref<?x?xf32, strided<[?, 1], offset: ?>, 9 : i32>)
              outs(%alloc_10 : memref<?x?xf32, 11 : i32>)

            // 获取输出缓冲区的子视图
            %subview_12 = memref.subview %alloc_7[%arg11, %arg12] [%8, %9] [1, 1]
              : memref<?x?xf32, 10 : i32> to memref<?x?xf32, strided<[?, 1], offset: ?>, 10 : i32>

            // 在向量输入缓冲区(VECIN - memory space 9)分配零值缓冲区
            // 用于ReLU操作: max(x, 0)
            %alloc_13 = memref.alloc(%8, %9) : memref<?x?xf32, 9 : i32>
            linalg.fill ins(%cst : f32) outs(%alloc_13 : memref<?x?xf32, 9 : i32>)

            // ReLU激活: output = max(add_result, 0)
            // 使用max_signed实现ReLU: max(x, 0)
            linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
              ins(%alloc_10, %alloc_13 : memref<?x?xf32, 11 : i32>, memref<?x?xf32, 9 : i32>)
              outs(%subview_12 : memref<?x?xf32, strided<[?, 1], offset: ?>, 10 : i32>)

            // 释放临时缓冲区
            memref.dealloc %alloc_13 : memref<?x?xf32, 9 : i32>
            memref.dealloc %alloc_10 : memref<?x?xf32, 11 : i32>
            memref.dealloc %alloc_9 : memref<?x?xf32, 9 : i32>
            memref.dealloc %alloc_8 : memref<?x?xf32, 7 : i32>
          }
        }

        // 将最终结果从VECOUT拷贝回输出矩阵
        memref.copy %alloc_7, %subview_4 : memref<?x?xf32, 10 : i32> to memref<?x?xf32, strided<[?, ?], offset: ?>>

        // 释放缓冲区
        memref.dealloc %alloc_7 : memref<?x?xf32, 10 : i32>
        memref.dealloc %alloc : memref<?x?xf32, 1 : i32>
        memref.dealloc %alloc_3 : memref<?x?xf32, 3 : i32>
        memref.dealloc %alloc_6 : memref<?x?xf32, 9 : i32>
      }
    }

    // 克隆输出作为返回值
    %5 = bufferization.clone %arg3 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    return %5 : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }
}
