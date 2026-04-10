// 计算剩余大小：min(剩余长度, 切分大小)
#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>

module {
  func.func @fc_relu(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: i64, %arg5: i64, %arg6: i64) -> tensor<?x?xf32> {
    //arg0: 左矩阵 (M×K)
    //arg1: 右矩阵 (K×N)
    //arg2: 偏置矩阵 (M×N)
    //arg3: 输出矩阵 (M×N)
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    // arg4、arg5、arg6分别表示M(%2)、N(%1)、K(%0)轴的切分大小
    %0 = arith.index_cast %arg6 : i64 to index
    %1 = arith.index_cast %arg5 : i64 to index
    %2 = arith.index_cast %arg4 : i64 to index

    // 构造全0矩阵，给ReLU使用，等价于max(x, 0)
    %dim = tensor.dim %arg3, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg3, %c1 : tensor<?x?xf32>
    %3 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>

    %dim_1 = tensor.dim %arg3, %c0 : tensor<?x?xf32>
    %dim_2 = tensor.dim %arg3, %c1 : tensor<?x?xf32>
    // 外层循环切M轴，内层循环切N轴，每个轴切分大小是step=%2和%1
    %5 = scf.for %arg7 = %c0 to %dim_1 step %2 iter_args(%arg8 = %arg3) -> (tensor<?x?xf32>) {
      %6 = scf.for %arg9 = %c0 to %dim_2 step %1 iter_args(%arg10 = %arg8) -> (tensor<?x?xf32>) {
        // 计算本次循环中，M轴和N轴的切分大小，确保不超过Tensor边界
        %7 = affine.min #map(%arg7)[%dim_1, %2]  // ==> affine.min (-%arg7 + %dim_1, %2)
        %8 = affine.min #map(%arg9)[%dim_2, %1]  // ==> affine.min (-%arg9 + %dim_2, %1)

        // K轴大小
        %dim_3 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
        // 从左矩阵的[%arg7, 0]开始，提取一个大小为[%7, %dim_3]的切片， Tile大小[%7, K]
        %extracted_slice = tensor.extract_slice %arg0[%arg7, 0] [%7, %dim_3] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        // 从右矩阵的[0, %arg9]开始，提取一个大小为[%dim_3, %8]的切片， Tile大小[K, %8]
        %extracted_slice_4 = tensor.extract_slice %arg1[0, %arg9] [%dim_3, %8] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        // 从输出矩阵的[%arg7, %arg9]开始，提取一个大小为[%7, %8]的切片， Tile大小[%7, %8]
        %extracted_slice_5 = tensor.extract_slice %arg10[%arg7, %arg9] [%7, %8] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

        // ====== Step1：执行Matmul，输入1=左矩阵，输入2=右矩阵，切K轴 ======
        // 切K轴，每个K轴切分大小是step=%0；另外：iter_args是迭代参数传递当前累积结果
        %9 = scf.for %arg11 = %c0 to %dim_3 step %0 iter_args(%arg12 = %extracted_slice_5) -> (tensor<?x?xf32>) {
          // 计算本次循环中，K轴的切分大小，确保不超过Tensor边界
          %12 = affine.min #map(%arg11)[%dim_3, %0]

          // 从左矩阵的[0, %arg11]开始，提取一个Tile大小为[%7, %12]的切片
          %extracted_slice_10 = tensor.extract_slice %extracted_slice[0, %arg11] [%7, %12] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          // 从右矩阵的[%arg11, 0]开始，提取一个Tile大小为[%12, %8]的切片
          %extracted_slice_11 = tensor.extract_slice %extracted_slice_4[%arg11, 0] [%12, %8] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          // 从输出矩阵的[0, 0]开始，提取当前累积结果，大小为[%7, %8]
          %extracted_slice_12 = tensor.extract_slice %arg12[0, 0] [%7, %8] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          // 执行linalg.matmul，是一个乘加操作，将结果累加到输出矩阵的[0, 0]位置
          %13 = linalg.matmul ins(%extracted_slice_10, %extracted_slice_11 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_12 : tensor<?x?xf32>) -> tensor<?x?xf32>
          // 将矩阵乘法结果插入到累积结果中，为下一次迭代准备，Tile大小[%7, %8]
          %inserted_slice_13 = tensor.insert_slice %13 into %arg12[0, 0] [%7, %8] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
          scf.yield %inserted_slice_13 : tensor<?x?xf32>
        }

        // ====== Step2：执行Add，输入1=Matmul的结果，输入2=Bias矩阵 ======
        // 从偏置矩阵的[%arg7, %arg9]开始，提取一个Tile大小为[%7, %8]的切片
        %extracted_slice_6 = tensor.extract_slice %arg2[%arg7, %arg9] [%7, %8] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        // 从输出矩阵的[%arg7, %arg9]开始，提取一个Tile大小为[%7, %8]的切片
        %extracted_slice_7 = tensor.extract_slice %arg10[%arg7, %arg9] [%7, %8] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        // 执行加法，将结果写入输出矩阵的[%arg7, %arg9]位置，Tile大小[%7, %8]
        %10 = linalg.elementwise kind=#linalg.elementwise_kind<add> ins(%9, %extracted_slice_6 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_7 : tensor<?x?xf32>) -> tensor<?x?xf32>

        // ====== Step3：执行ReLU(max)，输入1=Add的结果，输入2=全0矩阵 ======
        // 从全0矩阵的[%arg7, %arg9]开始，提取一个Tile大小为[%7, %8]的切片
        %extracted_slice_8 = tensor.extract_slice %4[%arg7, %arg9] [%7, %8] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        // 从输出矩阵的[%arg7, %arg9]开始，提取一个Tile大小为[%7, %8]的切片
        %extracted_slice_9 = tensor.extract_slice %arg10[%arg7, %arg9] [%7, %8] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        // 执行ReLU(max)，将结果写入输出矩阵的[%arg7, %arg9]位置，Tile大小[%7, %8]
        %11 = linalg.elementwise kind=#linalg.elementwise_kind<max_signed> ins(%10, %extracted_slice_8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_9 : tensor<?x?xf32>) -> tensor<?x?xf32>
        // 将ReLU(max)结果写入输出矩阵的[%arg7, %arg9]位置，Tile大小[%7, %8]
        %inserted_slice = tensor.insert_slice %11 into %arg10[%arg7, %arg9] [%7, %8] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

        scf.yield %inserted_slice : tensor<?x?xf32>
      }
      scf.yield %6 : tensor<?x?xf32>
    }
    return %5 : tensor<?x?xf32>
  }
}

