#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0)[s0, s1] -> (-(d0 * (s0 ceildiv s1)) + s0, s0 ceildiv s1)>
#map2 = affine_map<(d0) -> (0, d0)>
#map3 = affine_map<(d0)[s0, s1] -> (d0 * (s0 ceildiv s1))>

module {
  func.func @fc_relu(
      %arg0: tensor<?x?xf32>,  // A [M, K]
      %arg1: tensor<?x?xf32>,  // B [K, N]
      %arg2: tensor<?x?xf32>,  // bias [M, N]
      %arg3: tensor<?x?xf32>,  // output tensor [M, N], 初始输入
      %arg4: i64,               // t_K  (CUBE 单次 K 步长)
      %arg5: i64,               // TB_N (核间 N 分块)
      %arg6: i64,               // TB_M (核间 M 分块)
      %arg7: i64,               // Tb_M (核内 M 分块)
      %arg8: i64                // Tb_N (核内 N 分块)
  ) -> tensor<?x?xf32> {

    // =======================
    // 常量定义
    // =======================
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32

    // =======================
    // tile size: i64 -> index
    // =======================
    %0 = arith.index_cast %arg8 : i64 to index   // Tb_N
    %1 = arith.index_cast %arg7 : i64 to index   // Tb_M
    %2 = arith.index_cast %arg6 : i64 to index   // TB_M
    %3 = arith.index_cast %arg5 : i64 to index   // TB_N
    %4 = arith.index_cast %arg4 : i64 to index   // t_K

    // =======================
    // 矩阵维度
    // =======================
    %dim = tensor.dim %arg3, %c0 : tensor<?x?xf32>    // output M
    %dim_0 = tensor.dim %arg3, %c1 : tensor<?x?xf32>  // output N

    // =======================
    // 初始化全零 tensor，用于 ReLU
    // =======================
    %5 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>

    // ============================================================
    // TB 层: 按 K 步长循环 (t 层) 外层循环
    //   step = t_K
    //   %arg9, %arg11 对应 TB tile 的起始 M / N 坐标
    // ============================================================
    %7 = scf.for %arg9 = %c0 to %dim step %4 iter_args(%arg10 = %arg3) -> (tensor<?x?xf32>) {
      %8 = scf.for %arg11 = %c0 to %dim_0 step %3 iter_args(%arg12 = %arg10) -> (tensor<?x?xf32>) {

        // ----------------------------
        // TB tile 实际大小裁剪 (尾块处理)
        // ----------------------------
        %9 = affine.min #map(%arg9)[%dim, %4]     // 当前 TB_M tile size
        %10 = affine.min #map(%arg11)[%dim_0, %3] // 当前 TB_N tile size

        // ----------------------------
        // 提取 A/B/Output slice，用于 Tb 层
        // ----------------------------
        %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32> // K 维度
        %extracted_slice   = tensor.extract_slice %arg0[%arg9, 0] [%9, %dim_1] [1,1]  : tensor<?x?xf32> to tensor<?x?xf32>  // A_TB
        %extracted_slice_2 = tensor.extract_slice %arg1[0, %arg11] [%dim_1, %10] [1,1] : tensor<?x?xf32> to tensor<?x?xf32>  // B_TB
        %extracted_slice_3 = tensor.extract_slice %arg12[%arg9, %arg11] [%9, %10] [1,1] : tensor<?x?xf32> to tensor<?x?xf32>  // C_TB (output)

        // ============================================================
        // Tb 层 M 方向: 单核内循环
        // ============================================================
        %11 = scf.for %arg13 = %c0 to %dim_1 step %2 iter_args(%arg14 = %extracted_slice_3) -> (tensor<?x?xf32>) {
          %14 = affine.min #map(%arg13)[%dim_1, %2]

          // 提取 Tb_M 内 A/B/Output slice
          %extracted_slice_6 = tensor.extract_slice %extracted_slice[0, %arg13] [%9, %14] [1,1] : tensor<?x?xf32> to tensor<?x?xf32>
          %extracted_slice_7 = tensor.extract_slice %extracted_slice_2[%arg13, 0] [%14, %10] [1,1] : tensor<?x?xf32> to tensor<?x?xf32>
          %extracted_slice_8 = tensor.extract_slice %arg14[0,0] [%9, %10] [1,1] : tensor<?x?xf32> to tensor<?x?xf32>

          // Tb 层 K 方向 matmul
          %15 = linalg.matmul ins(%extracted_slice_6, %extracted_slice_7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_8 : tensor<?x?xf32>) -> tensor<?x?xf32>

          // 写回 Tb_M 累加 tensor
          %inserted_slice_9 = tensor.insert_slice %15 into %arg14[0,0] [%9, %10] [1,1] : tensor<?x?xf32> into tensor<?x?xf32>
          scf.yield %inserted_slice_9 : tensor<?x?xf32>
        }

        // ----------------------------
        // Add bias
        // ----------------------------
        %extracted_slice_4 = tensor.extract_slice %arg2[%arg9, %arg11] [%9, %10] [1,1] : tensor<?x?xf32> to tensor<?x?xf32>
        %12 = linalg.elementwise kind=#linalg.elementwise_kind<add> ins(%11, %extracted_slice_4 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_3 : tensor<?x?xf32>) -> tensor<?x?xf32>

        // ----------------------------
        // ReLU: max(add, 0)
        // ----------------------------
        %extracted_slice_5 = tensor.extract_slice %6[%arg9, %arg11] [%9, %10] [1,1] : tensor<?x?xf32> to tensor<?x?xf32>
        %13 = scf.forall (%arg13, %arg14) in (%1, %0) shared_outs(%arg15 = %extracted_slice_3) -> (tensor<?x?xf32>) {
          %14 = affine.min #map1(%arg13)[%9, %1]
          %15 = affine.max #map2(%14)
          %16 = affine.min #map1(%arg14)[%10, %0]
          %17 = affine.max #map2(%16)
          %18 = affine.apply #map3(%arg13)[%9, %1]
          %19 = affine.apply #map3(%arg14)[%10, %0]

          %extracted_slice_6 = tensor.extract_slice %12[%18, %19] [%15, %17] [1,1] : tensor<?x?xf32> to tensor<?x?xf32>
          %extracted_slice_7 = tensor.extract_slice %extracted_slice_5[%18, %19] [%15, %17] [1,1] : tensor<?x?xf32> to tensor<?x?xf32>
          %extracted_slice_8 = tensor.extract_slice %arg15[%18, %19] [%15, %17] [1,1] : tensor<?x?xf32> to tensor<?x?xf32>

          %20 = linalg.elementwise kind=#linalg.elementwise_kind<max_signed> ins(%extracted_slice_6, %extracted_slice_7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_8 : tensor<?x?xf32>) -> tensor<?x?xf32>

          // TB 层并行写回 (多核无冲突)
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %20 into %arg15[%18, %19] [%15, %17] [1,1] : tensor<?x?xf32> into tensor<?x?xf32>
          }
        } {mapping = [#gpu.block<y>, #gpu.block<x>]}  // TB 层映射到 GPU block
        %inserted_slice = tensor.insert_slice %13 into %arg12[%arg9, %arg11] [%9, %10] [1,1] : tensor<?x?xf32> into tensor<?x?xf32>

        scf.yield %inserted_slice : tensor<?x?xf32>
      }
      scf.yield %8 : tensor<?x?xf32>
    }

    return %7 : tensor<?x?xf32>
  }
}