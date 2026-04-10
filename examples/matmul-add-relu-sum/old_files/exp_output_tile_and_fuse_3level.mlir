#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)> // tile boundary min guard

module {
  func.func @fc_relu(
    %A: tensor<?x?xf32>,         // 输入矩阵 A  (M x K)
    %B: tensor<?x?xf32>,         // 输入矩阵 B  (K x N)
    %Bias: tensor<?x?xf32>,      // Bias (M x N)
    %OutputInit: tensor<?x?xf32>, // 初始输出 buffer
    %tileM_outer: i64,           // 外层 M tile
    %tileN_outer: i64,           // 外层 N tile
    %tileM_inner: i64,           // 内层 M tile
    %tileN_inner: i64,           // 内层 N tile
    %tileK: i64                  // K 方向 tile
  ) -> tensor<?x?xf32> {

    // ====================================================
    // 常量
    // ====================================================
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32

    // ====================================================
    // index cast
    // ====================================================
    %TK = arith.index_cast %tileK : i64 to index
    %TN_inner = arith.index_cast %tileN_inner : i64 to index
    %TM_inner = arith.index_cast %tileM_inner : i64 to index
    %TN_outer = arith.index_cast %tileN_outer : i64 to index
    %TM_outer = arith.index_cast %tileM_outer : i64 to index

    // ====================================================
    // 输出矩阵尺寸
    // ====================================================
    %M = tensor.dim %OutputInit, %c0
    %N = tensor.dim %OutputInit, %c1

    // ====================================================
    // 创建零张量（用于 ReLU）
    // ====================================================
    %ZeroTensor = tensor.empty(%M, %N) : tensor<?x?xf32>
    %ZeroTensorFilled = linalg.fill ins(%zero : f32) outs(%ZeroTensor : tensor<?x?xf32>) -> tensor<?x?xf32>

    // ====================================================
    // 多面体调度声明（Polyhedral Scheduling）
    // Iteration domain:
    //   D = { (i,j,k) | 0 ≤ i < M ∧ 0 ≤ j < N ∧ 0 ≤ k < K }
    // Schedule function:
    //   θ(i,j,k) = (floordiv(i, TM_outer), floordiv(j, TN_outer),
    //               floordiv(i, TM_inner), floordiv(j, TN_inner), k)
    // affine.min 保证 tile 边界合法
    // ====================================================
    %Result =
    scf.for %m_outer = %c0 to %M step %TM_outer iter_args(%OutAcc0 = %OutputInit) -> (tensor<?x?xf32>) {
        scf.for %n_outer = %c0 to %N step %TN_outer iter_args(%OutAcc1 = %OutAcc0) -> (tensor<?x?xf32>) {

          // 当前 outer tile 实际大小（边界处理）
          %m_outer_size = affine.min #map(%m_outer)[%M, %TM_outer]
          %n_outer_size = affine.min #map(%n_outer)[%N, %TN_outer]

          // K 维长度
          %K = tensor.dim %A, %c1

          // ====================================================
          // Tile-based kernel 映射 (Outer Tile → L2)
          // ====================================================
          %A_outer = tensor.extract_slice %A[%m_outer, 0][%m_outer_size, %K][1, 1]
          %B_outer = tensor.extract_slice %B[0, %n_outer][%K, %n_outer_size][1, 1]
          %Out_outer = tensor.extract_slice %OutAcc1[%m_outer, %n_outer][%m_outer_size, %n_outer_size][1, 1]
          %Bias_outer = tensor.extract_slice %Bias[%m_outer, %n_outer][%m_outer_size, %n_outer_size][1, 1]
          %Zero_outer = tensor.extract_slice %ZeroTensorFilled[%m_outer, %n_outer][%m_outer_size, %n_outer_size][1, 1]

          // ====================================================
          // 内层 Tile (M_inner × N_inner) → UB/Shared Memory Blocking
          // ====================================================
          %Updated_outer =
          scf.for %m_inner = %c0 to %m_outer_size step %TM_inner iter_args(%OutAcc2 = %Out_outer) -> (tensor<?x?xf32>) {
              scf.for %n_inner = %c0 to %n_outer_size step %TN_inner iter_args(%OutAcc3 = %OutAcc2) -> (tensor<?x?xf32>) {
                %m_inner_size = affine.min #map(%m_inner)[%m_outer_size, %TM_inner]
                %n_inner_size = affine.min #map(%n_inner)[%n_outer_size, %TN_inner]

                %A_inner = tensor.extract_slice %A_outer[%m_inner, 0][%m_inner_size, %K][1, 1]
                %B_inner = tensor.extract_slice %B_outer[0, %n_inner][%K, %n_inner_size][1, 1]
                %Out_inner = tensor.extract_slice %OutAcc3[%m_inner, %n_inner][%m_inner_size, %n_inner_size][1, 1]

                // ====================================================
                // K 方向分块 → Register Blocking
                // capacity constraints:
                //   TM_inner * TK <= UB_capacity_A
                //   TK * TN_inner <= UB_capacity_B
                //   TM_inner * TN_inner <= UB_capacity_C
                // ====================================================
                %MatmulResult =
                  scf.for %k = %c0 to %K step %TK iter_args(%OutAcc4 = %Out_inner) -> (tensor<?x?xf32>) {
                    %k_size = affine.min #map(%k)[%K, %TK]
                    %A_k = tensor.extract_slice %A_inner[0, %k][%m_inner_size, %k_size][1, 1]
                    %B_k = tensor.extract_slice %B_inner[%k, 0][%k_size, %n_inner_size][1, 1]
                    %PartialOut = tensor.extract_slice %OutAcc4[0, 0][%m_inner_size, %n_inner_size][1, 1]

                    // ====================================================
                    // Stage 1: GEMM
                    // ====================================================
                    %MatmulTile = linalg.matmul ins(%A_k, %B_k) outs(%PartialOut)

                    %OutNext = tensor.insert_slice %MatmulTile into %OutAcc4[0, 0][%m_inner_size, %n_inner_size][1, 1]
                    scf.yield %OutNext
                  }

                // ====================================================
                // Stage 2: Bias Add
                // ====================================================
                %Bias_inner = tensor.extract_slice %Bias_outer[%m_inner, %n_inner][%m_inner_size, %n_inner_size][1, 1]
                %AddResult = linalg.elementwise kind = #linalg.elementwise_kind<add> ins(%MatmulResult, %Bias_inner) outs(%Out_inner)

                // ====================================================
                // Stage 3: ReLU
                // ====================================================
                %Zero_inner = tensor.extract_slice %Zero_outer[%m_inner, %n_inner][%m_inner_size, %n_inner_size][1, 1]
                %FinalInner = linalg.elementwise kind = #linalg.elementwise_kind<max_signed> ins(%AddResult, %Zero_inner) outs(%Out_inner)
                %OutUpdated = tensor.insert_slice %FinalInner into %OutAcc3[%m_inner, %n_inner][%m_inner_size, %n_inner_size][1, 1]

                scf.yield %OutUpdated
              }

              scf.yield %n_inner
            }

          %OutFinal = tensor.insert_slice %Updated_outer into %OutAcc1[%m_outer, %n_outer][%m_outer_size, %n_outer_size][1, 1]
          scf.yield %OutFinal
        }
        scf.yield %n_outer
      }
    return %Result : tensor<?x?xf32>
  }
}