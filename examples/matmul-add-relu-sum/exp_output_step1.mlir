#tile_guard = affine_map<(d0)[upper, tile] -> (-d0 + upper, tile)>

// ============================================================================
//  Fully Connected + Bias + ReLU
//
//  ================= 核心技术 =================
//
//  1. 多面体调度分析
//     利用多面体模型描述迭代域 D(i,j,k)
//     通过合法 schedule θ(i,j,k) 保证数据依赖正确
//     affine.min 实现 tile 边界合法化
//
//  2. Tile-based Kernel 映射
//     GM → Outer Tile (L2)
//         → Inner Tile (UB)
//             → K Tile (Register / Cube pipeline)
//     显式表达分层数据复用
//
//  3. 容量约束与阶段延迟优化
//     tileM_inner * tileK   <= UB_A_capacity
//     tileK * tileN_inner   <= UB_B_capacity
//     tileM_inner * tileN_inner <= UB_C_capacity
//     平衡 Cube 计算阶段与 Vector Epilogue 延迟
// ============================================================================

module {
  func.func @fc_relu(
      %A: tensor<?x?xf32>,        // 输入 A (M x K)
      %B: tensor<?x?xf32>,        // 输入 B (K x N)
      %Bias: tensor<?x?xf32>,     // Bias (M x N)
      %OutInit: tensor<?x?xf32>,  // 初始输出 (M x N)
      %tileM_outer_i64: i64,
      %tileN_outer_i64: i64,
      %tileM_inner_i64: i64,
      %tileN_inner_i64: i64,
      %tileK_i64: i64
  ) -> tensor<?x?xf32> {

    // ============================================================
    // 基础常量
    // ============================================================
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero_f32 = arith.constant 0.000000e+00 : f32

    // ============================================================
    // tile 参数转换
    // ============================================================
    %tileK = arith.index_cast %tileK_i64 : i64 to index
    %tileN_inner = arith.index_cast %tileN_inner_i64 : i64 to index
    %tileM_inner = arith.index_cast %tileM_inner_i64 : i64 to index
    %tileN_outer = arith.index_cast %tileN_outer_i64 : i64 to index
    %tileM_outer = arith.index_cast %tileM_outer_i64 : i64 to index

    // ============================================================
    // 输出矩阵尺寸
    // ============================================================
    %M = tensor.dim %OutInit, %c0 : tensor<?x?xf32>
    %N = tensor.dim %OutInit, %c1 : tensor<?x?xf32>

    // 构造 ReLU 需要的 zero tensor
    %ZeroTensor = tensor.empty(%M, %N) : tensor<?x?xf32>
    %ZeroFilled = linalg.fill
        ins(%zero_f32 : f32)
        outs(%ZeroTensor : tensor<?x?xf32>) -> tensor<?x?xf32>

    // ============================================================
    // ===================== Outer Tile (L2 Blocking) =====================
    // 多面体调度第一层
    // ============================================================

    %OutAfterOuter =
    scf.for %m_outer = %c0 to %M step %tileM_outer
        iter_args(%Out_acc_outer = %OutInit)
        -> (tensor<?x?xf32>) {

      %OutAfterNOuter =
      scf.for %n_outer = %c0 to %N step %tileN_outer
          iter_args(%Out_acc_tile = %Out_acc_outer)
          -> (tensor<?x?xf32>) {

        // tile 边界保护
        %M_outer_size = affine.min #tile_guard(%m_outer)[%M, %tileM_outer]
        %N_outer_size = affine.min #tile_guard(%n_outer)[%N, %tileN_outer]

        %K = tensor.dim %A, %c1 : tensor<?x?xf32>

        // ---------------- GM → L2 tile ----------------

        %A_outer = tensor.extract_slice %A[%m_outer, 0][%M_outer_size, %K][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %B_outer = tensor.extract_slice %B[0, %n_outer][%K, %N_outer_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %Out_outer = tensor.extract_slice %Out_acc_tile[%m_outer, %n_outer][%M_outer_size, %N_outer_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %Bias_outer = tensor.extract_slice %Bias[%m_outer, %n_outer][%M_outer_size, %N_outer_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %Zero_outer = tensor.extract_slice %ZeroFilled[%m_outer, %n_outer][%M_outer_size, %N_outer_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

        // ============================================================
        // ===================== Inner Tile (UB Blocking) =================
        // ============================================================

        %OutAfterInner =
        scf.for %m_inner = %c0 to %M_outer_size step %tileM_inner iter_args(%Out_acc_inner = %Out_outer) -> (tensor<?x?xf32>) {

          %OutAfterNInner =
          scf.for %n_inner = %c0 to %N_outer_size step %tileN_inner iter_args(%Out_acc_block = %Out_acc_inner) -> (tensor<?x?xf32>) {

            %M_inner_size = affine.min #tile_guard(%m_inner)[%M_outer_size, %tileM_inner]
            %N_inner_size = affine.min #tile_guard(%n_inner)[%N_outer_size, %tileN_inner]

            %A_inner = tensor.extract_slice %A_outer[%m_inner, 0][%M_inner_size, %K][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %B_inner = tensor.extract_slice %B_outer[0, %n_inner][%K, %N_inner_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %Out_inner = tensor.extract_slice %Out_acc_block[%m_inner, %n_inner][%M_inner_size, %N_inner_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

            // ============================================================
            // ===================== K Blocking (Register/Cube) ============
            // ============================================================

            %OutAfterK =
            scf.for %k = %c0 to %K step %tileK iter_args(%Out_acc_k = %Out_inner) -> (tensor<?x?xf32>) {

              %K_tile_size = affine.min #tile_guard(%k)[%K, %tileK]

              %A_k = tensor.extract_slice %A_inner[0, %k][%M_inner_size, %K_tile_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %B_k = tensor.extract_slice %B_inner[%k, 0][%K_tile_size, %N_inner_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %Out_tile = tensor.extract_slice %Out_acc_k[0, 0][%M_inner_size, %N_inner_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

              // ---------------- Stage 1: Cube GEMM ----------------

              %MatmulTile = linalg.matmul {ascendc.unit = "AiCore.Cube"}
                ins(%A_k, %B_k : tensor<?x?xf32>, tensor<?x?xf32>)
                outs(%Out_tile : tensor<?x?xf32>) -> tensor<?x?xf32>

              %Out_acc_k_next = tensor.insert_slice %MatmulTile into %Out_acc_k[0, 0][%M_inner_size, %N_inner_size][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

              scf.yield %Out_acc_k_next : tensor<?x?xf32>
            } {ascendc.prologue = "lhs:A1->A2,rhs:B1->B2",
               ascendc.epilogue = "acc:CO1->VECIN"}

            // ---------------- Stage 2: Bias Add (Vector) ----------------

            %Bias_inner = tensor.extract_slice %Bias_outer[%m_inner, %n_inner][%M_inner_size, %N_inner_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

            %AddResult = linalg.elementwise kind=#linalg.elementwise_kind<add> {ascendc.unit = "AiCore.Vector"}
              ins(%OutAfterK, %Bias_inner : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%Out_inner : tensor<?x?xf32>) -> tensor<?x?xf32>

            // ---------------- Stage 3: ReLU (Vector) ----------------

            %Zero_inner = tensor.extract_slice %Zero_outer[%m_inner, %n_inner][%M_inner_size, %N_inner_size][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

            %ReluResult = linalg.elementwise kind=#linalg.elementwise_kind<max_signed> {ascendc.unit = "AiCore.Vector"}
              ins(%AddResult, %Zero_inner : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%Out_acc_block : tensor<?x?xf32>) -> tensor<?x?xf32>

            %Out_block_next = tensor.insert_slice %ReluResult into %Out_acc_block[%m_inner, %n_inner][%M_inner_size, %N_inner_size][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

            scf.yield %Out_block_next : tensor<?x?xf32>
          }
          scf.yield %OutAfterNInner : tensor<?x?xf32>
        }

        %Out_tile_next = tensor.insert_slice %OutAfterInner into %Out_acc_tile[%m_outer, %n_outer][%M_outer_size, %N_outer_size][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

        scf.yield %Out_tile_next : tensor<?x?xf32>
      } {ascendc.parallel = true,
         ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN",
         ascendc.epilogue = "result:VECOUT->GM"}

      scf.yield %OutAfterNOuter : tensor<?x?xf32>
    } {ascendc.parallel = true}

    return %OutAfterOuter : tensor<?x?xf32>
  }
}