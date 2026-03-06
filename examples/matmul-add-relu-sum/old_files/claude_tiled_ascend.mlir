// AscendNPU 三级 Tiling: TB / Tb / t
//
// 硬件语义:
//   TB  : 分核并行 (scf.forall 二维 [TB_M, TB_N])
//         每个 AI Core 独立处理一个 (TB_M x TB_N) 输出 tile
//   Tb  : 单核内循环 (Tb_M x Tb_N 两层 scf.for)
//         驱动 GM -> L1 Buffer 数据搬运 & 双缓冲流水基础
//         Tb_N 内层: A 片 (Tb_M x K) 保持不动, 复用于所有 Tb_N 迭代
//   t   : 单次处理量 (原 %9, K 方向 scf.for, step=TB_K)
//         对应一次 CUBE 指令处理的矩阵块, 操作 L0A/L0B/L0C
//
// 参数说明:
//   %arg4 = t_K  : t 级 K 步长 (原 TB_K, CUBE 单次 K 粒度)
//   %arg5 = TB_N : TB 级 N tile size (核间 N 方向分块)
//   %arg6 = TB_M : TB 级 M tile size (核间 M 方向分块)
//   %arg7 = Tb_M : Tb 级 M tile size (核内 M 方向分块, 须满足 TB_M % Tb_M == 0)
//   %arg8 = Tb_N : Tb 级 N tile size (核内 N 方向分块, 须满足 TB_N % Tb_N == 0)
//
// 典型昇腾910B参数示例:
//   TB_M=256, TB_N=256 (按实际核数和矩阵大小确定)
//   Tb_M=128, Tb_N=64  (L1 Buffer 容量决定)
//   t_K=16             (CUBE Core 单次 K 粒度)

#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>

module {
  func.func @fc_relu(
      %arg0: tensor<?x?xf32>,  // A [M, K]
      %arg1: tensor<?x?xf32>,  // B [K, N]
      %arg2: tensor<?x?xf32>,  // bias [M, N]
      %arg3: tensor<?x?xf32>,  // output [M, N]
      %arg4: i64,              // t_K  (原 TB_K, t 级 K 步长)
      %arg5: i64,              // TB_N (核间 N 分块)
      %arg6: i64,              // TB_M (核间 M 分块)
      %arg7: i64,              // Tb_M (核内 M 分块)
      %arg8: i64               // Tb_N (核内 N 分块)
  ) -> tensor<?x?xf32> {

    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32

    // tile size: i64 -> index
    %t_K  = arith.index_cast %arg4 : i64 to index
    %TB_N = arith.index_cast %arg5 : i64 to index
    %TB_M = arith.index_cast %arg6 : i64 to index
    %Tb_M = arith.index_cast %arg7 : i64 to index
    %Tb_N = arith.index_cast %arg8 : i64 to index

    // 全局维度
    %dim_M = tensor.dim %arg3, %c0 : tensor<?x?xf32>
    %dim_N = tensor.dim %arg3, %c1 : tensor<?x?xf32>
    %dim_K = tensor.dim %arg0, %c1 : tensor<?x?xf32>

    // zero tensor: 用于 relu 的比较基准 (linalg.fill 在核外完成)
    %zero_buf    = tensor.empty(%dim_M, %dim_N) : tensor<?x?xf32>
    %zero_filled = linalg.fill ins(%cst : f32)
                               outs(%zero_buf : tensor<?x?xf32>) -> tensor<?x?xf32>

    // =========================================================
    // TB 层: scf.forall 二维并行 [TB_M 方向, TB_N 方向]
    //   每个 AI Core 独立计算输出矩阵的一个 (sz_TB_M x sz_TB_N) tile
    //   in_parallel 块用 tensor.parallel_insert_slice 无冲突写回
    // =========================================================
    %result = scf.forall (%iv_TB_M, %iv_TB_N)
                  in (%dim_M, %dim_N)        // 总迭代空间 = 完整 M x N
                  step (%TB_M, %TB_N)        // 每核步长 = TB tile size
                  shared_outs(%out = %arg3)  // 输出张量
              -> tensor<?x?xf32> {

      // 当前 TB tile 的实际大小 (尾块边界裁剪)
      %sz_TB_M = affine.min #map(%iv_TB_M)[%dim_M, %TB_M]
      %sz_TB_N = affine.min #map(%iv_TB_N)[%dim_N, %TB_N]

      // 从全局 bias/zero 中取出本核负责的 tile (静态 offset, 无写冲突)
      %bias_TB = tensor.extract_slice %arg2[%iv_TB_M, %iv_TB_N]
                                            [%sz_TB_M, %sz_TB_N][1, 1]
                   : tensor<?x?xf32> to tensor<?x?xf32>

      %zero_TB = tensor.extract_slice %zero_filled[%iv_TB_M, %iv_TB_N]
                                                   [%sz_TB_M, %sz_TB_N][1, 1]
                   : tensor<?x?xf32> to tensor<?x?xf32>

      // 本核输出 tile 初始值 (从 shared_outs 取出)
      %out_TB_init = tensor.extract_slice %out[%iv_TB_M, %iv_TB_N]
                                              [%sz_TB_M, %sz_TB_N][1, 1]
                       : tensor<?x?xf32> to tensor<?x?xf32>

      // =========================================================
      // Tb 层 (M 方向): 单核内 M 方向循环
      //   驱动 A 矩阵 GM -> L1 搬运
      //   步长 Tb_M, 在 TB tile 范围 [0, sz_TB_M) 内迭代
      // =========================================================
      %out_Tb = scf.for %iv_Tb_M = %c0 to %sz_TB_M step %Tb_M
                    iter_args(%acc_Tb = %out_TB_init) -> tensor<?x?xf32> {

        // 当前 Tb_M 的实际大小
        %sz_Tb_M = affine.min #map(%iv_Tb_M)[%sz_TB_M, %Tb_M]

        // A 的 L1 tile: 本次 Tb_M 对应的行, 完整 K 列
        // 绝对行偏移 = TB 级偏移 + Tb 级偏移
        %abs_M = arith.addi %iv_TB_M, %iv_Tb_M : index
        %A_L1  = tensor.extract_slice %arg0[%abs_M, 0][%sz_Tb_M, %dim_K][1, 1]
                   : tensor<?x?xf32> to tensor<?x?xf32>

        // =========================================================
        // Tb 层 (N 方向): 单核内 N 方向循环
        //   驱动 B 矩阵 GM -> L1 搬运
        //   A_L1 在此层所有迭代中保持不动 (A 片复用)
        // =========================================================
        %out_Tb_N = scf.for %iv_Tb_N = %c0 to %sz_TB_N step %Tb_N
                        iter_args(%acc_Tb_N = %acc_Tb) -> tensor<?x?xf32> {

          // 当前 Tb_N 的实际大小
          %sz_Tb_N = affine.min #map(%iv_Tb_N)[%sz_TB_N, %Tb_N]

          // B 的 L1 tile: 完整 K 行, 本次 Tb_N 对应的列
          // 绝对列偏移 = TB 级偏移 + Tb 级偏移
          %abs_N = arith.addi %iv_TB_N, %iv_Tb_N : index
          %B_L1  = tensor.extract_slice %arg1[0, %abs_N][%dim_K, %sz_Tb_N][1, 1]
                     : tensor<?x?xf32> to tensor<?x?xf32>

          // 当前 (Tb_M x Tb_N) 输出 tile (从累加张量中取出)
          %C_Tb_init = tensor.extract_slice %acc_Tb_N[%iv_Tb_M, %iv_Tb_N]
                                                      [%sz_Tb_M, %sz_Tb_N][1, 1]
                         : tensor<?x?xf32> to tensor<?x?xf32>

          // =====================================================
          // t 层: K 方向 reduction (原 %9, 保持不动)
          //   step = t_K, 对应 CUBE Core 单次指令处理的 K 粒度
          //   操作 L0A (A_K_slice) / L0B (B_K_slice) / L0C (C_Tb)
          // =====================================================
          %C_Tb_matmul = scf.for %iv_K = %c0 to %dim_K step %t_K
                             iter_args(%C_acc = %C_Tb_init) -> tensor<?x?xf32> {

            %sz_K = affine.min #map(%iv_K)[%dim_K, %t_K]

            // L0A: A_L1 的 K 切片 [0:sz_Tb_M, iv_K:sz_K]
            %A_L0 = tensor.extract_slice %A_L1[0, %iv_K][%sz_Tb_M, %sz_K][1, 1]
                      : tensor<?x?xf32> to tensor<?x?xf32>

            // L0B: B_L1 的 K 切片 [iv_K:sz_K, 0:sz_Tb_N]
            %B_L0 = tensor.extract_slice %B_L1[%iv_K, 0][%sz_K, %sz_Tb_N][1, 1]
                      : tensor<?x?xf32> to tensor<?x?xf32>

            // L0C: 当前累加结果
            %C_L0 = tensor.extract_slice %C_acc[0, 0][%sz_Tb_M, %sz_Tb_N][1, 1]
                      : tensor<?x?xf32> to tensor<?x?xf32>

            // CUBE Core 矩阵乘法
            %C_L0_out = linalg.matmul
                ins(%A_L0, %B_L0 : tensor<?x?xf32>, tensor<?x?xf32>)
                outs(%C_L0       : tensor<?x?xf32>) -> tensor<?x?xf32>

            %C_acc_new = tensor.insert_slice %C_L0_out into %C_acc
                             [0, 0][%sz_Tb_M, %sz_Tb_N][1, 1]
                           : tensor<?x?xf32> into tensor<?x?xf32>
            scf.yield %C_acc_new : tensor<?x?xf32>
          } // end t (K-reduction)

          // -------------------------------------------------
          // Add: matmul 结果 + bias (Vector Core)
          //   bias/zero 均已在 TB 层切出, 这里再取 Tb 粒度子块
          // -------------------------------------------------
          %bias_Tb = tensor.extract_slice %bias_TB[%iv_Tb_M, %iv_Tb_N]
                                                   [%sz_Tb_M, %sz_Tb_N][1, 1]
                       : tensor<?x?xf32> to tensor<?x?xf32>

          %out_for_add = tensor.extract_slice %acc_Tb_N[%iv_Tb_M, %iv_Tb_N]
                                                        [%sz_Tb_M, %sz_Tb_N][1, 1]
                           : tensor<?x?xf32> to tensor<?x?xf32>

          %add_result = linalg.elementwise kind=#linalg.elementwise_kind<add>
              ins(%C_Tb_matmul, %bias_Tb : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%out_for_add          : tensor<?x?xf32>) -> tensor<?x?xf32>

          // -------------------------------------------------
          // ReLU: max(add, 0) (Vector Core)
          // -------------------------------------------------
          %zero_Tb = tensor.extract_slice %zero_TB[%iv_Tb_M, %iv_Tb_N]
                                                   [%sz_Tb_M, %sz_Tb_N][1, 1]
                       : tensor<?x?xf32> to tensor<?x?xf32>

          %out_for_relu = tensor.extract_slice %acc_Tb_N[%iv_Tb_M, %iv_Tb_N]
                                                         [%sz_Tb_M, %sz_Tb_N][1, 1]
                            : tensor<?x?xf32> to tensor<?x?xf32>

          %relu_result = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
              ins(%add_result, %zero_Tb : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%out_for_relu        : tensor<?x?xf32>) -> tensor<?x?xf32>

          // Tb tile 结果写回核内累加张量
          %acc_Tb_N_new = tensor.insert_slice %relu_result
                              into %acc_Tb_N[%iv_Tb_M, %iv_Tb_N]
                                            [%sz_Tb_M, %sz_Tb_N][1, 1]
                            : tensor<?x?xf32> into tensor<?x?xf32>
          scf.yield %acc_Tb_N_new : tensor<?x?xf32>
        } // end Tb_N

        scf.yield %out_Tb_N : tensor<?x?xf32>
      } // end Tb_M

      // =========================================================
      // TB 层写回: in_parallel 块
      //   tensor.parallel_insert_slice 保证多核无写冲突
      //   每个核写回自己负责的 (sz_TB_M x sz_TB_N) 区域
      // =========================================================
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %out_Tb
            into %out[%iv_TB_M, %iv_TB_N][%sz_TB_M, %sz_TB_N][1, 1]
          : tensor<?x?xf32> into tensor<?x?xf32>
      }
    } // end scf.forall (TB)

    return %result : tensor<?x?xf32>
  }
}
