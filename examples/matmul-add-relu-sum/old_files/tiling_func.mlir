// ============================================================
// tiling_func.mlir — Host端分块函数
//
// 功能：在Host CPU上计算分块参数(Tiling Parameters)
//
// 输入:
//   %M (i64): 行数
//   %N (i64): 列数
//   %K (i64): 内维大小
//
// 输出:
//   TilingData 结构体，包含:
//     - TB_M, TB_N: 核间分块大小 (outer tile)
//     - Tb_M, Tb_N: 核内分块大小 (inner tile per AiCore)
//     - t_K: K维度分块大小
//
// 算法说明：
//   - 根据UB容量计算每核能处理的矩阵块大小
//   - 矩阵乘法需要A[M,K]、B[K,N]、C[M,N]三个缓冲区
//   - 考虑Cube单元的数据对齐要求
//
// 硬件假设：
//   - CORE_NUM = 20 (AiCore数量)
//   - UB_BYTES = 256KB (Unified Buffer容量)
//   - L0_BUFFER = 64KB (L0缓冲区容量)
//   - ELEM_BYTES = 4 (f32 = 4字节)
// ============================================================

// TilingData结构体定义 (6个i64字段)
//   [0] TB_M — 外层M分块 (核间分发)
//   [1] TB_N — 外层N分块 (核间分发)
//   [2] Tb_M — 内层M分块 (每核UB缓冲区)
//   [3] Tb_N — 内层N分块 (每核UB缓冲区)
//   [4] t_K  — K维度分块
//   [5] core_num — 实际使用的核数
!TilingData = !llvm.struct<"TilingData", (i64, i64, i64, i64, i64, i64)>

module {

  // ----------------------------------------------------------
  // @tiling_func: 根据运行时形状计算分块参数
  // ----------------------------------------------------------
  func.func @tiling_func(%M : i64, %N : i64, %K : i64) -> !TilingData {

    // ---- 硬件常量定义 ----
    %UB_BYTES = arith.constant 262144 : i64   // 256KB UB容量
    %L0_BYTES = arith.constant 65536 : i64    // 64KB L0缓冲区
    %CORE_NUM = arith.constant 20 : i64       // AiCore数量
    %ELEM_BYTES = arith.constant 4 : i64      // f32 = 4字节
    %CUBE_ALIGN = arith.constant 16 : i64     // Cube对齐要求

    // ---- 计算t_K: K维度分块 ----
    // 根据L0缓冲区容量计算
    // L0需要容纳A的M×K块和B的K×N块
    %l0_elems = arith.divsi %L0_BYTES, %ELEM_BYTES : i64
    %t_K_raw = arith.divsi %l0_elems, %CUBE_ALIGN : i64
    %t_K = arith.muli %t_K_raw, %CUBE_ALIGN : i64

    // ---- 计算Tb_M, Tb_N: 核内分块 ----
    // 根据UB容量计算
    // UB需要容纳A[M,K]、B[K,N]、C[M,N]三个缓冲区
    %ub_elems = arith.divsi %UB_BYTES, %ELEM_BYTES : i64

    // 简化计算：假设K维度使用t_K分块
    // 则每行需要 K/t_K 个迭代
    %k_iters = arith.addi
                 (arith.divsi %K, %t_K : i64),
                 (arith.constant 1 : i64) : i64

    // 每核能处理的元素数
    %per_core_elems = arith.divsi %ub_elems, %k_iters : i64

    // 假设M和N维度相等，计算Tb
    %Tb_raw = arith.divsi %per_core_elems, %CUBE_ALIGN : i64
    %Tb = arith.muli %Tb_raw, %CUBE_ALIGN : i64
    %Tb_M = %Tb : i64
    %Tb_N = %Tb : i64

    // ---- 计算TB_M, TB_N: 核间分块 ----
    // 简化为：TB = Tb (每核处理一个Tb批次)
    %TB_M = %Tb_M : i64
    %TB_N = %Tb_N : i64

    // ---- 计算实际使用的核数 ----
    %m_cores = arith.addi
                 (arith.divsi %M, %TB_M : i64),
                 (arith.constant 1 : i64) : i64
    %n_cores = arith.addi
                 (arith.divsi %N, %TB_N : i64),
                 (arith.constant 1 : i64) : i64
    %total_cores = arith.muli %m_cores, %n_cores : i64
    %core_num = arith.minsi %total_cores, %CORE_NUM : i64

    // ---- 打包到TilingData结构体 ----
    %td = llvm.mlir.undef : !TilingData
    %td1 = llvm.insertvalue %TB_M, %td[0] : !TilingData
    %td2 = llvm.insertvalue %TB_N, %td1[1] : !TilingData
    %td3 = llvm.insertvalue %Tb_M, %td2[2] : !TilingData
    %td4 = llvm.insertvalue %Tb_N, %td3[3] : !TilingData
    %td5 = llvm.insertvalue %t_K, %td4[4] : !TilingData
    %td6 = llvm.insertvalue %core_num, %td5[5] : !TilingData

    return %td6 : !TilingData
  }

  // ----------------------------------------------------------
  // @runtime_dispatch: Host端调度入口
  // ----------------------------------------------------------
  func.func @runtime_dispatch(
      %input_a: memref<?x?xf32>,   // 输入A [M,K]
      %input_b: memref<?x?xf32>,   // 输入B [K,N]
      %bias: memref<?x?xf32>,       // 偏置 [M,N]
      %output: memref<?x?xf32>     // 输出 [M,N]
  ) {
    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index

    // ---- 获取运行时形状 ----
    %M_idx = memref.dim %input_a, %idx_0 : memref<?x?xf32>
    %K_idx = memref.dim %input_a, %idx_1 : memref<?x?xf32>
    %N_idx = memref.dim %input_b, %idx_1 : memref<?x?xf32>

    %M = arith.index_cast %M_idx : index to i64
    %N = arith.index_cast %N_idx : index to i64
    %K = arith.index_cast %K_idx : index to i64

    // ---- 计算TilingData ----
    %tiling = func.call @tiling_func(%M, %N, %K)
              : (i64, i64, i64) -> !TilingData

    // ---- 解包 ----
    %TB_M = llvm.extractvalue %tiling[0] : !TilingData
    %TB_N = llvm.extractvalue %tiling[1] : !TilingData
    %Tb_M = llvm.extractvalue %tiling[2] : !TilingData
    %Tb_N = llvm.extractvalue %tiling[3] : !TilingData
    %t_K = llvm.extractvalue %tiling[4] : !TilingData
    %core_num = llvm.extractvalue %tiling[5] : !TilingData

    // ---- 内核启动 (概念性) ----
    // acl_launch_kernel("fc_relu", core_num,
    //                   input_a, input_b, bias, output,
    //                   TB_M, TB_N, Tb_M, Tb_N, t_K)

    return
  }

} // module
