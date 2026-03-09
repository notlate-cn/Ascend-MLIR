// ============================================================
// tiling_func.mlir — Host端分块函数
//
// 功能：在Host CPU上计算分块参数(Tiling Parameters)
//
// 输入:
//   %M (i64): 行数
//   %N (i64): 列数
//
// 输出:
//   TilingData 结构体，包含:
//     - TB: 核间分块大小 (outer tile)
//     - Tb: 核内分块大小 (inner tile per AiCore)
//
// 算法说明：
//   - 归约轴(N)不做分块，每个AiCore处理完整的N列
//   - 根据UB容量计算每核能处理的行数
//   - 需要3个缓冲区：A切片、B矩阵块、结果
//
// 硬件假设：
//   - CORE_NUM = 20 (AiCore数量)
//   - UB_BYTES = 256KB (Unified Buffer容量)
//   - ELEM_BYTES = 2 (f16 = 2字节)
// ============================================================

// TilingData结构体定义 (2个i64字段)
//   [0] TB — 外层分块 (核间分发)
//   [1] Tb — 内层分块 (每核UB缓冲区)
!TilingData = !llvm.struct<"TilingData", (i64, i64)>

module {

  // ----------------------------------------------------------
  // @tiling_func: 根据运行时形状计算分块参数
  // ----------------------------------------------------------
  func.func @tiling_func(%M: i64, %N: i64) -> !TilingData {

    // ---- 硬件常量定义 ----
    %CORE_NUM = arith.constant 20 : i64      // AiCore数量
    %UB_BYTES = arith.constant 262144 : i64  // 256KB UB容量
    %ELEM_BYTES = arith.constant 2 : i64     // f16 = 2字节

    // ---- 计算Tb: 每核UB能容纳的行数 ----
    // 公式: UB / (3 × N × 2)
    // 需要3个缓冲区: A切片 + B矩阵块 + 结果
    %const_1 = arith.constant 1 : i64
    %const_3 = arith.constant 3 : i64

    // 总元素数 = UB_BYTES / ELEM_BYTES = 131072
    %total_elems = arith.divsi %UB_BYTES, %ELEM_BYTES : i64

    // 每缓冲区元素数 = total_elems / 3 = 43690
    %per_buf = arith.divsi %total_elems, %const_3 : i64

    // 行数 = per_buf / N (至少1行)
    %rows_raw = arith.divsi %per_buf, %N : i64
    %rows_clamped = arith.maxsi %rows_raw, %const_1 : i64
    %Tb = %rows_clamped : i64

    // ---- 计算TB: 外层分块 ----
    // 简化为: TB = Tb (每核处理一个Tb批次)
    %TB = %Tb : i64

    // ---- 打包到TilingData结构体 ----
    %td_0 = llvm.mlir.undef : !TilingData
    %td_1 = llvm.insertvalue %TB, %td_0[0] : !TilingData
    %td_2 = llvm.insertvalue %Tb, %td_1[1] : !TilingData

    return %td_2 : !TilingData
  }

  // ----------------------------------------------------------
  // @runtime_dispatch: Host端调度入口 (概念性)
  // ----------------------------------------------------------
  func.func @runtime_dispatch(
      %A: memref<?xf16>,      // 广播源 (长度M)
      %B: memref<?x?xf16>,    // 输入矩阵 (M × N)
      %out: memref<?xf16>     // 输出归约结果 (长度M)
  ) {
    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index

    // ---- 获取运行时形状 ----
    %M_idx = memref.dim %A, %idx_0 : memref<?xf16>
    %N_idx = memref.dim %B, %idx_1 : memref<?x?xf16>
    %M = arith.index_cast %M_idx : index to i64
    %N = arith.index_cast %N_idx : index to i64

    // ---- 计算分块参数 ----
    %tiling = func.call @tiling_func(%M, %N)
              : (i64, i64) -> !TilingData

    // ---- 解包 ----
    %TB = llvm.extractvalue %tiling[0] : !TilingData
    %Tb = llvm.extractvalue %tiling[1] : !TilingData

    // ---- 内核启动 (概念性) ----
    // acl_launch_kernel("broadcast_add_reducesum", core_num,
    //                   A, B, out, &tiling_data)

    return
  }

} // module
