// ============================================================
// STAGE 0: High-Level IR - 高层抽象表示
// 计算图：Gather(索引聚合) + Elementwise + Broadcast
//
// 数据流：
//   Op1 (Gather):  data[M, N] 按 indices[K] 在 N 轴抽取 → gathered[M, K]
//                  语义：gathered[i, j] = data[i, indices[j]]
//   Op2 (BroadcastAdd): gathered[M, K] + broadcast_row(bias[M]) → out[M, K]
//                  bias 沿 K 轴广播，每行加同一个 bias 值
//
// 注意：这是一个完全符号化的表示，M、N、K 均为动态维度
//
// Gather 的 linalg.generic 表达：
//   迭代器类型: [Parallel(M), Parallel(K)]
//   ins: indices[K] (i32, 广播到 M), data[M, N] (全 MN 访问)
//   outs: gathered[M, K]
//   body: gathered[i,j] = data[i, indices[j]]
//   这里用 library_call = "gather_by_index" 标记特殊语义
//   实际在 linalg.generic 中无法直接表达 gather（非仿射索引），
//   因此我们用专用 library_call 标记 + 辅助 indices tensor 来表达 gather 意图
//
// BroadcastAdd 的 linalg.generic 表达：
//   迭代器类型: [Parallel(M), Parallel(K)]
//   bias[M] 通过 row_broadcast_map (d0,d1)->d0 沿 K 轴广播
//   library_call = "broadcast_add_gathered"
//
// library_call 属性用于区分 Op1 和 Op2，以便 Transform 脚本分别匹配
// ============================================================
// RUN: afir-opt %s | FileCheck %s
// CHECK: func.func @ewop_broadcast_gather

// 行广播映射：bias[M] 沿 K 轴广播，(d0,d1)->d0
#row_broadcast_map = affine_map<(d0, d1) -> (d0)>
// indices 广播映射：indices[K] 沿 M 轴广播，(d0,d1)->d1
#col_broadcast_map = affine_map<(d0, d1) -> (d1)>
// 完整访问映射：访问 (d0, d1)
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @ewop_broadcast_gather(
      %data    : tensor<?x?xf16>,   // [M, N] - 源数据
      %indices : tensor<?xi32>,     // [K]    - 索引（沿 N 轴抽取）
      %bias    : tensor<?xf16>      // [M]    - 行方向 bias（广播加）
  ) -> tensor<?x?xf16> {

    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index

    %dim_m = tensor.dim %data, %idx_0 : tensor<?x?xf16>    // M
    %dim_n = tensor.dim %data, %idx_1 : tensor<?x?xf16>    // N
    %dim_k = tensor.dim %indices, %idx_0 : tensor<?xi32>   // K

    // ── Op1: Gather data[M,N] 按 indices[K] 抽取 → gathered[M,K] ─
    // 迭代器类型: [Parallel(M), Parallel(K)]
    // indices 用列广播映射 (d0,d1)->d1（仅依赖 d1=K）
    // data 用完整访问（虽然 gather 本身是非仿射的，这里简化为 full_access_map
    //   占位，实际 gather 语义通过 library_call = "gather_by_index" 标记，
    //   让 Transform 和 ComputeConversion 能识别并做专门处理）
    // body: 这里用 arith.index_cast + linalg.yield 模拟 gather 写出
    //   实际 ComputeConversion 通过 library_call 识别 gather 模式并生成 gather_l2
    %empty_gathered = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %gathered = linalg.generic {
      indexing_maps = [#col_broadcast_map, #full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "gather_by_index"
    } ins(%indices, %data : tensor<?xi32>, tensor<?x?xf16>)
      outs(%empty_gathered : tensor<?x?xf16>) {
    ^bb0(%idx_val: i32, %data_val: f16, %out_val: f16):
      // 占位 body：实际 gather_l2 由 ComputeConversion 专门处理
      // data_val 已经是 "正确行" 的值（data 全访问作为参考），
      // 实际转换时忽略此 body，改用 gather_l2(dst, src_row, src_offset, 0, K)
      linalg.yield %data_val : f16
    } -> tensor<?x?xf16>

    // ── Op2: gathered[M,K] + broadcast_row(bias[M]) → out[M,K] ───
    // 迭代器类型: [Parallel(M), Parallel(K)]
    // bias[M] 通过行广播映射扩展到 [M, K]
    // library_call = "broadcast_add_gathered" 用于 Transform 匹配
    %empty_out = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %out = linalg.generic {
      indexing_maps = [#full_access_map, #row_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "broadcast_add_gathered"
    } ins(%gathered, %bias : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_out : tensor<?x?xf16>) {
    ^bb0(%g_val: f16, %b_val: f16, %o_out: f16):
      %sum = arith.addf %g_val, %b_val : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>

    return %out : tensor<?x?xf16>
  }
}
