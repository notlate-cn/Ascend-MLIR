// ============================================================
// STAGE 1: Operator Fusion - 算子融合阶段
//
// 融合策略：
//   由于 Broadcast + Add 都是 [Parallel, Parallel] 类型，可以融合
//   ReduceSum 是 [Parallel, Reduction] 类型，无法与前两个算子融合
//
// 融合结果：
//   - 将 Broadcast + Add + ReduceSum 融合为单个 linalg.generic
//   - 消除了中间张量 C 和 D，直接从 A 和 B 计算到 E
//   - 在 basic block 中内联广播逻辑：A[d0] + B[d0,d1]
//
// 性能收益：
//   - 减少内存访问：避免写入/读取中间张量 C 和 D
//   - 更好的数据局部性：计算在寄存器/缓存中完成
// ============================================================

// 广播映射：只访问d0轴
#broadcast_map = affine_map<(d0, d1) -> (d0)>
// 完整访问映射：访问d0,d1
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module {
  // 融合后的主函数
  // 将三个算子融合为单个 linalg.generic
  func.func @broadcast_add_reducesum(
      %input_a: tensor<?xf16>,      // 输入A [M]
      %input_b: tensor<?x?xf16>     // 输入B [M,N]
  ) -> tensor<?xf16> {

    // 常量定义
    %zero = arith.constant 0.000000e+00 : f16
    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index

    // 获取维度
    %dim_m = tensor.dim %input_a, %idx_0 : tensor<?xf16>      // M维度

    // ── 融合后的单个算子 ─────────────────────────────────────
    // 输入: A[M] (广播), B[M,N] (完整访问)
    // 输出: E[M] (归约结果)
    // 迭代器: [Parallel, Reduction]
    //   - d0 (M轴): Parallel - 每个输出行独立计算
    //   - d1 (N轴): Reduction - 沿N轴求和
    %empty_output = tensor.empty(%dim_m) : tensor<?xf16>
    %init_output = linalg.fill ins(%zero : f16)
                   outs(%empty_output : tensor<?xf16>) -> tensor<?xf16>

    %output = linalg.generic {
      indexing_maps = [#broadcast_map, #full_access_map, #broadcast_map],
      iterator_types = ["parallel", "reduction"]
    } ins(%input_a, %input_b : tensor<?xf16>, tensor<?x?xf16>)
      outs(%init_output : tensor<?xf16>) {
    ^bb0(%a_value: f16, %b_value: f16, %accumulator: f16):
      // 内联广播加法：A[d0] + B[d0,d1]
      %sum = arith.addf %a_value, %b_value : f16
      // 累加到结果
      %new_accumulator = arith.addf %accumulator, %sum : f16
      linalg.yield %new_accumulator : f16
    } -> tensor<?xf16>

    return %output : tensor<?xf16>
  }
}
