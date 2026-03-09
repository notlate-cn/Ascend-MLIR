// ============================================================
// STAGE 0: High-Level IR - 高层抽象表示
// 计算图：广播(Broadcast) + 加法(Add) + 归约求和(ReduceSum)
//
// 数据流：
//   输入A[M] → Broadcast → C[M,N]
//   C[M,N] + B[M,N] → D[M,N] (逐元素加法)
//   D[M,N] → ReduceSum(axis=1) → 输出E[M] (按行求和)
//
// 注意：这是一个完全符号化的表示，M和N是动态维度
// 三个算子的迭代器类型：
//   - Broadcast: [Parallel, Parallel] - 两个维度都是并行
//   - Add: [Parallel, Parallel] - 两个维度都是并行
//   - ReduceSum: [Parallel, Reduction] - d0并行，d1归约
//
// 由于ReduceSum的d1轴是Reduction类型，而前两个算子是Parallel，
// 导致轴分组不一致，无法全局融合，需要分段处理
// ============================================================
// RUN: afir-opt %s | FileCheck %s
// CHECK: func.func @broadcast_add_reducesum

// 定义广播映射：输入(d0,d1)只访问d0，实现沿d1轴广播
#broadcast_map = affine_map<(d0, d1) -> (d0)>
// 定义完整访问映射：输入(d0,d1)访问对应位置
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module {
  // 主函数：广播加法归约
  // 参数:
  //   %input_a: 输入张量A，形状[M]，将被广播到[M,N]
  //   %input_b: 输入张量B，形状[M,N]
  // 返回:
  //   输出张量E，形状[M]，是ReduceSum的结果
  func.func @broadcast_add_reducesum(
      %input_a : tensor<?xf16>,      // [M] - 一维输入，将被广播
      %input_b : tensor<?x?xf16>     // [M, N] - 二维输入
  ) -> tensor<?xf16> {

    // 定义常量索引
    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index

    // 获取动态维度大小
    %dim_m = tensor.dim %input_a, %idx_0 : tensor<?xf16>       // M维度大小
    %dim_n = tensor.dim %input_b, %idx_1 : tensor<?x?xf16>     // N维度大小

    // ── 算子1: Broadcast - 将A[M]广播为C[M,N] ─────────────────────
    // 迭代器类型: [Parallel, Parallel]
    // 索引映射: A只访问d0，输出C访问d0,d1
    // 效果: A的每一行被复制N次，形成[M,N]的矩阵
    %empty_tensor_c = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %tensor_c = linalg.generic {
      indexing_maps = [#broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%input_a : tensor<?xf16>)
      outs(%empty_tensor_c : tensor<?x?xf16>) {
    ^bb0(%a_value: f16, %c_output: f16):
      linalg.yield %a_value : f16
    } -> tensor<?x?xf16>

    // ── 算子2: Add - 逐元素加法 C[M,N] + B[M,N] = D[M,N] ────────────────
    // 迭代器类型: [Parallel, Parallel]
    // 所有张量都完整访问对应位置
    %empty_tensor_d = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %tensor_d = linalg.generic {
      indexing_maps = [#full_access_map, #full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%tensor_c, %input_b : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%empty_tensor_d : tensor<?x?xf16>) {
    ^bb0(%c_value: f16, %b_value: f16, %d_output: f16):
      %sum = arith.addf %c_value, %b_value : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>

    // ── 算子3: ReduceSum(axis=1) - 按行求和 D[M,N] → E[M] ───────────────
    // 迭代器类型: [Parallel, Reduction]
    // d0是并行轴(输出维度)，d1是归约轴(被求和消除)
    // 注意: d1从Parallel变为Reduction，导致无法与前两个算子融合
    %zero_value = arith.constant 0.0 : f16
    %empty_tensor_e = tensor.empty(%dim_m) : tensor<?xf16>
    %init_tensor_e = linalg.fill ins(%zero_value : f16)
                 outs(%empty_tensor_e : tensor<?xf16>) -> tensor<?xf16>
    %tensor_e = linalg.generic {
      indexing_maps = [#full_access_map, #broadcast_map],
      iterator_types = ["parallel", "reduction"]
    } ins(%tensor_d : tensor<?x?xf16>)
      outs(%init_tensor_e : tensor<?xf16>) {
    ^bb0(%d_value: f16, %accumulator: f16):
      %new_accumulator = arith.addf %accumulator, %d_value : f16
      linalg.yield %new_accumulator : f16
    } -> tensor<?xf16>

    return %tensor_e : tensor<?xf16>
  }
}
