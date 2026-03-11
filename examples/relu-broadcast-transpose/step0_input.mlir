// ============================================================
// STAGE 0: High-Level IR - 高层抽象表示
// 计算图：逐元素运算(Elementwise) + 广播(Broadcast) + 转置(Transpose)
//
// 数据流：
//   输入: A[M,N], bias[N], scale[M]
//   Op1: relu(A[M,N]) + broadcast_col(bias[N]) → B[M,N]
//        （逐元素 ReLU + 列方向广播加 bias）
//   Op2: Transpose(B[M,N]) → C[N,M]
//        （2D 转置，行列互换）
//   Op3: C[N,M] * broadcast_col(scale[M]) → D[N,M]
//        （逐元素乘 scale，scale[M] 沿 N 轴广播）
//
// 注意：这是一个完全符号化的表示，M 和 N 是动态维度
//
// 迭代器类型：
//   Op1: [Parallel, Parallel] - relu+broadcast_col+add，d0=M, d1=N
//   Op2: [Parallel, Parallel] - 转置 generic，ins 访问 (d1,d0)
//   Op3: [Parallel, Parallel] - 乘以列广播 scale，d0=N, d1=M
//
// Transpose 用 linalg.generic 表达：
//   ins 的 indexing_map = (d0,d1) -> (d1,d0)，即读 B[d1,d0]
//   outs 的 indexing_map = (d0,d1) -> (d0,d1)，即写 C[d0,d1]
//   body 直接 yield → 等价于 C[i,j] = B[j,i]
//   library_call = "transpose" 用于 Transform 脚本匹配
//
// Op3 中 scale[M] 的广播：
//   迭代空间 (d0,d1) 对应 C[N,M]，d0=N, d1=M
//   scale[M] 通过 col_broadcast_map (d0,d1)->d1 沿 N 轴广播
//
// library_call 属性用于 Transform 脚本区分各 Op：
//   Op1: library_call = "relu_bias_add"
//   Op2: library_call = "transpose"
//   Op3: library_call = "scale_mul"
// ============================================================
// RUN: afir-opt %s | FileCheck %s
// CHECK: func.func @ewop_broadcast_transpose

// 列广播映射：1D 输入只访问 d1，沿 d0 轴广播
#col_broadcast_map = affine_map<(d0, d1) -> (d1)>
// 完整访问映射：访问 (d0,d1)
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>
// 转置映射：读 (d1,d0)，实现行列互换
#transpose_map = affine_map<(d0, d1) -> (d1, d0)>

module {
  func.func @ewop_broadcast_transpose(
      %input_a : tensor<?x?xf16>,   // [M, N] - 主输入，做 ReLU
      %bias    : tensor<?xf16>,     // [N]    - 列方向 bias
      %scale   : tensor<?xf16>      // [M]    - 转置后乘以的 scale
  ) -> tensor<?x?xf16> {

    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index

    %dim_m = tensor.dim %input_a, %idx_0 : tensor<?x?xf16>   // M
    %dim_n = tensor.dim %input_a, %idx_1 : tensor<?x?xf16>   // N

    // ── Op1: relu(A[M,N]) + broadcast_col(bias[N]) → B[M,N] ──────
    // 迭代器类型: [Parallel, Parallel]
    // bias[N] 通过列广播映射 (d0,d1)->d1 沿 M 轴广播
    // relu = max(x, 0)，用 arith.maximumf 实现
    %zero_f16 = arith.constant 0.0 : f16
    %empty_b = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %tensor_b = linalg.generic {
      indexing_maps = [#full_access_map, #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "relu_bias_add"
    } ins(%input_a, %bias : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_b : tensor<?x?xf16>) {
    ^bb0(%a_val: f16, %bias_val: f16, %b_out: f16):
      %relu_a = arith.maximumf %a_val, %zero_f16 : f16
      %result  = arith.addf %relu_a, %bias_val : f16
      linalg.yield %result : f16
    } -> tensor<?x?xf16>

    // ── Op2: Transpose(B[M,N]) → C[N,M] ─────────────────────────
    // 迭代器类型: [Parallel, Parallel]（迭代空间 [N,M]）
    // ins 的 indexing_map = (d0,d1) -> (d1,d0)：读 B[d1,d0]（即 B[M,N]）
    // outs 的 indexing_map = (d0,d1) -> (d0,d1)：写 C[d0,d1]（即 C[N,M]）
    // 结果: C[i,j] = B[j,i]，实现转置
    %empty_c = tensor.empty(%dim_n, %dim_m) : tensor<?x?xf16>
    %tensor_c = linalg.generic {
      indexing_maps = [#transpose_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "transpose"
    } ins(%tensor_b : tensor<?x?xf16>)
      outs(%empty_c : tensor<?x?xf16>) {
    ^bb0(%b_val: f16, %c_out: f16):
      linalg.yield %b_val : f16
    } -> tensor<?x?xf16>

    // ── Op3: C[N,M] * broadcast_col(scale[M]) → D[N,M] ──────────
    // 迭代器类型: [Parallel, Parallel]（迭代空间 [N,M]）
    // scale[M] 通过列广播映射 (d0,d1)->d1 沿 N 轴广播
    // d0=N, d1=M，scale 只访问 d1，实现列广播
    %empty_d = tensor.empty(%dim_n, %dim_m) : tensor<?x?xf16>
    %tensor_d = linalg.generic {
      indexing_maps = [#full_access_map, #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "scale_mul"
    } ins(%tensor_c, %scale : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_d : tensor<?x?xf16>) {
    ^bb0(%c_val: f16, %s_val: f16, %d_out: f16):
      %prod = arith.mulf %c_val, %s_val : f16
      linalg.yield %prod : f16
    } -> tensor<?x?xf16>

    return %tensor_d : tensor<?x?xf16>
  }
}
