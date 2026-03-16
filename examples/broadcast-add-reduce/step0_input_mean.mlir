// ============================================================
// STAGE 0: High-Level IR - 高层抽象表示
// 计算图：广播(Broadcast) + 加法(Add) + 归约求平均(ReduceMean)
//
// 数据流：
//   输入A[M] → Broadcast → C[M,N]
//   C[M,N] + B[M,N] → D[M,N] (逐元素加法)
//   D[M,N] → ReduceMean(axis=1) → 输出E[M] (按行求平均)
//
// 注意：ReduceMean = ReduceSum / count
// MLIR中没有直接的reduce_mean op，需要分两步：
//   1. ReduceSum累加
//   2. 除法：sum / N (N为归约轴长度)
// ============================================================
// RUN: afir-opt %s | FileCheck %s
// CHECK: func.func @broadcast_add_reducemean

#broadcast_map = affine_map<(d0, d1) -> (d0)>
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @broadcast_add_reducemean(
      %input_a : tensor<?xf16>,      // [M] - 一维输入，将被广播
      %input_b : tensor<?x?xf16>     // [M, N] - 二维输入
  ) -> tensor<?xf16> {

    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index

    %dim_m = tensor.dim %input_a, %idx_0 : tensor<?xf16>       // M维度大小
    %dim_n = tensor.dim %input_b, %idx_1 : tensor<?x?xf16>     // N维度大小

    // ── 算子1: Broadcast - 将A[M]广播为C[M,N] ─────────────────────
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

    // ── 算子3: ReduceMean(axis=1) - 按行求平均 D[M,N] → E[M] ─────────────
    // 步骤3a: ReduceSum累加
    %zero_value = arith.constant 0.0 : f16
    %empty_tensor_sum = tensor.empty(%dim_m) : tensor<?xf16>
    %init_tensor_sum = linalg.fill ins(%zero_value : f16)
                      outs(%empty_tensor_sum : tensor<?xf16>) -> tensor<?xf16>
    %tensor_sum = linalg.generic {
      indexing_maps = [#full_access_map, #broadcast_map],
      iterator_types = ["parallel", "reduction"]
    } ins(%tensor_d : tensor<?x?xf16>)
      outs(%init_tensor_sum : tensor<?xf16>) {
    ^bb0(%d_value: f16, %accumulator: f16):
      %new_accumulator = arith.addf %accumulator, %d_value : f16
      linalg.yield %new_accumulator : f16
    } -> tensor<?xf16>

    // 步骤3b: 除以N得到平均值 mean = sum / N
    %dim_n_i32 = arith.index_cast %dim_n : index to i32
    %dim_n_f16 = arith.sitofp %dim_n_i32 : i32 to f16
    %empty_tensor_mean = tensor.empty(%dim_m) : tensor<?xf16>
    %tensor_mean = linalg.generic {
      indexing_maps = [#broadcast_map, #broadcast_map],
      iterator_types = ["parallel"]
    } ins(%tensor_sum : tensor<?xf16>)
      outs(%empty_tensor_mean : tensor<?xf16>) {
    ^bb0(%sum_value: f16, %output: f16):
      %mean_value = arith.divf %sum_value, %dim_n_f16 : f16
      linalg.yield %mean_value : f16
    } -> tensor<?xf16>

    return %tensor_mean : tensor<?xf16>
  }
}