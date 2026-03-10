// ============================================================
// STAGE 0: High-Level IR - 高层抽象表示
// 计算图：逐元素运算(Elementwise) + 广播(Broadcast) + 拼接(Concat)
//
// 数据流：
//   Op1: input_a[M] + input_b[M,N] → C[M,N]  （广播加法）
//   Op2: input_c[M] * input_d[M,N] → D[M,N]  （广播乘法）
//   Concat(C, D, axis=0) → output[2M, N]       （沿行方向拼接）
//
// 注意：这是一个完全符号化的表示，M 和 N 是动态维度
//
// Op1 和 Op2 的迭代器类型：
//   - [Parallel, Parallel] - 两个维度都是并行
//   1D 输入使用 broadcast_map（只访问 d0），实现沿 d1 轴的隐式广播
//
// Concat 通过预先创建 output[2M,N]，Op1/Op2 分别写入对应 slice：
//   Op1 outs = extract_slice output[0:M, :]   → 写入 output 前半
//   Op2 outs = extract_slice output[M:2M, :]  → 写入 output 后半
// bufferization 后 extract_slice/insert_slice 均 in-place，无中间 alloc
//
// library_call 属性用于区分 Op1 和 Op2，以便 Transform 脚本分别匹配
// ============================================================
// RUN: afir-opt %s | FileCheck %s
// CHECK: func.func @ewop_broadcast_concat

// 广播映射：1D 输入(d0,d1)只访问 d0，实现沿 d1 轴广播
#broadcast_map = affine_map<(d0, d1) -> (d0)>
// 完整访问映射：输入(d0,d1)访问对应位置
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @ewop_broadcast_concat(
      %input_a : tensor<?xf16>,      // [M] - 广播加法的 1D 输入
      %input_b : tensor<?x?xf16>,    // [M, N] - 广播加法的 2D 输入
      %input_c : tensor<?xf16>,      // [M] - 广播乘法的 1D 输入
      %input_d : tensor<?x?xf16>     // [M, N] - 广播乘法的 2D 输入
  ) -> tensor<?x?xf16> {

    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index
    %c2    = arith.constant 2 : index

    %dim_m = tensor.dim %input_a, %idx_0 : tensor<?xf16>
    %dim_n = tensor.dim %input_b, %idx_1 : tensor<?x?xf16>

    // ── 预先创建输出 tensor[2M, N]，Op1/Op2 原地写入各自的 slice ───
    %dim_2m = arith.muli %dim_m, %c2 : index
    %empty_out = tensor.empty(%dim_2m, %dim_n) : tensor<?x?xf16>

    // ── Op1: 广播加法 input_a[M] + input_b[M,N] → output[0:M, :] ──
    // extract_slice 作为 outs：bufferization 后 in-place 写入 output 前半
    // library_call = "broadcast_add" 用于 Transform 脚本中的属性匹配
    %slice_c = tensor.extract_slice %empty_out[0, 0][%dim_m, %dim_n][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %tensor_c = linalg.generic {
      indexing_maps = [#broadcast_map, #full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "broadcast_add"
    } ins(%input_a, %input_b : tensor<?xf16>, tensor<?x?xf16>)
      outs(%slice_c : tensor<?x?xf16>) {
    ^bb0(%a_val: f16, %b_val: f16, %c_out: f16):
      %sum = arith.addf %a_val, %b_val : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>
    %out_with_c = tensor.insert_slice %tensor_c into %empty_out
        [0, 0] [%dim_m, %dim_n] [1, 1]
        : tensor<?x?xf16> into tensor<?x?xf16>

    // ── Op2: 广播乘法 input_c[M] * input_d[M,N] → output[M:2M, :] ─
    // extract_slice 作为 outs：bufferization 后 in-place 写入 output 后半
    // library_call = "broadcast_mul" 用于 Transform 脚本中的属性匹配
    %slice_d = tensor.extract_slice %out_with_c[%dim_m, 0][%dim_m, %dim_n][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %tensor_d = linalg.generic {
      indexing_maps = [#broadcast_map, #full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "broadcast_mul"
    } ins(%input_c, %input_d : tensor<?xf16>, tensor<?x?xf16>)
      outs(%slice_d : tensor<?x?xf16>) {
    ^bb0(%c_val: f16, %d_val: f16, %e_out: f16):
      %prod = arith.mulf %c_val, %d_val : f16
      linalg.yield %prod : f16
    } -> tensor<?x?xf16>
    %out_final = tensor.insert_slice %tensor_d into %out_with_c
        [%dim_m, 0] [%dim_m, %dim_n] [1, 1]
        : tensor<?x?xf16> into tensor<?x?xf16>

    return %out_final : tensor<?x?xf16>
  }
}
