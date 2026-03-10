// ============================================================
// STAGE 0: High-Level IR - 高层抽象表示
// 计算图：逐元素运算(Elementwise) + 广播(Broadcast) + 分割(Split)
//
// 数据流：
//   Op1: relu(A[M,N]) + broadcast_row(bias[M]) → C[M,N]
//        （逐元素 ReLU + 行方向广播加 bias）
//   Op2: split0[M, N/2] = C[:, 0:N/2] * broadcast_col(scale0[N/2])
//        （从 C 取前半列，乘以列方向广播的 scale）
//   Op3: split1[M, N/2] = C[:, N/2:N] * broadcast_col(scale1[N/2])
//        （从 C 取后半列，乘以列方向广播的 scale）
//
// 注意：这是一个完全符号化的表示，M 和 N 是动态维度
//
// Op1 的迭代器类型：[Parallel, Parallel]
//   - bias[M] 用行广播映射 (d0,d1)->d0，实现沿 d1（N）轴广播
//   - relu 通过 arith.maximumf 与零实现
//
// Op2、Op3 的迭代器类型：[Parallel, Parallel]
//   - 输入 C 通过 tensor.extract_slice 取对应半列（half-N slice）
//   - scale[N/2] 用列广播映射 (d0,d1)->d1，实现沿 d0（M）轴广播
//   - 这与 add-broadcast-concat 的行广播对称，测试两种广播方向
//
// Split 语义：通过 tensor.extract_slice 从中间 tensor C 取 slice 作为 ins
//   - 与 add-broadcast-concat 中 Concat 的 insert_slice outs 对称
//   - bufferization 后 extract_slice → memref.subview（只读），无中间 alloc
//
// library_call 属性用于区分 Op2 和 Op3，以便 Transform 脚本分别匹配
// ============================================================
// RUN: afir-opt %s | FileCheck %s
// CHECK: func.func @ewop_broadcast_split

// 行广播映射：1D 输入(d0,d1)只访问 d0，实现沿 d1（N）轴广播
#row_broadcast_map = affine_map<(d0, d1) -> (d0)>
// 列广播映射：1D 输入(d0,d1)只访问 d1，实现沿 d0（M）轴广播
#col_broadcast_map = affine_map<(d0, d1) -> (d1)>
// 完整访问映射：输入(d0,d1)访问对应位置
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @ewop_broadcast_split(
      %input_a  : tensor<?x?xf16>,   // [M, N] - 主输入，做 ReLU
      %bias     : tensor<?xf16>,     // [M]    - 行方向 bias
      %scale0   : tensor<?xf16>,     // [N/2]  - 前半列 scale
      %scale1   : tensor<?xf16>      // [N/2]  - 后半列 scale
  ) -> (tensor<?x?xf16>, tensor<?x?xf16>) {

    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index

    %dim_m  = tensor.dim %input_a, %idx_0 : tensor<?x?xf16>
    %dim_n  = tensor.dim %input_a, %idx_1 : tensor<?x?xf16>
    %dim_hn = tensor.dim %scale0, %idx_0  : tensor<?xf16>   // N/2

    // ── Op1: relu(A) + broadcast_row(bias) → C[M, N] ─────────────
    // 迭代器类型: [Parallel, Parallel]
    // bias 只访问 d0，实现沿 N 轴广播
    // relu = max(x, 0)，用 arith.maximumf 实现
    %zero_f16 = arith.constant 0.0 : f16
    %empty_c = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %tensor_c = linalg.generic {
      indexing_maps = [#full_access_map, #row_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%input_a, %bias : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_c : tensor<?x?xf16>) {
    ^bb0(%a_val: f16, %b_val: f16, %c_out: f16):
      %relu_a = arith.maximumf %a_val, %zero_f16 : f16
      %result  = arith.addf %relu_a, %b_val : f16
      linalg.yield %result : f16
    } -> tensor<?x?xf16>

    // ── Op2: C[:, 0:N/2] * broadcast_col(scale0) → out0[M, N/2] ──
    // Split 前半列：extract_slice 取 C 的 [0:M, 0:N/2]
    // scale0[N/2] 通过列广播映射扩展到 [M, N/2]
    // library_call = "split_scale0" 用于 Transform 脚本属性匹配
    %c_slice0 = tensor.extract_slice %tensor_c[0, 0][%dim_m, %dim_hn][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %empty_out0 = tensor.empty(%dim_m, %dim_hn) : tensor<?x?xf16>
    %out0 = linalg.generic {
      indexing_maps = [#full_access_map, #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "split_scale0"
    } ins(%c_slice0, %scale0 : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_out0 : tensor<?x?xf16>) {
    ^bb0(%c_val: f16, %s_val: f16, %o_out: f16):
      %prod = arith.mulf %c_val, %s_val : f16
      linalg.yield %prod : f16
    } -> tensor<?x?xf16>

    // ── Op3: C[:, N/2:N] * broadcast_col(scale1) → out1[M, N/2] ──
    // Split 后半列：extract_slice 取 C 的 [0:M, N/2:N]
    // scale1[N/2] 通过列广播映射扩展到 [M, N/2]
    // library_call = "split_scale1" 用于 Transform 脚本属性匹配
    %c_slice1 = tensor.extract_slice %tensor_c[0, %dim_hn][%dim_m, %dim_hn][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %empty_out1 = tensor.empty(%dim_m, %dim_hn) : tensor<?x?xf16>
    %out1 = linalg.generic {
      indexing_maps = [#full_access_map, #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "split_scale1"
    } ins(%c_slice1, %scale1 : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_out1 : tensor<?x?xf16>) {
    ^bb0(%c_val: f16, %s_val: f16, %o_out: f16):
      %prod = arith.mulf %c_val, %s_val : f16
      linalg.yield %prod : f16
    } -> tensor<?x?xf16>

    return %out0, %out1 : tensor<?x?xf16>, tensor<?x?xf16>
  }
}
