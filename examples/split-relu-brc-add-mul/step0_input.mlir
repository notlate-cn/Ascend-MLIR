// ============================================================
// STAGE 0: High-Level IR - 高层抽象表示
// 计算图：行分割(Row-Split) + 逐元素运算 + 广播 + 行合并(Row-Concat)
//
// 数据流：
//   Split:  input_a[0:M/2, :]  → a0[M/2, N]  （头半行，零开销 view）
//           input_a[M/2:M, :]  → a1[M/2, N]  （尾半行，零开销 view）
//   Op0: relu(a0) + broadcast_row(bias0[M/2]) * broadcast_col(scale0[N]) → out0[M/2, N]
//   Op1: relu(a1) + broadcast_row(bias1[M/2]) * broadcast_col(scale1[N]) → out1[M/2, N]
//   Concat: concat dim(0) [out0, out1] → output[M, N]
//           (bufferize 后 epilogue data_copy 直写 output 对应行区间，零开销)
//
// 设计原则（面向 AscendNPU 优化）：
//   - 行分割：两条链各处理 M/2 行，迭代空间 [M/2, N]（与 add-broadcast-concat 对称）
//   - bias 分为 bias0[M/2] / bias1[M/2]，各对应一条链（行广播）
//   - scale 分为 scale0[N] / scale1[N]，各对应一条链（列广播）
//   - concat dim(0) + decompose_concat：epilogue data_copy 直写 output 行区间，零 memcpy
//
// 迭代器类型（每条链）：[Parallel, Parallel]，d0=M/2, d1=N（N 须为 16 的倍数）
//   relu:     full × full
//   brc_add:  row_brc × full × full  (bias[M/2] 沿 N 轴广播)
//   brc_mul:  col_brc × full × full  (scale[N] 沿 M/2 轴广播)
// ============================================================
// RUN: afir-opt %s | FileCheck %s
// CHECK: func.func @ewop_broadcast_split

// 行广播映射：1D 输入只访问 d0，沿 d1 轴广播
#row_broadcast_map = affine_map<(d0, d1) -> (d0)>
// 列广播映射：1D 输入只访问 d1，沿 d0 轴广播
#col_broadcast_map = affine_map<(d0, d1) -> (d1)>
// 完整访问映射
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @ewop_broadcast_split(
      %input_a : tensor<?x?xf16>,   // [M, N]    - 主输入
      %bias0   : tensor<?xf16>,     // [M/2]     - 链0 行方向 bias
      %bias1   : tensor<?xf16>,     // [M/2]     - 链1 行方向 bias
      %scale0  : tensor<?xf16>,     // [N]       - 链0 列方向 scale
      %scale1  : tensor<?xf16>      // [N]       - 链1 列方向 scale
  ) -> tensor<?x?xf16> {

    %c0     = arith.constant 0 : index
    %c1     = arith.constant 1 : index
    %zero   = arith.constant 0.0 : f16

    %dim_m  = tensor.dim %input_a, %c0 : tensor<?x?xf16>
    %dim_n  = tensor.dim %input_a, %c1 : tensor<?x?xf16>
    %dim_hm = tensor.dim %bias0,   %c0 : tensor<?xf16>    // M/2

    // ── Split: input_a[M,N] → a0[M/2,N] + a1[M/2,N] ─────────────
    // 纯 view，bufferize 后为 GM subview，零开销
    %a0 = tensor.extract_slice %input_a[0,       0][%dim_hm, %dim_n][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %a1 = tensor.extract_slice %input_a[%dim_hm, 0][%dim_hm, %dim_n][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>

    // ── 链0：relu(a0) → brc_add(bias0) → brc_mul(scale0) → out0 ──
    %empty0 = tensor.empty(%dim_hm, %dim_n) : tensor<?x?xf16>
    %out0 = linalg.generic {
      indexing_maps = [#full_access_map, #row_broadcast_map,
                       #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %bias0, %scale0 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>)
      outs(%empty0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %b: f16, %s: f16, %out: f16):
      %r = arith.maximumf %in, %zero : f16
      %a = arith.addf %r, %b : f16
      %v = arith.mulf %a, %s : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    // ── 链1：relu(a1) → brc_add(bias1) → brc_mul(scale1) → out1 ──
    %empty1 = tensor.empty(%dim_hm, %dim_n) : tensor<?x?xf16>
    %out1 = linalg.generic {
      indexing_maps = [#full_access_map, #row_broadcast_map,
                       #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%a1, %bias1, %scale1 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>)
      outs(%empty1 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %b: f16, %s: f16, %out: f16):
      %r = arith.maximumf %in, %zero : f16
      %a = arith.addf %r, %b : f16
      %v = arith.mulf %a, %s : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    // ── Concat dim(0): out0[M/2,N] + out1[M/2,N] → output[M,N] ──
    // decompose_concat → insert_slice，epilogue data_copy 直写 output 行区间，零 memcpy
    %output = tensor.concat dim(0) %out0, %out1
        : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>

    return %output : tensor<?x?xf16>
  }
}
