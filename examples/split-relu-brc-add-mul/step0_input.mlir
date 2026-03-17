// ============================================================
// STAGE 0: High-Level IR - 高层抽象表示
// 计算图：分割(Split) + 逐元素运算(Elementwise) + 广播(Broadcast)
//
// 数据流：
//   Split:  input_a[:, 0:N/2]  → a0[M, N/2]   （尾轴 split，零开销 view）
//           input_a[:, N/2:N]  → a1[M, N/2]
//   Op1a: relu(a0) + broadcast_row(bias[M]) * broadcast_col(scale0[N/2]) → out0[M, N/2]
//   Op1b: relu(a1) + broadcast_row(bias[M]) * broadcast_col(scale1[N/2]) → out1[M, N/2]
//
// 设计原则（面向 AscendNPU 优化）：
//   - Split 移到最前，对 input_a 做 extract_slice（仅指针偏移，无拷贝）
//   - 每条链迭代空间均为 [M, N/2]，linalg-fuse-elementwise-ops 可将
//     relu + brc_add + brc_mul 融合为单个 generic
//   - 无中间 [M, N] 完整 buffer，输出直接是 out0/out1
//   - scale 分为 scale0[N/2] / scale1[N/2]，各对应一条链
//   - 可扩展到 split=K 份：K 个 extract_slice + K 条独立融合链
//
// 迭代器类型（每条链）：[Parallel, Parallel]，d0=M, d1=N/2
//   relu:     full × full
//   brc_add:  full × row_brc × full  (bias[M] 沿 N/2 轴广播)
//   brc_mul:  full × col_brc × full  (scale[N/2] 沿 M 轴广播)
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
      %input_a : tensor<?x?xf16>,   // [M, N]   - 主输入
      %bias    : tensor<?xf16>,     // [M]      - 行方向 bias（两条链共享）
      %scale0  : tensor<?xf16>,     // [N/2]    - 前半列 scale
      %scale1  : tensor<?xf16>      // [N/2]    - 后半列 scale
  ) -> (tensor<?x?xf16>, tensor<?x?xf16>) {

    %c0     = arith.constant 0 : index
    %c1     = arith.constant 1 : index
    %zero   = arith.constant 0.0 : f16

    %dim_m  = tensor.dim %input_a, %c0 : tensor<?x?xf16>
    %dim_n  = tensor.dim %input_a, %c1 : tensor<?x?xf16>
    %dim_hn = tensor.dim %scale0,  %c0 : tensor<?xf16>    // N/2

    // ── Split: input_a[M,N] → a0[M,N/2] + a1[M,N/2] ─────────────
    // 纯 view，bufferize 后为 GM subview，零开销
    %a0 = tensor.extract_slice %input_a[0, 0      ][%dim_m, %dim_hn][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %a1 = tensor.extract_slice %input_a[0, %dim_hn][%dim_m, %dim_hn][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>

    // ── 链0：relu(a0) → brc_add(bias) → brc_mul(scale0) → out0 ───
    // Op relu0
    %empty_r0 = tensor.empty(%dim_m, %dim_hn) : tensor<?x?xf16>
    %relu0 = linalg.generic {
      indexing_maps = [#full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0 : tensor<?x?xf16>)
      outs(%empty_r0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      %v = arith.maximumf %in, %zero : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    // Op brc_add0
    %empty_c0 = tensor.empty(%dim_m, %dim_hn) : tensor<?x?xf16>
    %add0 = linalg.generic {
      indexing_maps = [#full_access_map, #row_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%relu0, %bias : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_c0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %b: f16, %out: f16):
      %v = arith.addf %in, %b : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    // Op brc_mul0
    %empty_out0 = tensor.empty(%dim_m, %dim_hn) : tensor<?x?xf16>
    %out0 = linalg.generic {
      indexing_maps = [#full_access_map, #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%add0, %scale0 : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_out0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %s: f16, %out: f16):
      %v = arith.mulf %in, %s : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    // ── 链1：relu(a1) → brc_add(bias) → brc_mul(scale1) → out1 ───
    // Op relu1
    %empty_r1 = tensor.empty(%dim_m, %dim_hn) : tensor<?x?xf16>
    %relu1 = linalg.generic {
      indexing_maps = [#full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%a1 : tensor<?x?xf16>)
      outs(%empty_r1 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      %v = arith.maximumf %in, %zero : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    // Op brc_add1
    %empty_c1 = tensor.empty(%dim_m, %dim_hn) : tensor<?x?xf16>
    %add1 = linalg.generic {
      indexing_maps = [#full_access_map, #row_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%relu1, %bias : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_c1 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %b: f16, %out: f16):
      %v = arith.addf %in, %b : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    // Op brc_mul1
    %empty_out1 = tensor.empty(%dim_m, %dim_hn) : tensor<?x?xf16>
    %out1 = linalg.generic {
      indexing_maps = [#full_access_map, #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%add1, %scale1 : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_out1 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %s: f16, %out: f16):
      %v = arith.mulf %in, %s : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    return %out0, %out1 : tensor<?x?xf16>, tensor<?x?xf16>
  }
}
