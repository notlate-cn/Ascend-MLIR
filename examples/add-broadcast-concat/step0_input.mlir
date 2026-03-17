// ============================================================
// STAGE 0: High-Level IR - 高层抽象表示
// 计算图：逐元素运算(Elementwise) + 广播(Broadcast) + 拼接(Concat)
//
// 数据流：
//   Op1: input_a[M] + input_b[M,N] → C[M,N]   （广播加法）
//   Op2: input_c[M] * input_d[M,N] → D[M,N]   （广播乘法）
//   tensor.concat dim(0) (C, D) → output[2M,N] （拼接）
//
// 注意：这是一个完全符号化的表示，M 和 N 是动态维度
//
// 设计原则（面向 AscendNPU 优化）：
//   - 用 tensor.concat 直接表达语义，与 stablehlo→linalg(enablePrimitiveOps)
//     的输出路径一致，语义清晰，便于上层框架直接生成
//   - tensor.concat 由 step2 Transform sequence 通过
//     transform.apply_patterns.tensor.decompose_concat 展开为
//     empty + insert_slice 链，转为 tiling 友好的形式
//   - axis=0（首轴）concat 展开后内存连续，DMA 友好
//
// Op1 和 Op2 的迭代器类型：
//   - [Parallel, Parallel] - 两个维度都是并行
//   1D 输入使用 broadcast_map（只访问 d0），实现沿 d1 轴的隐式广播
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

    %dim_m = tensor.dim %input_a, %idx_0 : tensor<?xf16>
    %dim_n = tensor.dim %input_b, %idx_1 : tensor<?x?xf16>

    // ── Op1: input_a[M] + input_b[M,N] → C[M,N] ─────────────
    %init_c = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %result_c = linalg.generic {
      indexing_maps = [#broadcast_map, #full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%input_a, %input_b : tensor<?xf16>, tensor<?x?xf16>)
      outs(%init_c : tensor<?x?xf16>) {
    ^bb0(%a_val: f16, %b_val: f16, %c_out: f16):
      %sum = arith.addf %a_val, %b_val : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>


    // ── Op2: input_c[M] * input_d[M,N] → D[M,N] ─────────────
    %init_d = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %result_d = linalg.generic {
      indexing_maps = [#broadcast_map, #full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%input_c, %input_d : tensor<?xf16>, tensor<?x?xf16>)
      outs(%init_d : tensor<?x?xf16>) {
    ^bb0(%c_val: f16, %d_val: f16, %e_out: f16):
      %prod = arith.mulf %c_val, %d_val : f16
      linalg.yield %prod : f16
    } -> tensor<?x?xf16>


    // ── Concat: [C; D] → output[2M, N] ───────────────────────
    // tensor.concat 由 step2 transform.apply_patterns.tensor.decompose_concat
    // 展开为 empty[2M,N] + insert_slice(C, 0..M) + insert_slice(D, M..2M)
    %output = tensor.concat dim(0) %result_c, %result_d
        : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>

    return %output : tensor<?x?xf16>
  }
}
