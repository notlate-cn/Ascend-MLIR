// ============================================================
// STAGE 0: High-Level IR
// 计算图：A[M] → Broadcast → C[M,N]
//          C[M,N] + B[M,N] → D[M,N]   (Add)
//          D[M,N] → ReduceSum(axis=1) → E[M]
//
// 完全符号化，M/N 是动态 Shape。
// 三个 Op：Broadcast（Parallel/Parallel）
//          Add        （Parallel/Parallel）
//          ReduceSum  （Parallel/Reduce）
// ============================================================
// RUN: afir-opt %s | FileCheck %s
// CHECK: func.func @broadcast_add_reducesum

func.func @broadcast_add_reducesum(
    %A : tensor<?xf16>,      // [M]
    %B : tensor<?x?xf16>     // [M, N]
) -> tensor<?xf16> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M  = tensor.dim %A, %c0 : tensor<?xf16>
  %N  = tensor.dim %B, %c1 : tensor<?x?xf16>

  // ── Op1: Broadcast  A[M] → C[M,N] ─────────────────────
  // iterator: [Parallel, Parallel]
  // d1 被广播（A 的 indexing_map 不含 d1）
  %empty_C = tensor.empty(%M, %N) : tensor<?x?xf16>
  %C = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0)>,      // A：只用 d0
      affine_map<(d0, d1) -> (d0, d1)>   // C：输出
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%A : tensor<?xf16>)
    outs(%empty_C : tensor<?x?xf16>) {
  ^bb0(%a_val: f16, %c_out: f16):
    linalg.yield %a_val : f16
  } -> tensor<?x?xf16>

  // ── Op2: Add  D[M,N] = C[M,N] + B[M,N] ────────────────
  // iterator: [Parallel, Parallel]
  %empty_D = tensor.empty(%M, %N) : tensor<?x?xf16>
  %D = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%C, %B : tensor<?x?xf16>, tensor<?x?xf16>)
    outs(%empty_D : tensor<?x?xf16>) {
  ^bb0(%c_val: f16, %b_val: f16, %d_out: f16):
    %sum = arith.addf %c_val, %b_val : f16
    linalg.yield %sum : f16
  } -> tensor<?x?xf16>

  // ── Op3: ReduceSum(axis=1)  E[M] = sum_j D[i,j] ────────
  // iterator: [Parallel, Reduction]  ← d1 从 Parallel 变成 Reduction
  // → 轴分组与 Op1/Op2 不一致，无法全局融合
  %zero    = arith.constant 0.0 : f16
  %empty_E = tensor.empty(%M) : tensor<?xf16>
  %init_E  = linalg.fill ins(%zero : f16)
               outs(%empty_E : tensor<?xf16>) -> tensor<?xf16>
  %E = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,  // D：输入
      affine_map<(d0, d1) -> (d0)>        // E：输出，d1 被规约
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%D : tensor<?x?xf16>)
    outs(%init_E : tensor<?xf16>) {
  ^bb0(%d_val: f16, %acc: f16):
    %new_acc = arith.addf %acc, %d_val : f16
    linalg.yield %new_acc : f16
  } -> tensor<?xf16>

  return %E : tensor<?xf16>
}
