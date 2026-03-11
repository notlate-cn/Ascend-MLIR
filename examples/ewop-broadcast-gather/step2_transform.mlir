// ============================================================
// STAGE 2: Transform Dialect Tiling - 使用变换方言进行分块
//
// 本阶段使用 Transform Dialect 描述分块策略：
//   输入：Op1 (gather_by_index) 和 Op2 (broadcast_add_gathered)
//   两个 generic 的迭代器类型均为：["parallel", "parallel"]
//     * d0 = M (Parallel) - 行维度，核间分发
//     * d1 = K (Parallel) - 列维度（gather 输出宽），不切分（整 K 在 UB 处理）
//
// Tiling 结构（仅对 d0=M 进行两级切分）：
//   - TB 层：核间并行，每核负责 TB_M 行（标记 ascendc.parallel）
//   - Tb 层：UB 批次，每批 Tb_M 行（搬运粒度）
//   - d1 (K) 轴：不切分，整 K 在 UB 内处理
//
// 注意：Op1 (gather) 的 data[M,N] 输入很大（N >> K），
//   Tb_M 应设为 1（每次处理一行），以便 gather_l2 处理
//   data 单行 [N] + indices [K] → gathered [K]
//
// Op1 和 Op2 通过 library_call 属性区分，各自独立 tile：
//   - Op1: library_call = "gather_by_index"
//   - Op2: library_call = "broadcast_add_gathered"
//
// 函数签名扩展：追加两个 index 参数 TB_M, Tb_M
//
// RUN: afir-opt --transform-interpreter %s --canonicalize --cse | FileCheck %s
// CHECK: scf.for
// CHECK: ascendc.parallel
// ============================================================

// 行广播映射：bias[M] 沿 K 轴广播，(d0,d1)->d0
#row_broadcast_map = affine_map<(d0, d1) -> (d0)>
// indices 广播映射：indices[K] 沿 M 轴广播，(d0,d1)->d1
#col_broadcast_map = affine_map<(d0, d1) -> (d1)>
// 完整访问映射：访问 (d0, d1)
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  // ==========================================================
  // 主计算函数：Gather + BroadcastAdd
  // ==========================================================
  func.func @ewop_broadcast_gather(
      %data    : tensor<?x?xf16>,
      %indices : tensor<?xi32>,
      %bias    : tensor<?xf16>
  ) -> tensor<?x?xf16> {

    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index

    %dim_m = tensor.dim %data, %idx_0 : tensor<?x?xf16>
    %dim_n = tensor.dim %data, %idx_1 : tensor<?x?xf16>
    %dim_k = tensor.dim %indices, %idx_0 : tensor<?xi32>

    // Op1: Gather
    %empty_gathered = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %gathered = linalg.generic {
      indexing_maps = [#col_broadcast_map, #full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "gather_by_index"
    } ins(%indices, %data : tensor<?xi32>, tensor<?x?xf16>)
      outs(%empty_gathered : tensor<?x?xf16>) {
    ^bb0(%idx_val: i32, %data_val: f16, %out_val: f16):
      linalg.yield %data_val : f16
    } -> tensor<?x?xf16>

    // Op2: BroadcastAdd
    %empty_out = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %out = linalg.generic {
      indexing_maps = [#full_access_map, #row_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "broadcast_add_gathered"
    } ins(%gathered, %bias : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_out : tensor<?x?xf16>) {
    ^bb0(%g_val: f16, %b_val: f16, %o_out: f16):
      %sum = arith.addf %g_val, %b_val : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>

    return %out : tensor<?x?xf16>
  }

  // ==========================================================
  // Transform 调度脚本
  // ==========================================================
  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}
  ) {
    // ---- Step 1: 匹配 func.func，追加 2 个 index 参数 ----
    // 参数: TB_M（核间分块大小），Tb_M（UB 批次大小）
    // 注意：Gather 场景建议 Tb_M = 1（每次处理一行，N 轴扫全部），
    //   这样 gather_l2 处理 data[row, 0..N-1] → gathered[row, 0..K-1]
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op

    %func_new, %tb_m_param, %tb_inner_m_param =
        transform.func.add_index_args %func, 2
            : (!transform.any_op)
            -> (!transform.any_op,
                !transform.any_op,
                !transform.any_op)

    // ---- Step 2: 分别匹配 Op1（gather）和 Op2（broadcast_add）----
    %op1 = transform.structured.match ops{["linalg.generic"]}
        attributes{library_call = "gather_by_index"} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %op2 = transform.structured.match ops{["linalg.generic"]}
        attributes{library_call = "broadcast_add_gathered"} in %func_new
        : (!transform.any_op) -> !transform.any_op

    // ---- 公共标注参数 ----
    %true_param = transform.param.constant true -> !transform.any_param
    %prologue_param = transform.param.constant
        "src:GM->VECIN" -> !transform.any_param
    %epilogue_param = transform.param.constant
        "dst:VECOUT->GM" -> !transform.any_param
    %vector_unit_param = transform.param.constant
        "AiCore.Vector" -> !transform.any_param

    // ══════════════ Op1（gather）Tiling ══════════════

    // TB 层切分（d0=M，tile_size=0 表示 d1=K 不切）
    %op1_tb, %loop_op1_tb =
        transform.structured.tile_using_for %op1
            tile_sizes [%tb_m_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_op1_tb "ascendc.parallel"
        = %true_param : !transform.any_op, !transform.any_param

    // Tb 层切分（UB 批次粒度，Gather 场景建议 Tb_M=1）
    %op1_inner, %loop_op1_inner =
        transform.structured.tile_using_for %op1_tb
            tile_sizes [%tb_inner_m_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_op1_inner "ascendc.prologue"
        = %prologue_param : !transform.any_op, !transform.any_param
    transform.annotate %loop_op1_inner "ascendc.epilogue"
        = %epilogue_param : !transform.any_op, !transform.any_param
    transform.annotate %op1_inner "ascendc.unit"
        = %vector_unit_param : !transform.any_op, !transform.any_param

    transform.loop.hoist_loop_invariant_subsets %loop_op1_inner
        : !transform.any_op

    // ══════════════ Op2（broadcast_add）Tiling ══════════════

    %op2_tb, %loop_op2_tb =
        transform.structured.tile_using_for %op2
            tile_sizes [%tb_m_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_op2_tb "ascendc.parallel"
        = %true_param : !transform.any_op, !transform.any_param

    %op2_inner, %loop_op2_inner =
        transform.structured.tile_using_for %op2_tb
            tile_sizes [%tb_inner_m_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_op2_inner "ascendc.prologue"
        = %prologue_param : !transform.any_op, !transform.any_param
    transform.annotate %loop_op2_inner "ascendc.epilogue"
        = %epilogue_param : !transform.any_op, !transform.any_param
    transform.annotate %op2_inner "ascendc.unit"
        = %vector_unit_param : !transform.any_op, !transform.any_param

    transform.loop.hoist_loop_invariant_subsets %loop_op2_inner
        : !transform.any_op

    transform.yield
  }
}
