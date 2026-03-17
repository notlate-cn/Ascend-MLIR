// ============================================================
// STAGE 2: Transform Dialect Tiling - 使用变换方言进行分块
//
// 输入：step1_fused.mlir（linalg-fuse-elementwise-ops 后）
//
// 融合结果：两条独立链各自融合为单个 linalg.generic
//   链0: relu(a0) + brc_add(bias) + brc_mul(scale0) → out0[M, N/2]
//   链1: relu(a1) + brc_add(bias) + brc_mul(scale1) → out1[M, N/2]
//   indexing_maps = [full, row_brc, col_brc, full]
//
// Split 已在 step0 消除：a0/a1 是 input_a 的 GM subview，无拷贝。
// 两条链的输出直接是 out0/out1，无中间 [M,N] buffer。
//
// Tiling 策略：
//   - d0=M：两级切分，TB 层（核间并行）+ Tb 层（UB 批次）
//   - d1=N/2：不切，整列进 UB
//
// 两条链迭代器相同，transform.foreach 统一处理。
//
// RUN: afir-opt --transform-interpreter %s --canonicalize --cse | FileCheck %s
// CHECK: scf.for
// CHECK: ascendc.parallel
// ============================================================

#row_broadcast_map = affine_map<(d0, d1) -> (d0)>
#col_broadcast_map = affine_map<(d0, d1) -> (d1)>
#full_access_map   = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  // ==========================================================
  // 主计算函数（step1 融合后）
  // ==========================================================
  func.func @ewop_broadcast_split(
      %input_a : tensor<?x?xf16>,
      %bias    : tensor<?xf16>,
      %scale0  : tensor<?xf16>,
      %scale1  : tensor<?xf16>
  ) -> (tensor<?x?xf16>, tensor<?x?xf16>) {

    %c0    = arith.constant 0 : index
    %zero  = arith.constant 0.0 : f16
    %dim_m = tensor.dim %input_a, %c0 : tensor<?x?xf16>
    %dim_hn = tensor.dim %scale0, %c0 : tensor<?xf16>

    %a0 = tensor.extract_slice %input_a[0, 0       ][%dim_m, %dim_hn][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %a1 = tensor.extract_slice %input_a[0, %dim_hn ][%dim_m, %dim_hn][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>

    %empty0 = tensor.empty(%dim_m, %dim_hn) : tensor<?x?xf16>
    %out0 = linalg.generic {
      indexing_maps = [#full_access_map, #row_broadcast_map,
                       #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %bias, %scale0 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>)
      outs(%empty0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %b: f16, %s: f16, %out: f16):
      %r = arith.maximumf %in, %zero : f16
      %a = arith.addf %r, %b : f16
      %v = arith.mulf %a, %s : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    %empty1 = tensor.empty(%dim_m, %dim_hn) : tensor<?x?xf16>
    %out1 = linalg.generic {
      indexing_maps = [#full_access_map, #row_broadcast_map,
                       #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%a1, %bias, %scale1 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>)
      outs(%empty1 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %b: f16, %s: f16, %out: f16):
      %r = arith.maximumf %in, %zero : f16
      %a = arith.addf %r, %b : f16
      %v = arith.mulf %a, %s : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    return %out0, %out1 : tensor<?x?xf16>, tensor<?x?xf16>
  }

  // ==========================================================
  // Transform 调度脚本
  // ==========================================================
  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}
  ) {
    // ---- Step 1: 追加 2 个 index 参数（TB_M, Tb_M）----
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op

    %func_new, %tb_m_param, %tb_inner_m_param =
        transform.func.add_index_args %func, 2
            : (!transform.any_op)
            -> (!transform.any_op,
                !transform.any_op,
                !transform.any_op)

    // ---- Step 2: 匹配两个 linalg.generic，统一 tiling ----
    %all_generics = transform.structured.match ops{["linalg.generic"]}
        in %func_new : (!transform.any_op) -> !transform.any_op

    // ---- 公共标注参数 ----
    %true_param       = transform.param.constant true -> !transform.any_param
    %prologue_param   = transform.param.constant "src:GM->VECIN"  -> !transform.any_param
    %epilogue_param   = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %vector_unit_param = transform.param.constant "AiCore.Vector" -> !transform.any_param

    // ══════════════ 两条链统一两级 tiling ══════════════
    transform.foreach %all_generics : !transform.any_op {
    ^bb0(%op : !transform.any_op):
      // TB 层：d0=M 切 TB_M，d1=N/2 不切
      %op_tb, %loop_tb =
          transform.structured.tile_using_for %op
              tile_sizes [%tb_m_param, 0]
                  : (!transform.any_op, !transform.any_op)
              -> (!transform.any_op, !transform.any_op)

      transform.annotate %loop_tb "ascendc.parallel"
          = %true_param : !transform.any_op, !transform.any_param

      // Tb 层：UB 批次
      %op_inner, %loop_inner =
          transform.structured.tile_using_for %op_tb
              tile_sizes [%tb_inner_m_param, 0]
                  : (!transform.any_op, !transform.any_op)
              -> (!transform.any_op, !transform.any_op)

      transform.annotate %loop_inner "ascendc.prologue"
          = %prologue_param : !transform.any_op, !transform.any_param
      transform.annotate %loop_inner "ascendc.epilogue"
          = %epilogue_param : !transform.any_op, !transform.any_param
      transform.annotate %op_inner "ascendc.unit"
          = %vector_unit_param : !transform.any_op, !transform.any_param

      transform.loop.hoist_loop_invariant_subsets %loop_inner
          : !transform.any_op
    }

    transform.yield
  }
}
