// ============================================================
// STAGE 2: Transform Dialect Tiling
//
// 输入：step1_fused.mlir（linalg-fuse-elementwise-ops + decompose_concat 后）
//
// 融合结果：两条链各融合为单个 linalg.generic + concat dim(0) 行合并
//   链0: relu(a0[M/2,N]) + brc_add(bias0[M/2]) + brc_mul(scale0[N]) → out0[M/2,N]
//   链1: relu(a1[M/2,N]) + brc_add(bias1[M/2]) + brc_mul(scale1[N]) → out1[M/2,N]
//   concat dim(0): → output[M,N]
//
// Tiling 策略：
//   - d0=M/2：两级切分，TB 层（核间并行，ascendc.parallel）+ Tb 层（UB 批次）
//   - d1=N：不切，整行进 UB
//
// Transform 步骤：
//   Step 0: decompose_concat dim(0) → insert_slice，使 epilogue 直写 output 行区间
//   Step 1: 追加 2 个 index 参数（TB_M, Tb_M）
//   Step 2: 对两个 generic 统一两级 tiling
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
  // 直接使用 step1_fused.mlir 的 IR 表示（两个 generic + tensor.concat dim(0)）
  func.func @ewop_broadcast_split(
      %arg0: tensor<?x?xf16>,   // input_a [M,N]
      %arg1: tensor<?xf16>,     // bias0   [M/2]
      %arg2: tensor<?xf16>,     // bias1   [M/2]
      %arg3: tensor<?xf16>,     // scale0  [N]
      %arg4: tensor<?xf16>      // scale1  [N]
  ) -> tensor<?x?xf16> {
    %c0    = arith.constant 0 : index
    %c1    = arith.constant 1 : index
    %zero  = arith.constant 0.0 : f16
    %dim_n  = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %dim_hm = tensor.dim %arg1, %c0 : tensor<?xf16>   // M/2

    %a0 = tensor.extract_slice %arg0[0,       0][%dim_hm, %dim_n][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %a1 = tensor.extract_slice %arg0[%dim_hm, 0][%dim_hm, %dim_n][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>

    %empty0 = tensor.empty(%dim_hm, %dim_n) : tensor<?x?xf16>
    %out0 = linalg.generic {
      indexing_maps = [#full_access_map, #row_broadcast_map,
                       #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %arg1, %arg3 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>)
      outs(%empty0 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %b: f16, %s: f16, %out: f16):
      %r = arith.maximumf %in, %zero : f16
      %a = arith.addf %r, %b : f16
      %v = arith.mulf %a, %s : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>
    %empty1 = tensor.empty(%dim_hm, %dim_n) : tensor<?x?xf16>
    %out1 = linalg.generic {
      indexing_maps = [#full_access_map, #row_broadcast_map,
                       #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%a1, %arg2, %arg4 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>)
      outs(%empty1 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %b: f16, %s: f16, %out: f16):
      %r = arith.maximumf %in, %zero : f16
      %a = arith.addf %r, %b : f16
      %v = arith.mulf %a, %s : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    %output = tensor.concat dim(0) %out0, %out1
        : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
    return %output : tensor<?x?xf16>
  }

  // ==========================================================
  // Transform 调度脚本
  // ==========================================================
  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}
  ) {
    // ---- Step 0: 展开 tensor.concat dim(0) ----
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.tensor.decompose_concat
    } : !transform.any_op

    // ---- Step 1: 追加 2 个 index 参数（TB_M, Tb_M）----
    %func_new, %tb_m_param, %tb_inner_m_param =
        transform.func.add_index_args %func, 2
            : (!transform.any_op)
            -> (!transform.any_op,
                !transform.any_op,
                !transform.any_op)

    // ---- Step 2: 匹配两个 linalg.generic，统一 tiling ----
    %all_generics = transform.structured.match ops{["linalg.generic"]}
        in %func_new : (!transform.any_op) -> !transform.any_op

    %true_param        = transform.param.constant true            -> !transform.any_param
    %prologue_param    = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %epilogue_param    = transform.param.constant "dst:VECOUT->GM"-> !transform.any_param
    %vector_unit_param = transform.param.constant "AiCore.Vector" -> !transform.any_param

    // ══════════════ 两条链统一两级 tiling ══════════════
    transform.foreach %all_generics : !transform.any_op {
    ^bb0(%op : !transform.any_op):
      // TB 层：d0=M/2 切 TB_M，d1=N 不切
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
