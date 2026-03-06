// ============================================================
// STAGE 2: Transform Dialect 驱动 Tiling
//
// 输入：step1_fused.mlir 的单个 linalg.generic
//   iterator_types = ["parallel", "reduction"]
//   d0 = M（Parallel，核间切分）
//   d1 = N（Reduction，片内规约）
//
// Tiling 结构（对 Parallel 轴 d0 做三级切分，Reduction 轴 d1 不切）：
//   TB 层：核间并行，每核负责 TB_M 行（标注 ascendc.parallel）
//   Tb 层：UB 批次，每批 Tb_M 行（搬运粒度）
//   d1 层：Reduction 轴完整遍历 N，向量化规约
//
// 函数签名追加 index 参数：TB_M, Tb_M
// ============================================================
// RUN: afir-opt --transform-interpreter %s | FileCheck %s
// CHECK: scf.for
// CHECK: ascendc.parallel

#map  = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {
  func.func @broadcast_add_reducesum(
      %arg0: tensor<?xf16>,
      %arg1: tensor<?x?xf16>
  ) -> tensor<?xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %c0  = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf16>
    %0   = tensor.empty(%dim) : tensor<?xf16>
    %1   = linalg.fill ins(%cst : f16) outs(%0 : tensor<?xf16>) -> tensor<?xf16>
    %2   = linalg.generic {
             indexing_maps  = [#map, #map1, #map],
             iterator_types = ["parallel", "reduction"]
           } ins(%arg0, %arg1 : tensor<?xf16>, tensor<?x?xf16>)
             outs(%1 : tensor<?xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %3 = arith.addf %in, %in_0 : f16
      %4 = arith.addf %out, %3 : f16
      linalg.yield %4 : f16
    } -> tensor<?xf16>
    return %2 : tensor<?xf16>
  }

  // ── Transform 调度脚本 ──────────────────────────────────
  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}
  ) {
    // Step1: 匹配 func.func，追加 2 个 index 参数: TB_M, Tb_M
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op

    %func_new, %TB_M, %Tb_M =
        transform.func.add_index_args %func, 2
            : (!transform.any_op)
            -> (!transform.any_op,
                !transform.any_op,
                !transform.any_op)

    // Step2: 匹配 linalg.generic
    %generic = transform.structured.match ops{["linalg.generic"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    // Step3: TB层切分（沿 d0=Parallel 轴，d1=Reduction 不切 → size=0）
    %tiled_TB, %for_TB =
        transform.structured.tile_using_for %generic
            tile_sizes [%TB_M, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // 标注 TB 循环为核间并行
    %p_true     = transform.param.constant true -> !transform.any_param
    transform.annotate %for_TB "ascendc.parallel"
        = %p_true : !transform.any_op, !transform.any_param


    // Step4: Tb层切分（沿 d0，UB批次粒度）
    %tiled_Tb, %for_Tb =
        transform.structured.tile_using_for %tiled_TB
            tile_sizes [%Tb_M, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    %p_prologue = transform.param.constant
        "src:GM->VECIN" -> !transform.any_param
    %p_epilogue = transform.param.constant
        "dst:VECOUT->GM" -> !transform.any_param
    transform.annotate %for_Tb "ascendc.prologue"
        = %p_prologue : !transform.any_op, !transform.any_param
    transform.annotate %for_Tb "ascendc.epilogue"
        = %p_epilogue : !transform.any_op, !transform.any_param

    // Step5: 标注 vector 执行单元
    %p_vector = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.annotate %tiled_Tb "ascendc.unit"
        = %p_vector : !transform.any_op, !transform.any_param

    // Step6: 提升循环不变切片（把不依赖 iv_Tb 的 slice 提到 for_TB 内）
    transform.loop.hoist_loop_invariant_subsets %for_Tb : !transform.any_op

    transform.yield
  }
}
