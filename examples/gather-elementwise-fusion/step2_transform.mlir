// ============================================================
// STAGE 2: Transform Dialect Tiling
//
// Input: step1b_fused.mlir (after --mark-structured-ops + --fuse-gather-elementwise)
//   Single fused gather generic carries {gather_dim = 1 : i64}
//   with relu + add inlined in body.
//
// Tiling strategy:
//   d0 = M: two-level tile (TB inter-core, Tb intra-core)
//   d1 = K: no tiling (full stays in UB)
//
// RUN: afir-opt --transform-interpreter %s --canonicalize --cse | FileCheck %s
// CHECK: scf.for
// CHECK: ascendc.parallel
// ============================================================

#col_broadcast_map = affine_map<(d0, d1) -> (d1)>
#full_map = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  func.func @relu_index_select_add(
      %data    : tensor<?x?xf16>,
      %indices : tensor<?xi64>,
      %bias    : tensor<?xf16>
  ) -> tensor<?x?xf16> {

    %c0   = arith.constant 0 : index
    %c1   = arith.constant 1 : index
    %zero = arith.constant 0.0 : f16

    %dim_m = tensor.dim %data,    %c0 : tensor<?x?xf16>
    %dim_n = tensor.dim %data,    %c1 : tensor<?x?xf16>
    %dim_k = tensor.dim %indices, %c0 : tensor<?xi64>

    // Fused op: relu + index_select(dim=1) + add in one gather generic.
    %empty_out = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %out = linalg.generic {
      indexing_maps = [#col_broadcast_map, #col_broadcast_map, #full_map],
      iterator_types = ["parallel", "parallel"],
      gather_dim = 1 : i64
    } ins(%indices, %bias : tensor<?xi64>, tensor<?xf16>)
      outs(%empty_out : tensor<?x?xf16>) {
    ^bb0(%idx: i64, %b: f16, %o: f16):
      %i = linalg.index 0 : index
      %idx_cast = arith.index_cast %idx : i64 to index
      %raw = tensor.extract %data[%i, %idx_cast] : tensor<?x?xf16>
      %relu = arith.maximumf %raw, %zero : f16
      %v = arith.addf %relu, %b : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    return %out : tensor<?x?xf16>
  }

  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}
  ) {
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op

    %func_new, %tb_m_param, %tb_inner_m_param =
        transform.func.add_index_args %func, 2
            : (!transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %all_generics = transform.structured.match ops{["linalg.generic"]}
        in %func_new : (!transform.any_op) -> !transform.any_op

    %true_param        = transform.param.constant true -> !transform.any_param
    %prologue_param    = transform.param.constant "src:GM->VECIN"  -> !transform.any_param
    %epilogue_param    = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %vector_unit_param = transform.param.constant "AiCore.Vector"  -> !transform.any_param

    transform.foreach %all_generics : !transform.any_op {
    ^bb0(%op : !transform.any_op):
      %op_tb, %loop_tb =
          transform.structured.tile_using_for %op
              tile_sizes [%tb_m_param, 0]
                  : (!transform.any_op, !transform.any_op)
              -> (!transform.any_op, !transform.any_op)

      transform.annotate %loop_tb "ascendc.parallel"
          = %true_param : !transform.any_op, !transform.any_param

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
