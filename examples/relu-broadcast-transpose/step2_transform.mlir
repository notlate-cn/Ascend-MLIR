// ============================================================
// STAGE 2: Transform Dialect Tiling — single fused generic
//
// Input: step0_input_out.mlir (one linalg.generic, no library_call)
// Computation: relu(data0[m,1]) + data1[n,m] (transpose+broadcast fused into map)
//
// Iteration space: [n, m] — d0=n, d1=m
// Tiling strategy: tile d0 (n-axis) at TB and Tb levels; d1 (m-axis) untiled
//   TB-level: inter-core parallelism (one AiCore per TB_N rows)
//   Tb-level: UB batch size (Tb_N rows per UB tile)
//
// Function params appended: TB_N (inter-core block), Tb_N (UB batch)
//
// RUN: afir-opt --transform-interpreter %s --canonicalize --cse | FileCheck %s
// CHECK: scf.for
// CHECK: ascendc.parallel
// ============================================================

#map_data0 = affine_map<(d0, d1) -> (d1, 0)>
#map_identity = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  func.func @relu_transpose_broadcast_add(
      %data0: tensor<?x1xf16>,
      %data1: tensor<?x?xf16>) -> tensor<?x?xf16> {

    %c0 = arith.constant 0 : index
    %m = tensor.dim %data0, %c0 : tensor<?x1xf16>
    %n = tensor.dim %data1, %c0 : tensor<?x?xf16>

    // Single fused generic: relu(data0[m,1]) broadcast+transpose+add with data1[n,m]
    %out_init = tensor.empty(%n, %m) : tensor<?x?xf16>
    %result = linalg.generic {
      indexing_maps = [#map_data0, #map_identity, #map_identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%data0, %data1 : tensor<?x1xf16>, tensor<?x?xf16>)
      outs(%out_init : tensor<?x?xf16>) {
    ^bb0(%v0: f16, %v1: f16, %vout: f16):
      %cst = arith.constant 0.000000e+00 : f16
      %relu = arith.maximumf %v0, %cst : f16
      %add  = arith.addf %relu, %v1 : f16
      linalg.yield %add : f16
    } -> tensor<?x?xf16>

    return %result : tensor<?x?xf16>
  }

  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}
  ) {
    // ---- Step 1: Match func.func, append 2 index params (TB_N, Tb_N) ----
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op

    %func_new, %tb_n_param, %tb_inner_n_param =
        transform.func.add_index_args %func, 2
            : (!transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ---- Step 2: Match the single fused linalg.generic ----
    %fused_op = transform.structured.match ops{["linalg.generic"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    // ---- Common annotation params ----
    %true_param = transform.param.constant true -> !transform.any_param
    %prologue_param = transform.param.constant
        "src:GM->VECIN" -> !transform.any_param
    %epilogue_param = transform.param.constant
        "dst:VECOUT->GM" -> !transform.any_param
    %vector_unit_param = transform.param.constant
        "AiCore.Vector" -> !transform.any_param

    // ══════ Fused generic tiling: d0 (n-axis) at TB and Tb levels ══════
    // d1 (m-axis) untiled: data0 has size 1 in that axis, processed whole in UB

    %op_tb, %loop_tb =
        transform.structured.tile_using_for %fused_op
            tile_sizes [%tb_n_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_tb "ascendc.parallel"
        = %true_param : !transform.any_op, !transform.any_param

    %op_inner, %loop_inner =
        transform.structured.tile_using_for %op_tb
            tile_sizes [%tb_inner_n_param, 0]
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

    transform.yield
  }
}
