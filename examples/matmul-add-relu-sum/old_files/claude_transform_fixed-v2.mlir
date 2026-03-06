// claude_transform_fixed-v3.mlir
//
// 修复 fuse_into_containing_op 报错的正确变换顺序
// 并使用 index args 传递 tile sizes，避免硬编码
//
// 执行:
//   mlir-opt claude.mlir \
//     --transform-interpreter=entry-point=__transform_main \
//     --canonicalize --cse \
//     -o out.mlir

module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(
      %module: !transform.any_op {transform.readonly}
  ) {

    // ============================================================
    // Step 0: 获取 func @fc_relu
    // ============================================================
    %func = transform.structured.match ops{["func.func"]}
              attributes{sym_name = "fc_relu"}
              in %module
            : (!transform.any_op) -> !transform.any_op

    // ============================================================
    // Step 1: 为 tile sizes 添加 index args
    //   这里用 TB_M, TB_N, Tb_M, Tb_N 四个 index arg 替代硬编码
    // ============================================================
    %transformed = transform.func.add_index_args %func {num_index_args = 4}
                   : (!transform.any_op) -> !transform.any_op

    %TB_M = transform.structured.get_index_arg %transformed[0] : (!transform.any_op) -> !transform.index
    %TB_N = transform.structured.get_index_arg %transformed[1] : (!transform.any_op) -> !transform.index
    %Tb_M = transform.structured.get_index_arg %transformed[2] : (!transform.any_op) -> !transform.index
    %Tb_N = transform.structured.get_index_arg %transformed[3] : (!transform.any_op) -> !transform.index

    // ============================================================
    // Step 2: TB 级并行化
    // ============================================================
    %matmul = transform.structured.match ops{["linalg.matmul"]}
                in %func
              : (!transform.any_op) -> !transform.any_op

    %tiled_matmul_TB, %forall =
        transform.structured.tile_using_forall %matmul
            tile_sizes [%TB_M, %TB_N]
            ( mapping = [#gpu.block<y>, #gpu.block<x>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 3: fuse elementwise 进 TB 级 forall
    // ============================================================
    %elementwise = transform.structured.match ops{["linalg.elementwise"]}
                     in %forall
                   : (!transform.any_op) -> !transform.any_op

    %fused_elems, %fused_loop =
        transform.structured.fuse_into_containing_op %elementwise
            into %forall
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 4: Tb 级 tiling
    //   对 forall 内 matmul 做 tile_using_for，使用 index args
    // ============================================================
    %matmul_in_forall = transform.structured.match ops{["linalg.matmul"]}
                          in %forall
                        : (!transform.any_op) -> !transform.any_op

    %tiled_matmul_Tb, %loops:2 =
        transform.structured.tile_using_for %matmul_in_forall
            tile_sizes [%Tb_M, %Tb_N, 0]  // K 方向不切
        : (!transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ============================================================
    // Step 5: fuse add/relu 进 Tb_N 最内层循环
    // ============================================================
    %add_op = transform.structured.match ops{["linalg.elementwise"]}
                attributes{kind = #linalg.elementwise_kind<add>}
                in %forall
              : (!transform.any_op) -> !transform.any_op

    %relu_op = transform.structured.match ops{["linalg.elementwise"]}
                 attributes{kind = #linalg.elementwise_kind<max_signed>}
                 in %forall
               : (!transform.any_op) -> !transform.any_op

    transform.structured.fuse_into_containing_op %add_op
        into %loops#1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.structured.fuse_into_containing_op %relu_op
        into %loops#1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 6: 清理
    // ============================================================
    %func_for_cleanup = transform.structured.match ops{["func.func"]}
                          in %module
                        : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %func_for_cleanup {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.apply_cse to %func_for_cleanup : !transform.any_op

    transform.yield
  }
}