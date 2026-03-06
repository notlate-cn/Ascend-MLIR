// transform_3level.mlir
//
// 在 transform_tile_and_fuse.mlir 基础上扩展，完成 TB/Tb/t 三级变换
//
// 新增参数说明 (在原有3个参数基础上增加2个):
//   %sz#0 = TB_M  : TB级 M 步长 (核间并行，forall M 方向)
//   %sz#1 = TB_N  : TB级 N 步长 (核间并行，forall N 方向)
//   %sz#2 = t_K   : t级  K 步长 (CUBE 单次 K 粒度，原有)
//   %sz#3 = Tb_M  : Tb级 M 步长 (核内 M 循环，新增)
//   %sz#4 = Tb_N  : Tb级 N 步长 (核内 N 循环，新增)
//
// 变换后循环结构:
//   scf.forall [iv_TB_M, iv_TB_N] (TB: 核间并行)
//     scf.for Tb_M                (Tb: 核内M循环，A片复用基础)
//       scf.for Tb_N              (Tb: 核内N循环，A片在此层复用)
//         scf.for K (step=t_K)    (t:  CUBE单次K-reduction)
//           linalg.matmul
//         linalg.elementwise add
//         linalg.elementwise max  (relu)
//     in_parallel: parallel_insert_slice
//
// 执行:
//   mlir-opt fc_add_relu.mlir \
//     --transform-interpreter=entry-point=__transform_main \
//     --canonicalize --cse \
//     -o out_3level.mlir

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}
  ) {

    // ============================================================
    // Step 1: 匹配 func.func
    // ============================================================
    %func = transform.structured.match ops{["func.func"]}
              in %arg1
            : (!transform.any_op) -> !transform.any_op

    // ============================================================
    // Step 2: 为 func.func 增加 5 个 index 入参
    //   %sz#0 = TB_M, %sz#1 = TB_N, %sz#2 = t_K (原有3个语义不变)
    //   %sz#3 = Tb_M, %sz#4 = Tb_N              (新增2个)
    // ============================================================
    %transformed, %sz:5 = transform.func.add_index_args %func, 5
      : (!transform.any_op)
        -> (!transform.any_op,
            !transform.any_op, !transform.any_op, !transform.any_op,
            !transform.any_op, !transform.any_op)

    // ============================================================
    // Step 3: 匹配 linalg ops
    // ============================================================
    %matmul = transform.structured.match ops{["linalg.matmul"]}
                in %transformed
              : (!transform.any_op) -> !transform.any_op

    %elementwise = transform.structured.match ops{["linalg.elementwise"]}
                     in %transformed
                   : (!transform.any_op) -> !transform.any_op

    %add, %max = transform.split_handle %elementwise
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 4 (继承原逻辑): 倒序 TileAndFuse，建立 Tb 级两层循环
    //
    //   先对 max 按 [Tb_M, Tb_N] tile → 产生 Tb_M x Tb_N 两层 scf.for
    //   再把 add、matmul 倒序 fuse 进内层循环 (loop1 = Tb_N 层)
    //   K 轴单独 tile step=t_K → 产生第三层 scf.for (t 级)
    //
    //   注意: 这里用 %sz#3/%sz#4 (Tb_M/Tb_N) 作为 tile size
    //         而不是原来的 %sz#0/%sz#1，因为 TB 级并行化在 Step 5 完成
    // ============================================================

    // tile max → 产生 Tb_M(%loop0) x Tb_N(%loop1) 两层 scf.for
    %tiled_max, %loop0, %loop1 = transform.structured.tile_using_for %max
        tile_sizes [%sz#3, %sz#4]
      : (!transform.any_op, !transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // fuse add 进 loop1 (Tb_N 内层)
    %add_fused, %loop2 = transform.structured.fuse_into_containing_op %add
        into %loop1
      : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)

    // fuse matmul 进 loop1 (Tb_N 内层)
    %matmul_fused, %loop3 = transform.structured.fuse_into_containing_op %matmul
        into %loop1
      : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)

    // K 轴 tile step=t_K → 产生第三层 scf.for (t 级 K-reduction)
    %matmul_fused_split:3 = transform.split_handle %matmul_fused
      : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %ret, %loop_K = transform.structured.tile_using_for
        %matmul_fused_split#0 tile_sizes [0, 0, %sz#2]
      : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 5 (新增): TB 级并行化 → scf.forall
    //
    //   此时 IR 结构为:
    //     scf.for Tb_M
    //       scf.for Tb_N
    //         scf.for K
    //           matmul / add / max
    //
    //   对最外层 scf.for (loop0 = Tb_M 层) 的父级 op (即整个计算块)
    //   做 tile_using_forall，tile_sizes = [TB_M, TB_N]
    //   产生 scf.forall 包在 Tb 两层循环外面
    //
    //   这里直接对 fuse 后的 max (tiled_max) 做 forall tiling，
    //   因为 max 是整个计算块最外层的 consumer，
    //   forall 会自动把内部的 Tb 循环包进去
    // ============================================================

    // 重新 match func 内的 max op (经过 fuse 后可能有新的 handle)
    %max_after_fuse = transform.structured.match ops{["linalg.elementwise"]}
                        attributes{kind = #linalg.elementwise_kind<max_signed>}
                        in %transformed
                      : (!transform.any_op) -> !transform.any_op

    %tiled_forall, %forall_op =
        transform.structured.tile_using_forall %max_after_fuse
            tile_sizes [%sz#0, %sz#1]
            ( mapping = [#gpu.block<y>, #gpu.block<x>] )
        : (!transform.any_op)
          -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 6: 清理
    // ============================================================
    transform.apply_patterns to %transformed {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.apply_cse to %transformed : !transform.any_op

    transform.yield
  }
}
