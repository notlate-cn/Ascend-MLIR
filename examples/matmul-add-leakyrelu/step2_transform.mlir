// examples/matmul-add-leakyrelu/step2_transform.mlir
// RUN: afir-opt --transform-interpreter %s | FileCheck %s
// CHECK: scf.for
// CHECK: ascendc.parallel

#bias_map = affine_map<(d0, d1) -> (d1)>
#full_map  = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  func.func @matmul_add_leakyrelu(
      %a    : tensor<?x?xf16>,
      %b    : tensor<?x?xf16>,
      %bias : tensor<?xf32>,
      %out  : tensor<?x?xf32>
  ) -> tensor<?x?xf32> {
    %idx0  = arith.constant 0 : index
    %idx1  = arith.constant 1 : index
    %dim_m = tensor.dim %out, %idx0 : tensor<?x?xf32>
    %dim_n = tensor.dim %out, %idx1 : tensor<?x?xf32>

    %c = linalg.matmul
      ins(%a, %b : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%out  : tensor<?x?xf32>)
      -> tensor<?x?xf32>

    %empty_d = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>
    %d = linalg.generic {
      indexing_maps = [#full_map, #bias_map, #full_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%c, %bias : tensor<?x?xf32>, tensor<?xf32>)
      outs(%empty_d : tensor<?x?xf32>) {
    ^bb0(%c_val: f32, %bias_val: f32, %o: f32):
      %s = arith.addf %c_val, %bias_val : f32
      linalg.yield %s : f32
    } -> tensor<?x?xf32>

    %alpha = arith.constant 1.0e-3 : f32
    %empty_e = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>
    %e = linalg.generic {
      indexing_maps = [#full_map, #full_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%d : tensor<?x?xf32>)
      outs(%empty_e : tensor<?x?xf32>) {
    ^bb0(%x: f32, %o: f32):
      %scaled = arith.mulf %x, %alpha : f32
      %result = arith.maximumf %x, %scaled : f32
      linalg.yield %result : f32
    } -> tensor<?x?xf32>

    return %e : tensor<?x?xf32>
  }

  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}
  ) {
    // ── Step 1: add 5 index args: TB_M, TB_N, Tb_M, Tb_N, t_K ──────
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %func_new, %TB_M, %TB_N, %Tb_M, %Tb_N, %t_K =
        transform.func.add_index_args %func, 5
            : (!transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op,
                !transform.any_op, !transform.any_op, !transform.any_op)

    // ── Step 2: 匹配原始 ops ─────────────────────────────────────────
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func_new
        : (!transform.any_op) -> !transform.any_op
    %generics = transform.structured.match ops{["linalg.generic"]} in %func_new
        : (!transform.any_op) -> !transform.any_op
    // IR 顺序: #0=add_bias, #1=leaky_relu
    %add, %leakyrelu = transform.split_handle %generics
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ── 预定义 param 常量 ────────────────────────────────────────────
    %p_true      = transform.param.constant true -> !transform.any_param
    %p_TB_pro    = transform.param.constant "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"
                       -> !transform.any_param
    %p_TB_epi    = transform.param.constant "result:VECOUT->GM"
                       -> !transform.any_param
    %p_K_pro     = transform.param.constant "lhs:A1->A2,rhs:B1->B2"
                       -> !transform.any_param
    %p_K_epi     = transform.param.constant "acc:CO1->VECIN"
                       -> !transform.any_param
    %p_cube      = transform.param.constant "AiCore.Cube"   -> !transform.any_param
    %p_vector    = transform.param.constant "AiCore.Vector" -> !transform.any_param

    // ── 第一轮: TB 层切分 leaky_relu [TB_M, TB_N] ────────────────────
    %tiled_lrelu_TB, %for_TB_M, %for_TB_N =
        transform.structured.tile_using_for %leakyrelu
            tile_sizes [%TB_M, %TB_N]
                : (!transform.any_op, !transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // fuse add into for_TB_N
    %add_fused_TB, %loop_add_TB =
        transform.structured.fuse_into_containing_op %add into %for_TB_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // fuse matmul into for_TB_N
    %matmul_fused_TB, %loop_matmul_TB =
        transform.structured.fuse_into_containing_op %matmul into %for_TB_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    %matmul_TB_split:3 = transform.split_handle %matmul_fused_TB
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 分核标注
    transform.annotate %for_TB_M "ascendc.parallel"
        = %p_true : !transform.any_op, !transform.any_param
    transform.annotate %for_TB_N "ascendc.parallel"
        = %p_true : !transform.any_op, !transform.any_param
    transform.annotate %for_TB_N "ascendc.prologue"
        = %p_TB_pro : !transform.any_op, !transform.any_param
    transform.annotate %for_TB_N "ascendc.epilogue"
        = %p_TB_epi : !transform.any_op, !transform.any_param

    // ── 第二轮: Tb 层切分 [Tb_M, Tb_N] ──────────────────────────────
    %tiled_lrelu_Tb, %for_Tb_M, %for_Tb_N =
        transform.structured.tile_using_for %tiled_lrelu_TB
            tile_sizes [%Tb_M, %Tb_N]
                : (!transform.any_op, !transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // fuse add into for_Tb_N
    %add_fused_Tb, %loop_add_Tb =
        transform.structured.fuse_into_containing_op %add_fused_TB into %for_Tb_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // fuse matmul into for_Tb_N
    %matmul_fused_Tb, %loop_matmul_Tb =
        transform.structured.fuse_into_containing_op %matmul_TB_split#0 into %for_Tb_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    %matmul_Tb_split:3 = transform.split_handle %matmul_fused_Tb
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ── 第三轮: K 轴切分 [0, 0, t_K] ────────────────────────────────
    %tiled_matmul_K, %for_K =
        transform.structured.tile_using_for %matmul_Tb_split#0
            tile_sizes [0, 0, %t_K]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    %matmul_final = transform.structured.match ops{["linalg.matmul"]} in %for_K
        : (!transform.any_op) -> !transform.any_op
    transform.annotate %matmul_final "ascendc.unit"
        = %p_cube : !transform.any_op, !transform.any_param
    transform.annotate %add_fused_Tb "ascendc.unit"
        = %p_vector : !transform.any_op, !transform.any_param
    transform.annotate %tiled_lrelu_Tb "ascendc.unit"
        = %p_vector : !transform.any_op, !transform.any_param

    transform.annotate %for_K "ascendc.prologue"
        = %p_K_pro : !transform.any_op, !transform.any_param
    transform.annotate %for_K "ascendc.epilogue"
        = %p_K_epi : !transform.any_op, !transform.any_param

    transform.yield
  }
}
