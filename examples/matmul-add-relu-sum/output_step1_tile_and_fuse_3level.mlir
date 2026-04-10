// ============================================================
// output_step1_tile_and_fuse_3level.mlir - 三级分块与融合后的 IR
//
// 本阶段应用了三级 Tiling 策略和算子融合：
//   - TB 层 (Outer Tile): 核间分块，映射到多核并行
//   - Tb 层 (Inner Tile): 核内分块，UB 缓冲区粒度
//   - t_K 层 (K Tile):    K 维度分块，Cube 计算粒度
//
// 计算图: output = ReLU(matmul(A, B) + bias)
//
// 融合结果：
//   - matmul + add + relu 融合为单个计算单元
//   - 消除中间张量，减少内存访问
//
// 执行单元标注：
//   - matmul:  ascendc.unit = "AiCore.Cube"
//   - add/max: ascendc.unit = "AiCore.Vector"
//
// 搬运时机标注：
//   - prologue: 循环入口数据搬运 (GM -> 片上内存)
//   - epilogue: 循环出口数据写回 (片上内存 -> GM)
// ============================================================

// 动态边界计算映射：min(剩余大小, tile大小)
#tile_guard = affine_map<(d0)[upper, tile] -> (-d0 + upper, tile)>

module attributes {transform.with_named_sequence} {
  // 主函数：三级分块的全连接 + ReLU
  func.func @fc_relu(
      %input_a: tensor<?x?xf32>,        // 输入 A [M, K]
      %input_b: tensor<?x?xf32>,        // 输入 B [K, N]
      %bias: tensor<?x?xf32>,            // 偏置 [M, N]
      %output_init: tensor<?x?xf32>,    // 初始输出 [M, N]
      %tile_m_outer_i64: i64,           // TB_M
      %tile_n_outer_i64: i64,           // TB_N
      %tile_m_inner_i64: i64,           // Tb_M
      %tile_n_inner_i64: i64,           // Tb_N
      %tile_k_i64: i64                  // t_K
  ) -> tensor<?x?xf32> {

    // 基础常量定义
    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index
    %zero_f32 = arith.constant 0.000000e+00 : f32

    // Tile 参数类型转换 (i64 -> index)
    %tile_k = arith.index_cast %tile_k_i64 : i64 to index
    %tile_n_inner = arith.index_cast %tile_n_inner_i64 : i64 to index
    %tile_m_inner = arith.index_cast %tile_m_inner_i64 : i64 to index
    %tile_n_outer = arith.index_cast %tile_n_outer_i64 : i64 to index
    %tile_m_outer = arith.index_cast %tile_m_outer_i64 : i64 to index

    // 获取输出矩阵维度
    %dim_m = tensor.dim %output_init, %idx_0 : tensor<?x?xf32>
    %dim_n = tensor.dim %output_init, %idx_1 : tensor<?x?xf32>

    // 构造 ReLU 需要的零张量
    %zero_tensor_empty = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>
    %zero_filled = linalg.fill
        ins(%zero_f32 : f32)
        outs(%zero_tensor_empty : tensor<?x?xf32>) -> tensor<?x?xf32>

    // 第一层: Outer Tile (TB 层 - 核间分块)
    %result_after_outer =
    scf.for %m_outer = %idx_0 to %dim_m step %tile_m_outer
        iter_args(%outer_acc = %output_init)
        -> (tensor<?x?xf32>) {

      %result_after_n_outer =
      scf.for %n_outer = %idx_0 to %dim_n step %tile_n_outer
          iter_args(%n_outer_acc = %outer_acc)
          -> (tensor<?x?xf32>) {

        // Tile 边界保护
        %m_outer_size = affine.min #tile_guard(%m_outer)[%dim_m, %tile_m_outer]
        %n_outer_size = affine.min #tile_guard(%n_outer)[%dim_n, %tile_n_outer]
        %dim_k = tensor.dim %input_a, %idx_1 : tensor<?x?xf32>

        // GM → L1 tile
        %a_outer = tensor.extract_slice %input_a[%m_outer, 0][%m_outer_size, %dim_k][1, 1]
                   : tensor<?x?xf32> to tensor<?x?xf32>
        %b_outer = tensor.extract_slice %input_b[0, %n_outer][%dim_k, %n_outer_size][1, 1]
                   : tensor<?x?xf32> to tensor<?x?xf32>
        %out_outer = tensor.extract_slice %n_outer_acc[%m_outer, %n_outer][%m_outer_size, %n_outer_size][1, 1]
                     : tensor<?x?xf32> to tensor<?x?xf32>
        %bias_outer = tensor.extract_slice %bias[%m_outer, %n_outer][%m_outer_size, %n_outer_size][1, 1]
                      : tensor<?x?xf32> to tensor<?x?xf32>
        %zero_outer = tensor.extract_slice %zero_filled[%m_outer, %n_outer][%m_outer_size, %n_outer_size][1, 1]
                      : tensor<?x?xf32> to tensor<?x?xf32>

        // 第二层: Inner Tile (Tb 层 - 核内分块)
        %result_after_inner =
        scf.for %m_inner = %idx_0 to %m_outer_size step %tile_m_inner
            iter_args(%inner_acc = %out_outer)
            -> (tensor<?x?xf32>) {

          %result_after_n_inner =
          scf.for %n_inner = %idx_0 to %n_outer_size step %tile_n_inner
              iter_args(%n_inner_acc = %inner_acc)
              -> (tensor<?x?xf32>) {

            %m_inner_size = affine.min #tile_guard(%m_inner)[%m_outer_size, %tile_m_inner]
            %n_inner_size = affine.min #tile_guard(%n_inner)[%n_outer_size, %tile_n_inner]

            %a_inner = tensor.extract_slice %a_outer[%m_inner, 0][%m_inner_size, %dim_k][1, 1]
                       : tensor<?x?xf32> to tensor<?x?xf32>
            %b_inner = tensor.extract_slice %b_outer[0, %n_inner][%dim_k, %n_inner_size][1, 1]
                       : tensor<?x?xf32> to tensor<?x?xf32>
            %out_inner = tensor.extract_slice %n_inner_acc[%m_inner, %n_inner][%m_inner_size, %n_inner_size][1, 1]
                         : tensor<?x?xf32> to tensor<?x?xf32>

            // 第三层: K Blocking (K 维度分块 - Cube 计算)
            %result_after_k =
            scf.for %k = %idx_0 to %dim_k step %tile_k
                iter_args(%k_acc = %out_inner)
                -> (tensor<?x?xf32>) {

              %k_tile_size = affine.min #tile_guard(%k)[%dim_k, %tile_k]

              %a_k = tensor.extract_slice %a_inner[0, %k][%m_inner_size, %k_tile_size][1, 1]
                     : tensor<?x?xf32> to tensor<?x?xf32>
              %b_k = tensor.extract_slice %b_inner[%k, 0][%k_tile_size, %n_inner_size][1, 1]
                     : tensor<?x?xf32> to tensor<?x?xf32>
              %out_tile = tensor.extract_slice %k_acc[0, 0][%m_inner_size, %n_inner_size][1, 1]
                          : tensor<?x?xf32> to tensor<?x?xf32>

              // Stage 1: Cube GEMM
              %matmul_tile = linalg.matmul {ascendc.unit = "AiCore.Cube"}
                ins(%a_k, %b_k : tensor<?x?xf32>, tensor<?x?xf32>)
                outs(%out_tile : tensor<?x?xf32>) -> tensor<?x?xf32>

              %k_acc_next = tensor.insert_slice %matmul_tile into %k_acc[0, 0][%m_inner_size, %n_inner_size][1, 1]
                            : tensor<?x?xf32> into tensor<?x?xf32>

              scf.yield %k_acc_next : tensor<?x?xf32>
            } {ascendc.prologue = "lhs:A1->A2,rhs:B1->B2",
               ascendc.epilogue = "acc:CO1->VECIN"}

            // Stage 2: Bias Add (Vector)
            %bias_inner = tensor.extract_slice %bias_outer[%m_inner, %n_inner][%m_inner_size, %n_inner_size][1, 1]
                          : tensor<?x?xf32> to tensor<?x?xf32>

            %add_result = linalg.elementwise kind=#linalg.elementwise_kind<add> {ascendc.unit = "AiCore.Vector"}
              ins(%result_after_k, %bias_inner : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%out_inner : tensor<?x?xf32>) -> tensor<?x?xf32>

            // Stage 3: ReLU (Vector)
            %zero_inner = tensor.extract_slice %zero_outer[%m_inner, %n_inner][%m_inner_size, %n_inner_size][1, 1]
                          : tensor<?x?xf32> to tensor<?x?xf32>

            %relu_result = linalg.elementwise kind=#linalg.elementwise_kind<max_signed> {ascendc.unit = "AiCore.Vector"}
              ins(%add_result, %zero_inner : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%n_inner_acc : tensor<?x?xf32>) -> tensor<?x?xf32>

            %block_next = tensor.insert_slice %relu_result into %n_inner_acc[%m_inner, %n_inner][%m_inner_size, %n_inner_size][1, 1]
                          : tensor<?x?xf32> into tensor<?x?xf32>

            scf.yield %block_next : tensor<?x?xf32>
          }
          scf.yield %result_after_n_inner : tensor<?x?xf32>
        }

        %tile_next = tensor.insert_slice %result_after_inner into %n_outer_acc[%m_outer, %n_outer][%m_outer_size, %n_outer_size][1, 1]
                     : tensor<?x?xf32> into tensor<?x?xf32>

        scf.yield %tile_next : tensor<?x?xf32>
      } {ascendc.parallel = true,
         ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN",
         ascendc.epilogue = "result:VECOUT->GM"}

      scf.yield %result_after_n_outer : tensor<?x?xf32>
    } {ascendc.parallel = true}

    return %result_after_outer : tensor<?x?xf32>
  }

  // Transform 调度脚本
  transform.named_sequence @__transform_main(
      %root_op: !transform.any_op {transform.readonly}
  ) {
    %matched_func = transform.structured.match ops{["func.func"]} in %root_op
        : (!transform.any_op) -> !transform.any_op
    %transformed_func, %new_arg_0, %new_arg_1, %new_arg_2, %new_arg_3, %new_arg_4 =
        transform.func.add_index_args %matched_func, 5
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %matched_matmul = transform.structured.match ops{["linalg.matmul"]} in %transformed_func
        : (!transform.any_op) -> !transform.any_op
    %tiled_tb, %loop_tb_m, %loop_tb_n = transform.structured.tile_using_for %matched_matmul [%new_arg_0, %new_arg_1]
        : (!transform.any_op, !transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %true_val = transform.param.constant true -> !transform.any_param
    transform.annotate %loop_tb_m "ascendc.parallel" = %true_val
        : !transform.any_op, !transform.any_param
    transform.annotate %loop_tb_n "ascendc.parallel" = %true_val
        : !transform.any_op, !transform.any_param
    %prologue_tb = transform.param.constant "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN" -> !transform.any_param
    transform.annotate %loop_tb_m "ascendc.prologue" = %prologue_tb
        : !transform.any_op, !transform.any_param
    %tiled_tb_inner, %loop_tb_m_inner, %loop_tb_n_inner = transform.structured.tile_using_for %tiled_tb [%new_arg_2, %new_arg_3]
        : (!transform.any_op, !transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_t, %loop_t_k = transform.structured.tile_using_for %tiled_tb_inner [0, 0, %new_arg_4]
        : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    %cube_unit = transform.param.constant "AiCore.Cube" -> !transform.any_param
    transform.annotate %tiled_t "ascendc.unit" = %cube_unit
        : !transform.any_op, !transform.any_param
    %prologue_k = transform.param.constant "lhs:A1->A2,rhs:B1->B2" -> !transform.any_param
    %epilogue_k = transform.param.constant "acc:CO1->VECIN" -> !transform.any_param
    transform.annotate %loop_t_k "ascendc.prologue" = %prologue_k
        : !transform.any_op, !transform.any_param
    transform.annotate %loop_t_k "ascendc.epilogue" = %epilogue_k
        : !transform.any_op, !transform.any_param
    %matched_add = transform.structured.match ops{["linalg.elementwise"]} attributes{kind = #linalg.elementwise_kind<add>} in %transformed_func
        : (!transform.any_op) -> !transform.any_op
    %matched_relu = transform.structured.match ops{["linalg.elementwise"]} attributes{kind = #linalg.elementwise_kind<max_signed>} in %transformed_func
        : (!transform.any_op) -> !transform.any_op
    %vector_unit = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.annotate %matched_add "ascendc.unit" = %vector_unit
        : !transform.any_op, !transform.any_param
    transform.annotate %matched_relu "ascendc.unit" = %vector_unit
        : !transform.any_op, !transform.any_param
    %epilogue_tb = transform.param.constant "result:VECOUT->GM" -> !transform.any_param
    transform.annotate %loop_tb_n "ascendc.epilogue" = %epilogue_tb
        : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loop_tb_m_inner : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %loop_tb_n_inner : !transform.any_op
    transform.yield
  }
}
