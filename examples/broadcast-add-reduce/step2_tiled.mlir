// ============================================================
// STAGE 2: Tiled IR - 分块后的中间表示
//
// 这是应用 Transform Dialect 调度后的结果
// 主要变化：
//   - 函数签名扩展，追加 TB_M 和 Tb_M 两个分块参数
//   - 生成嵌套的 scf.for 循环结构
//   - 外层循环标记 ascendc.parallel (核间并行)
//   - 内层循环标记 ascendc.prologue/epilogue (数据搬运)
//
// 循环结构：
//   scf.for %outer (TB层, ascendc.parallel)  // 核间分发
//     scf.for %inner (Tb层)                  // UB批次
//       linalg.generic (向量化计算)
//
// 内存访问模式：
//   - 使用 tensor.extract_slice 进行切片
//   - 使用 tensor.insert_slice 写回结果
// ============================================================

// 动态维度边界计算：min(剩余大小, 分块大小)
#dynamic_bound = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
// 广播映射
#broadcast_map = affine_map<(d0, d1) -> (d0)>
// 完整访问映射
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  // 主函数：分块后的广播加法归约
  func.func @broadcast_add_reducesum(
      %input_a: tensor<?xf16>,      // 输入 A [M]
      %input_b: tensor<?x?xf16>,    // 输入 B [M,N]
      %tb_m_param: i64,             // TB_M: 核间分块大小
      %tb_inner_m_param: i64        // Tb_M: UB 批次大小
  ) -> tensor<?xf16> {

    // 常量定义
    %idx_1 = arith.constant 1 : index
    %idx_0 = arith.constant 0 : index
    %zero = arith.constant 0.000000e+00 : f16

    // 将 i64 参数转换为 index 类型
    %tb_inner_m = arith.index_cast %tb_inner_m_param : i64 to index
    %tb_m = arith.index_cast %tb_m_param : i64 to index

    // 获取维度
    %dim_m = tensor.dim %input_a, %idx_0 : tensor<?xf16>
    %dim_n = tensor.dim %input_b, %idx_1 : tensor<?x?xf16>

    // 初始化输出张量
    %empty_output = tensor.empty(%dim_m) : tensor<?xf16>
    %init_output = linalg.fill ins(%zero : f16)
                   outs(%empty_output : tensor<?xf16>) -> tensor<?xf16>

    // ---- 外层循环: TB 层 (核间并行) ----
    %result_after_outer = scf.for %outer_iv = %idx_0 to %dim_m step %tb_m
        iter_args(%outer_acc = %init_output) -> (tensor<?xf16>) {

      // 计算实际分块大小 (边界处理)
      %outer_size = affine.min #dynamic_bound(%outer_iv)[%dim_m, %tb_m]

      // 切片输入 A
      %slice_a = tensor.extract_slice %input_a[%outer_iv] [%outer_size] [1]
                 : tensor<?xf16> to tensor<?xf16>
      // 切片输入 B
      %slice_b = tensor.extract_slice %input_b[%outer_iv, 0]
                 [%outer_size, %dim_n] [1, 1]
                 : tensor<?x?xf16> to tensor<?x?xf16>
      // 切片输出累加器
      %slice_acc = tensor.extract_slice %outer_acc[%outer_iv] [%outer_size] [1]
                   : tensor<?xf16> to tensor<?xf16>

      // ---- 内层循环: Tb 层 (UB 批次) ----
      %result_after_inner = scf.for %inner_iv = %idx_0 to %outer_size step %tb_inner_m
          iter_args(%inner_acc = %slice_acc) -> (tensor<?xf16>) {

        // 计算内层实际大小
        %inner_size = affine.min #dynamic_bound(%inner_iv)[%outer_size, %tb_inner_m]

        // 进一步切片
        %inner_slice_a = tensor.extract_slice %slice_a[%inner_iv] [%inner_size] [1]
                         : tensor<?xf16> to tensor<?xf16>
        %inner_slice_b = tensor.extract_slice %slice_b[%inner_iv, 0]
                         [%inner_size, %dim_n] [1, 1]
                         : tensor<?x?xf16> to tensor<?x?xf16>
        %inner_slice_acc = tensor.extract_slice %inner_acc[%inner_iv] [%inner_size] [1]
                           : tensor<?xf16> to tensor<?xf16>

        // ---- 核心计算: 向量化 linalg.generic ----
        %computed = linalg.generic {
          indexing_maps = [#broadcast_map, #full_access_map, #broadcast_map],
          iterator_types = ["parallel", "reduction"]
        } ins(%inner_slice_a, %inner_slice_b : tensor<?xf16>, tensor<?x?xf16>)
          outs(%inner_slice_acc : tensor<?xf16>)
          attrs = {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%a_val: f16, %b_val: f16, %acc: f16):
          %sum = arith.addf %a_val, %b_val : f16
          %new_acc = arith.addf %acc, %sum : f16
          linalg.yield %new_acc : f16
        } -> tensor<?xf16>

        // 写回内层结果
        %inserted_inner = tensor.insert_slice %computed into %inner_acc[%inner_iv] [%inner_size] [1]
                          : tensor<?xf16> into tensor<?xf16>
        scf.yield %inserted_inner : tensor<?xf16>

      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}

      // 写回外层结果
      %inserted_outer = tensor.insert_slice %result_after_inner into %outer_acc[%outer_iv] [%outer_size] [1]
                        : tensor<?xf16> into tensor<?xf16>
      scf.yield %inserted_outer : tensor<?xf16>

    } {ascendc.parallel = true}

    return %result_after_outer : tensor<?xf16>
  }

  // Transform 调度脚本 (保留用于参考)
  transform.named_sequence @__transform_main(
      %root_op: !transform.any_op {transform.readonly}
  ) {
    // 匹配函数
    %matched_func = transform.structured.match ops{["func.func"]} in %root_op
        : (!transform.any_op) -> !transform.any_op

    // 添加索引参数
    %transformed_func, %new_arg_0, %new_arg_1 =
        transform.func.add_index_args %matched_func, 2
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 匹配 linalg.generic
    %matched_generic = transform.structured.match ops{["linalg.generic"]} in %transformed_func
        : (!transform.any_op) -> !transform.any_op

    // TB 层切分
    %tiled_op, %outer_loop = transform.structured.tile_using_for %matched_generic
        tile_sizes [%new_arg_0, 0]
        : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)

    // 标记并行
    %true_val = transform.param.constant true -> !transform.any_param
    transform.annotate %outer_loop "ascendc.parallel" = %true_val
        : !transform.any_op, !transform.any_param

    // Tb 层切分
    %tiled_inner_op, %inner_loop = transform.structured.tile_using_for %tiled_op
        tile_sizes [%new_arg_1, 0]
        : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)

    // 标记 prologue 和 epilogue
    %prologue_val = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %epilogue_val = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    transform.annotate %inner_loop "ascendc.prologue" = %prologue_val
        : !transform.any_op, !transform.any_param
    transform.annotate %inner_loop "ascendc.epilogue" = %epilogue_val
        : !transform.any_op, !transform.any_param

    // 标记执行单元
    %unit_val = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.annotate %tiled_inner_op "ascendc.unit" = %unit_val
        : !transform.any_op, !transform.any_param

    // 提升循环不变量
    transform.loop.hoist_loop_invariant_subsets %inner_loop : !transform.any_op

    transform.yield
  }
}
