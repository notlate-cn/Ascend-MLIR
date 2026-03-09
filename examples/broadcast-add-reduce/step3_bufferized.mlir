// ============================================================
// STAGE 3: Bufferized IR - 缓冲区化后的中间表示
//
// 本阶段将 tensor 方言转换为 memref 方言：
//   - tensor.extract_slice → memref.subview
//   - tensor.insert_slice → memref.copy
//   - tensor.empty → memref.alloc
//   - 所有张量操作变为内存引用操作
//
// 关键变化：
//   - 函数参数从 tensor 变为 memref
//   - 使用 memref.subview 进行零拷贝切片
//   - 使用 memref.alloc 分配输出缓冲区
//   - 使用 memref.copy 进行数据搬运
//
// 内存层级：
//   - GM (Global Memory): 主内存，通过 memref 访问
//   - 后续阶段将引入 UB (Unified Buffer) 等片上内存
// ============================================================

// 动态维度边界计算
#dynamic_bound = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
// 广播映射
#broadcast_map = affine_map<(d0, d1) -> (d0)>
// 完整访问映射
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  // 主函数：缓冲区化后的广播加法归约
  // 注意：所有 tensor 参数已变为 memref 参数
  func.func @broadcast_add_reducesum(
      %input_a: memref<?xf16>,       // 输入 A [M] - 全局内存
      %input_b: memref<?x?xf16>,     // 输入 B [M,N] - 全局内存
      %tb_m_param: i64,              // TB_M: 核间分块大小
      %tb_inner_m_param: i64         // Tb_M: UB 批次大小
  ) -> memref<?xf16> {

    // 常量定义
    %idx_1 = arith.constant 1 : index
    %idx_0 = arith.constant 0 : index
    %zero = arith.constant 0.000000e+00 : f16

    // 类型转换
    %tb_inner_m = arith.index_cast %tb_inner_m_param : i64 to index
    %tb_m = arith.index_cast %tb_m_param : i64 to index

    // 获取维度
    %dim_m = memref.dim %input_a, %idx_0 : memref<?xf16>
    %dim_n = memref.dim %input_b, %idx_1 : memref<?x?xf16>

    // 分配输出缓冲区 (GM)
    %output_buffer = memref.alloc(%dim_m) {alignment = 64 : i64} : memref<?xf16>
    // 初始化输出为0
    linalg.fill ins(%zero : f16) outs(%output_buffer : memref<?xf16>)

    // ---- 外层循环: TB 层 (核间并行) ----
    %result_after_outer = scf.for %outer_iv = %idx_0 to %dim_m step %tb_m
        iter_args(%outer_acc = %output_buffer) -> (memref<?xf16>) {

      // 计算实际分块大小
      %outer_size = affine.min #dynamic_bound(%outer_iv)[%dim_m, %tb_m]

      // 创建子视图 (零拷贝切片)
      %subview_a = memref.subview %input_a[%outer_iv] [%outer_size] [1]
                   : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_b = memref.subview %input_b[%outer_iv, 0] [%outer_size, %dim_n] [1, 1]
                   : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_acc = memref.subview %outer_acc[%outer_iv] [%outer_size] [1]
                     : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>

      // ---- 内层循环: Tb 层 (UB 批次) ----
      %result_after_inner = scf.for %inner_iv = %idx_0 to %outer_size step %tb_inner_m
          iter_args(%inner_acc = %subview_acc) -> (memref<?xf16, strided<[1], offset: ?>>) {

        // 计算内层实际大小
        %inner_size = affine.min #dynamic_bound(%inner_iv)[%outer_size, %tb_inner_m]

        // 进一步切片
        %inner_subview_a = memref.subview %subview_a[%inner_iv] [$inner_size] [1]
                           : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %inner_subview_b = memref.subview %subview_b[%inner_iv, 0] [%inner_size, %dim_n] [1, 1]
                           : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %inner_subview_acc = memref.subview %inner_acc[%inner_iv] [%inner_size] [1]
                             : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>

        // ---- 核心计算 ----
        linalg.generic {
          indexing_maps = [#broadcast_map, #full_access_map, #broadcast_map],
          iterator_types = ["parallel", "reduction"]
        } ins(%inner_subview_a, %inner_subview_b
              : memref<?xf16, strided<[1], offset: ?>>,
                memref<?x?xf16, strided<[?, 1], offset: ?>>)
          outs(%inner_subview_acc : memref<?xf16, strided<[1], offset: ?>>)
          attrs = {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%a_val: f16, %b_val: f16, %acc: f16):
          %sum = arith.addf %a_val, %b_val : f16
          %new_acc = arith.addf %acc, %sum : f16
          linalg.yield %new_acc : f16
        }

        // 写回结果 (memref.copy)
        memref.copy %inner_subview_acc, %inner_subview_acc
            : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>

        scf.yield %inner_acc : memref<?xf16, strided<[1], offset: ?>>

      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}

      // 写回外层结果
      memref.copy %result_after_inner, %subview_acc
          : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>

      scf.yield %outer_acc : memref<?xf16>

    } {ascendc.parallel = true}

    return %result_after_outer : memref<?xf16>
  }

  // Transform 调度脚本 (保留)
  transform.named_sequence @__transform_main(
      %root_op: !transform.any_op {transform.readonly}
  ) {
    %matched_func = transform.structured.match ops{["func.func"]} in %root_op
        : (!transform.any_op) -> !transform.any_op
    %transformed_func, %new_arg_0, %new_arg_1 =
        transform.func.add_index_args %matched_func, 2
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %matched_generic = transform.structured.match ops{["linalg.generic"]} in %transformed_func
        : (!transform.any_op) -> !transform.any_op
    %tiled_op, %outer_loop = transform.structured.tile_using_for %matched_generic
        tile_sizes [%new_arg_0, 0]
        : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    %true_val = transform.param.constant true -> !transform.any_param
    transform.annotate %outer_loop "ascendc.parallel" = %true_val
        : !transform.any_op, !transform.any_param
    %tiled_inner_op, %inner_loop = transform.structured.tile_using_for %tiled_op
        tile_sizes [%new_arg_1, 0]
        : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    %prologue_val = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %epilogue_val = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    transform.annotate %inner_loop "ascendc.prologue" = %prologue_val
        : !transform.any_op, !transform.any_param
    transform.annotate %inner_loop "ascendc.epilogue" = %epilogue_val
        : !transform.any_op, !transform.any_param
    %unit_val = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.annotate %tiled_inner_op "ascendc.unit" = %unit_val
        : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %inner_loop : !transform.any_op
    transform.yield
  }
}
