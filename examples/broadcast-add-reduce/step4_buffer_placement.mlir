// ============================================================
// STAGE 4: Buffer Placement - 缓冲区放置优化
//
// 本阶段引入片上内存层级，优化数据搬运：
//   - 在 UB (Unified Buffer) 上分配临时缓冲区
//   - 使用 memref.alloc 指定 memory_space 属性
//   - memory_space 9  = VECIN (向量输入缓冲区)
//   - memory_space 10 = VECOUT (向量输出缓冲区)
//
// 数据流优化：
//   - 内层循环前: GM → VECIN (数据预取)
//   - 计算: 在 VECIN/VECOUT 上执行
//   - 内层循环后: VECOUT → GM (写回)
//
// 关键变化：
//   - 在内层循环中分配 UB 缓冲区
//   - 使用 memref.copy 进行 GM ↔ UB 数据搬运
//   - 计算操作在 UB 上执行
// ============================================================

// 动态维度边界计算
#dynamic_bound = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
// 广播映射
#broadcast_map = affine_map<(d0, d1) -> (d0)>
// 完整访问映射
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  // 主函数：带缓冲区放置的广播加法归约
  func.func @broadcast_add_reducesum(
      %input_a: memref<?xf16>,       // 输入 A [M] - GM
      %input_b: memref<?x?xf16>,     // 输入 B [M,N] - GM
      %tb_m_param: i64,              // TB_M
      %tb_inner_m_param: i64         // Tb_M
  ) -> memref<?xf16> {

    // 常量
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
    linalg.fill ins(%zero : f16) outs(%output_buffer : memref<?xf16>)

    // ---- 外层循环: TB 层 ----
    %result_after_outer = scf.for %outer_iv = %idx_0 to %dim_m step %tb_m
        iter_args(%outer_acc = %output_buffer) -> (memref<?xf16>) {

      %outer_size = affine.min #dynamic_bound(%outer_iv)[%dim_m, %tb_m]

      // 创建 GM 子视图
      %subview_a = memref.subview %input_a[%outer_iv] [%outer_size] [1]
                   : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_b = memref.subview %input_b[%outer_iv, 0] [%outer_size, %dim_n] [1, 1]
                   : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_acc = memref.subview %outer_acc[%outer_iv] [$outer_size] [1]
                     : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>

      // ---- 内层循环: Tb 层 (带 UB 缓冲区) ----
      %result_after_inner = scf.for %inner_iv = %idx_0 to %outer_size step %tb_inner_m
          iter_args(%inner_acc = %subview_acc) -> (memref<?xf16, strided<[1], offset: ?>>) {

        %inner_size = affine.min #dynamic_bound(%inner_iv)[%outer_size, %tb_inner_m]

        // 创建 GM 子视图
        %inner_subview_a = memref.subview %subview_a[%inner_iv] [%inner_size] [1]
                           : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %inner_subview_acc = memref.subview %inner_acc[%inner_iv] [%inner_size] [1]
                             : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>

        // ---- 分配 UB 缓冲区 ----
        // VECIN (memory_space = 9): 用于输入数据
        %ub_buffer_a = memref.alloc(%inner_size) : memref<?xf16, 9 : i32>
        // VECOUT (memory_space = 10): 用于输出数据
        %ub_buffer_acc = memref.alloc(%inner_size) : memref<?xf16, 10 : i32>

        // ---- 数据搬运: GM → VECIN ----
        memref.copy %inner_subview_a, %ub_buffer_a
            : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, 9 : i32>

        // 创建 B 的子视图 (仍在 GM)
        %inner_subview_b = memref.subview %subview_b[%inner_iv, 0] [%inner_size, %dim_n] [1, 1]
                           : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>

        // ---- 核心计算 (在 UB 上) ----
        linalg.generic {
          indexing_maps = [#broadcast_map, #full_access_map, #broadcast_map],
          iterator_types = ["parallel", "reduction"]
        } ins(%ub_buffer_a, %inner_subview_b
              : memref<?xf16, 9 : i32>,
                memref<?x?xf16, strided<[?, 1], offset: ?>>)
          outs(%ub_buffer_acc : memref<?xf16, 10 : i32>) {
        ^bb0(%a_val: f16, %b_val: f16, %acc: f16):
          %sum = arith.addf %a_val, %b_val : f16
          %new_acc = arith.addf %acc, %sum : f16
          linalg.yield %new_acc : f16
        }

        // ---- 数据搬运: VECOUT → GM ----
        memref.copy %ub_buffer_acc, %inner_subview_acc
            : memref<?xf16, 10 : i32> to memref<?xf16, strided<[1], offset: ?>>

        // 释放 UB 缓冲区
        memref.dealloc %ub_buffer_a : memref<?xf16, 9 : i32>
        memref.dealloc %ub_buffer_acc : memref<?xf16, 10 : i32>

        scf.yield %inner_acc : memref<?xf16, strided<[1], offset: ?>>

      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}

      // 写回外层结果
      memref.copy %result_after_inner, %subview_acc
          : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>

      scf.yield %outer_acc : memref<?xf16>

    } {ascendc.parallel = true}

    return %result_after_outer : memref<?xf16>
  }

  // Transform 调度脚本
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
