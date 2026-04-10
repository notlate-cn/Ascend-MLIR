// ============================================================
// fc_add_relu.mlir - 全连接层 + 偏置 + ReLU 激活函数
//
// 计算图: output = ReLU(matmul(input_a, input_b) + bias)
//
// 本文件展示了一个典型的深度学习算子组合:
//   1. MatMul: 矩阵乘法 (全连接层的核心计算)
//   2. Add:    逐元素加法 (添加偏置)
//   3. ReLU:   激活函数 (max(x, 0))
//
// 这是一个高层 IR 表示，使用 tensor 方言和 linalg 方言
// 后续会通过编译流程逐步降级到 AscendC 内核代码
//
// 输入维度:
//   - input_a: [M, K] - 输入特征
//   - input_b: [K, N] - 权重矩阵
//   - bias:    [M, N] - 偏置矩阵
//   - output:  [M, N] - 输出矩阵
// ============================================================

// 主函数: 全连接层 + ReLU
// 参数:
//   %input_a:  输入张量 A [M, K]
//   %input_b:  输入张量 B [K, N] (权重)
//   %bias:     偏置张量 [M, N]
//   %output:   输出张量 [M, N]
func.func @fc_relu(
    %input_a: tensor<?x?xf32>,   // 输入特征 [M, K]
    %input_b: tensor<?x?xf32>,   // 权重矩阵 [K, N]
    %bias: tensor<?x?xf32>,       // 偏置矩阵 [M, N]
    %output: tensor<?x?xf32>     // 输出矩阵 [M, N]
) -> tensor<?x?xf32> {

  // ── 操作 1: 矩阵乘法 ──────────────────────────────────────
  // matmul_result = input_a @ input_b
  // 计算: [M, K] × [K, N] → [M, N]
  %matmul_result = linalg.matmul
    ins(%input_a, %input_b : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%output : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  // ── 操作 2: 加偏置 (逐元素加法) ────────────────────────────
  // biased = matmul_result + bias
  %biased = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%matmul_result, %bias : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%output : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  // ── 操作 3: ReLU 激活函数 ──────────────────────────────────
  // ReLU(x) = max(x, 0)
  // 需要创建一个全零张量用于比较

  // 常量定义
  %zero_f32 = arith.constant 0.0 : f32     // 浮点零值
  %idx_0 = arith.constant 0 : index         // 索引 0
  %idx_1 = arith.constant 1 : index         // 索引 1

  // 获取输出维度
  %dim_m = tensor.dim %biased, %idx_0 : tensor<?x?xf32>  // M 维度
  %dim_n = tensor.dim %biased, %idx_1 : tensor<?x?xf32>  // N 维度

  // 创建全零张量
  %empty_zero = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>
  %zero_tensor = linalg.fill
    ins(%zero_f32 : f32)
    outs(%empty_zero : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  // ReLU: max(biased, 0)
  %relu_result = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
    ins(%biased, %zero_tensor : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%output : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  return %relu_result : tensor<?x?xf32>
}
