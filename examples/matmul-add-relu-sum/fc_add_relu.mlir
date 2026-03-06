// 文件: fc_add_relu.mlir
// 函数: fc_relu
// 功能: 全连接层 + 偏置 + ReLU 激活函数

func.func @fc_relu(%lhs: tensor<?x?xf32>,
                   %rhs: tensor<?x?xf32>,
                   %bias: tensor<?x?xf32>,
                   %output: tensor<?x?xf32>)
                   -> tensor<?x?xf32> {
  // 操作 1: 矩阵乘法
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%output : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  // 操作 2: 加偏置 (逐元素加法)
  %biased = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%matmul, %bias : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%output : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  // 操作 3: ReLU (逐元素 max(x, 0))
  %c0f = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %biased, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %biased, %c1 : tensor<?x?xf32>
  %zero_tensor = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %filled_zero = linalg.fill ins(%c0f : f32) outs(%zero_tensor : tensor<?x?xf32>) -> tensor<?x?xf32>
  %relued = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
    ins(%biased, %filled_zero : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%output : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  return %relued : tensor<?x?xf32>
}