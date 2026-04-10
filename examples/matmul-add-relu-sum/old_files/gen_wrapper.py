TB_M, TB_N, tb_M, tb_N, t_K = 128, 128, 64, 64, 32

print(f"""
module {{
  // 声明 fc_relu 为外部函数（private，链接时解析）
  func.func private @fc_relu(
      memref<?x?xf32, strided<[?, ?], offset: ?>>,
      memref<?x?xf32, strided<[?, ?], offset: ?>>,
      memref<?x?xf32, strided<[?, ?], offset: ?>>,
      memref<?x?xf32, strided<[?, ?], offset: ?>>,
      i64, i64, i64, i64, i64
  ) -> memref<?x?xf32, strided<[?, ?], offset: ?>>

  func.func @fc_relu_spec(
      %a:    memref<?x?xf32, strided<[?, ?], offset: ?>>,
      %b:    memref<?x?xf32, strided<[?, ?], offset: ?>>,
      %bias: memref<?x?xf32, strided<[?, ?], offset: ?>>,
      %c:    memref<?x?xf32, strided<[?, ?], offset: ?>>
  ) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {{
    %TB_M = arith.constant {TB_M} : i64
    %TB_N = arith.constant {TB_N} : i64
    %tb_M = arith.constant {tb_M} : i64
    %tb_N = arith.constant {tb_N} : i64
    %t_K  = arith.constant {t_K}  : i64
    %r = func.call @fc_relu(%a, %b, %bias, %c,
                             %TB_M, %TB_N, %tb_M, %tb_N, %t_K)
       : (memref<?x?xf32, strided<[?, ?], offset: ?>>,
          memref<?x?xf32, strided<[?, ?], offset: ?>>,
          memref<?x?xf32, strided<[?, ?], offset: ?>>,
          memref<?x?xf32, strided<[?, ?], offset: ?>>,
          i64, i64, i64, i64, i64)
       -> memref<?x?xf32, strided<[?, ?], offset: ?>>
    return %r : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }}
}}
""")