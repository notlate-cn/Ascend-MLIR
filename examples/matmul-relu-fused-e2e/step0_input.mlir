// CV-fusion probe: matmul + relu epilogue, fp16 inputs / fp32 acc.
//   C[M,N] = relu(A[M,K] @ B[K,N])
func.func @mm_relu(%a: tensor<32x16xf16>,
                    %b: tensor<16x64xf16>,
                    %init: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %c = linalg.matmul
    ins(%a, %b : tensor<32x16xf16>, tensor<16x64xf16>)
    outs(%init : tensor<32x64xf32>) -> tensor<32x64xf32>

  %empty = tensor.empty() : tensor<32x64xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
                     affine_map<(d0,d1)->(d0,d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%c : tensor<32x64xf32>) outs(%empty : tensor<32x64xf32>) {
  ^bb0(%v: f32, %_: f32):
    %z = arith.constant 0.0 : f32
    %t = arith.maximumf %v, %z : f32
    linalg.yield %t : f32
  } -> tensor<32x64xf32>
  return %r : tensor<32x64xf32>
}
