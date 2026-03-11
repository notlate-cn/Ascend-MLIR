// RUN: afir-opt --linalg-mark %s | FileCheck %s
module {
  func.func @test() -> (tensor<2xf32>, tensor<4x3xf32>, tensor<2x3xf32>, tensor<3x2xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32

    %A = arith.constant dense<[[1.0, 2.0],
                                [5.0, 6.0],
                                [9.0, 10.0],
                                [13.0, 14.0]]> : tensor<4x2xf32>
    %B = arith.constant dense<[[0.1, 0.2, 0.3],
                                [0.5, 0.6, 0.7]]> : tensor<2x3xf32>
    %C = arith.constant dense<0.0> : tensor<2xf32>
    %D = arith.constant dense<0.0> : tensor<4x3xf32>  
    %E = arith.constant dense<0.0> : tensor<2x3xf32>   
    %F = arith.constant dense<0.0> : tensor<3x2xf32>
    
    // CHECK: {namedKind = 0 : index}
    %generic_result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1)->(d0, d1)>, affine_map<(d0, d1)->(d1, d0)>, affine_map<(d0, d1)->(d1)>],
        iterator_types = ["reduction", "parallel"]
    } ins(%F, %B : tensor<3x2xf32>, tensor<2x3xf32>)
      outs(%C : tensor<2xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %sum = arith.addf %arg0, %arg1 : f32
      %acc = arith.addf %arg2, %sum : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>
    // CHECK: {namedKind = 1 : index}
    %matmul_result = linalg.matmul ins(%A, %B : tensor<4x2xf32>, tensor<2x3xf32>)
                                 outs(%D : tensor<4x3xf32>) -> tensor<4x3xf32>

    

    %A2x3 = arith.constant dense<[[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
    // CHECK: {namedKind = 2 : index}
    %add_result = linalg.add ins(%A2x3, %B : tensor<2x3xf32>, tensor<2x3xf32>)
                           outs(%E : tensor<2x3xf32>) -> tensor<2x3xf32>
    %G = arith.constant dense<0.0> : tensor<3x2xf32>
    // CHECK: {namedKind = 3 : index}
    %transpose_result = linalg.transpose ins(%A2x3 : tensor<2x3xf32>)
                                        outs(%G : tensor<3x2xf32>)
                                        permutation = [1, 0]

    return %generic_result, %matmul_result, %add_result, %transpose_result : tensor<2xf32>, tensor<4x3xf32>, tensor<2x3xf32>, tensor<3x2xf32>
  }
}