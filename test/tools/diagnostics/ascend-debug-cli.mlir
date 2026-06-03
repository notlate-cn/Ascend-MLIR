// RUN: bash %S/test_ascend_debug_cli.sh %s | FileCheck %s

func.func @add_mul_relu(
    %a: tensor<?x?x?xf32>,
    %b: tensor<?x?x?xf32>,
    %c: tensor<?x?x?xf32>,
    %out: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %a, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %a, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %a, %c2 : tensor<?x?x?xf32>
  %tmp_empty = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>

  // b * c
  %tmp = linalg.mul
    ins(%b, %c : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs(%tmp_empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  // a + b*c
  %add_empty = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %add = linalg.add
    ins(%a, %tmp : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs(%add_empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  // relu = max(a + b*c, 0)
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>,
                     affine_map<(d0,d1,d2)->(d0,d1,d2)>],
    iterator_types = ["parallel","parallel","parallel"]}
    ins(%add : tensor<?x?x?xf32>) outs(%out : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %unused: f32):
    %zero = arith.constant 0.0 : f32
    %r = arith.maximumf %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<?x?x?xf32>
  return %result : tensor<?x?x?xf32>
}

// CHECK: ascend-debug.collect.out=
// CHECK: STAGES_OK=6
// CHECK: OPEN_OK=1
