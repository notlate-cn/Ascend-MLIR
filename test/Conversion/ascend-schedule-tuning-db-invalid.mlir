// RUN: echo '# ascend.schedule.tuning_db schema=2' > %t.bad
// RUN: not ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default tuning-db-in=%t.bad' 2>&1 | FileCheck %s

func.func @invalid_tuning_db(%arg0: tensor<8xf32>,
                             %arg1: tensor<8xf32>,
                             %out: tensor<8xf32>) -> tensor<8xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%out : tensor<8xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %r = arith.addf %a, %b : f32
    linalg.yield %r : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK: failed to read ascend schedule tuning database file
