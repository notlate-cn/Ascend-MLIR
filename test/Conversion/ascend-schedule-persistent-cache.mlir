// RUN: rm -f %t.cache
// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default tuning-cache-out=%t.cache' | FileCheck %s --check-prefix=IR
// RUN: test -s %t.cache
// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default tuning-cache-in=%t.cache dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=CACHE

func.func @persistent_cache_vector(%arg0: tensor<64xf32>,
                                   %arg1: tensor<64xf32>,
                                   %out: tensor<64xf32>) -> tensor<64xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>)
    outs(%out : tensor<64xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %r = arith.addf %a, %b : f32
    linalg.yield %r : f32
  } -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// IR: ascend.schedule.decision_id
// CACHE: ScheduleCache:
// CACHE: persistent_tuning_hits = 1
