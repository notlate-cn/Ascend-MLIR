// RUN: rm -f %t.seed.db %t.out.db
// RUN: echo '# ascend.schedule.tuning_db schema=1' > %t.seed.db
// RUN: echo 'record schema=1 target=Ascend910B2 policy=legacy-default signature=vector_generic|single_tile_per_block|64|32 family=vector_generic template=single_tile_per_block result=64 tile=32 score=123 cycle_count=123 profile=/tmp/profile.json source=autotuner' >> %t.seed.db
// RUN: echo 'negative schema=1 target=Ascend910B2 policy=legacy-default signature=vector_generic|single_tile_per_block|64|64 reason=validation_fail score=999 profile=/tmp/bad-profile.json source=autotuner' >> %t.seed.db
// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default tuning-db-in=%t.seed.db tuning-db-out=%t.out.db dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=CACHE
// RUN: FileCheck %s --input-file=%t.out.db --check-prefix=DB

func.func @tuning_db_profile_vector(%arg0: tensor<64xf32>,
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

// CACHE: ScheduleCache:
// CACHE: persistent_tuning_hits = 1
// DB: record schema=1 target=Ascend910B2 policy=legacy-default
// DB-SAME: signature=vector_generic|single_tile_per_block|64|32
// DB-SAME: score=123
// DB-SAME: cycle_count=123
// DB-SAME: profile=/tmp/profile.json
// DB-SAME: source=autotuner
// DB: negative schema=1 target=Ascend910B2 policy=legacy-default
// DB-SAME: signature=vector_generic|single_tile_per_block|64|64
// DB-SAME: reason=validation_fail
// DB-SAME: score=999
// DB-SAME: profile=/tmp/bad-profile.json
// DB-SAME: source=autotuner
