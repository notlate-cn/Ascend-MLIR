// RUN: not afir-opt %s --vector-plan-codegen 2>&1 | FileCheck %s

// LinalgToAscendC must fail loudly on a compute op it can't lower (here
// arith.divf), instead of silently emitting a kernel that drops it.

// CHECK: error: LinalgToAscendC: unsupported op in linalg.generic body: arith.divf
func.func @divkernel(%a: tensor<64xf32>, %b: tensor<64xf32>, %o: tensor<64xf32>) -> tensor<64xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<64xf32>, tensor<64xf32>) outs(%o : tensor<64xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %d = arith.divf %x, %y : f32
    linalg.yield %d : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}
