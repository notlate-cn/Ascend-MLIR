// RUN: not afir-opt %s --auto-fuse-codegen 2>&1 | FileCheck %s

// LinalgToAscendC must fail loudly on a compute op it can't lower (here
// math.cos), instead of silently emitting a kernel that drops it.
// (arith.divf and math.erf are now supported — see the GELU vector path.)

// CHECK: error: LinalgToAscendC: unsupported op in linalg.generic body: math.cos
func.func @coskernel(%a: tensor<64xf32>, %o: tensor<64xf32>) -> tensor<64xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
                       iterator_types = ["parallel"]}
       ins(%a : tensor<64xf32>) outs(%o : tensor<64xf32>) {
  ^bb0(%x: f32, %out: f32):
    %d = math.cos %x : f32
    linalg.yield %d : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}
