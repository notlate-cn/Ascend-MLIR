// RUN: afir-opt --auto-fuse-codegen %s | afir-translate -mlir-to-cann | FileCheck %s

// Bias-add broadcast [192] -> [8,2,192] (broadcast on the two LEADING axes).
// The DecomposeMultiAxisBroadcast pass must peel the inner axis (1) before the
// outer axis (0); peeling axis 0 first strands axis 1 (non-1 prefix AND suffix)
// and the broadcast falls back to PyAsc's broken default printer
// (reinterpret_cast<uint64_t> on an i32, which is illegal C++).

#m  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#mb = affine_map<(d0, d1, d2) -> (d2)>

func.func @bias_bcast(%a: tensor<8x2x192xf32>, %bias: tensor<192xf32>,
                      %init: tensor<8x2x192xf32>) -> tensor<8x2x192xf32> {
  %0 = linalg.generic {indexing_maps = [#m, #mb, #m],
                       iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%a, %bias : tensor<8x2x192xf32>, tensor<192xf32>)
       outs(%init : tensor<8x2x192xf32>) {
  ^bb0(%x: f32, %b: f32, %o: f32):
    %1 = arith.addf %x, %b : f32
    linalg.yield %1 : f32
  } -> tensor<8x2x192xf32>
  return %0 : tensor<8x2x192xf32>
}

// Folds to valid 2D row broadcasts; never the illegal reinterpret_cast form.
// CHECK: AscendC::Broadcast<float, 2, 0>
// CHECK-NOT: Broadcast{{.*}}reinterpret_cast
