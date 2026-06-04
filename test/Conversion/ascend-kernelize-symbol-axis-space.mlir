// RUN: sed -n '/\/\/ PERMUTED-BEGIN/,/\/\/ PERMUTED-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s --check-prefix=PERMUTED

// PERMUTED-BEGIN
func.func @kernelize_symbol_axis_space_transpose(
    %arg0: tensor<?x?xf32>, %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf32>)
      outs(%out : tensor<?x?xf32>) {
    ^bb0(%x: f32, %outv: f32):
      linalg.yield %x : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// PERMUTED-END

// PERMUTED: DependencyAnalysis
// PERMUTED: GlobalAxisSpace
// PERMUTED-DAG: axis_id = 0 sym = "arg0_dim0" kind = "parallel"
// PERMUTED-DAG: axis_id = 1 sym = "arg0_dim1" kind = "parallel"
// PERMUTED: OpAxisMap
// PERMUTED: op_id = 0 axes = [1:"arg0_dim1", 0:"arg0_dim0"]
