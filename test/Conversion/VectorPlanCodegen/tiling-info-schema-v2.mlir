// RUN: afir-opt %s --vector-plan-codegen 2>&1 | FileCheck %s

// Dynamic elementwise: shape-equal output, two inputs, one tunable axis.
// We use a dynamic dim so the kernel materializes tensor.dim ops -- those are
// what TilePlanGen's schema-author scans for `shape_derived` fields.
//
// The output is a single-line module attr; dict keys are sorted
// alphabetically (`args`, ..., `fields`, `kernel_id`, `schema_version`).
// CHECK-SAME order follows that.
//
// CHECK: vector_plan.tiling_infos
// CHECK-SAME: role = "input"
// CHECK-SAME: role = "tile_param"
// CHECK-SAME: role = "output"
// CHECK-SAME: shape_expr = ["arg0_dim0"]
// CHECK-SAME: kind = "tunable"
// CHECK-SAME: name = "XBLOCK"
// CHECK-SAME: kind = "shape_derived"
// CHECK-SAME: schema_version = 2

func.func @add_1d_dyn(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %d  = tensor.dim %a, %c0 : tensor<?xf32>
  %o  = tensor.empty(%d) : tensor<?xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>,
                     affine_map<(i) -> (i)>,
                     affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]}
    ins(%a, %b : tensor<?xf32>, tensor<?xf32>)
    outs(%o : tensor<?xf32>) {
  ^bb0(%x: f32, %y: f32, %_: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<?xf32>
  return %r : tensor<?xf32>
}
