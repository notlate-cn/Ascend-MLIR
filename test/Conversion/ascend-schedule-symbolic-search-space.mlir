// RUN: sed -n '/\/\/ BASIC-BEGIN/,/\/\/ BASIC-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=BASIC
// RUN: sed -n '/\/\/ BUDGET-BEGIN/,/\/\/ BUDGET-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule max-search-budget=1' 2>&1 | FileCheck %s --check-prefix=BUDGET

// BASIC-BEGIN
func.func @symbolic_dynamic_vector(%arg0: tensor<?x?xf16>,
                                   %arg1: tensor<?x?xf16>,
                                   %out: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %add = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%out : tensor<?x?xf16>) {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %v = arith.addf %x, %y : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>
  return %add : tensor<?x?xf16>
}

// BASIC: ScheduleSearch:
// BASIC:   generated =
// BASIC:   kept = 4
// BASIC:   compile_time_top_k = 4
// BASIC: ScheduleGuards:
// BASIC-NEXT:   kernel = kernel_0
// BASIC-NEXT:   candidate_guards = 4
// BASIC-NEXT:   decision_guards = 0
// BASIC-NEXT:   guard_budget = 8
// BASIC-NEXT:   pruned_by_guard_budget = 0
// BASIC-NEXT:   candidate_guard = T_arg0_dim0 > 0
// BASIC-NEXT:   candidate_guard = T_arg0_dim0 <= arg0_dim0
// BASIC-NEXT:   candidate_guard = T_arg0_dim1 > 0
// BASIC-NEXT:   candidate_guard = T_arg0_dim1 <= arg0_dim1
// BASIC-NEXT:   kept_instance = kernel_0.vector_generic.1
// BASIC-NEXT:   candidate_guard = T_arg0_dim1 > 0
// BASIC-NEXT:   candidate_guard = T_arg0_dim1 <= arg0_dim1
// BASIC-NEXT:   kept_instance = kernel_0.vector_generic.2
// BASIC-NEXT:   candidate_guard = T_arg0_dim0 > 0
// BASIC-NEXT:   candidate_guard = T_arg0_dim0 <= arg0_dim0
// BASIC: ScheduleDecisionSet:
// BASIC:   tile_params = [name=T_arg0_dim0 axis=0 binding=runtime
// BASIC-SAME: [name=T_arg0_dim1 axis=1 binding=runtime
// BASIC-END

// BUDGET-BEGIN
func.func @symbolic_rank5_guard_budget(%arg0: tensor<?x?x?x?x?xf16>,
                                       %arg1: tensor<?x?x?x?x?xf16>,
                                       %out: tensor<?x?x?x?x?xf16>)
                                       -> tensor<?x?x?x?x?xf16> {
  %add = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?x?x?x?xf16>, tensor<?x?x?x?x?xf16>)
      outs(%out : tensor<?x?x?x?x?xf16>) {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %v = arith.addf %x, %y : f16
      linalg.yield %v : f16
    } -> tensor<?x?x?x?x?xf16>
  return %add : tensor<?x?x?x?x?xf16>
}

// BUDGET: ScheduleProblem:
// BUDGET:   guard_budget = 8
// BUDGET: ScheduleSearch:
// BUDGET-NEXT:   kernel = kernel_0
// BUDGET:   kept = 2
// BUDGET: ScheduleGuards:
// BUDGET-NEXT:   kernel = kernel_0
// BUDGET-NEXT:   candidate_guards = 2
// BUDGET-NEXT:   decision_guards = 0
// BUDGET-NEXT:   guard_budget = 8
// BUDGET-NEXT:   pruned_by_guard_budget = 1
// BUDGET-NEXT:   candidate_guard = T_arg0_dim0 > 0
// BUDGET-NEXT:   candidate_guard = T_arg0_dim0 <= arg0_dim0
// BUDGET: ScheduleDecisionSet:
// BUDGET:   selected = kernel_0.decision.0
// BUDGET:   tile_params = [name=T_arg0_dim0 axis=0 binding=runtime
// BUDGET-SAME: [name=T_arg0_dim1 axis=1 binding=extent
// BUDGET-SAME: [name=T_arg0_dim4 axis=4 binding=extent
// BUDGET-END
