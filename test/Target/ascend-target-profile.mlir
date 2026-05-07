// REQUIRES: ascend_env
// RUN: afir-opt --ascend-print-target-profile='soc=Ascend910B2 cann-root=%cann_root' %s 2>&1 | FileCheck %s

module {}

// CHECK: TargetProfile
// CHECK: soc = "Ascend910B2"
// CHECK: memory_place = "GM"
// CHECK: memory_place = "L1"
// CHECK: memory_place = "UB"
// CHECK: intrinsic = "Intrinsic_data_move_out2l1"
