// REQUIRES: ascend_env
// RUN: afir-opt --ascend-print-target-profile='soc=Ascend910B2 cann-root=%cann_root' %s 2>&1 | FileCheck %s

module {}

// CHECK: TargetProfile
// CHECK: soc = "Ascend910B2"
// CHECK: memory_place = "GM"
// CHECK: memory_place = "A1"
// CHECK: memory_place = "A2"
// CHECK: memory_place = "B1"
// CHECK: memory_place = "B2"
// CHECK: memory_place = "CO1"
// CHECK: memory_place = "VECIN"
// CHECK: memory_place = "VECOUT"
// CHECK: memory_place = "VECCALC"
// CHECK: intrinsic = "Intrinsic_data_move_out2l1"
// CHECK: intrinsic = "Intrinsic_mmad"
// CHECK: intrinsic = "Intrinsic_vadd"
