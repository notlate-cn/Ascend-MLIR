// RUN: afir-opt %s --ascendc-decompose-multi-axis-broadcast 2>&1 | FileCheck %s

// =============================================================================
// Multi-axis broadcast: src [1, D1, 1] -> dst [D0, D1, D2]
// Broadcast axes are 0 and 2; the pass must rewrite the single 3-D broadcast
// into a chain of two single-axis broadcasts with an intermediate VECCALC tile.
// =============================================================================

// CHECK-LABEL: func.func @multi_axis_broadcast
// CHECK: %[[PIPE:.*]] = ascendc.pipe
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[D0:.*]] = arith.constant 4 : i32
// CHECK: %[[D1:.*]] = arith.constant 8 : i32
// CHECK: %[[D2:.*]] = arith.constant 16 : i32
// First broadcast in the chain allocates an intermediate VECCALC tile and
// expands axis 0 only; src/dst shapes still carry the constant-1 in axis 2.
// CHECK: ascendc.tbuf : <veccalc>
// CHECK: ascendc.pipe.init_buffer
// CHECK: %[[TMP:.*]] = ascendc.tbuf.get_tensor
// CHECK: ascendc.broadcast_l2 %[[TMP]], %{{.*}}, %[[D0]], %[[D1]], %[[ONE]], %[[ONE]], %[[D1]], %[[ONE]]
// Last broadcast writes to the original dst and expands axis 2.
// CHECK: ascendc.broadcast_l2 %{{.*}}, %[[TMP]], %[[D0]], %[[D1]], %[[D2]], %[[D0]], %[[D1]], %[[ONE]]
// Original 3-axis broadcast is gone (dstShape carries 3 non-const-1 entries).
// CHECK-NOT: ascendc.broadcast_l2 %{{.*}}, %{{.*}}, %[[D0]], %[[D1]], %[[D2]], %[[ONE]], %[[D1]], %[[ONE]]

func.func @multi_axis_broadcast(%src: !ascendc.local_tensor<*xf32>,
                                  %dst: !ascendc.local_tensor<*xf32>) {
  %pipe = ascendc.pipe
  %c1   = arith.constant 1  : i32
  %d0   = arith.constant 4  : i32
  %d1   = arith.constant 8  : i32
  %d2   = arith.constant 16 : i32
  ascendc.broadcast_l2 %dst, %src, %d0, %d1, %d2, %c1, %d1, %c1
    {constRank = 3 : i32, operandSegmentSizes = array<i32: 1, 1, 3, 3>}
    : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>,
      i32, i32, i32, i32, i32, i32
  return
}

// =============================================================================
// Single-axis broadcast must be left untouched.
// =============================================================================

// CHECK-LABEL: func.func @single_axis_broadcast
// CHECK: ascendc.broadcast_l2 %arg1, %arg0, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
// CHECK-NOT: ascendc.tbuf
// CHECK-NOT: ascendc.broadcast_l2 %{{.*}}, %{{.*}}

func.func @single_axis_broadcast(%src: !ascendc.local_tensor<*xf32>,
                                   %dst: !ascendc.local_tensor<*xf32>) {
  %pipe = ascendc.pipe
  %c1   = arith.constant 1  : i32
  %d0   = arith.constant 4  : i32
  %d1   = arith.constant 8  : i32
  %d2   = arith.constant 16 : i32
  // src [D0, 1, D2] -> dst [D0, D1, D2], broadcast only on axis 1.
  ascendc.broadcast_l2 %dst, %src, %d0, %d1, %d2, %d0, %c1, %d2
    {constRank = 3 : i32, operandSegmentSizes = array<i32: 1, 1, 3, 3>}
    : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>,
      i32, i32, i32, i32, i32, i32
  return
}
