// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

// CHECK-LABEL: func.func @named_rank2_leading_swap_gm
// CHECK: ascendc.data_copy_l2
// CHECK: ascendc.transpose
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: memref.load
// CHECK-NOT: memref.store
// CHECK-NOT: linalg.transpose
func.func @named_rank2_leading_swap_gm(%src: memref<384x128xf32>,
                                       %dst: memref<128x384xf32>) {
  linalg.transpose ins(%src : memref<384x128xf32>)
      outs(%dst : memref<128x384xf32>)
      permutation = [1, 0]
  return
}

// CHECK-LABEL: func.func @named_rank2_leading_swap_gm_selected_tile
// CHECK: scf.for %{{.*}} = %c0 to %c128 step %c32
// CHECK-NOT: memref.subview
// CHECK: ascendc.global_tensor.set_global_buffer %{{.*}}, %arg0
// CHECK: emitasc.verbatim
// CHECK-SAME: GetPhyAddr
// CHECK-SAME: DataCopyPad($0, _afir_src
// CHECK: ascendc.transpose
// CHECK: ascendc.global_tensor.set_global_buffer %{{.*}}, %arg1
// CHECK: ascendc.global_tensor.bracket
// CHECK-NOT: linalg.transpose
func.func @named_rank2_leading_swap_gm_selected_tile(
    %src: memref<384x128xf32>,
    %dst: memref<128x384xf32>) {
  linalg.transpose
      {ascend.schedule.selected_tile_shape = array<i64: 32, 384>}
      ins(%src : memref<384x128xf32>)
      outs(%dst : memref<128x384xf32>)
      permutation = [1, 0]
  return
}

// CHECK-LABEL: func.func @named_rank3_leading_swap_gm
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: linalg.transpose
func.func @named_rank3_leading_swap_gm(%src: memref<?x?x128xf32>,
                                       %dst: memref<?x?x128xf32>) {
  linalg.transpose ins(%src : memref<?x?x128xf32>)
      outs(%dst : memref<?x?x128xf32>)
      permutation = [1, 0, 2]
  return
}

// CHECK-LABEL: func.func @named_rank5_permutation_gm
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: linalg.transpose
func.func @named_rank5_permutation_gm(%src: memref<1x2x3x4x5xf32>,
                                      %dst: memref<4x2x3x1x5xf32>) {
  linalg.transpose ins(%src : memref<1x2x3x4x5xf32>)
      outs(%dst : memref<4x2x3x1x5xf32>)
      permutation = [3, 1, 2, 0, 4]
  return
}

// CHECK-LABEL: func.func @named_rank4_non_involutive_permutation_gm
// CHECK: scf.for [[D0:%[^ ]+]] =
// CHECK: scf.for [[D1:%[^ ]+]] =
// CHECK: scf.for [[D2:%[^ ]+]] =
// CHECK: scf.for [[D3:%[^ ]+]] =
// CHECK: memref.load %arg0{{\[}}[[D1]], [[D2]], [[D0]], [[D3]]{{\]}}
// CHECK: memref.store {{.*}}, %arg1{{\[}}[[D0]], [[D1]], [[D2]], [[D3]]{{\]}}
// CHECK-NOT: linalg.transpose
func.func @named_rank4_non_involutive_permutation_gm(
    %src: memref<1x4x16x32xf32>,
    %dst: memref<16x1x4x32xf32>) {
  linalg.transpose ins(%src : memref<1x4x16x32xf32>)
      outs(%dst : memref<16x1x4x32xf32>)
      permutation = [2, 0, 1, 3]
  return
}
