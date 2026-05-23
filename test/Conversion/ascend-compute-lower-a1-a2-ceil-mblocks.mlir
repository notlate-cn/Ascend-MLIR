// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

// Regression: A1->A2 LoadData src_stride must not floor dynamic M/16 to zero.
// Small transformer tiles can have M < 16, so the packed A matrix still needs
// one M block.
// CHECK-LABEL: func.func @copy_dynamic_m_a1_to_a2
// CHECK-DAG: %[[C15:.*]] = arith.constant 15 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[PAD_NUM:.*]] = arith.addi %{{.*}}, %[[C15]] : index
// CHECK: %[[PAD_BLOCKS:.*]] = arith.divui %[[PAD_NUM]], %[[C16]] : index
// CHECK: %[[PAD_M:.*]] = arith.muli %[[PAD_BLOCKS]], %[[C16]] : index
// CHECK: %[[PAD_ELEMS:.*]] = arith.muli %[[PAD_M]], %[[C128]] : index
// CHECK: %[[PAD_BYTES:.*]] = arith.muli %[[PAD_ELEMS]], %[[C4]] : index
// CHECK: ascendc.pipe.init_queue {{.*}}, %[[PAD_BYTES]]
// CHECK: ascendc.que_bind.deque_tensor
// CHECK: %[[M_BLOCKS_NUM:.*]] = arith.addi %{{.*}}, %[[C15]] : index
// CHECK: %[[M_BLOCKS:.*]] = arith.divui %[[M_BLOCKS_NUM]], %[[C16]] : index
// CHECK: %[[M_BLOCKS_I16:.*]] = arith.index_cast %[[M_BLOCKS]] : index to i16
// CHECK: ascendc.construct !ascendc.load_data_2d_params({{.*}}%[[M_BLOCKS_I16]]
// CHECK: ascendc.load_data_l0
// CHECK-NOT: memref.copy
func.func @copy_dynamic_m_a1_to_a2(%m: index) {
  %src = memref.alloc(%m) : memref<?x128xf32, 1 : i32>
  %dst = memref.alloc(%m) : memref<?x128xf32, 2 : i32>
  memref.copy %src, %dst
      : memref<?x128xf32, 1 : i32> to memref<?x128xf32, 2 : i32>
  return
}

// CHECK-LABEL: func.func @copy_dynamic_m_gm_to_a1
// CHECK-DAG: %[[C15_GM:.*]] = arith.constant 15 : index
// CHECK-DAG: %[[C16_GM:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C128_GM:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C4_GM:.*]] = arith.constant 4 : index
// CHECK: %[[PAD_NUM_GM:.*]] = arith.addi %{{.*}}, %[[C15_GM]] : index
// CHECK: %[[PAD_BLOCKS_GM:.*]] = arith.divui %[[PAD_NUM_GM]], %[[C16_GM]] : index
// CHECK: %[[PAD_M_GM:.*]] = arith.muli %[[PAD_BLOCKS_GM]], %[[C16_GM]] : index
// CHECK: %[[PAD_ELEMS_GM:.*]] = arith.muli %[[PAD_M_GM]], %[[C128_GM]] : index
// CHECK: %[[PAD_BYTES_GM:.*]] = arith.muli %[[PAD_ELEMS_GM]], %[[C4_GM]] : index
// CHECK: ascendc.pipe.init_queue {{.*}}, %[[PAD_BYTES_GM]]
// CHECK: %[[DST_STRIDE_NUM:.*]] = arith.addi %{{.*}}, %[[C15_GM]] : index
// CHECK: %[[DST_STRIDE_BLOCKS:.*]] = arith.divui %[[DST_STRIDE_NUM]], %[[C16_GM]] : index
// CHECK: %[[DST_STRIDE:.*]] = arith.muli %[[DST_STRIDE_BLOCKS]], %[[C16_GM]] : index
// CHECK: %[[DST_STRIDE_I16:.*]] = arith.index_cast %[[DST_STRIDE]] : index to i16
// CHECK: ascendc.construct !ascendc.nd2nz_params({{.*}}%[[DST_STRIDE_I16]]
func.func @copy_dynamic_m_gm_to_a1(%src: memref<?x128xf32>, %m: index) {
  %dst = memref.alloc(%m) : memref<?x128xf32, 1 : i32>
  memref.copy %src, %dst
      : memref<?x128xf32> to memref<?x128xf32, 1 : i32>
  return
}
