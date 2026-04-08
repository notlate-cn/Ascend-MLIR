// RUN: afir-opt %s --linalg-to-ascendc | FileCheck %s

#map_par_reduce_lhs = affine_map<(d0, d1) -> (d0)>
#map_par_reduce_rhs = affine_map<(d0, d1) -> (d0, d1)>

//===----------------------------------------------------------------------===//
// Data-move: GM → A1 (data_copy_nd2nz)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @copy_gm_to_a1
// CHECK: ascendc.pipe
// CHECK: ascendc.queue
// CHECK: ascendc.que_bind.alloc_tensor
// CHECK: ascendc.global_tensor
// CHECK: ascendc.global_tensor.set_global_buffer
// CHECK: ascendc.construct !ascendc.nd2nz_params
// CHECK: ascendc.data_copy_nd2nz
// CHECK: ascendc.que_bind.enque_tensor
// CHECK-NOT: memref.copy
func.func @copy_gm_to_a1(%src: memref<?x?xf32>) {
  %dst = memref.alloc() : memref<16x16xf32, 1 : i32>
  memref.copy %src, %dst : memref<?x?xf32> to memref<16x16xf32, 1 : i32>
  return
}

//===----------------------------------------------------------------------===//
// Data-move: GM → VECIN (data_copy_l2)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @copy_gm_to_vecin
// CHECK: ascendc.que_bind.alloc_tensor
// CHECK: ascendc.global_tensor
// CHECK: ascendc.global_tensor.set_global_buffer
// CHECK: ascendc.data_copy_l2
// CHECK: ascendc.que_bind.enque_tensor
// CHECK-NOT: memref.copy
func.func @copy_gm_to_vecin(%src: memref<?x?xf32>) {
  %dst = memref.alloc() : memref<16x16xf32, 9 : i32>
  memref.copy %src, %dst : memref<?x?xf32> to memref<16x16xf32, 9 : i32>
  return
}

//===----------------------------------------------------------------------===//
// Data-move: A1 → A2 (load_data_l0)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @copy_a1_to_a2
// CHECK: ascendc.que_bind.deque_tensor
// CHECK: ascendc.que_bind.alloc_tensor
// CHECK: ascendc.construct !ascendc.load_data_2d_params
// CHECK: ascendc.load_data_l0
// CHECK: ascendc.que_bind.enque_tensor
// CHECK: ascendc.que_bind.free_tensor
// CHECK-NOT: memref.copy
func.func @copy_a1_to_a2() {
  %src = memref.alloc() : memref<16x16xf32, 1 : i32>
  %dst = memref.alloc() : memref<16x16xf32, 2 : i32>
  memref.copy %src, %dst : memref<16x16xf32, 1 : i32> to memref<16x16xf32, 2 : i32>
  return
}

//===----------------------------------------------------------------------===//
// Data-move: B1 → B2 (load_data_with_transpose)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @copy_b1_to_b2
// CHECK: ascendc.construct !ascendc.load_data_2d_transpose_params
// CHECK: ascendc.load_data_with_transpose
// CHECK-NOT: memref.copy
func.func @copy_b1_to_b2() {
  %src = memref.alloc() : memref<16x16xf32, 3 : i32>
  %dst = memref.alloc() : memref<16x16xf32, 4 : i32>
  memref.copy %src, %dst : memref<16x16xf32, 3 : i32> to memref<16x16xf32, 4 : i32>
  return
}

//===----------------------------------------------------------------------===//
// Data-move: CO1 → VECIN (data_copy_co12dst)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @copy_co1_to_vecin
// CHECK: ascendc.construct !ascendc.data_copy_co12dst_params
// CHECK: ascendc.data_copy_co12dst
// CHECK-NOT: memref.copy
func.func @copy_co1_to_vecin() {
  %src = memref.alloc() : memref<16x16xf32, 7 : i32>
  %dst = memref.alloc() : memref<16x16xf32, 9 : i32>
  memref.copy %src, %dst : memref<16x16xf32, 7 : i32> to memref<16x16xf32, 9 : i32>
  return
}

//===----------------------------------------------------------------------===//
// Data-move: VECOUT → GM (data_copy_l2)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @copy_vecout_to_gm
// CHECK: ascendc.que_bind.deque_tensor
// CHECK: ascendc.global_tensor
// CHECK: ascendc.global_tensor.set_global_buffer
// CHECK: ascendc.data_copy_l2
// CHECK: ascendc.que_bind.free_tensor
// CHECK-NOT: memref.copy
func.func @copy_vecout_to_gm(%dst: memref<?x?xf32>) {
  %src = memref.alloc() : memref<16x16xf32, 10 : i32>
  memref.copy %src, %dst : memref<16x16xf32, 10 : i32> to memref<?x?xf32>
  return
}

//===----------------------------------------------------------------------===//
// Compute: linalg.fill on CO1 → elided (mmad auto-zeroes CO1)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @test_fill_co1
// CHECK-NOT: linalg.fill
// CHECK-NOT: ascendc.duplicate_l2
func.func @test_fill_co1() {
  %alloc = memref.alloc() : memref<32x32xf32, 7 : i32>
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32, 7 : i32>)
  return
}

//===----------------------------------------------------------------------===//
// Compute: linalg.fill on GM → NOT converted
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @test_fill_gm
// CHECK: linalg.fill
func.func @test_fill_gm(%alloc: memref<32x32xf32>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
  return
}

//===----------------------------------------------------------------------===//
// Compute: linalg.matmul (A2/B2/CO1) → mmad
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @test_matmul
// CHECK-NOT: linalg.matmul
// CHECK: ascendc.que_bind.deque_tensor
// CHECK: ascendc.que_bind.alloc_tensor
// CHECK: ascendc.construct !ascendc.mmad_params
// CHECK: ascendc.mmad {{.*}} {ascendc.unit = "AiCore.Cube"}
// CHECK: ascendc.que_bind.enque_tensor
func.func @test_matmul() {
  %A2  = memref.alloc() : memref<32x64xf32, 2 : i32>
  %B2  = memref.alloc() : memref<64x32xf32, 4 : i32>
  %CO1 = memref.alloc() : memref<32x32xf32, 7 : i32>
  linalg.matmul {ascendc.unit = "AiCore.Cube"}
    ins(%A2, %B2 : memref<32x64xf32, 2 : i32>, memref<64x32xf32, 4 : i32>)
    outs(%CO1 : memref<32x32xf32, 7 : i32>)
  return
}

//===----------------------------------------------------------------------===//
// Compute: linalg.matmul with wrong memory_space → NOT converted
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @test_matmul_no_convert
// CHECK: linalg.matmul
func.func @test_matmul_no_convert(
    %A: memref<32x64xf32>, %B: memref<64x32xf32>, %C: memref<32x32xf32>) {
  linalg.matmul
    ins(%A, %B : memref<32x64xf32>, memref<64x32xf32>)
    outs(%C : memref<32x32xf32>)
  return
}

//===----------------------------------------------------------------------===//
// Compute: linalg.elementwise add → add_l2
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @test_add
// CHECK-NOT: linalg.elementwise
// CHECK: ascendc.add_l2 {{.*}} {ascendc.unit = "AiCore.Vector"}
func.func @test_add() {
  %src0 = memref.alloc() : memref<32x32xf32, 9 : i32>
  %src1 = memref.alloc() : memref<32x32xf32, 9 : i32>
  %dst  = memref.alloc() : memref<32x32xf32, 11 : i32>
  linalg.elementwise kind=#linalg.elementwise_kind<add> {ascendc.unit = "AiCore.Vector"}
    ins(%src0, %src1 : memref<32x32xf32, 9 : i32>, memref<32x32xf32, 9 : i32>)
    outs(%dst : memref<32x32xf32, 11 : i32>)
  return
}

//===----------------------------------------------------------------------===//
// Compute: linalg.fill(0) on VECIN(9) + max_signed → duplicate_l2 + max_l2
// (ReLU).  The fill on VECOUT(10) is elided (max_l2 writes it directly).
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @test_relu
// CHECK-NOT: linalg.elementwise
// CHECK: ascendc.duplicate_l2
// CHECK: ascendc.max_l2 {{.*}} {ascendc.unit = "AiCore.Vector"}
func.func @test_relu() {
  %src      = memref.alloc() : memref<32x32xf32, 11 : i32>
  %zero_buf = memref.alloc() : memref<32x32xf32, 9 : i32>
  %dst      = memref.alloc() : memref<32x32xf32, 10 : i32>
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%zero_buf : memref<32x32xf32, 9 : i32>)
  linalg.elementwise kind=#linalg.elementwise_kind<max_signed> {ascendc.unit = "AiCore.Vector"}
    ins(%src, %zero_buf : memref<32x32xf32, 11 : i32>, memref<32x32xf32, 9 : i32>)
    outs(%dst : memref<32x32xf32, 10 : i32>)
  return
}

//===----------------------------------------------------------------------===//
// Compute: linalg.generic {parallel, reduction} → add_l2 + reduce_sum_2d_l2
// Regression: the yielded addf must map back to the accumulator SSA value
// instead of reading a nonexistent AddL2Op result.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @test_parallel_reduction_add
// CHECK-NOT: linalg.generic
// CHECK: ascendc.add_l2 {{.*}} {ascendc.unit = "AiCore.Vector"}
// CHECK: ascendc.add_l2 {{.*}} {ascendc.unit = "AiCore.Vector"}
// CHECK: ascendc.reduce_sum_2d_l2 {{.*}} {ascendc.unit = "AiCore.Vector"{{.*}}}
func.func @test_parallel_reduction_add() {
  %lhs = memref.alloc() : memref<8xf32, 9 : i32>
  %rhs = memref.alloc() : memref<8x4xf32, 9 : i32>
  %out = memref.alloc() : memref<8xf32, 10 : i32>
  linalg.generic {indexing_maps = [#map_par_reduce_lhs, #map_par_reduce_rhs, #map_par_reduce_lhs], iterator_types = ["parallel", "reduction"], ascendc.unit = "AiCore.Vector"}
    ins(%lhs, %rhs : memref<8xf32, 9 : i32>, memref<8x4xf32, 9 : i32>)
    outs(%out : memref<8xf32, 10 : i32>) {
  ^bb0(%in: f32, %in_0: f32, %acc: f32):
    %sum = arith.addf %in, %in_0 : f32
    %next = arith.addf %acc, %sum : f32
    linalg.yield %next : f32
  }
  return
}

//===----------------------------------------------------------------------===//
// Single pipe per function: all queue/tbuf share one pipe
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @test_single_pipe
// CHECK:       [[PIPE:%.*]] = ascendc.pipe
// CHECK:       ascendc.queue
// CHECK:       ascendc.mmad
// CHECK-NOT:   ascendc.pipe
func.func @test_single_pipe() {
  %a1  = memref.alloc() : memref<16x16xf32, 1 : i32>
  %a2  = memref.alloc() : memref<16x16xf32, 2 : i32>
  %b1  = memref.alloc() : memref<16x16xf32, 3 : i32>
  %b2  = memref.alloc() : memref<16x16xf32, 4 : i32>
  %co1 = memref.alloc() : memref<16x16xf32, 7 : i32>
  memref.copy %a1, %a2  : memref<16x16xf32, 1 : i32> to memref<16x16xf32, 2 : i32>
  memref.copy %b1, %b2  : memref<16x16xf32, 3 : i32> to memref<16x16xf32, 4 : i32>
  linalg.matmul
    ins(%a2, %b2 : memref<16x16xf32, 2 : i32>, memref<16x16xf32, 4 : i32>)
    outs(%co1 : memref<16x16xf32, 7 : i32>)
  return
}
