// RUN: afir-opt %s --linalg-to-ascendc | FileCheck %s

#map_par_reduce_lhs = affine_map<(d0, d1) -> (d0)>
#map_par_reduce_rhs = affine_map<(d0, d1) -> (d0, d1)>
#map_bt_lhs = affine_map<(d0, d1) -> (d1, 0)>
#map_2d = affine_map<(d0, d1) -> (d0, d1)>
#map_tail = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>

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
// Compute: linalg.matmul on GM buffers → scalar loop fallback
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @test_matmul_no_convert
// CHECK-NOT: linalg.matmul
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: memref.store
// CHECK-NOT: linalg.matmul
// CHECK: return
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
// Compute: all-parallel generic with VECOUT output and temporary GM input.
// Regression for real NPU: the final tensor must come from the VECOUT queue,
// and temporary VECIN tensors created for GM inputs must be freed.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @test_parallel_vecout_queue_and_temp_free
// CHECK-DAG: %[[OUT_Q:.*]] = ascendc.queue : <vecout, 1>
// CHECK: %[[TMP_Q:.*]] = ascendc.queue : <vecin, 1>
// CHECK: %[[TMP:.*]] = ascendc.que_bind.deque_tensor %[[TMP_Q]]
// CHECK: %[[OUT:.*]] = ascendc.que_bind.alloc_tensor %[[OUT_Q]]
// CHECK: ascendc.add_l2 %[[OUT]], {{.*}}, %[[TMP]]
// CHECK: ascendc.que_bind.enque_tensor %[[OUT_Q]], %[[OUT]]
// CHECK: ascendc.que_bind.free_tensor %[[TMP_Q]], %[[TMP]]
func.func @test_parallel_vecout_queue_and_temp_free(%rhs: memref<?x?xf16>, %m: index, %n: index) {
  %lhs = memref.alloc(%m) : memref<?x1xf16, 9 : i32>
  %out = memref.alloc(%m, %n) : memref<?x?xf16, 10 : i32>
  linalg.generic {indexing_maps = [#map_bt_lhs, #map_2d, #map_2d], iterator_types = ["parallel", "parallel"], ascendc.unit = "AiCore.Vector"}
    ins(%lhs, %rhs : memref<?x1xf16, 9 : i32>, memref<?x?xf16>)
    outs(%out : memref<?x?xf16, 10 : i32>) {
  ^bb0(%in: f16, %in_0: f16, %out_0: f16):
    %sum = arith.addf %in, %in_0 : f16
    linalg.yield %sum : f16
  }
  return
}

//===----------------------------------------------------------------------===//
// Queue-backed allocs should not retain a standalone TBuf if all accesses use
// que_bind.alloc_tensor / deque_tensor. The dead TBuf still consumes real UB
// after hoisting and can push multi-branch kernels over the device limit.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @test_queue_backed_vecout_drops_dead_tbuf
// CHECK: %[[OUT_Q:.*]] = ascendc.queue : <vecout, 1>
// CHECK-NOT: ascendc.tbuf : <vecout>
// CHECK: ascendc.que_bind.alloc_tensor %[[OUT_Q]]
func.func @test_queue_backed_vecout_drops_dead_tbuf(%lhs: memref<?x?xf16, 9 : i32>, %rhs: memref<?x?xf16, 9 : i32>, %m: index, %n: index) {
  %out = memref.alloc(%m, %n) : memref<?x?xf16, 10 : i32>
  linalg.generic {indexing_maps = [#map_2d, #map_2d, #map_2d], iterator_types = ["parallel", "parallel"], ascendc.unit = "AiCore.Vector"}
    ins(%lhs, %rhs : memref<?x?xf16, 9 : i32>, memref<?x?xf16, 9 : i32>)
    outs(%out : memref<?x?xf16, 10 : i32>) {
  ^bb0(%in: f16, %in_0: f16, %out_0: f16):
    %sum = arith.addf %in, %in_0 : f16
    linalg.yield %sum : f16
  }
  return
}

//===----------------------------------------------------------------------===//
// Reduction chunks must accumulate into the previous partial output instead of
// overwriting each N chunk result.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @test_reduction_chunk_copy_accumulates_previous_partial
// CHECK: scf.for %[[R:.*]] =
// CHECK: %[[NOT_FIRST:.*]] = arith.cmpi ne, %[[R]],
// CHECK: scf.if %[[NOT_FIRST]]
// CHECK: ascendc.data_copy_l2 {{.*}} : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, index
// CHECK: ascendc.add_l2
// CHECK: ascendc.data_copy_l2 {{.*}} : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf16>, index
func.func @test_reduction_chunk_copy_accumulates_previous_partial(%a: memref<?xf16>, %b: memref<?x?xf16>, %out: memref<?xf16>, %m: index, %n: index, %rn: index) {
  %c0 = arith.constant 0 : index
  %partial = memref.alloc(%m) : memref<?xf16, 10 : i32>
  scf.for %r = %c0 to %n step %rn {
    %b_chunk = memref.subview %b[0, %r] [%m, %rn] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    linalg.generic {indexing_maps = [#map_par_reduce_lhs, #map_par_reduce_rhs, #map_par_reduce_lhs], iterator_types = ["parallel", "reduction"], ascendc.unit = "AiCore.Vector"}
      ins(%a, %b_chunk : memref<?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>)
      outs(%partial : memref<?xf16, 10 : i32>) {
    ^bb0(%in: f16, %in_0: f16, %out_0: f16):
      %sum = arith.addf %in, %in_0 : f16
      %acc = arith.addf %out_0, %sum : f16
      linalg.yield %acc : f16
    }
    memref.copy %partial, %out : memref<?xf16, 10 : i32> to memref<?xf16>
  }
  return
}

//===----------------------------------------------------------------------===//
// Buffer init for tail-tiled allocs should use the loop step upper bound.
// Using affine.min(remaining, step) keeps InitBuffer inside the loop and
// repeatedly consumes real UB across loop iterations.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @test_tail_alloc_init_uses_loop_step
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[ELEMS:.*]] = arith.muli %arg1, %arg2
// CHECK: %[[BYTES:.*]] = arith.muli %[[ELEMS]], %[[C2]]
// CHECK: ascendc.pipe.init_queue {{.*}}, {{.*}}, {{.*}}, %[[BYTES]]
// CHECK: scf.for
// CHECK: affine.min
// CHECK-NOT: ascendc.pipe.init_buffer
// CHECK: memref.alloc
func.func @test_tail_alloc_init_uses_loop_step(%n: index, %step: index, %m: index) {
  %c0 = arith.constant 0 : index
  scf.for %i = %c0 to %n step %step {
    %tile = affine.min #map_tail(%i)[%n, %step]
    %alloc = memref.alloc(%tile, %m) : memref<?x?xf16, 10 : i32>
    memref.dealloc %alloc : memref<?x?xf16, 10 : i32>
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
