// RUN: afir-opt %s --ascendc-buffer-placement | FileCheck %s

// Test 1: GM memory_space - GM (=0) is default and not printed
// CHECK-LABEL: func.func @test_gm_args(
// CHECK-SAME: %arg0: memref<?x?xf32>
func.func @test_gm_args(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) {
  return
}

// Test 2: Matmul output alloc -> CO1 (7 : i32)
// CHECK-LABEL: func.func @test_matmul_co1(
func.func @test_matmul_co1(%arg0: memref<128x256xf32>, %arg1: memref<256x128xf32>) {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<128x128xf32, 7 : i32>
  // CHECK: linalg.matmul
  // CHECK-SAME: outs(%[[ALLOC]] : memref<128x128xf32, 7 : i32>)
  %0 = memref.alloc() : memref<128x128xf32>
  linalg.matmul {ascendc.unit = "AiCore.Cube"}
      ins(%arg0, %arg1 : memref<128x256xf32>, memref<256x128xf32>)
      outs(%0 : memref<128x128xf32>)
  return
}

// Test 3: Annotation cleanup - all ascendc.* attributes removed
// CHECK-LABEL: func.func @test_annotation_cleanup(
// CHECK-NOT: ascendc.prologue
// CHECK-NOT: ascendc.epilogue
// CHECK-NOT: ascendc.unit
func.func @test_annotation_cleanup(%arg0: memref<128x128xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %0 = memref.alloc() : memref<128x128xf32>
  linalg.matmul {ascendc.unit = "AiCore.Cube"}
      ins(%arg0, %arg0 : memref<128x128xf32>, memref<128x128xf32>)
      outs(%0 : memref<128x128xf32>)
  scf.for %i = %c0 to %c128 step %c1 {
  } {ascendc.prologue = "lhs:GM->A1", ascendc.epilogue = "result:VECOUT->GM"}
  return
}

// Test 4: Prologue GM->A1,B1 inserts allocs with 1:i32, 3:i32 and memref.copy
// CHECK-LABEL: func.func @test_prologue_gm_l1(
func.func @test_prologue_gm_l1(%arg0: memref<128x256xf32>, %arg1: memref<256x128xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c0 to %c128 step %c64 {
    // CHECK: memref.alloc() : memref<128x256xf32, 1 : i32>
    // CHECK: memref.copy
    // CHECK: memref.alloc() : memref<256x128xf32, 3 : i32>
    // CHECK: memref.copy
  } {ascendc.prologue = "lhs:GM->A1,rhs:GM->B1"}
  return
}

// Test 5: Prologue dynamic shapes (GM->A1, B1, VECIN)
// CHECK-LABEL: func.func @test_prologue_dynamic(
func.func @test_prologue_dynamic(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                                  %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c0 to %c128 step %c64 {
    scf.for %j = %c0 to %c128 step %c64 {
      // CHECK: memref.alloc{{.*}} : memref<?x?xf32, 1 : i32>
      // CHECK: memref.copy
      // CHECK: memref.alloc{{.*}} : memref<?x?xf32, 3 : i32>
      // CHECK: memref.copy
      // CHECK: memref.alloc{{.*}} : memref<?x?xf32, 9 : i32>
      // CHECK: memref.copy
      // CHECK-NOT: ascendc.prologue
      // CHECK-NOT: ascendc.epilogue
    } {ascendc.parallel = true,
       ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN",
       ascendc.epilogue = "result:VECOUT->GM"}
  } {ascendc.parallel = true}
  return
}

// Test 6: Vector op TPosition inference (VECCALC vs VECOUT)
// add's output has a Vector consumer (max) -> VECCALC (11 : i32)
// max's output has no Vector consumer -> VECOUT (10 : i32)
// CHECK-LABEL: func.func @test_vector_tposition(
func.func @test_vector_tposition(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>,
                                  %arg2: memref<128x128xf32>) {
  // CHECK: %[[VECCALC:.*]] = memref.alloc() : memref<128x128xf32, 11 : i32>
  %0 = memref.alloc() : memref<128x128xf32>
  // CHECK: linalg.elementwise{{.*}}add
  // CHECK-SAME: outs(%[[VECCALC]] : memref<128x128xf32, 11 : i32>)
  linalg.elementwise kind=#linalg.elementwise_kind<add>
      {ascendc.unit = "AiCore.Vector"}
      ins(%arg0, %arg1 : memref<128x128xf32>, memref<128x128xf32>)
      outs(%0 : memref<128x128xf32>)

  // CHECK: %[[VECOUT:.*]] = memref.alloc() : memref<128x128xf32, 10 : i32>
  %1 = memref.alloc() : memref<128x128xf32>
  // CHECK: linalg.elementwise{{.*}}max_signed
  // CHECK-SAME: outs(%[[VECOUT]] : memref<128x128xf32, 10 : i32>)
  linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
      {ascendc.unit = "AiCore.Vector"}
      ins(%0, %arg2 : memref<128x128xf32>, memref<128x128xf32>)
      outs(%1 : memref<128x128xf32>)
  return
}

// Test 7: A1->A2, B1->B2 prologue: alloc sized from L1 subview (tile), matmul ins redirected.
// A2/B2 dealloc inserted at end of for_K body.
// CHECK-LABEL: func.func @test_l1_to_l0_prologue(
func.func @test_l1_to_l0_prologue(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>,
                                   %arg1: memref<?x?xf32, strided<[?, ?], offset: ?>>,
                                   %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c0 to %c128 step %c32 {
    scf.for %j = %c0 to %c128 step %c32 {
      // Tile subviews tracing to arg0 (lhs) and arg1 (rhs)
      %sv_a = memref.subview %arg0[%i, 0] [%c32, %c128] [1, 1]
              : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %sv_b = memref.subview %arg1[0, %j] [%c128, %c32] [1, 1]
              : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %sv_c = memref.subview %arg2[%i, %j] [%c32, %c32] [1, 1]
              : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %co1 = memref.alloc() : memref<32x32xf32>
      memref.copy %sv_c, %co1 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
      scf.for %k = %c0 to %c128 step %c64 {
        // K-slice subviews — lhs traces to arg0 via sv_a, rhs traces to arg1 via sv_b
        %sv_a_k = memref.subview %sv_a[0, %k] [%c32, %c64] [1, 1]
                : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %sv_b_k = memref.subview %sv_b[%k, 0] [%c64, %c32] [1, 1]
                : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %sv_co1 = memref.subview %co1[0, 0] [32, 32] [1, 1]
                : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
        // CHECK: %[[A2:.*]] = memref.alloc{{.*}} : memref<?x?xf32, 2 : i32>
        // CHECK: memref.copy
        // CHECK: %[[B2:.*]] = memref.alloc{{.*}} : memref<?x?xf32, 4 : i32>
        // CHECK: memref.copy
        // CHECK: linalg.matmul
        // CHECK-SAME: ins(%[[A2]], %[[B2]]
        // CHECK: memref.dealloc %[[A2]]
        // CHECK: memref.dealloc %[[B2]]
        linalg.matmul {ascendc.unit = "AiCore.Cube"}
            ins(%sv_a_k, %sv_b_k : memref<?x?xf32, strided<[?, ?], offset: ?>>,
                                   memref<?x?xf32, strided<[?, ?], offset: ?>>)
            outs(%sv_co1 : memref<32x32xf32, strided<[32, 1]>>)
      } {ascendc.prologue = "lhs:A1->A2,rhs:B1->B2", ascendc.epilogue = "acc:CO1->VECIN"}
    } {ascendc.parallel = true,
       ascendc.prologue = "lhs:GM->A1,rhs:GM->B1",
       ascendc.epilogue = "result:VECOUT->GM"}
  } {ascendc.parallel = true}
  return
}

// Test 8: CO1 initialization replaced with linalg.fill(0), CO1 dealloc inserted.
// The bufferizer produces memref.copy(C_subview -> CO1); pass must replace it.
// CHECK-LABEL: func.func @test_co1_zero_init(
func.func @test_co1_zero_init(%arg0: memref<64x128xf32>, %arg1: memref<128x64xf32>,
                               %arg2: memref<64x64xf32, strided<[64, 1], offset: 0>>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  scf.for %i = %c0 to %c64 step %c64 {
    scf.for %j = %c0 to %c64 step %c64 {
      %sv_c = memref.subview %arg2[%i, %j] [64, 64] [1, 1]
              : memref<64x64xf32, strided<[64, 1], offset: 0>>
              to memref<64x64xf32, strided<[64, 1], offset: ?>>
      %co1 = memref.alloc() : memref<64x64xf32>
      // Simulates bufferizer init: copy C tile into CO1.
      memref.copy %sv_c, %co1 : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<64x64xf32>
      linalg.matmul {ascendc.unit = "AiCore.Cube"}
          ins(%arg0, %arg1 : memref<64x128xf32>, memref<128x64xf32>)
          outs(%co1 : memref<64x64xf32>)
      // CHECK: %[[CO1:.*]] = memref.alloc() : memref<64x64xf32, 7 : i32>
      // CHECK-NOT: memref.copy {{.*}}, %[[CO1]]
      // CHECK: linalg.fill ins({{.*}}) outs(%[[CO1]]
      // CHECK: memref.dealloc %[[CO1]]
    } {ascendc.parallel = true,
       ascendc.epilogue = "result:VECOUT->GM"}
  } {ascendc.parallel = true}
  return
}

// Test 9: bias VECIN redirect works for non-add elementwise ops (e.g. mul).
// CHECK-LABEL: func.func @test_bias_vecin_generic(
func.func @test_bias_vecin_generic(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>,
                                    %arg1: memref<?x?xf32, strided<[?, ?], offset: ?>>,
                                    %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>,
                                    %arg3: memref<?x?xf32, strided<[?, ?], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c0 to %c128 step %c64 {
    scf.for %j = %c0 to %c128 step %c64 {
      %bias_sv = memref.subview %arg2[%i, %j] [%c64, %c64] [1, 1]
              : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %acc = memref.alloc() : memref<64x64xf32>
      %out = memref.alloc() : memref<64x64xf32>
      // CHECK: %[[VECIN:.*]] = memref.alloc{{.*}} : memref<?x?xf32, 9 : i32>
      // CHECK: memref.copy {{.*}}, %[[VECIN]]
      // mul uses bias-like arg2 as ins[1] (not add — tests kind-agnostic redirect)
      // Pass creates a subview of VECIN matching the elementwise tile offsets.
      // CHECK: %[[VECIN_SV:.*]] = memref.subview %[[VECIN]][
      // CHECK: linalg.elementwise{{.*}}mul
      // CHECK-SAME: ins({{.*}}, %[[VECIN_SV]]
      linalg.elementwise kind=#linalg.elementwise_kind<mul>
          {ascendc.unit = "AiCore.Vector"}
          ins(%acc, %bias_sv : memref<64x64xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>)
          outs(%out : memref<64x64xf32>)
    } {ascendc.parallel = true,
       ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN",
       ascendc.epilogue = "result:VECOUT->GM"}
  } {ascendc.parallel = true}
  return
}

// Test 10: Matmul with static shapes -> CO1
// CHECK-LABEL: func.func @test_matmul_static(
func.func @test_matmul_static(%arg0: memref<128x256xf32>, %arg1: memref<256x128xf32>) {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<128x128xf32, 7 : i32>
  %0 = memref.alloc() : memref<128x128xf32>
  %1 = memref.subview %arg0[0, 0] [128, 256] [1, 1] : memref<128x256xf32> to memref<128x256xf32, strided<[256, 1]>>
  %2 = memref.subview %arg1[0, 0] [256, 128] [1, 1] : memref<256x128xf32> to memref<256x128xf32, strided<[128, 1]>>
  // CHECK: linalg.matmul
  // CHECK-SAME: outs(%[[ALLOC]] : memref<128x128xf32, 7 : i32>)
  linalg.matmul {ascendc.unit = "AiCore.Cube"}
      ins(%1, %2 : memref<128x256xf32, strided<[256, 1]>>, memref<256x128xf32, strided<[128, 1]>>)
      outs(%0 : memref<128x128xf32>)
  return
}

