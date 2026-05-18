// RUN: afir-opt %s --ascendc-flatten-gm-ptr 2>&1 | FileCheck %s

// 1D subview: set_global_buffer via subview should become flat ptr + i32 offset.
// CHECK-LABEL: func.func @test_1d
// CHECK: emitasc.reinterpret_cast
// CHECK-NOT: memref.subview

// GM alloc promoted to func arg.
// CHECK-LABEL: func.func @test_alloc_promoted
// CHECK-SAME: memref<16xf32
// CHECK-NOT: memref.alloc

func.func @test_1d(%arg0: memref<1024xf32>) {
  %gt = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
  %c8 = arith.constant 8 : index
  %sub = memref.subview %arg0[%c8][16][1]
    : memref<1024xf32> to memref<16xf32, strided<[1], offset: ?>>
  %cast = memref.cast %sub
    : memref<16xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
  ascendc.global_tensor.set_global_buffer %gt, %cast
    : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[?], offset: ?>>
  return
}

func.func @test_alloc_promoted() {
  %alloc = memref.alloc() : memref<16xf32>
  %cst = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  %val = memref.load %alloc[%c0] : memref<16xf32>
  return
}

// 2D subview: flat offset = row * dim[1] + col.
// CHECK-LABEL: func.func @test_2d
// CHECK: memref.dim
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: emitasc.reinterpret_cast
// CHECK-NOT: memref.subview

func.func @test_2d(%arg0: memref<32x32xf32>) {
  %gt = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %sub = memref.subview %arg0[%c2, %c4][4, 4][1, 1]
    : memref<32x32xf32> to memref<4x4xf32, strided<[32, 1], offset: ?>>
  ascendc.global_tensor.set_global_buffer %gt, %sub
    : !ascendc.global_tensor<*xf32>, memref<4x4xf32, strided<[32, 1], offset: ?>>
  return
}

// 3D subview: flat offset = row * dim[1] * dim[2] (general N-D row-major formula).
// CHECK-LABEL: func.func @test_3d
// CHECK: memref.dim
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: emitasc.reinterpret_cast
// CHECK-NOT: memref.subview

func.func @test_3d(%arg0: memref<4x8x32xf32>) {
  %gt = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %sub = memref.subview %arg0[%c1, %c0, %c0][1, 8, 32][1, 1, 1]
    : memref<4x8x32xf32> to memref<1x8x32xf32, strided<[256, 32, 1], offset: ?>>
  ascendc.global_tensor.set_global_buffer %gt, %sub
    : !ascendc.global_tensor<*xf32>, memref<1x8x32xf32, strided<[256, 32, 1], offset: ?>>
  return
}

// GM->GM copy should become memmove.
// CHECK-LABEL: func.func @test_gm_copy
// CHECK: memmove
// CHECK-NOT: memref.copy

func.func @test_gm_copy(%arg0: memref<16xf32>) {
  %alloc = memref.alloc() : memref<16xf32>
  memref.copy %arg0, %alloc : memref<16xf32> to memref<16xf32>
  return
}

// Bare memref.collapse_shape (no surrounding subview): the bcast pattern
// "load the whole operand" feeds collapse_shape directly into set_global_buffer.
// CHECK-LABEL: func.func @test_bare_collapse
// CHECK: emitasc.reinterpret_cast
// CHECK-NOT: memref.collapse_shape
// CHECK-NOT: memref.subview
func.func @test_bare_collapse(%arg0: memref<4x32xf32>) {
  %gt = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
  %col = memref.collapse_shape %arg0 [[0, 1]]
       : memref<4x32xf32> into memref<128xf32>
  ascendc.global_tensor.set_global_buffer %gt, %col
    : !ascendc.global_tensor<*xf32>, memref<128xf32>
  return
}

// Bare block arg with identity layout, directly into set_global_buffer.
// CHECK-LABEL: func.func @test_bare_blockarg
// CHECK: emitasc.reinterpret_cast
// CHECK-NOT: memref.collapse_shape
func.func @test_bare_blockarg(%arg0: memref<128xf32>) {
  %gt = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
  ascendc.global_tensor.set_global_buffer %gt, %arg0
    : !ascendc.global_tensor<*xf32>, memref<128xf32>
  return
}
