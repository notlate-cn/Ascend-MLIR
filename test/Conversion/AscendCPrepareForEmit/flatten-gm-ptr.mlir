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
