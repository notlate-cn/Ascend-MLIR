#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module {
  func.func @fc_relu(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg1: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg3: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg8 : i64 to index
    %1 = arith.index_cast %arg7 : i64 to index
    %2 = arith.index_cast %arg6 : i64 to index
    %3 = arith.index_cast %arg5 : i64 to index
    %4 = arith.index_cast %arg4 : i64 to index
    %dim = memref.dim %arg3, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<?x?xf32>)
    scf.for %arg9 = %c0 to %dim step %4 {
      scf.for %arg10 = %c0 to %dim_0 step %3 {
        %6 = affine.min #map(%arg9)[%dim, %4]
        %7 = affine.min #map(%arg10)[%dim_0, %3]
        %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview = memref.subview %arg0[%arg9, 0] [%6, %dim_1] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_2 = memref.subview %arg1[0, %arg10] [%dim_1, %7] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_3 = memref.subview %arg3[%arg9, %arg10] [%6, %7] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_4 = memref.subview %arg2[%arg9, %arg10] [%6, %7] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_5 = memref.subview %alloc[%arg9, %arg10] [%6, %7] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %alloc_6 = memref.alloc(%6, %7) {alignment = 64 : i64} : memref<?x?xf32>
        memref.copy %subview_3, %alloc_6 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
        scf.for %arg11 = %c0 to %6 step %2 {
          scf.for %arg12 = %c0 to %7 step %1 {
            %8 = affine.min #map(%arg11)[%6, %2]
            %9 = affine.min #map(%arg12)[%7, %1]
            %subview_7 = memref.subview %subview[%arg11, 0] [%8, %dim_1] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            %subview_8 = memref.subview %subview_2[0, %arg12] [%dim_1, %9] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            %subview_9 = memref.subview %subview_3[%arg11, %arg12] [%8, %9] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            %alloc_10 = memref.alloc(%8, %9) {alignment = 64 : i64} : memref<?x?xf32>
            memref.copy %subview_9, %alloc_10 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
            scf.for %arg13 = %c0 to %dim_1 step %0 {
              %10 = affine.min #map(%arg13)[%dim_1, %0]
              %subview_15 = memref.subview %subview_7[0, %arg13] [%8, %10] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %subview_16 = memref.subview %subview_8[%arg13, 0] [%10, %9] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %subview_17 = memref.subview %alloc_10[0, 0] [%8, %9] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
              linalg.matmul {ascendc.unit = "AiCore.Cube"} ins(%subview_15, %subview_16 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%subview_17 : memref<?x?xf32, strided<[?, 1]>>)
            } {ascendc.epilogue = "acc:CO1->VECIN", ascendc.prologue = "lhs:A1->A2,rhs:B1->B2"}
            %subview_11 = memref.subview %subview_4[%arg11, %arg12] [%8, %9] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            %alloc_12 = memref.alloc(%8, %9) {alignment = 64 : i64} : memref<?x?xf32>
            linalg.elementwise kind=#linalg.elementwise_kind<add> {ascendc.unit = "AiCore.Vector"} ins(%alloc_10, %subview_11 : memref<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_12 : memref<?x?xf32>)
            %subview_13 = memref.subview %subview_5[%arg11, %arg12] [%8, %9] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %subview_14 = memref.subview %alloc_6[%arg11, %arg12] [%8, %9] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            linalg.elementwise kind=#linalg.elementwise_kind<max_signed> {ascendc.unit = "AiCore.Vector"} ins(%alloc_12, %subview_13 : memref<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_14 : memref<?x?xf32, strided<[?, 1], offset: ?>>)
            memref.dealloc %alloc_10 : memref<?x?xf32>
            memref.dealloc %alloc_12 : memref<?x?xf32>
          }
        }
        memref.copy %alloc_6, %subview_3 : memref<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        memref.dealloc %alloc_6 : memref<?x?xf32>
      } {ascendc.epilogue = "result:VECOUT->GM", ascendc.parallel = true, ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"}
    } {ascendc.parallel = true}
    %5 = bufferization.clone %arg3 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    memref.dealloc %alloc : memref<?x?xf32>
    return %5 : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }
}

