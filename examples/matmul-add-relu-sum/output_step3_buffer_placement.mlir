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
    scf.for %arg9 = %c0 to %dim step %4 {
      scf.for %arg10 = %c0 to %dim_0 step %3 {
        %6 = affine.min #map(%arg9)[%dim, %4]
        %7 = affine.min #map(%arg10)[%dim_0, %3]
        %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview = memref.subview %arg0[%arg9, 0] [%6, %dim_1] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %alloc = memref.alloc(%6, %dim_1) : memref<?x?xf32, 1 : i32>
        memref.copy %subview, %alloc : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, 1 : i32>
        %subview_2 = memref.subview %arg1[0, %arg10] [%dim_1, %7] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %alloc_3 = memref.alloc(%dim_1, %7) : memref<?x?xf32, 3 : i32>
        memref.copy %subview_2, %alloc_3 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, 3 : i32>
        %subview_4 = memref.subview %arg3[%arg9, %arg10] [%6, %7] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %subview_5 = memref.subview %arg2[%arg9, %arg10] [%6, %7] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %alloc_6 = memref.alloc(%6, %7) : memref<?x?xf32, 9 : i32>
        memref.copy %subview_5, %alloc_6 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, 9 : i32>
        %alloc_7 = memref.alloc(%6, %7) : memref<?x?xf32, 10 : i32>
        linalg.fill ins(%cst : f32) outs(%alloc_7 : memref<?x?xf32, 10 : i32>)
        scf.for %arg11 = %c0 to %6 step %2 {
          scf.for %arg12 = %c0 to %7 step %1 {
            %8 = affine.min #map(%arg11)[%6, %2]
            %9 = affine.min #map(%arg12)[%7, %1]
            %alloc_8 = memref.alloc(%8, %9) : memref<?x?xf32, 7 : i32>
            linalg.fill ins(%cst : f32) outs(%alloc_8 : memref<?x?xf32, 7 : i32>)
            scf.for %arg13 = %c0 to %dim_1 step %0 {
              %10 = affine.min #map(%arg13)[%dim_1, %0]
              %subview_14 = memref.subview %alloc[0, %arg13] [%8, %10] [1, 1] : memref<?x?xf32, 1 : i32> to memref<?x?xf32, strided<[?, 1], offset: ?>, 1 : i32>
              %alloc_15 = memref.alloc(%8, %10) : memref<?x?xf32, 2 : i32>
              memref.copy %subview_14, %alloc_15 : memref<?x?xf32, strided<[?, 1], offset: ?>, 1 : i32> to memref<?x?xf32, 2 : i32>
              %subview_16 = memref.subview %alloc_3[%arg13, 0] [%10, %9] [1, 1] : memref<?x?xf32, 3 : i32> to memref<?x?xf32, strided<[?, 1], offset: ?>, 3 : i32>
              %alloc_17 = memref.alloc(%10, %9) : memref<?x?xf32, 4 : i32>
              memref.copy %subview_16, %alloc_17 : memref<?x?xf32, strided<[?, 1], offset: ?>, 3 : i32> to memref<?x?xf32, 4 : i32>
              %subview_18 = memref.subview %alloc_8[0, 0] [%8, %9] [1, 1] : memref<?x?xf32, 7 : i32> to memref<?x?xf32, strided<[?, 1]>, 7 : i32>
              linalg.matmul ins(%alloc_15, %alloc_17 : memref<?x?xf32, 2 : i32>, memref<?x?xf32, 4 : i32>) outs(%subview_18 : memref<?x?xf32, strided<[?, 1]>, 7 : i32>)
              memref.dealloc %alloc_15 : memref<?x?xf32, 2 : i32>
              memref.dealloc %alloc_17 : memref<?x?xf32, 4 : i32>
            }
            %alloc_9 = memref.alloc(%8, %9) : memref<?x?xf32, 9 : i32>
            memref.copy %alloc_8, %alloc_9 : memref<?x?xf32, 7 : i32> to memref<?x?xf32, 9 : i32>
            %alloc_10 = memref.alloc(%8, %9) : memref<?x?xf32, 11 : i32>
            %subview_11 = memref.subview %alloc_6[%arg11, %arg12] [%8, %9] [1, 1] : memref<?x?xf32, 9 : i32> to memref<?x?xf32, strided<[?, 1], offset: ?>, 9 : i32>
            linalg.elementwise kind=#linalg.elementwise_kind<add> ins(%alloc_9, %subview_11 : memref<?x?xf32, 9 : i32>, memref<?x?xf32, strided<[?, 1], offset: ?>, 9 : i32>) outs(%alloc_10 : memref<?x?xf32, 11 : i32>)
            %subview_12 = memref.subview %alloc_7[%arg11, %arg12] [%8, %9] [1, 1] : memref<?x?xf32, 10 : i32> to memref<?x?xf32, strided<[?, 1], offset: ?>, 10 : i32>
            %alloc_13 = memref.alloc(%8, %9) : memref<?x?xf32, 9 : i32>
            linalg.fill ins(%cst : f32) outs(%alloc_13 : memref<?x?xf32, 9 : i32>)
            linalg.elementwise kind=#linalg.elementwise_kind<max_signed> ins(%alloc_10, %alloc_13 : memref<?x?xf32, 11 : i32>, memref<?x?xf32, 9 : i32>) outs(%subview_12 : memref<?x?xf32, strided<[?, 1], offset: ?>, 10 : i32>)
            memref.dealloc %alloc_13 : memref<?x?xf32, 9 : i32>
            memref.dealloc %alloc_10 : memref<?x?xf32, 11 : i32>
            memref.dealloc %alloc_9 : memref<?x?xf32, 9 : i32>
            memref.dealloc %alloc_8 : memref<?x?xf32, 7 : i32>
          }
        }
        memref.copy %alloc_7, %subview_4 : memref<?x?xf32, 10 : i32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        memref.dealloc %alloc_7 : memref<?x?xf32, 10 : i32>
        memref.dealloc %alloc : memref<?x?xf32, 1 : i32>
        memref.dealloc %alloc_3 : memref<?x?xf32, 3 : i32>
        memref.dealloc %alloc_6 : memref<?x?xf32, 9 : i32>
      }
    }
    %5 = bufferization.clone %arg3 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    return %5 : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }
}

