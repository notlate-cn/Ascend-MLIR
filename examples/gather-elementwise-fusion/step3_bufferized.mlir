#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0)>
module {
  func.func @relu_index_select_add(%arg0: memref<?x?xf16>, %arg1: memref<?xi64>, %arg2: memref<?xf16>, %arg3: i64, %arg4: i64) -> memref<?x?xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg4 : i64 to index
    %1 = arith.index_cast %arg3 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf16>
    %dim_0 = memref.dim %arg1, %c0 : memref<?xi64>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf16>
    %2 = scf.for %arg5 = %c0 to %dim step %1 iter_args(%arg6 = %alloc) -> (memref<?x?xf16>) {
      %3 = affine.min #map(%arg5)[%dim, %1]
      %subview = memref.subview %arg1[0] [%dim_0] [1] : memref<?xi64> to memref<?xi64, strided<[1]>>
      %subview_1 = memref.subview %arg2[0] [%dim_0] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_2 = memref.subview %arg6[%arg5, 0] [%3, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %4 = scf.for %arg7 = %c0 to %3 step %0 iter_args(%arg8 = %subview_2) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
        %5 = affine.min #map(%arg7)[%3, %0]
        %subview_3 = memref.subview %subview[0] [%dim_0] [1] : memref<?xi64, strided<[1]>> to memref<?xi64, strided<[1]>>
        %subview_4 = memref.subview %subview_1[0] [%dim_0] [1] : memref<?xf16, strided<[1]>> to memref<?xf16, strided<[1]>>
        %subview_5 = memref.subview %arg8[%arg7, 0] [%5, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_3, %subview_4 : memref<?xi64, strided<[1]>>, memref<?xf16, strided<[1]>>) outs(%subview_5 : memref<?x?xf16, strided<[?, 1], offset: ?>>) attrs =  {ascendc.unit = "AiCore.Vector", gather_dim = 1 : i64} {
        ^bb0(%in: i64, %in_6: f16, %out: f16):
          %6 = linalg.index 0 : index
          %7 = affine.apply #map3(%arg5, %arg7)[%6]
          %8 = arith.index_cast %in : i64 to index
          %9 = memref.load %arg0[%7, %8] : memref<?x?xf16>
          %10 = arith.maximumf %9, %cst : f16
          %11 = arith.addf %10, %in_6 : f16
          linalg.yield %11 : f16
        }
        memref.copy %subview_5, %subview_5 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %arg8 : memref<?x?xf16, strided<[?, 1], offset: ?>>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      memref.copy %4, %subview_2 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.yield %arg6 : memref<?x?xf16>
    } {ascendc.parallel = true}
    return %2 : memref<?x?xf16>
  }
}

