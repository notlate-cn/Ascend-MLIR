// RUN: afir-opt --annotate-mix-matmul-semantics %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_add_leakyrelu
// CHECK: abi_matmul_epilogue_kind = "BiasAddLeakyRelu"
// CHECK: abi_matmul_has_bias = true
// CHECK: abi_matmul_layout_a = "ND"
// CHECK: abi_matmul_layout_b = "ND"
// CHECK: abi_matmul_layout_c = "ND"
// CHECK: abi_matmul_op_kind = "matmul"
// CHECK: abi_matmul_trans_a = false
// CHECK: abi_matmul_trans_b = false
// CHECK-LABEL: func.func @matmul_transpose_b
// CHECK: abi_matmul_epilogue_kind = "None"
// CHECK: abi_matmul_has_bias = false
// CHECK: abi_matmul_layout_a = "ND"
// CHECK: abi_matmul_layout_b = "ND"
// CHECK: abi_matmul_layout_c = "ND"
// CHECK: abi_matmul_op_kind = "matmul"
// CHECK: abi_matmul_trans_a = false
// CHECK: abi_matmul_trans_b = true
// CHECK-LABEL: func.func @matmul_transpose_a
// CHECK: abi_matmul_epilogue_kind = "None"
// CHECK: abi_matmul_has_bias = false
// CHECK: abi_matmul_layout_a = "ND"
// CHECK: abi_matmul_layout_b = "ND"
// CHECK: abi_matmul_layout_c = "ND"
// CHECK: abi_matmul_op_kind = "matmul"
// CHECK: abi_matmul_trans_a = true
// CHECK: abi_matmul_trans_b = false
// CHECK-LABEL: func.func @batch_matmul
// CHECK: abi_matmul_batch_shape = [2]
// CHECK: abi_matmul_epilogue_kind = "None"
// CHECK: abi_matmul_has_bias = false
// CHECK: abi_matmul_layout_a = "ND"
// CHECK: abi_matmul_layout_b = "ND"
// CHECK: abi_matmul_layout_c = "ND"
// CHECK: abi_matmul_op_kind = "batch_matmul"
// CHECK: abi_matmul_trans_a = false
// CHECK: abi_matmul_trans_b = false
// CHECK-LABEL: func.func @batch_matmul_transpose_b
// CHECK: abi_matmul_batch_shape = [2]
// CHECK: abi_matmul_epilogue_kind = "None"
// CHECK: abi_matmul_has_bias = false
// CHECK: abi_matmul_layout_a = "ND"
// CHECK: abi_matmul_layout_b = "ND"
// CHECK: abi_matmul_layout_c = "ND"
// CHECK: abi_matmul_op_kind = "batch_matmul"
// CHECK: abi_matmul_trans_a = false
// CHECK: abi_matmul_trans_b = true
// CHECK-LABEL: func.func @cube_only
// CHECK-NOT: abi_matmul_
// CHECK-LABEL: func.func @unrelated_generic
// CHECK-NOT: abi_matmul_
// CHECK-LABEL: func.func @non_identity_layout
// CHECK-NOT: abi_matmul_

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#layout = affine_map<(d0, d1) -> (d1, d0)>

module {
  func.func @matmul_add_leakyrelu(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>,
                                  %arg2: memref<?xf32>, %arg3: memref<?x?xf32>)
      attributes {ascendc.kernel_kind = "mix"} {
    %cst = arith.constant 1.000000e-03 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim0 = memref.dim %arg3, %c0 : memref<?x?xf32>
    %dim1 = memref.dim %arg3, %c1 : memref<?x?xf32>
    %acc = memref.alloc(%dim0, %dim1) : memref<?x?xf32>
    linalg.matmul {ascendc.unit = "AiCore.Cube"}
        ins(%arg0, %arg1 : memref<?x?xf16>, memref<?x?xf16>)
        outs(%acc : memref<?x?xf32>)
    %bias_added = memref.alloc(%dim0, %dim1) : memref<?x?xf32>
    linalg.generic
        {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]}
        ins(%acc, %arg2 : memref<?x?xf32>, memref<?xf32>)
        outs(%bias_added : memref<?x?xf32>) attrs = {ascendc.unit = "AiCore.Vector"} {
      ^bb0(%in: f32, %bias: f32, %out: f32):
        %sum = arith.addf %in, %bias : f32
        linalg.yield %sum : f32
    }
    linalg.generic
        {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%bias_added : memref<?x?xf32>)
        outs(%arg3 : memref<?x?xf32>) attrs = {ascendc.unit = "AiCore.Vector"} {
      ^bb0(%in: f32, %out: f32):
        %scaled = arith.mulf %in, %cst : f32
        %relu = arith.maximumf %in, %scaled : f32
        linalg.yield %relu : f32
    }
    return
  }

  func.func @matmul_transpose_b(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>,
                                %arg2: memref<?x?xf32>)
      attributes {ascendc.kernel_kind = "mix"} {
    linalg.matmul_transpose_b {ascendc.unit = "AiCore.Cube"}
        ins(%arg0, %arg1 : memref<?x?xf16>, memref<?x?xf16>)
        outs(%arg2 : memref<?x?xf32>)
    return
  }

  func.func @matmul_transpose_a(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>,
                                %arg2: memref<?x?xf32>)
      attributes {ascendc.kernel_kind = "mix"} {
    linalg.matmul_transpose_a {ascendc.unit = "AiCore.Cube"}
        ins(%arg0, %arg1 : memref<?x?xf16>, memref<?x?xf16>)
        outs(%arg2 : memref<?x?xf32>)
    return
  }

  func.func @batch_matmul(%arg0: memref<2x4x8xf16>, %arg1: memref<2x8x16xf16>,
                          %arg2: memref<2x4x16xf32>)
      attributes {ascendc.kernel_kind = "mix"} {
    linalg.batch_matmul {ascendc.unit = "AiCore.Cube"}
        ins(%arg0, %arg1 : memref<2x4x8xf16>, memref<2x8x16xf16>)
        outs(%arg2 : memref<2x4x16xf32>)
    return
  }

  func.func @batch_matmul_transpose_b(%arg0: memref<2x4x8xf16>,
                                      %arg1: memref<2x16x8xf16>,
                                      %arg2: memref<2x4x16xf32>)
      attributes {ascendc.kernel_kind = "mix"} {
    linalg.batch_matmul_transpose_b {ascendc.unit = "AiCore.Cube"}
        ins(%arg0, %arg1 : memref<2x4x8xf16>, memref<2x16x8xf16>)
        outs(%arg2 : memref<2x4x16xf32>)
    return
  }

  func.func @cube_only(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>,
                       %arg2: memref<?x?xf32>) attributes {ascendc.kernel_kind = "cube"} {
    linalg.matmul {ascendc.unit = "AiCore.Cube"}
        ins(%arg0, %arg1 : memref<?x?xf16>, memref<?x?xf16>)
        outs(%arg2 : memref<?x?xf32>)
    return
  }

  func.func @unrelated_generic(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>,
                               %arg2: memref<?xf32>, %arg3: memref<?x?xf32>,
                               %arg4: memref<?x?xf32>) attributes {ascendc.kernel_kind = "mix"} {
    %cst = arith.constant 1.000000e-03 : f32
    linalg.matmul {ascendc.unit = "AiCore.Cube"}
        ins(%arg0, %arg1 : memref<?x?xf16>, memref<?x?xf16>)
        outs(%arg3 : memref<?x?xf32>)
    linalg.generic
        {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]}
        ins(%arg4, %arg2 : memref<?x?xf32>, memref<?xf32>)
        outs(%arg3 : memref<?x?xf32>) attrs = {ascendc.unit = "AiCore.Vector"} {
      ^bb0(%in: f32, %bias: f32, %out: f32):
        %sum = arith.addf %in, %bias : f32
        linalg.yield %sum : f32
    }
    linalg.generic
        {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%arg3 : memref<?x?xf32>)
        outs(%arg4 : memref<?x?xf32>) attrs = {ascendc.unit = "AiCore.Vector"} {
      ^bb0(%in: f32, %out: f32):
        %scaled = arith.mulf %in, %cst : f32
        %relu = arith.maximumf %in, %scaled : f32
        linalg.yield %relu : f32
    }
    return
  }

  func.func @non_identity_layout(%arg0: memref<?x?xf16, #layout>,
                                 %arg1: memref<?x?xf16>,
                                 %arg2: memref<?x?xf32>) attributes {ascendc.kernel_kind = "mix"} {
    linalg.matmul {ascendc.unit = "AiCore.Cube"}
        ins(%arg0, %arg1 : memref<?x?xf16, #layout>, memref<?x?xf16>)
        outs(%arg2 : memref<?x?xf32>)
    return
  }
}
