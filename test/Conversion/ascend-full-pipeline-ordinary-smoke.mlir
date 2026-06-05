// RUN: sed -n '/\/\/ POSITIVE-BEGIN/,/\/\/ POSITIVE-END/p' %s | ascend-mlir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate dump-report=true debug-stage=realize' --ascend-compute-lower 2>&1 | FileCheck %s
// RUN: sed -n '/\/\/ POSITIVE-BEGIN/,/\/\/ POSITIVE-END/p' %s | ascend-mlir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='placement-mode=target-aware cann-root=%S/Inputs/ascend-target-aware-placement-cann soc=SyntheticSoC materialization-mode=memory-space-annotate' --ascend-compute-lower --ascend-parallelize --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | FileCheck %s --check-prefix=ABI
// RUN: sed -n '/\/\/ POSITIVE-BEGIN/,/\/\/ POSITIVE-END/p' %s | ascend-mlir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' | FileCheck %s --check-prefix=REALIZE
// RUN: sed -n '/\/\/ NON-IDENTITY-BEGIN/,/\/\/ NON-IDENTITY-END/p' %s | ascend-mlir-opt --ascend-realize='materialization-mode=memory-space-annotate dump-report=true debug-stage=realize' 2>&1 | FileCheck %s --check-prefix=NON-IDENTITY
// RUN: sed -n '/\/\/ SUBVIEW-BEGIN/,/\/\/ SUBVIEW-END/p' %s | ascend-mlir-opt --ascend-realize='materialization-mode=memory-space-annotate dump-report=true debug-stage=realize' 2>&1 | FileCheck %s --check-prefix=SUBVIEW
// RUN: sed -n '/\/\/ CAST-BEGIN/,/\/\/ CAST-END/p' %s | ascend-mlir-opt --ascend-realize='materialization-mode=memory-space-annotate dump-report=true debug-stage=realize' 2>&1 | FileCheck %s --check-prefix=CAST

// POSITIVE-BEGIN

func.func @ordinary_elementwise(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> {
  %empty = tensor.empty() : tensor<64xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty : tensor<64xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>
  return %out : tensor<64xf16>
}

// POSITIVE-END

// NON-IDENTITY-BEGIN

func.func @non_identity_output_map(%arg0: tensor<4x4xf16>,
                                   %arg1: tensor<4x4xf16>) -> tensor<4x4xf16>
    attributes {ascend.normalized = true} {
  %empty = tensor.empty() : tensor<4x4xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d1, d0)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x4xf16>, tensor<4x4xf16>)
    outs(%empty : tensor<4x4xf16>)
    attrs = {
      ascend.kernel = "kernel_non_identity",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_non_identity.decision.0",
      ascend.schedule.schedule_contract = "generic_tiled_loop"
    } {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x4xf16>
  return %out : tensor<4x4xf16>
}

// NON-IDENTITY-END

// SUBVIEW-BEGIN

func.func @view_mediated_downstream_user(%arg0: tensor<64xf16>,
                                         %arg1: tensor<64xf16>) -> tensor<32xf16>
    attributes {ascend.normalized = true} {
  %empty0 = tensor.empty() : tensor<64xf16>
  %mid = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty0 : tensor<64xf16>)
    attrs = {
      ascend.kernel = "kernel_producer",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_producer.decision.0",
      ascend.schedule.schedule_contract = "generic_tiled_loop"
    } {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>

  %extract = tensor.extract_slice %mid[0] [32] [1] : tensor<64xf16> to tensor<32xf16>
  %empty1 = tensor.empty() : tensor<32xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%extract : tensor<32xf16>)
    outs(%empty1 : tensor<32xf16>)
    attrs = {
      ascend.kernel = "kernel_consumer",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_consumer.decision.0",
      ascend.schedule.schedule_contract = "generic_tiled_loop"
    } {
  ^bb0(%x: f16, %o: f16):
    %v = arith.maximumf %x, %x : f16
    linalg.yield %v : f16
  } -> tensor<32xf16>
  return %out : tensor<32xf16>
}

// SUBVIEW-END

// CAST-BEGIN

func.func @cast_mediated_downstream_user(%arg0: memref<64xf16>,
                                         %arg1: memref<64xf16>) -> memref<64xf16>
    attributes {ascend.normalized = true} {
  %mid = memref.alloc() : memref<64xf16>
  linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : memref<64xf16>, memref<64xf16>)
    outs(%mid : memref<64xf16>)
    attrs = {
      ascend.kernel = "kernel_cast_producer",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_cast_producer.decision.0",
      ascend.schedule.schedule_contract = "generic_tiled_loop"
    } {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  }

  %cast = memref.cast %mid : memref<64xf16> to memref<?xf16>
  %out = memref.alloc() : memref<64xf16>
  linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%cast : memref<?xf16>)
    outs(%out : memref<64xf16>)
    attrs = {
      ascend.kernel = "kernel_cast_consumer",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_cast_consumer.decision.0",
      ascend.schedule.schedule_contract = "generic_tiled_loop"
    } {
  ^bb0(%x: f16, %o: f16):
    %v = arith.maximumf %x, %x : f16
    linalg.yield %v : f16
  }
  return %out : memref<64xf16>
}

// CAST-END

// CHECK: MemoryRealizationPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "memory_space_materialize"
// CHECK:   memory_space_annotations = 0
// CHECK-NEXT:   materialized_allocs = 1
// CHECK-NEXT:   materialized_copies = 1
// CHECK: ascendc.add_l2
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: linalg.generic

// ABI-LABEL: func.func @ordinary_elementwise
// ABI-SAME: %{{.*}}: memref<ui8>
// ABI-SAME: %{{.*}}: !emitasc.py_struct<"TilingData"
// ABI-SAME: cann.num_inputs = 2 : i32
// ABI: ascendc.add_l2
// ABI: ascendc.data_copy_l2
// ABI-NOT: linalg.generic

// REALIZE-LABEL: func.func @ordinary_elementwise
// REALIZE: %[[GM:.*]] = memref.alloc() {{.*}} : memref<64xf16>
// REALIZE: %[[VECOUT:.*]] = memref.alloc() {{.*}} : memref<64xf16, 10 : i32>
// REALIZE: linalg.generic
// REALIZE-SAME: outs(%[[VECOUT]] : memref<64xf16, 10 : i32>)
// REALIZE: memref.copy %[[VECOUT]], %[[GM]] : memref<64xf16, 10 : i32> to memref<64xf16>
// REALIZE-NOT: memref<64xf16, 11 : i32>

// NON-IDENTITY: MemoryRealizationPlan:
// NON-IDENTITY-NEXT:   kernel = kernel_non_identity
// NON-IDENTITY-NEXT:   mode = "memory_space_annotate"
// NON-IDENTITY:   materialized_allocs = 0
// NON-IDENTITY-NEXT:   materialized_copies = 0
// NON-IDENTITY-NOT: memref<4x4xf16, 10 : i32>

// SUBVIEW: MemoryRealizationPlan:
// SUBVIEW-NEXT:   kernel = kernel_consumer
// SUBVIEW-NEXT:   mode = "memory_space_materialize"
// SUBVIEW:   materialized_allocs = 1
// SUBVIEW-NEXT:   materialized_copies = 1
// SUBVIEW: MemoryRealizationPlan:
// SUBVIEW-NEXT:   kernel = kernel_producer
// SUBVIEW-NEXT:   mode = "memory_space_annotate"
// SUBVIEW:   materialized_allocs = 0
// SUBVIEW-NEXT:   materialized_copies = 0

// CAST: MemoryRealizationPlan:
// CAST-NEXT:   kernel = kernel_cast_consumer
// CAST-NEXT:   mode = "memory_space_materialize"
// CAST:   materialized_allocs = 1
// CAST-NEXT:   materialized_copies = 1
// CAST: MemoryRealizationPlan:
// CAST-NEXT:   kernel = kernel_cast_producer
// CAST-NEXT:   mode = "memory_space_annotate"
// CAST:   materialized_allocs = 0
// CAST-NEXT:   materialized_copies = 0
