#pragma once
#include "Conversion/VectorPlan/TilePlan.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::afir {

/// CV-fusion Phase 4 entry point: emit the cube tile nest + annotations
/// inside `func`, consuming the cube tunables already on `plan` (see
/// `buildCubePlan` in TilePlanGen.cpp).  Target IR shape mirrors AF's
/// matmul-vec mix kernel (matmul-add-leakyrelu/step2_tiled.mlir):
///
///   scf.for M_outer step XBLOCK_M    {ascendc.parallel,
///                                     ascendc.prologue/epilogue dataflow}
///     scf.for N_outer step XBLOCK_N  {ascendc.parallel}
///       scf.for M_inner step M_INNER
///         scf.for N_inner step N_INNER
///           scf.for K_inner step K_INNER
///             linalg.matmul {ascendc.unit = "AiCore.Cube"}
///           linalg.generic <vec-epilogue> {ascendc.unit = "AiCore.Vector"}
///
/// Implementation strategy: reuse `scf::tileConsumerAndFuseProducersUsingSCF`
/// (TileUsingInterface.h) to tile each level by fusing producers into the
/// tiled consumer.  This is the same primitive `transform.structured.
/// tile_using_for + fuse_into_containing_op` from the matmul-add-leakyrelu
/// transform spec uses underneath.
///
/// Returns `failure()` only when the IR doesn't match the expected
/// matmul + trailing-vec shape (e.g. unsupported op kind); the caller
/// should fall back gracefully.
LogicalResult emitCubeKernel(mlir::func::FuncOp func,
                              const mlir::vector_plan::TilePlan &plan);

} // namespace mlir::afir
