//===- ComputeLoweringPreconditions.cpp - Compute preconditions -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ComputeLoweringInternal.h"

#include "../../PreEmit/PreEmitInternalPasses.h"
#include "../../../Kernelize/KernelizeInternalPasses.h"

using namespace mlir;

namespace mlir::ascend {

LogicalResult prepareComputeLoweringPreconditions(func::FuncOp funcOp) {
  if (failed(annotateAscendKernelKind(funcOp)))
    return failure();
  return annotateMixMatmulSemantics(funcOp);
}

} // namespace mlir::ascend
