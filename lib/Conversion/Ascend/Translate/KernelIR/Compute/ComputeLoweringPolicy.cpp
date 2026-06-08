//===- ComputeLoweringPolicy.cpp - Ascend compute lowering policy ---------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ComputeLoweringInternal.h"

#include "Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

using namespace mlir;

namespace mlir::ascend {
namespace {

bool bodyUsesOutputBlockArguments(linalg::GenericOp op) {
  Block &body = *op.getBody();
  unsigned firstOutputArg = op.getNumDpsInputs();
  auto isOutputArg = [&](Value value) {
    auto arg = dyn_cast<BlockArgument>(value);
    return arg && arg.getOwner() == &body &&
           arg.getArgNumber() >= firstOutputArg;
  };
  for (Operation &bodyOp : body.getOperations())
    for (Value operand : bodyOp.getOperands())
      if (isOutputArg(operand))
        return true;
  return false;
}

bool isSupportedVectorIndexingMap(AffineMap map, unsigned iterRank) {
  if (map.isIdentity())
    return true;

  SmallVector<bool, 8> seen(iterRank, false);
  for (AffineExpr expr : map.getResults()) {
    if (isa<AffineConstantExpr>(expr))
      continue;
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr || dimExpr.getPosition() >= iterRank)
      return false;
    if (seen[dimExpr.getPosition()])
      return false;
    seen[dimExpr.getPosition()] = true;
  }
  return true;
}

} // namespace

GmOutputNativeLoweringKind getGmOutputNativeLoweringKind(Operation *op) {
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic)
    return GmOutputNativeLoweringKind::None;

  if (!isGmAllParallelGeneric(generic) || isPureYieldGeneric(generic))
    return GmOutputNativeLoweringKind::None;

  // Reading the DPS init block argument requires copying the GM output into
  // the accumulator first. Keep that path scalar until it is implemented.
  if (bodyUsesOutputBlockArguments(generic))
    return GmOutputNativeLoweringKind::None;

  unsigned iterRank = generic.getIteratorTypesArray().size();
  auto maps = generic.getIndexingMapsArray();
  for (unsigned i = 0, e = generic.getNumDpsInputs(); i < e; ++i)
    if (!isSupportedVectorIndexingMap(maps[i], iterRank))
      return GmOutputNativeLoweringKind::None;

  backend::AscendBackendSupportMatrix matrix;
  backend::ComputeKind kind =
      backend::classifyLinalgComputeKind(generic.getOperation(), matrix);
  if (kind == backend::ComputeKind::FusedElementwise)
    return GmOutputNativeLoweringKind::FusedElementwiseVector;
  return GmOutputNativeLoweringKind::None;
}

} // namespace mlir::ascend
