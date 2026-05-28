//===- LinalgBodyClassifier.cpp - Ascend linalg body classifier -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.h"

#include "Conversion/Ascend/Translate/KernelIR/Capabilities/ElementwiseBodyOpRegistry.h"
#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::ascend::backend {
namespace {

bool hasVectorRole(Operation *op) {
  auto role = op->getAttrOfType<StringAttr>(kOpRoleAttr);
  return role && role.getValue() == kOpRoleVector;
}

int64_t getIntegerMemorySpace(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType)
    return -1;
  Attribute space = memrefType.getMemorySpace();
  if (!space)
    return 0;
  if (auto intAttr = dyn_cast<IntegerAttr>(space))
    return intAttr.getInt();
  return -1;
}

bool hasOnChipOutput(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumDpsInits() == 0)
    return false;
  return getIntegerMemorySpace(linalgOp.getDpsInitOperand(0)->get().getType()) >
         0;
}

bool isRank2SwapPermutation(ArrayRef<int64_t> permutation) {
  return permutation.size() == 2 && permutation[0] == 1 &&
         permutation[1] == 0;
}

bool isValidPermutation(ArrayRef<int64_t> permutation) {
  SmallVector<bool, 8> seen(permutation.size(), false);
  for (int64_t position : permutation) {
    if (position < 0 || position >= static_cast<int64_t>(permutation.size()))
      return false;
    if (seen[position])
      return false;
    seen[position] = true;
  }
  return true;
}

bool hasOnlyParallelIterators(linalg::LinalgOp linalgOp) {
  return llvm::all_of(linalgOp.getIteratorTypesArray(),
                      [](utils::IteratorType iteratorType) {
                        return iteratorType == utils::IteratorType::parallel;
                      });
}

void collectLinalgElementTypes(linalg::LinalgOp linalgOp,
                               SmallVectorImpl<Type> &inputTypes,
                               SmallVectorImpl<Type> &outputTypes) {
  for (Value input : linalgOp.getDpsInputs())
    if (auto shaped = dyn_cast<ShapedType>(input.getType()))
      inputTypes.push_back(shaped.getElementType());

  for (Value init : linalgOp.getDpsInits())
    if (auto shaped = dyn_cast<ShapedType>(init.getType()))
      outputTypes.push_back(shaped.getElementType());
}

bool hasSupportedDtypes(linalg::LinalgOp linalgOp, ComputeKind kind,
                        const AscendBackendSupportMatrix &matrix) {
  SmallVector<Type, 4> inputTypes;
  SmallVector<Type, 2> outputTypes;
  collectLinalgElementTypes(linalgOp, inputTypes, outputTypes);
  return matrix.isSupportedDtype(kind, inputTypes, outputTypes);
}

ComputeKind classifyElementwiseBodyOp(Operation &bodyOp) {
  const ElementwiseBodyOpEntry *entry =
      lookupElementwiseBodyOp(bodyOp.getName().getStringRef());
  return entry ? entry->kind : ComputeKind::Unknown;
}

bool isAvailableOperand(Value value, Value previousResult) {
  if (isa<BlockArgument>(value))
    return true;
  if (value.getDefiningOp<arith::ConstantOp>())
    return true;
  return previousResult && value == previousResult;
}

bool isSupportedFusedElementwiseBody(
    linalg::GenericOp generic, const AscendBackendSupportMatrix &matrix) {
  if (!hasOnlyParallelIterators(generic))
    return false;

  Block *body = generic.getBody();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return false;

  Operation *lastArithOp = nullptr;
  Value previousResult;
  for (Operation &bodyOp : body->without_terminator()) {
    if (isa<arith::ConstantOp>(bodyOp))
      continue;

    ComputeKind kind = classifyElementwiseBodyOp(bodyOp);
    if (!matrix.isSupportedComputeKind(kind) ||
        !hasSupportedDtypes(cast<linalg::LinalgOp>(generic.getOperation()),
                            kind, matrix))
      return false;
    unsigned numOperands = bodyOp.getNumOperands();
    if ((numOperands != 1 && numOperands != 2) || bodyOp.getNumResults() != 1)
      return false;
    if (!llvm::all_of(bodyOp.getOperands(), [&](Value value) {
          return isAvailableOperand(value, previousResult);
        }))
      return false;

    previousResult = bodyOp.getResult(0);
    lastArithOp = &bodyOp;
  }

  return lastArithOp && yieldOp.getOperand(0) == lastArithOp->getResult(0);
}

bool isSupportedPureYieldBody(linalg::GenericOp generic) {
  if (!hasOnlyParallelIterators(generic))
    return false;
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
    return false;
  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  unsigned outputMapIndex = generic.getNumDpsInputs();
  if (maps.size() <= outputMapIndex || !maps[outputMapIndex].isIdentity())
    return false;
  unsigned lastDim = 0;
  bool hasLastDim = false;
  for (AffineExpr expr : maps[0].getResults()) {
    if (isa<AffineConstantExpr>(expr))
      continue;
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return false;
    unsigned position = dimExpr.getPosition();
    if (hasLastDim && position <= lastDim)
      return false;
    lastDim = position;
    hasLastDim = true;
  }

  Block *body = generic.getBody();
  if (body->getOperations().size() != 1)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(&body->front());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return false;
  auto blockArg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
  return blockArg && blockArg.getArgNumber() == 0;
}

bool isSimpleDimOrConstantMap(AffineMap map, unsigned rank) {
  for (AffineExpr expr : map.getResults()) {
    if (isa<AffineConstantExpr>(expr))
      continue;
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr || dimExpr.getPosition() >= rank)
      return false;
  }
  return true;
}

bool hasOnlyParallelOrReductionIterators(linalg::LinalgOp linalgOp) {
  return llvm::all_of(linalgOp.getIteratorTypesArray(),
                      [](utils::IteratorType iteratorType) {
                        return iteratorType == utils::IteratorType::parallel ||
                               iteratorType == utils::IteratorType::reduction;
                      });
}

bool isSupportedGmScalarGeneric(linalg::GenericOp generic) {
  if (!hasOnlyParallelOrReductionIterators(generic))
    return false;
  if (generic.getNumDpsInits() == 0)
    return false;

  unsigned rank = generic.getIteratorTypesArray().size();
  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  unsigned firstOutputMap = generic.getNumDpsInputs();
  size_t expectedMapCount = static_cast<size_t>(firstOutputMap) +
                            static_cast<size_t>(generic.getNumDpsInits());
  if (maps.size() != expectedMapCount)
    return false;

  for (unsigned i = 0, e = generic.getNumDpsInputs(); i < e; ++i)
    if (!isSimpleDimOrConstantMap(maps[i], rank))
      return false;

  for (unsigned i = 0, e = generic.getNumDpsInits(); i < e; ++i) {
    Value output = generic.getDpsInitOperand(i)->get();
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!outputType || getIntegerMemorySpace(output.getType()) !=
                           static_cast<int64_t>(MemorySpace::GM))
      return false;
    AffineMap outputMap = maps[firstOutputMap + i];
    if (!isSimpleDimOrConstantMap(outputMap, rank) ||
        outputMap.getNumResults() != static_cast<unsigned>(outputType.getRank()))
      return false;
  }

  Block *body = generic.getBody();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != generic.getNumDpsInits())
    return false;
  for (Operation &bodyOp : body->without_terminator()) {
    if (isa<linalg::IndexOp>(bodyOp))
      continue;
    if (bodyOp.getNumRegions() != 0)
      return false;
  }
  return true;
}

bool isSupportedVectorGatherBody(linalg::GenericOp generic,
                                 const AscendBackendSupportMatrix &matrix) {
  if (!matrix.isSupportedComputeKind(ComputeKind::VectorGather))
    return false;
  if (!generic->hasAttr(kGatherDimAttr))
    return false;
  if (!hasOnlyParallelIterators(generic))
    return false;

  bool sawLoad = false;
  Value previousResult;
  for (Operation &bodyOp : generic.getBody()->without_terminator()) {
    if (isa<linalg::IndexOp, arith::IndexCastOp>(bodyOp))
      continue;

    if (isa<memref::LoadOp>(bodyOp)) {
      if (sawLoad)
        return false;
      sawLoad = true;
      previousResult = bodyOp.getResult(0);
      continue;
    }

    ComputeKind kind = classifyElementwiseBodyOp(bodyOp);
    if (kind == ComputeKind::Unknown)
      return false;
    if (!sawLoad || !matrix.isSupportedComputeKind(kind))
      return false;
    if (bodyOp.getNumOperands() != 2 || bodyOp.getNumResults() != 1)
      return false;
    if (!llvm::all_of(bodyOp.getOperands(), [&](Value value) {
          return isAvailableOperand(value, previousResult);
        }))
      return false;

    previousResult = bodyOp.getResult(0);
  }

  auto yieldOp = dyn_cast<linalg::YieldOp>(generic.getBody()->getTerminator());
  return sawLoad && previousResult && yieldOp && yieldOp.getNumOperands() == 1 &&
         yieldOp.getOperand(0) == previousResult;
}

bool isSupportedTransposeOp(linalg::TransposeOp transpose,
                            const AscendBackendSupportMatrix &matrix) {
  bool hasOnChip = hasOnChipOutput(transpose);
  // On-chip (UB) transpose is limited to 2D swap; the hardware intrinsic only
  // supports rank-2. Off-chip (GM) transpose is lowered via DMA which handles
  // arbitrary permutations.
  return matrix.isSupportedComputeKind(ComputeKind::Transpose) &&
         ((hasOnChip && isRank2SwapPermutation(transpose.getPermutation())) ||
          (!hasOnChip && isValidPermutation(transpose.getPermutation())));
}

bool isSupportedTransposeGeneric(linalg::GenericOp generic,
                                 const AscendBackendSupportMatrix &matrix) {
  if (!matrix.isSupportedComputeKind(ComputeKind::Transpose))
    return false;
  // Generic transpose is only recognized for on-chip (UB) output, which
  // requires a 2D swap permutation. Off-chip generic ops use linalg.transpose.
  if (!hasOnChipOutput(generic))
    return false;
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
    return false;

  auto maps = generic.getIndexingMapsArray();
  if (maps.size() != 2)
    return false;

  AffineMap inMap = maps[0];
  AffineMap outMap = maps[1];
  unsigned rank = generic.getIteratorTypesArray().size();
  if (!outMap.isIdentity() || inMap.getNumResults() != rank)
    return false;

  SmallVector<int64_t, 8> permutation;
  llvm::SmallSet<unsigned, 8> seen;
  for (unsigned r = 0; r < rank; ++r) {
    auto dimExpr = dyn_cast<AffineDimExpr>(inMap.getResult(r));
    if (!dimExpr)
      return false;
    unsigned position = dimExpr.getPosition();
    if (position >= rank || !seen.insert(position).second)
      return false;
    permutation.push_back(position);
  }
  if (!isRank2SwapPermutation(permutation))
    return false;

  Block &body = *generic.getBody();
  if (body.getOperations().size() != 1)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(&body.front());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return false;
  auto blockArg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
  return blockArg && blockArg.getArgNumber() == 0;
}

} // namespace

bool hasIdentityOutputMaps(linalg::LinalgOp linalgOp) {
  SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
  unsigned firstOutputMap = linalgOp.getNumDpsInputs();
  size_t expectedMapCount = static_cast<size_t>(firstOutputMap) +
                            static_cast<size_t>(linalgOp.getNumDpsInits());
  if (maps.size() < expectedMapCount)
    return false;

  for (unsigned i = 0, e = linalgOp.getNumDpsInits(); i < e; ++i)
    if (!maps[firstOutputMap + i].isIdentity())
      return false;
  return true;
}

ComputeKind classifyBackendReductionBody(
    linalg::GenericOp generic, const AscendBackendSupportMatrix &matrix) {
  if (!llvm::is_contained(generic.getIteratorTypesArray(),
                          utils::IteratorType::reduction))
    return ComputeKind::Unknown;

  Block *body = generic.getBody();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return ComputeKind::Unknown;

  // Find all non-constant compute ops; all must be the same homogeneous kind
  // and the yield must use the last op's result. Multiple ops of the same kind
  // are allowed (e.g. two arith.addf for a pre-accumulation + reduce pattern).
  Operation *lastOp = nullptr;
  Operation *kindOp = nullptr; // first op, used for kind determination
  for (Operation &bodyOp : body->without_terminator()) {
    if (isa<arith::ConstantOp>(bodyOp))
      continue;
    if (kindOp && bodyOp.getName() != kindOp->getName())
      return ComputeKind::Unknown; // mixed op kinds in body
    if (!kindOp)
      kindOp = &bodyOp;
    lastOp = &bodyOp;
  }
  if (!lastOp)
    return ComputeKind::Unknown;
  if (yieldOp.getOperand(0) != lastOp->getResult(0))
    return ComputeKind::Unknown;

  ComputeKind kind = ComputeKind::Unknown;
  if (isa<arith::AddFOp>(kindOp))      kind = ComputeKind::ReductionAdd;
  if (isa<arith::MaximumFOp>(kindOp))  kind = ComputeKind::ReductionMax;
  if (isa<arith::MinimumFOp>(kindOp))  kind = ComputeKind::ReductionMin;
  if (isa<arith::MulFOp>(kindOp))      kind = ComputeKind::ReductionMul;
  if (kind == ComputeKind::Unknown || !matrix.isSupportedComputeKind(kind) ||
      !hasSupportedDtypes(cast<linalg::LinalgOp>(generic.getOperation()), kind,
                          matrix))
    return ComputeKind::Unknown;
  return kind;
}

ComputeKind
classifyLinalgComputeKind(Operation *op,
                          const AscendBackendSupportMatrix &matrix) {
  if (isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
          linalg::MatmulTransposeBOp>(op))
    return ComputeKind::Matmul;
  if (isa<linalg::BatchMatmulOp>(op))
    return ComputeKind::BatchMatmul;
  if (auto batchTranspose = dyn_cast<linalg::BatchMatmulTransposeAOp>(op))
    if (!hasOnChipOutput(batchTranspose))
      return ComputeKind::BatchMatmul;
  if (auto batchTranspose = dyn_cast<linalg::BatchMatmulTransposeBOp>(op))
    if (!hasOnChipOutput(batchTranspose))
      return ComputeKind::BatchMatmul;
  if (isa<linalg::FillOp>(op))
    return ComputeKind::Fill;

  if (auto transpose = dyn_cast<linalg::TransposeOp>(op))
    if (isSupportedTransposeOp(transpose, matrix))
      return ComputeKind::Transpose;

  if (auto elementwise = dyn_cast<linalg::ElementwiseOp>(op)) {
    auto kind = elementwise.getKind();
    ComputeKind computeKind = ComputeKind::Unknown;
    if (kind == linalg::ElementwiseKind::add)
      computeKind = ComputeKind::ElementwiseAdd;
    if (kind == linalg::ElementwiseKind::mul)
      computeKind = ComputeKind::ElementwiseMul;
    if (kind == linalg::ElementwiseKind::max_signed)
      computeKind = ComputeKind::ElementwiseMax;
    if (kind == linalg::ElementwiseKind::sub)
      computeKind = ComputeKind::ElementwiseSub;
    if (kind == linalg::ElementwiseKind::div)
      computeKind = ComputeKind::ElementwiseDiv;
    if (kind == linalg::ElementwiseKind::min_signed)
      computeKind = ComputeKind::ElementwiseMin;
    if (computeKind != ComputeKind::Unknown &&
        matrix.isSupportedComputeKind(computeKind) &&
        hasSupportedDtypes(cast<linalg::LinalgOp>(elementwise.getOperation()),
                           computeKind, matrix))
      return computeKind;
    return ComputeKind::Unknown;
  }

  if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
    if (isSupportedTransposeGeneric(generic, matrix))
      return ComputeKind::Transpose;
    if (isSupportedPureYieldBody(generic))
      return ComputeKind::TensorCopy;
    if (isSupportedVectorGatherBody(generic, matrix))
      return ComputeKind::VectorGather;
    if (ComputeKind rk = classifyBackendReductionBody(generic, matrix);
        rk != ComputeKind::Unknown)
      return rk;
    if (isSupportedFusedElementwiseBody(generic, matrix))
      return ComputeKind::FusedElementwise;
    if (isSupportedGmScalarGeneric(generic))
      return ComputeKind::ScalarGeneric;
  }

  return ComputeKind::Unknown;
}

bool isSupportedBackendVectorOutput(
    linalg::LinalgOp linalgOp, const AscendBackendSupportMatrix &matrix) {
  if (!hasVectorRole(linalgOp.getOperation()))
    return false;
  if (!hasOnlyParallelIterators(linalgOp))
    return false;
  if (!hasIdentityOutputMaps(linalgOp))
    return false;

  ComputeKind kind = classifyLinalgComputeKind(linalgOp.getOperation(), matrix);
  switch (kind) {
  case ComputeKind::ElementwiseAdd:
  case ComputeKind::ElementwiseMul:
  case ComputeKind::ElementwiseMax:
  case ComputeKind::ElementwiseSub:
  case ComputeKind::ElementwiseDiv:
  case ComputeKind::ElementwiseNeg:
  case ComputeKind::ElementwiseExp:
  case ComputeKind::ElementwiseExp2:
  case ComputeKind::ElementwiseLog:
  case ComputeKind::ElementwiseSqrt:
  case ComputeKind::ElementwiseRsqrt:
  case ComputeKind::ElementwiseTanh:
  case ComputeKind::ElementwiseErf:
  case ComputeKind::ElementwiseAbs:
  case ComputeKind::ElementwiseSin:
  case ComputeKind::ElementwiseCos:
  case ComputeKind::ElementwiseFma:
  case ComputeKind::ElementwiseReciprocal:
  case ComputeKind::ElementwiseRelu:
  case ComputeKind::ElementwiseSelect:
  case ComputeKind::ElementwiseMin:
  case ComputeKind::FusedElementwise:
    return matrix.isSupportedComputeKind(kind) &&
           hasSupportedDtypes(linalgOp, kind, matrix);
  default:
    return false;
  }
}

bool isSupportedBackendGatherOutput(
    linalg::LinalgOp linalgOp, const AscendBackendSupportMatrix &matrix) {
  auto generic = dyn_cast<linalg::GenericOp>(linalgOp.getOperation());
  if (!generic || !hasIdentityOutputMaps(generic))
    return false;
  return classifyLinalgComputeKind(generic.getOperation(), matrix) ==
         ComputeKind::VectorGather;
}

bool isSupportedBackendFinalOutput(linalg::LinalgOp linalgOp,
                                  const AscendBackendSupportMatrix &matrix) {
  if (isSupportedBackendVectorOutput(linalgOp, matrix))
    return true;
  if (isSupportedBackendGatherOutput(linalgOp, matrix))
    return true;

  auto generic = dyn_cast<linalg::GenericOp>(linalgOp.getOperation());
  return generic && classifyBackendReductionBody(generic, matrix) != ComputeKind::Unknown;
}

} // namespace mlir::ascend::backend
