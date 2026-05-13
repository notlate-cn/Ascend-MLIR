//===- MemoryRealizationDriver.cpp - Ascend memory realization -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/MemoryRealizationDriver.h"

#include "Target/Ascend/TargetProfile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::afir::ascend::realize {
namespace {

constexpr int64_t kVecCalcMemorySpace =
    static_cast<int64_t>(::mlir::ascend::MemoryPlace::VECCALC);
constexpr int64_t kVecOutMemorySpace =
    static_cast<int64_t>(::mlir::ascend::MemoryPlace::VECOUT);

static StringRef getKernelId(Operation *op) {
  auto kernelAttr = op->getAttrOfType<StringAttr>(kKernelAttr);
  return kernelAttr ? kernelAttr.getValue() : StringRef();
}

static bool isVectorOp(Operation *op) {
  auto role = op->getAttrOfType<StringAttr>(kOpRoleAttr);
  return role && role.getValue() == "vector";
}

static bool hasOnlyKernelUses(Value value, StringRef kernelId) {
  for (Operation *user : value.getUsers()) {
    if (getKernelId(user) != kernelId)
      return false;
  }
  return true;
}

static MemRefType withMemorySpace(MemRefType type, Attribute memorySpace) {
  return MemRefType::get(type.getShape(), type.getElementType(),
                         type.getLayout(), memorySpace);
}

static bool isSupportedPhase5GenericBody(linalg::GenericOp generic) {
  Block *body = generic.getBody();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return false;

  Operation *lastArithOp = nullptr;
  Value previousResult;
  for (Operation &bodyOp : body->without_terminator()) {
    if (isa<arith::ConstantOp>(bodyOp))
      continue;
    if (!isa<arith::AddFOp, arith::MulFOp, arith::MaximumFOp>(bodyOp))
      return false;
    if (bodyOp.getNumOperands() != 2 || bodyOp.getNumResults() != 1)
      return false;

    auto isAvailableOperand = [&](Value value) {
      if (isa<BlockArgument>(value))
        return true;
      if (value.getDefiningOp<arith::ConstantOp>())
        return true;
      return previousResult && value == previousResult;
    };
    if (!llvm::all_of(bodyOp.getOperands(), isAvailableOperand))
      return false;

    previousResult = bodyOp.getResult(0);
    lastArithOp = &bodyOp;
  }

  return lastArithOp && yieldOp.getOperand(0) == lastArithOp->getResult(0);
}

static bool isSupportedPhase5ReductionBody(linalg::GenericOp generic) {
  if (!llvm::is_contained(generic.getIteratorTypesArray(),
                          utils::IteratorType::reduction))
    return false;

  Block *body = generic.getBody();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return false;

  Operation *lastAdd = nullptr;
  for (Operation &bodyOp : body->without_terminator()) {
    if (isa<arith::ConstantOp>(bodyOp))
      continue;
    if (!isa<arith::AddFOp>(bodyOp))
      return false;
    lastAdd = &bodyOp;
  }

  return lastAdd && yieldOp.getOperand(0) == lastAdd->getResult(0);
}

static bool hasIdentityOutputMaps(linalg::LinalgOp linalgOp) {
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

static bool isSupportedPhase5VectorOutput(linalg::LinalgOp linalgOp) {
  if (!isVectorOp(linalgOp.getOperation()))
    return false;
  if (!llvm::all_of(linalgOp.getIteratorTypesArray(),
                    [](utils::IteratorType iteratorType) {
                      return iteratorType == utils::IteratorType::parallel;
                    }))
    return false;
  if (!hasIdentityOutputMaps(linalgOp))
    return false;

  if (auto generic = dyn_cast<linalg::GenericOp>(linalgOp.getOperation()))
    return isSupportedPhase5GenericBody(generic);

  if (auto elementwise =
          dyn_cast<linalg::ElementwiseOp>(linalgOp.getOperation())) {
    auto kind = elementwise.getKind();
    return kind == linalg::ElementwiseKind::add ||
           kind == linalg::ElementwiseKind::mul ||
           kind == linalg::ElementwiseKind::max_signed;
  }

  return false;
}

static bool isReductionInitFillForWriter(Operation *user, Operation *writer,
                                         Value output) {
  auto fillOp = dyn_cast<linalg::FillOp>(user);
  auto generic = dyn_cast<linalg::GenericOp>(writer);
  if (!fillOp || !generic || !isSupportedPhase5ReductionBody(generic))
    return false;

  if (!llvm::is_contained(fillOp.getOutputs(), output))
    return false;
  return llvm::is_contained(generic.getDpsInits(), output);
}

static bool isSupportedPhase5FinalOutput(linalg::LinalgOp linalgOp) {
  if (isSupportedPhase5VectorOutput(linalgOp))
    return true;

  auto generic = dyn_cast<linalg::GenericOp>(linalgOp.getOperation());
  return generic && isSupportedPhase5ReductionBody(generic);
}

static bool isAllowedExternalOutputUse(Operation *user,
                                       llvm::DenseSet<Operation *> &visited) {
  if (!visited.insert(user).second)
    return false;

  if (!getKernelId(user).empty())
    return false;

  if (isa<func::ReturnOp>(user))
    return true;

  if (auto castOp = dyn_cast<memref::CastOp>(user)) {
    bool hasForwardedUse = false;
    for (Operation *forwardedUser : castOp.getResult().getUsers()) {
      hasForwardedUse = true;
      if (!isAllowedExternalOutputUse(forwardedUser, visited))
        return false;
    }
    return hasForwardedUse;
  }

  return false;
}

static bool isFinalKernelOutput(Value value, Operation *writer) {
  bool hasExternalUse = false;
  llvm::DenseSet<Operation *> visited;
  for (Operation *user : value.getUsers()) {
    if (user == writer)
      continue;
    if (isReductionInitFillForWriter(user, writer, value))
      continue;

    if (auto castOp = dyn_cast<memref::CastOp>(user))
      if (castOp.getResult().use_empty())
        continue;

    if (!isAllowedExternalOutputUse(user, visited))
      return false;

    hasExternalUse = true;
  }
  return hasExternalUse;
}

struct AnnotatableAlloc {
  memref::AllocOp alloc;
  std::string kernelId;
};

struct Phase5BridgeOutput {
  linalg::LinalgOp linalgOp;
  OpOperand *initOperand;
  memref::AllocOp gmAlloc;
  memref::CopyOp concatCopy;
  std::string kernelId;
};

static bool isConstantOpFoldResult(OpFoldResult ofr, int64_t expected) {
  std::optional<int64_t> value = getConstantIntValue(ofr);
  return value && *value == expected;
}

static bool hasReturnUse(Value value, llvm::DenseSet<Operation *> &visited) {
  for (Operation *user : value.getUsers()) {
    if (!visited.insert(user).second)
      continue;
    if (isa<func::ReturnOp>(user))
      return true;
    if (auto castOp = dyn_cast<memref::CastOp>(user))
      if (hasReturnUse(castOp.getResult(), visited))
        return true;
  }
  return false;
}

static bool hasReturnUse(Value value) {
  llvm::DenseSet<Operation *> visited;
  return hasReturnUse(value, visited);
}

static bool isSupportedConcatTargetSubview(Value target, MemRefType sourceType) {
  auto subview = target.getDefiningOp<memref::SubViewOp>();
  if (!subview)
    return false;

  auto targetType = dyn_cast<MemRefType>(target.getType());
  auto concatOutputType = dyn_cast<MemRefType>(subview.getSource().getType());
  if (!targetType || !concatOutputType)
    return false;
  if (sourceType.getMemorySpace() || targetType.getMemorySpace() ||
      concatOutputType.getMemorySpace())
    return false;
  if (!hasReturnUse(subview.getSource()))
    return false;
  if (sourceType.getRank() != targetType.getRank() ||
      sourceType.getRank() != concatOutputType.getRank())
    return false;
  if (sourceType.getElementType() != targetType.getElementType() ||
      sourceType.getElementType() != concatOutputType.getElementType())
    return false;
  for (auto [sourceDim, targetDim] :
       llvm::zip(sourceType.getShape(), targetType.getShape()))
    if (!ShapedType::isDynamic(sourceDim) &&
        !ShapedType::isDynamic(targetDim) && sourceDim != targetDim)
      return false;

  for (OpFoldResult stride : subview.getMixedStrides())
    if (!isConstantOpFoldResult(stride, 1))
      return false;

  SmallVector<OpFoldResult> offsets = subview.getMixedOffsets();
  if (offsets.size() != static_cast<size_t>(sourceType.getRank()))
    return false;
  for (unsigned i = 1, e = offsets.size(); i < e; ++i)
    if (!isConstantOpFoldResult(offsets[i], 0))
      return false;

  return true;
}

static memref::CopyOp findSupportedConcatCopyUse(Value init, Operation *writer) {
  auto sourceType = dyn_cast<MemRefType>(init.getType());
  if (!sourceType || sourceType.getMemorySpace())
    return {};

  memref::CopyOp concatCopy;
  for (Operation *user : init.getUsers()) {
    if (user == writer)
      continue;
    if (isa<memref::DimOp>(user))
      continue;

    auto copyOp = dyn_cast<memref::CopyOp>(user);
    if (!copyOp || copyOp.getSource() != init ||
        copyOp->getBlock() != writer->getBlock() ||
        !writer->isBeforeInBlock(copyOp.getOperation()) ||
        !isSupportedConcatTargetSubview(copyOp.getTarget(), sourceType))
      return {};
    if (concatCopy)
      return {};
    concatCopy = copyOp;
  }

  return concatCopy;
}

static std::optional<Value> dynamicSizeForDim(memref::AllocOp allocOp,
                                              int64_t dim) {
  auto type = cast<MemRefType>(allocOp.getType());
  if (dim < 0 || dim >= type.getRank())
    return std::nullopt;
  if (!ShapedType::isDynamic(type.getDimSize(dim)))
    return std::nullopt;

  unsigned dynamicIndex = 0;
  for (int64_t i = 0; i < dim; ++i)
    if (ShapedType::isDynamic(type.getDimSize(i)))
      ++dynamicIndex;

  if (dynamicIndex >= allocOp.getDynamicSizes().size())
    return std::nullopt;
  return allocOp.getDynamicSizes()[dynamicIndex];
}

static LogicalResult replaceAllocDimUses(IRRewriter &rewriter,
                                         memref::AllocOp allocOp) {
  auto type = cast<MemRefType>(allocOp.getType());
  SmallVector<memref::DimOp, 4> dimUsers;
  for (Operation *user : allocOp.getResult().getUsers())
    if (auto dimOp = dyn_cast<memref::DimOp>(user))
      dimUsers.push_back(dimOp);

  for (memref::DimOp dimOp : dimUsers) {
    std::optional<int64_t> dim = getConstantIntValue(dimOp.getIndex());
    if (!dim || *dim < 0 || *dim >= type.getRank())
      return failure();

    rewriter.setInsertionPoint(dimOp);
    if (!ShapedType::isDynamic(type.getDimSize(*dim))) {
      Value replacement =
          rewriter.create<arith::ConstantIndexOp>(dimOp.getLoc(),
                                                  type.getDimSize(*dim));
      rewriter.replaceOp(dimOp, replacement);
      continue;
    }

    std::optional<Value> dynamicSize = dynamicSizeForDim(allocOp, *dim);
    if (!dynamicSize)
      return failure();
    rewriter.replaceOp(dimOp, *dynamicSize);
  }

  return success();
}

} // namespace

FailureOr<MemoryRealizationPlan>
MemoryRealizationDriver::materialize(const PlacementPlan &placement,
                                     const StaticMemoryPlan &staticMemory,
                                     const MovementPlan &movement) const {
  if (staticMemory.kernelId != placement.kernelId ||
      movement.kernelId != placement.kernelId)
    return failure();

  MemoryRealizationPlan plan;
  plan.kernelId = placement.kernelId;
  plan.mode = "read_only_freeze";
  plan.frozen = true;
  plan.verificationScope = "plan_identity_only";
  plan.planIdsVerified = true;
  plan.materializedAllocCount = 0;
  plan.materializedCopyCount = 0;
  return plan;
}

FailureOr<llvm::StringMap<unsigned>>
MemoryRealizationDriver::annotateMemorySpaces(ModuleOp module) const {
  MLIRContext *context = module.getContext();
  Attribute vecCalcSpace = IntegerAttr::get(IntegerType::get(context, 32),
                                            kVecCalcMemorySpace);

  SmallVector<AnnotatableAlloc, 4> allocsToAnnotate;
  llvm::DenseSet<Operation *> seenAllocs;
  module.walk([&](linalg::LinalgOp linalgOp) {
    Operation *op = linalgOp.getOperation();
    if (!isVectorOp(op))
      return;
    StringRef kernelId = getKernelId(op);
    if (kernelId.empty())
      return;

    for (Value init : linalgOp.getDpsInits()) {
      auto allocOp = init.getDefiningOp<memref::AllocOp>();
      if (!allocOp || seenAllocs.contains(allocOp.getOperation()))
        continue;

      auto allocType = dyn_cast<MemRefType>(allocOp.getType());
      if (!allocType || allocType.getMemorySpace())
        continue;

      if (!hasOnlyKernelUses(allocOp.getResult(), kernelId))
        continue;

      allocsToAnnotate.push_back({allocOp, kernelId.str()});
      seenAllocs.insert(allocOp.getOperation());
    }
  });

  IRRewriter rewriter(context);
  llvm::StringMap<unsigned> annotationCounts;
  for (const AnnotatableAlloc &item : allocsToAnnotate) {
    memref::AllocOp allocOp = item.alloc;
    auto oldType = cast<MemRefType>(allocOp.getType());
    auto newType = withMemorySpace(oldType, vecCalcSpace);

    rewriter.setInsertionPoint(allocOp);
    auto newAlloc =
        rewriter.create<memref::AllocOp>(allocOp.getLoc(), newType,
                                         allocOp.getDynamicSizes(),
                                         allocOp.getSymbolOperands(),
                                         allocOp.getAlignmentAttr());
    newAlloc->setAttrs(allocOp->getAttrs());

    rewriter.replaceAllUsesWith(allocOp.getResult(), newAlloc.getResult());
    rewriter.eraseOp(allocOp);
    ++annotationCounts[item.kernelId];
  }

  return annotationCounts;
}

FailureOr<llvm::StringMap<Phase5BridgeMaterializationCounts>>
MemoryRealizationDriver::materializePhase5Bridge(ModuleOp module) const {
  MLIRContext *context = module.getContext();
  Attribute vecOutSpace = IntegerAttr::get(IntegerType::get(context, 32),
                                           kVecOutMemorySpace);

  SmallVector<Phase5BridgeOutput, 4> outputsToBridge;
  module.walk([&](linalg::LinalgOp linalgOp) {
    if (!isSupportedPhase5FinalOutput(linalgOp))
      return;

    Operation *op = linalgOp.getOperation();
    StringRef kernelId = getKernelId(op);
    if (kernelId.empty())
      return;

    for (unsigned i = 0, e = linalgOp.getNumDpsInits(); i < e; ++i) {
      OpOperand *initOperand = linalgOp.getDpsInitOperand(i);
      Value init = initOperand->get();
      auto allocOp = init.getDefiningOp<memref::AllocOp>();
      if (!allocOp)
        continue;

      auto allocType = dyn_cast<MemRefType>(allocOp.getType());
      if (!allocType || allocType.getMemorySpace())
        continue;

      memref::CopyOp concatCopy;
      if (!isFinalKernelOutput(init, op)) {
        concatCopy = findSupportedConcatCopyUse(init, op);
        if (!concatCopy)
          continue;
      }

      outputsToBridge.push_back({linalgOp, initOperand, allocOp,
                                 concatCopy, kernelId.str()});
    }
  });

  IRRewriter rewriter(context);
  llvm::StringMap<Phase5BridgeMaterializationCounts> counts;
  for (Phase5BridgeOutput &item : outputsToBridge) {
    memref::AllocOp gmAlloc = item.gmAlloc;
    auto gmType = cast<MemRefType>(gmAlloc.getType());
    auto vecOutType = withMemorySpace(gmType, vecOutSpace);

    rewriter.setInsertionPoint(item.linalgOp);
    auto vecOutAlloc = rewriter.create<memref::AllocOp>(
        item.linalgOp.getLoc(), vecOutType, gmAlloc.getDynamicSizes(),
        gmAlloc.getSymbolOperands(), gmAlloc.getAlignmentAttr());
    vecOutAlloc->setAttrs(gmAlloc->getAttrs());

    if (item.concatCopy) {
      if (failed(replaceAllocDimUses(rewriter, gmAlloc)))
        return failure();

      item.initOperand->set(vecOutAlloc.getResult());
      item.concatCopy->setOperand(0, vecOutAlloc.getResult());
      if (gmAlloc.getResult().use_empty())
        rewriter.eraseOp(gmAlloc);
    } else {
      item.initOperand->set(vecOutAlloc.getResult());
      rewriter.setInsertionPointAfter(item.linalgOp);
      rewriter.create<memref::CopyOp>(item.linalgOp.getLoc(),
                                      vecOutAlloc.getResult(),
                                      gmAlloc.getResult());
    }

    ++counts[item.kernelId].materializedAllocCount;
    ++counts[item.kernelId].materializedCopyCount;
  }

  return counts;
}

void MemoryRealizationDriver::markMemorySpaceMaterialized(
    MemoryRealizationPlan &plan, unsigned annotationCount,
    const Phase5BridgeMaterializationCounts &materializationCounts) const {
  bool hasMaterialization =
      materializationCounts.materializedAllocCount != 0 ||
      materializationCounts.materializedCopyCount != 0;
  plan.mode =
      hasMaterialization ? "memory_space_materialize" : "memory_space_annotate";
  plan.frozen = true;
  plan.verificationScope = hasMaterialization ? "memory_space_materialization"
                                              : "memory_space_annotation";
  plan.planIdsVerified = true;
  plan.memorySpaceAnnotationCount = annotationCount;
  plan.materializedAllocCount = materializationCounts.materializedAllocCount;
  plan.materializedCopyCount = materializationCounts.materializedCopyCount;
}

} // namespace mlir::afir::ascend::realize
