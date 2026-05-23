//===- MemoryRealizationDriver.cpp - Ascend memory realization -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "MemoryRealizationDriver.h"
#include "TranslateMemoryBridge.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
static StringRef getKernelId(Operation *op) {
  auto kernelAttr = op->getAttrOfType<StringAttr>(kKernelAttr);
  return kernelAttr ? kernelAttr.getValue() : StringRef();
}

static bool opRolesAttrHasRole(Operation *op, StringRef roleName) {
  auto roles = op->getAttrOfType<ArrayAttr>(kOpRolesAttr);
  if (!roles)
    return false;
  for (Attribute attr : roles) {
    auto role = dyn_cast<StringAttr>(attr);
    if (role && role.getValue() == roleName)
      return true;
  }
  return false;
}

static bool hasRole(Operation *op, StringRef legacyRoleName,
                    StringRef kernelizeRoleName) {
  auto role = op->getAttrOfType<StringAttr>(kOpRoleAttr);
  if (role && role.getValue() == legacyRoleName)
    return true;
  return opRolesAttrHasRole(op, kernelizeRoleName);
}

static bool isVectorOp(Operation *op) {
  return hasRole(op, kOpRoleVector, kKernelizeOpRoleVector);
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

static Attribute getMemorySpaceAttr(MLIRContext *context,
                                    int64_t memorySpace) {
  return IntegerAttr::get(IntegerType::get(context, 32), memorySpace);
}

struct AnnotatableAlloc {
  memref::AllocOp alloc;
  std::string kernelId;
};

struct MovementUseRewrite {
  OpOperand *use = nullptr;
  SmallVector<Operation *, 2> viewChain;
};

struct MovementMaterializationItem {
  const MovementStep *step = nullptr;
  const StaticMemoryWorkspaceSlot *slot = nullptr;
  Value source;
  SmallVector<MovementUseRewrite, 4> uses;
  Operation *firstUser = nullptr;
  bool hasDynamicViewChain = false;
};

static void addMaterializationCounts(TranslateBridgeMaterializationCounts &lhs,
                                     const TranslateBridgeMaterializationCounts
                                         &rhs) {
  lhs.materializedAllocCount += rhs.materializedAllocCount;
  lhs.materializedCopyCount += rhs.materializedCopyCount;
}

static std::optional<int64_t> getMemorySpaceValue(MemRefType type) {
  Attribute memorySpace = type.getMemorySpace();
  if (!memorySpace)
    return std::nullopt;
  if (auto intAttr = dyn_cast<IntegerAttr>(memorySpace))
    return intAttr.getInt();
  return std::nullopt;
}

static bool hasMemorySpace(MemRefType type, MemoryPlace place) {
  std::optional<int64_t> memorySpace = getMemorySpaceValue(type);
  return memorySpace && *memorySpace == static_cast<int64_t>(place);
}

static bool isGmMemref(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && !type.getMemorySpace();
}

static bool hasDynamicSubViewOperand(memref::SubViewOp subview) {
  for (OpFoldResult offset : subview.getMixedOffsets())
    if (isa<Value>(offset))
      return true;
  for (OpFoldResult size : subview.getMixedSizes())
    if (isa<Value>(size))
      return true;
  for (OpFoldResult stride : subview.getMixedStrides())
    if (isa<Value>(stride))
      return true;
  return false;
}

static bool hasDynamicViewChain(ArrayRef<Operation *> viewChain) {
  for (Operation *op : viewChain)
    if (auto subview = dyn_cast<memref::SubViewOp>(op))
      if (hasDynamicSubViewOperand(subview))
        return true;
  return false;
}

static bool hasDynamicViewChain(ArrayRef<MovementUseRewrite> uses) {
  for (const MovementUseRewrite &rewrite : uses)
    if (hasDynamicViewChain(rewrite.viewChain))
      return true;
  return false;
}

static void appendUniqueGmMovementSource(SmallVectorImpl<Value> &sources,
                                         llvm::DenseSet<Value> &seen,
                                         Value value) {
  if (!isGmMemref(value) || !seen.insert(value).second)
    return;
  sources.push_back(value);
}

static SmallVector<Value, 8> collectGmMovementSourcesByValueId(
    ModuleOp module, StringRef kernelId) {
  SmallVector<Value, 8> sources;
  llvm::DenseSet<Value> seen;
  module.walk([&](linalg::LinalgOp linalgOp) {
    if (getKernelId(linalgOp.getOperation()) != kernelId)
      return;
    for (OpOperand *input : linalgOp.getDpsInputOperands())
      appendUniqueGmMovementSource(sources, seen, input->get());
    for (OpOperand &init : linalgOp.getDpsInitsMutable())
      appendUniqueGmMovementSource(sources, seen, init.get());
  });
  return sources;
}

static bool collectViewChainToSource(
    Value value, Value source, SmallVectorImpl<Operation *> &viewChain) {
  if (value == source)
    return true;

  Operation *def = value.getDefiningOp();
  if (!def || def->getNumResults() != 1)
    return false;

  if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
    if (!collectViewChainToSource(subview.getSource(), source, viewChain))
      return false;
    viewChain.push_back(def);
    return true;
  }

  if (auto castOp = dyn_cast<memref::CastOp>(def)) {
    if (!collectViewChainToSource(castOp.getSource(), source, viewChain))
      return false;
    viewChain.push_back(def);
    return true;
  }

  if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(def)) {
    if (!collectViewChainToSource(reshapeOp.getSource(), source, viewChain))
      return false;
    viewChain.push_back(def);
    return true;
  }

  if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(def)) {
    if (!collectViewChainToSource(expandOp.getSrc(), source, viewChain))
      return false;
    viewChain.push_back(def);
    return true;
  }

  if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(def)) {
    if (!collectViewChainToSource(collapseOp.getSrc(), source, viewChain))
      return false;
    viewChain.push_back(def);
    return true;
  }

  return false;
}

static LogicalResult collectMovementSourceUses(
    ModuleOp module, StringRef kernelId, Value source,
    SmallVectorImpl<MovementUseRewrite> &uses, Operation *&firstUser) {
  Block *block = nullptr;
  bool unsupportedUseBlock = false;
  module.walk([&](linalg::LinalgOp linalgOp) {
    if (getKernelId(linalgOp.getOperation()) != kernelId)
      return;

    for (OpOperand *input : linalgOp.getDpsInputOperands()) {
      SmallVector<Operation *, 2> viewChain;
      if (!collectViewChainToSource(input->get(), source, viewChain))
        continue;

      Operation *user = linalgOp.getOperation();
      if (!block)
        block = user->getBlock();
      if (block != user->getBlock()) {
        unsupportedUseBlock = true;
        return;
      }

      uses.push_back({input, std::move(viewChain)});
      if (!firstUser || user->isBeforeInBlock(firstUser))
        firstUser = user;
    }
  });

  if (unsupportedUseBlock || uses.empty() || !firstUser)
    return failure();
  for (const MovementUseRewrite &rewrite : uses)
    if (rewrite.use->getOwner()->getBlock() != firstUser->getBlock())
      return failure();
  return success();
}

static SmallVector<Value, 4> buildDynamicSizes(OpBuilder &builder,
                                               Location loc, Value source) {
  SmallVector<Value, 4> dynamicSizes;
  auto type = cast<MemRefType>(source.getType());
  for (auto [index, dim] : llvm::enumerate(type.getShape())) {
    if (!ShapedType::isDynamic(dim))
      continue;
    dynamicSizes.push_back(
        builder.create<memref::DimOp>(loc, source, index));
  }
  return dynamicSizes;
}

static memref::AllocOp createMemorySpaceAllocLike(IRRewriter &rewriter,
                                                  Location loc, Value source,
                                                  Attribute memorySpace) {
  auto sourceType = cast<MemRefType>(source.getType());
  auto allocType = withMemorySpace(sourceType, memorySpace);
  SmallVector<Value, 4> dynamicSizes =
      buildDynamicSizes(rewriter, loc, source);
  return rewriter.create<memref::AllocOp>(loc, allocType, dynamicSizes);
}

static bool isStaticIdentityMemRef(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.hasStaticShape() && type.getLayout().isIdentity();
}

static std::optional<uint64_t> getElementByteWidth(MemRefType type) {
  unsigned elementBits = type.getElementTypeBitWidth();
  if (elementBits == 0 || elementBits % 8 != 0)
    return std::nullopt;
  return elementBits / 8;
}

static std::optional<uint64_t>
getElementOffset(const StaticMemoryWorkspaceSlot &slot, MemRefType sourceType) {
  std::optional<uint64_t> elementBytes = getElementByteWidth(sourceType);
  if (!elementBytes || *elementBytes == 0)
    return std::nullopt;
  if (slot.offset % *elementBytes != 0 || slot.byteSize % *elementBytes != 0)
    return std::nullopt;
  return slot.offset / *elementBytes;
}

static std::optional<uint64_t> getPackedWorkspaceElementCount(
    ArrayRef<MovementMaterializationItem> items) {
  if (items.empty())
    return std::nullopt;

  auto sourceType = dyn_cast<MemRefType>(items.front().source.getType());
  if (!sourceType)
    return std::nullopt;

  std::optional<uint64_t> elementBytes = getElementByteWidth(sourceType);
  if (!elementBytes || *elementBytes == 0)
    return std::nullopt;

  uint64_t packedBytes = 0;
  for (const MovementMaterializationItem &item : items) {
    if (!item.slot || !item.slot->staticByteSizeKnown)
      return std::nullopt;
    auto itemType = dyn_cast<MemRefType>(item.source.getType());
    if (!itemType || itemType.getElementType() != sourceType.getElementType())
      return std::nullopt;
    if (item.slot->offset % *elementBytes != 0 ||
        item.slot->byteSize % *elementBytes != 0)
      return std::nullopt;
    packedBytes =
        std::max<uint64_t>(packedBytes, item.slot->offset + item.slot->byteSize);
  }

  return packedBytes / *elementBytes;
}

static SmallVector<int64_t, 4> getIdentityStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t, 4> strides(shape.size(), 1);
  int64_t runningStride = 1;
  for (int64_t index = static_cast<int64_t>(shape.size()) - 1; index >= 0;
       --index) {
    strides[index] = runningStride;
    runningStride *= shape[index];
  }
  return strides;
}

static bool canShareMovementWorkspace(const MovementMaterializationItem &lhs,
                                      const MovementMaterializationItem &rhs) {
  if (!lhs.step || !rhs.step || !lhs.slot || !rhs.slot ||
      lhs.step->dstPlace != rhs.step->dstPlace)
    return false;
  if (!lhs.firstUser || !rhs.firstUser ||
      lhs.firstUser->getBlock() != rhs.firstUser->getBlock())
    return false;
  if (!isStaticIdentityMemRef(lhs.source) ||
      !isStaticIdentityMemRef(rhs.source))
    return false;

  auto lhsType = cast<MemRefType>(lhs.source.getType());
  auto rhsType = cast<MemRefType>(rhs.source.getType());
  if (lhsType.getElementType() != rhsType.getElementType())
    return false;
  return lhs.slot->staticByteSizeKnown && rhs.slot->staticByteSizeKnown &&
         getElementOffset(*lhs.slot, lhsType) &&
         getElementOffset(*rhs.slot, rhsType);
}

static Operation *
getEarliestUserInBlock(ArrayRef<MovementMaterializationItem> items) {
  Operation *earliest = items.front().firstUser;
  for (const MovementMaterializationItem &item : items)
    if (item.firstUser->isBeforeInBlock(earliest))
      earliest = item.firstUser;
  return earliest;
}

static memref::AllocOp createPackedMovementWorkspaceAlloc(
    IRRewriter &rewriter, Location loc,
    ArrayRef<MovementMaterializationItem> items, Attribute memorySpace) {
  auto sourceType = cast<MemRefType>(items.front().source.getType());
  std::optional<uint64_t> elementCount =
      getPackedWorkspaceElementCount(items);
  if (!elementCount)
    return {};

  auto workspaceType = MemRefType::get(
      {static_cast<int64_t>(*elementCount)}, sourceType.getElementType(),
      MemRefLayoutAttrInterface{}, memorySpace);
  return rewriter.create<memref::AllocOp>(loc, workspaceType);
}

static Value createPackedMovementWorkspaceView(
    IRRewriter &rewriter, Location loc, Value workspace,
    const MovementMaterializationItem &item, Attribute memorySpace) {
  auto sourceType = cast<MemRefType>(item.source.getType());
  std::optional<uint64_t> elementOffset =
      getElementOffset(*item.slot, sourceType);
  if (!elementOffset)
    return {};

  SmallVector<int64_t, 4> sizes(sourceType.getShape().begin(),
                                sourceType.getShape().end());
  SmallVector<int64_t, 4> strides = getIdentityStrides(sizes);
  auto layout = StridedLayoutAttr::get(
      rewriter.getContext(), static_cast<int64_t>(*elementOffset), strides);
  auto viewType = MemRefType::get(sizes, sourceType.getElementType(), layout,
                                  memorySpace);
  return rewriter
      .create<memref::ReinterpretCastOp>(
          loc, viewType, workspace,
          /*offset=*/static_cast<int64_t>(*elementOffset), sizes, strides)
      .getResult();
}

static FailureOr<Value>
materializeMovementUseViewChain(IRRewriter &rewriter, Location loc, Value base,
                                ArrayRef<Operation *> viewChain,
                                Attribute memorySpace) {
  Value current = base;
  for (Operation *viewOp : viewChain) {
    if (auto subview = dyn_cast<memref::SubViewOp>(viewOp)) {
      auto oldType = dyn_cast<MemRefType>(subview.getType());
      if (!oldType)
        return failure();
      auto newType = withMemorySpace(oldType, memorySpace);
      current = rewriter
                    .create<memref::SubViewOp>(
                        loc, newType, current, subview.getMixedOffsets(),
                        subview.getMixedSizes(), subview.getMixedStrides())
                    .getResult();
      continue;
    }

    if (auto castOp = dyn_cast<memref::CastOp>(viewOp)) {
      auto oldType = dyn_cast<MemRefType>(castOp.getType());
      if (!oldType)
        return failure();
      auto newType = withMemorySpace(oldType, memorySpace);
      current =
          rewriter.create<memref::CastOp>(loc, newType, current).getResult();
      continue;
    }

    if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(viewOp)) {
      auto oldType = dyn_cast<MemRefType>(reshapeOp.getType());
      if (!oldType)
        return failure();
      auto newType = withMemorySpace(oldType, memorySpace);
      current = rewriter
                    .create<memref::ReshapeOp>(loc, newType, current,
                                                reshapeOp.getShape())
                    .getResult();
      continue;
    }

    if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(viewOp)) {
      auto oldType = dyn_cast<MemRefType>(expandOp.getType());
      if (!oldType)
        return failure();
      auto newType = withMemorySpace(oldType, memorySpace);
      current =
          rewriter
              .create<memref::ExpandShapeOp>(
                  loc, newType, current, expandOp.getReassociationIndices(),
                  expandOp.getMixedOutputShape())
              .getResult();
      continue;
    }

    if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(viewOp)) {
      auto oldType = dyn_cast<MemRefType>(collapseOp.getType());
      if (!oldType)
        return failure();
      auto newType = withMemorySpace(oldType, memorySpace);
      current =
          rewriter
              .create<memref::CollapseShapeOp>(
                  loc, newType, current, collapseOp.getReassociationIndices())
              .getResult();
      continue;
    }

    return failure();
  }
  return current;
}

static LogicalResult preflightMovementUseViewChain(
    ArrayRef<Operation *> viewChain, Attribute memorySpace) {
  for (Operation *viewOp : viewChain) {
    auto resultType =
        viewOp->getNumResults() == 1
            ? dyn_cast<MemRefType>(viewOp->getResult(0).getType())
            : MemRefType();
    if (!resultType)
      return failure();

    if (isa<memref::SubViewOp, memref::CastOp, memref::ReshapeOp,
            memref::ExpandShapeOp, memref::CollapseShapeOp>(viewOp)) {
      (void)withMemorySpace(resultType, memorySpace);
      continue;
    }

    return failure();
  }
  return success();
}

static LogicalResult
preflightSingleMovementItem(const MovementMaterializationItem &item,
                            Attribute targetSpace) {
  if (!item.step || !item.slot || !item.firstUser ||
      !isa<MemRefType>(item.source.getType()))
    return failure();
  for (const MovementUseRewrite &rewrite : item.uses) {
    if (!rewrite.use || !rewrite.use->getOwner() ||
        rewrite.use->getOwner()->getBlock() != item.firstUser->getBlock())
      return failure();
    if (failed(preflightMovementUseViewChain(rewrite.viewChain, targetSpace)))
      return failure();
  }
  return success();
}

static LogicalResult
preflightMovementWorkspaceGroup(ArrayRef<MovementMaterializationItem> items,
                                Attribute targetSpace) {
  if (items.empty() || !getEarliestUserInBlock(items))
    return failure();
  for (const MovementMaterializationItem &item : items) {
    if (failed(preflightSingleMovementItem(item, targetSpace)))
      return failure();
    auto sourceType = dyn_cast<MemRefType>(item.source.getType());
    if (!sourceType || !getElementOffset(*item.slot, sourceType))
      return failure();
  }
  return success();
}

static LogicalResult materializeSingleMovementItem(
    IRRewriter &rewriter, const MovementMaterializationItem &item,
    Attribute targetSpace, TranslateBridgeMaterializationCounts &counts) {
  rewriter.setInsertionPoint(item.firstUser);
  memref::AllocOp localAlloc = createMemorySpaceAllocLike(
      rewriter, item.firstUser->getLoc(), item.source, targetSpace);
  rewriter.create<memref::CopyOp>(item.firstUser->getLoc(), item.source,
                                  localAlloc.getResult());
  for (const MovementUseRewrite &rewrite : item.uses) {
    FailureOr<Value> replacement = materializeMovementUseViewChain(
        rewriter, item.firstUser->getLoc(), localAlloc.getResult(),
        rewrite.viewChain, targetSpace);
    if (failed(replacement))
      return failure();
    rewrite.use->set(*replacement);
  }

  auto localType = cast<MemRefType>(localAlloc.getType());
  if (!hasMemorySpace(localType, item.step->dstPlace))
    return failure();

  ++counts.materializedAllocCount;
  ++counts.materializedCopyCount;
  return success();
}

static LogicalResult materializeMovementWorkspaceGroup(
    IRRewriter &rewriter, ArrayRef<MovementMaterializationItem> items,
    Attribute targetSpace, TranslateBridgeMaterializationCounts &counts) {
  Operation *firstUser = getEarliestUserInBlock(items);
  rewriter.setInsertionPoint(firstUser);
  memref::AllocOp workspace = createPackedMovementWorkspaceAlloc(
      rewriter, firstUser->getLoc(), items, targetSpace);
  if (!workspace)
    return failure();

  if (!hasMemorySpace(cast<MemRefType>(workspace.getType()),
                      items.front().step->dstPlace))
    return failure();

  for (const MovementMaterializationItem &item : items) {
    rewriter.setInsertionPoint(item.firstUser);
    Value localView = createPackedMovementWorkspaceView(
        rewriter, item.firstUser->getLoc(), workspace.getResult(), item,
        targetSpace);
    if (!localView)
      return failure();
    rewriter.create<memref::CopyOp>(item.firstUser->getLoc(), item.source,
                                    localView);
    for (const MovementUseRewrite &rewrite : item.uses) {
      FailureOr<Value> replacement = materializeMovementUseViewChain(
          rewriter, item.firstUser->getLoc(), localView, rewrite.viewChain,
          targetSpace);
      if (failed(replacement))
        return failure();
      rewrite.use->set(*replacement);
    }
    ++counts.materializedCopyCount;
  }

  ++counts.materializedAllocCount;
  return success();
}

static unsigned
countDynamicViewChainRewrites(ArrayRef<MovementMaterializationItem> items) {
  unsigned count = 0;
  for (const MovementMaterializationItem &item : items) {
    if (!item.hasDynamicViewChain)
      continue;
    for (const MovementUseRewrite &rewrite : item.uses)
      if (hasDynamicViewChain(rewrite.viewChain))
        ++count;
  }
  return count;
}

static unsigned
countDynamicViewChainRewrites(const MovementMaterializationItem &item) {
  unsigned count = 0;
  if (!item.hasDynamicViewChain)
    return count;
  for (const MovementUseRewrite &rewrite : item.uses)
    if (hasDynamicViewChain(rewrite.viewChain))
      ++count;
  return count;
}

static const StaticMemoryWorkspaceSlot *
lookupWorkspaceSlot(const StaticMemoryPlan &staticMemory, unsigned slotId) {
  for (const StaticMemoryWorkspaceSlot &slot : staticMemory.workspaceSlots)
    if (slot.slotId == slotId)
      return &slot;
  return nullptr;
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

LogicalResult
MemoryRealizationDriver::materialize(ModuleOp module,
                                     MutableArrayRef<RealizePlanBundle> bundles,
                                     MemoryRealizationMode mode) const {
  for (RealizePlanBundle &bundle : bundles) {
    FailureOr<MemoryRealizationPlan> realization =
        materialize(bundle.placement, bundle.staticMemory, bundle.movement);
    if (failed(realization))
      return failure();
    bundle.realization = std::move(*realization);
  }

  if (mode == MemoryRealizationMode::PlanOnly)
    return success();

  FailureOr<llvm::StringMap<TranslateBridgeMaterializationCounts>>
      movementCounts = materializeMovementSteps(module, bundles);
  if (failed(movementCounts))
    return failure();

  FailureOr<llvm::StringMap<unsigned>> annotationCounts =
      annotateMemorySpaces(module);
  if (failed(annotationCounts))
    return failure();

  FailureOr<llvm::StringMap<TranslateBridgeMaterializationCounts>>
      translateBridgeCounts = materializeTranslateMemoryBridge(module);
  if (failed(translateBridgeCounts))
    return failure();

  for (RealizePlanBundle &bundle : bundles) {
    unsigned annotationCount = 0;
    auto countIt = annotationCounts->find(bundle.kernel.kernelId);
    if (countIt != annotationCounts->end())
      annotationCount = countIt->second;
    TranslateBridgeMaterializationCounts materializationCount;
    auto bridgeIt = translateBridgeCounts->find(bundle.kernel.kernelId);
    if (bridgeIt != translateBridgeCounts->end())
      materializationCount = bridgeIt->second;
    auto movementIt = movementCounts->find(bundle.kernel.kernelId);
    if (movementIt != movementCounts->end())
      addMaterializationCounts(materializationCount, movementIt->second);
    markMemorySpaceMaterialized(bundle.realization, annotationCount,
                                materializationCount);
  }

  return success();
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

FailureOr<llvm::StringMap<TranslateBridgeMaterializationCounts>>
MemoryRealizationDriver::materializeMovementSteps(
    ModuleOp module, MutableArrayRef<RealizePlanBundle> bundles) const {
  MLIRContext *context = module.getContext();
  IRRewriter rewriter(context);
  llvm::StringMap<TranslateBridgeMaterializationCounts> counts;

  for (RealizePlanBundle &bundle : bundles) {
    StringRef kernelId = bundle.kernel.kernelId;
    if (kernelId.empty())
      continue;

    SmallVector<Value, 8> sources =
        collectGmMovementSourcesByValueId(module, kernelId);
    SmallVector<MovementMaterializationItem, 8> items;
    for (const MovementStep &step : bundle.movement.movementSteps) {
      if (!step.pathSelected || step.pathSelectionDeferred ||
          step.srcPlace != MemoryPlace::GM)
        continue;
      if (step.valueId >= sources.size())
        return failure();
      const StaticMemoryWorkspaceSlot *slot =
          lookupWorkspaceSlot(bundle.staticMemory, step.slotId);
      if (!slot)
        return failure();

      Value source = sources[step.valueId];
      SmallVector<MovementUseRewrite, 4> uses;
      Operation *firstUser = nullptr;
      if (failed(collectMovementSourceUses(module, kernelId, source, uses,
                                           firstUser)))
        return failure();

      bool dynamicViewChain = hasDynamicViewChain(uses);
      items.push_back(
          {&step, slot, source, std::move(uses), firstUser, dynamicViewChain});
    }

    SmallVector<bool, 8> materialized(items.size(), false);
    for (unsigned i = 0, e = items.size(); i < e; ++i) {
      if (materialized[i])
        continue;

      Attribute targetSpace = getMemorySpaceAttr(
          context, static_cast<int64_t>(items[i].step->dstPlace));
      SmallVector<MovementMaterializationItem, 4> group;
      SmallVector<unsigned, 4> groupIndices;
      group.push_back(items[i]);
      groupIndices.push_back(i);
      for (unsigned j = i + 1; j < e; ++j) {
        if (materialized[j] || !canShareMovementWorkspace(items[i], items[j]))
          continue;
        group.push_back(items[j]);
        groupIndices.push_back(j);
      }

      if (group.size() > 1) {
        if (failed(preflightMovementWorkspaceGroup(group, targetSpace))) {
          unsigned deferred = countDynamicViewChainRewrites(group);
          if (deferred == 0)
            return failure();
          bundle.movement.deferredViewChainRewriteCount += deferred;
          for (unsigned index : groupIndices)
            materialized[index] = true;
          continue;
        }
        if (failed(materializeMovementWorkspaceGroup(
                rewriter, group, targetSpace, counts[kernelId])))
          return failure();
        bundle.movement.dynamicViewChainRewriteCount +=
            countDynamicViewChainRewrites(group);
        for (unsigned index : groupIndices)
          materialized[index] = true;
        continue;
      }

      if (failed(preflightSingleMovementItem(items[i], targetSpace))) {
        unsigned deferred = countDynamicViewChainRewrites(items[i]);
        if (deferred == 0)
          return failure();
        bundle.movement.deferredViewChainRewriteCount += deferred;
        materialized[i] = true;
        continue;
      }
      if (failed(materializeSingleMovementItem(rewriter, items[i], targetSpace,
                                               counts[kernelId])))
        return failure();
      bundle.movement.dynamicViewChainRewriteCount +=
          countDynamicViewChainRewrites(items[i]);
      materialized[i] = true;
    }
  }

  return counts;
}



void MemoryRealizationDriver::markMemorySpaceMaterialized(
    MemoryRealizationPlan &plan, unsigned annotationCount,
    const TranslateBridgeMaterializationCounts &materializationCounts) const {
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
