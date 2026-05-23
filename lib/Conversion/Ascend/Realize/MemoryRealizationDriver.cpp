//===- MemoryRealizationDriver.cpp - Ascend memory realization -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "MemoryRealizationDriver.h"

#include "Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.h"
#include "Target/Ascend/TargetProfile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
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
constexpr int64_t kA1MemorySpace =
    static_cast<int64_t>(::mlir::ascend::MemoryPlace::A1);
constexpr int64_t kA2MemorySpace =
    static_cast<int64_t>(::mlir::ascend::MemoryPlace::A2);
constexpr int64_t kB1MemorySpace =
    static_cast<int64_t>(::mlir::ascend::MemoryPlace::B1);
constexpr int64_t kB2MemorySpace =
    static_cast<int64_t>(::mlir::ascend::MemoryPlace::B2);
constexpr int64_t kCo1MemorySpace =
    static_cast<int64_t>(::mlir::ascend::MemoryPlace::CO1);
constexpr int64_t kVecInMemorySpace =
    static_cast<int64_t>(::mlir::ascend::MemoryPlace::VECIN);

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

static bool isCubeOp(Operation *op) {
  return hasRole(op, kOpRoleCube, kKernelizeOpRoleCube);
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

static void annotateAscendCUnits(ModuleOp module) {
  MLIRContext *context = module.getContext();
  module.walk([&](linalg::LinalgOp linalgOp) {
    Operation *op = linalgOp.getOperation();
    if (op->hasAttr(kAscendCUnitAttr))
      return;
    if (isCubeOp(op)) {
      op->setAttr(kAscendCUnitAttr,
                  StringAttr::get(context, kAscendCUnitCube));
      return;
    }
    if (isVectorOp(op))
      op->setAttr(kAscendCUnitAttr,
                  StringAttr::get(context, kAscendCUnitVector));
  });
}

static bool isReductionInitFillForWriter(Operation *user, Operation *writer,
                                         Value output,
                                         const backend::AscendBackendSupportMatrix
                                             &matrix) {
  auto fillOp = dyn_cast<linalg::FillOp>(user);
  auto generic = dyn_cast<linalg::GenericOp>(writer);
  if (!fillOp || !generic ||
      backend::classifyPhase5ReductionBody(generic, matrix) == backend::ComputeKind::Unknown)
    return false;

  if (!llvm::is_contained(fillOp.getOutputs(), output))
    return false;
  return llvm::is_contained(generic.getDpsInits(), output);
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

static bool isFinalKernelOutput(Value value, Operation *writer,
                                const backend::AscendBackendSupportMatrix
                                    &matrix) {
  bool hasExternalUse = false;
  llvm::DenseSet<Operation *> visited;
  for (Operation *user : value.getUsers()) {
    if (user == writer)
      continue;
    if (isReductionInitFillForWriter(user, writer, value, matrix))
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

struct Phase5CubeBridge {
  linalg::LinalgOp linalgOp;
  Value lhs;
  Value rhs;
  OpOperand *initOperand;
  Value originalOutput;
  SmallVector<OpOperand *, 4> vectorInputUses;
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

static void addMaterializationCounts(Phase5BridgeMaterializationCounts &lhs,
                                     const Phase5BridgeMaterializationCounts
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

static LogicalResult verifyReplaceableAllocDimUses(memref::AllocOp allocOp) {
  auto type = cast<MemRefType>(allocOp.getType());
  for (Operation *user : allocOp.getResult().getUsers()) {
    auto dimOp = dyn_cast<memref::DimOp>(user);
    if (!dimOp)
      continue;

    std::optional<int64_t> dim = getConstantIntValue(dimOp.getIndex());
    if (!dim || *dim < 0 || *dim >= type.getRank())
      return failure();

    if (ShapedType::isDynamic(type.getDimSize(*dim)) &&
        !dynamicSizeForDim(allocOp, *dim))
      return failure();
  }
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
    Attribute targetSpace, Phase5BridgeMaterializationCounts &counts) {
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
    Attribute targetSpace, Phase5BridgeMaterializationCounts &counts) {
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

static bool isBridgeableCubeCompute(
    linalg::LinalgOp linalgOp,
    const backend::AscendBackendSupportMatrix &matrix) {
  if (!isCubeOp(linalgOp.getOperation()))
    return false;

  backend::ComputeKind computeKind =
      backend::classifyLinalgComputeKind(linalgOp.getOperation(), matrix);
  std::optional<int64_t> expectedRank;
  switch (computeKind) {
  case backend::ComputeKind::Matmul:
    expectedRank = 2;
    break;
  case backend::ComputeKind::BatchMatmul:
    expectedRank = 3;
    break;
  default:
    return false;
  }

  if (linalgOp.getNumDpsInputs() != 2 || linalgOp.getNumDpsInits() != 1)
    return false;
  auto lhsType =
      dyn_cast<MemRefType>(linalgOp.getDpsInputOperand(0)->get().getType());
  auto rhsType =
      dyn_cast<MemRefType>(linalgOp.getDpsInputOperand(1)->get().getType());
  auto outType =
      dyn_cast<MemRefType>(linalgOp.getDpsInitOperand(0)->get().getType());
  if (!lhsType || !rhsType || !outType ||
      lhsType.getRank() != *expectedRank ||
      rhsType.getRank() != *expectedRank ||
      outType.getRank() != *expectedRank)
    return false;
  return !lhsType.getMemorySpace() && !rhsType.getMemorySpace() &&
         !outType.getMemorySpace();
}

static bool isDpsInputOperand(linalg::LinalgOp linalgOp,
                              OpOperand *operand) {
  for (OpOperand *input : linalgOp.getDpsInputOperands())
    if (input == operand)
      return true;
  return false;
}

static bool collectSafeCubeVectorUses(linalg::LinalgOp linalgOp,
                                      SmallVectorImpl<OpOperand *> &uses,
                                      const backend::AscendBackendSupportMatrix
                                          &matrix,
                                      DominanceInfo &dominance) {
  Operation *cubeOp = linalgOp.getOperation();
  StringRef kernelId = getKernelId(cubeOp);
  Value output = linalgOp.getDpsInitOperand(0)->get();

  for (OpOperand &use : output.getUses()) {
    Operation *user = use.getOwner();
    if (user == cubeOp)
      continue;
    if (isa<memref::DimOp, memref::DeallocOp>(user))
      continue;

    auto linalgUser = dyn_cast<linalg::LinalgOp>(user);
    if (!linalgUser || getKernelId(user) != kernelId ||
        !dominance.properlyDominates(cubeOp, user) ||
        !backend::isSupportedPhase5VectorOutput(linalgUser, matrix) ||
        !isDpsInputOperand(linalgUser, &use))
      return false;

    uses.push_back(&use);
  }

  return !uses.empty();
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

  FailureOr<llvm::StringMap<Phase5BridgeMaterializationCounts>>
      movementCounts = materializeMovementSteps(module, bundles);
  if (failed(movementCounts))
    return failure();

  FailureOr<llvm::StringMap<unsigned>> annotationCounts =
      annotateMemorySpaces(module);
  if (failed(annotationCounts))
    return failure();

  FailureOr<llvm::StringMap<Phase5BridgeMaterializationCounts>>
      phase5BridgeCounts = materializePhase5Bridge(module);
  if (failed(phase5BridgeCounts))
    return failure();

  for (RealizePlanBundle &bundle : bundles) {
    unsigned annotationCount = 0;
    auto countIt = annotationCounts->find(bundle.kernel.kernelId);
    if (countIt != annotationCounts->end())
      annotationCount = countIt->second;
    Phase5BridgeMaterializationCounts materializationCount;
    auto bridgeIt = phase5BridgeCounts->find(bundle.kernel.kernelId);
    if (bridgeIt != phase5BridgeCounts->end())
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

FailureOr<llvm::StringMap<Phase5BridgeMaterializationCounts>>
MemoryRealizationDriver::materializeMovementSteps(
    ModuleOp module, MutableArrayRef<RealizePlanBundle> bundles) const {
  MLIRContext *context = module.getContext();
  IRRewriter rewriter(context);
  llvm::StringMap<Phase5BridgeMaterializationCounts> counts;

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

FailureOr<llvm::StringMap<Phase5BridgeMaterializationCounts>>
MemoryRealizationDriver::materializePhase5Bridge(ModuleOp module) const {
  MLIRContext *context = module.getContext();
  backend::AscendBackendSupportMatrix matrix;

  Attribute a1Space = getMemorySpaceAttr(context, kA1MemorySpace);
  Attribute a2Space = getMemorySpaceAttr(context, kA2MemorySpace);
  Attribute b1Space = getMemorySpaceAttr(context, kB1MemorySpace);
  Attribute b2Space = getMemorySpaceAttr(context, kB2MemorySpace);
  Attribute co1Space = getMemorySpaceAttr(context, kCo1MemorySpace);
  Attribute vecInSpace = getMemorySpaceAttr(context, kVecInMemorySpace);
  Attribute vecOutSpace = getMemorySpaceAttr(context, kVecOutMemorySpace);
  DominanceInfo dominance(module);

  SmallVector<Phase5CubeBridge, 4> cubeBridges;
  module.walk([&](linalg::LinalgOp linalgOp) {
    StringRef kernelId = getKernelId(linalgOp.getOperation());
    if (kernelId.empty() || !isBridgeableCubeCompute(linalgOp, matrix))
      return;

    Value originalOutput = linalgOp.getDpsInitOperand(0)->get();
    SmallVector<OpOperand *, 4> vectorInputUses;
    if (!collectSafeCubeVectorUses(linalgOp, vectorInputUses, matrix,
                                   dominance))
      return;

    cubeBridges.push_back({linalgOp, linalgOp.getDpsInputOperand(0)->get(),
                           linalgOp.getDpsInputOperand(1)->get(),
                           linalgOp.getDpsInitOperand(0), originalOutput,
                           std::move(vectorInputUses), kernelId.str()});
  });

  SmallVector<Phase5BridgeOutput, 4> outputsToBridge;
  module.walk([&](linalg::LinalgOp linalgOp) {
    if (!backend::isSupportedPhase5FinalOutput(linalgOp, matrix))
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
      if (!isFinalKernelOutput(init, op, matrix)) {
        concatCopy = findSupportedConcatCopyUse(init, op);
        if (!concatCopy)
          continue;
      }

      outputsToBridge.push_back({linalgOp, initOperand, allocOp,
                                 concatCopy, kernelId.str()});
    }
  });

  for (const Phase5BridgeOutput &item : outputsToBridge)
    if (item.concatCopy && failed(verifyReplaceableAllocDimUses(item.gmAlloc)))
      return failure();

  annotateAscendCUnits(module);

  IRRewriter rewriter(context);
  llvm::StringMap<Phase5BridgeMaterializationCounts> counts;
  for (Phase5CubeBridge &item : cubeBridges) {
    linalg::LinalgOp linalgOp = item.linalgOp;
    if (!linalgOp)
      continue;

    Location loc = linalgOp.getLoc();
    rewriter.setInsertionPoint(linalgOp);
    memref::AllocOp a1 =
        createMemorySpaceAllocLike(rewriter, loc, item.lhs, a1Space);
    rewriter.create<memref::CopyOp>(loc, item.lhs, a1.getResult());
    memref::AllocOp a2 =
        createMemorySpaceAllocLike(rewriter, loc, a1.getResult(), a2Space);
    rewriter.create<memref::CopyOp>(loc, a1.getResult(), a2.getResult());

    memref::AllocOp b1 =
        createMemorySpaceAllocLike(rewriter, loc, item.rhs, b1Space);
    rewriter.create<memref::CopyOp>(loc, item.rhs, b1.getResult());
    memref::AllocOp b2 =
        createMemorySpaceAllocLike(rewriter, loc, b1.getResult(), b2Space);
    rewriter.create<memref::CopyOp>(loc, b1.getResult(), b2.getResult());

    memref::AllocOp co1 = createMemorySpaceAllocLike(
        rewriter, loc, item.originalOutput, co1Space);
    linalgOp.getDpsInputOperand(0)->set(a2.getResult());
    linalgOp.getDpsInputOperand(1)->set(b2.getResult());
    item.initOperand->set(co1.getResult());

    rewriter.setInsertionPointAfter(linalgOp);
    memref::AllocOp vecIn = createMemorySpaceAllocLike(
        rewriter, loc, co1.getResult(), vecInSpace);
    rewriter.create<memref::CopyOp>(loc, co1.getResult(),
                                    vecIn.getResult());
    for (OpOperand *use : item.vectorInputUses)
      use->set(vecIn.getResult());

    counts[item.kernelId].materializedAllocCount += 6;
    counts[item.kernelId].materializedCopyCount += 5;
  }

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
