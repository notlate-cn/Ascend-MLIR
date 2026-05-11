//===- MemoryRealizationDriver.cpp - Ascend memory realization -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/MemoryRealizationDriver.h"

#include "Target/Ascend/TargetProfile.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::afir::ascend::realize {
namespace {

constexpr int64_t kVecCalcMemorySpace =
    static_cast<int64_t>(::mlir::ascend::MemoryPlace::VECCALC);

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

struct AnnotatableAlloc {
  memref::AllocOp alloc;
  std::string kernelId;
};

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

void MemoryRealizationDriver::markMemorySpaceAnnotated(
    MemoryRealizationPlan &plan, unsigned annotationCount) const {
  plan.mode = "memory_space_annotate";
  plan.frozen = true;
  plan.verificationScope = "memory_space_annotation";
  plan.planIdsVerified = true;
  plan.memorySpaceAnnotationCount = annotationCount;
  plan.materializedAllocCount = 0;
  plan.materializedCopyCount = 0;
}

} // namespace mlir::afir::ascend::realize
