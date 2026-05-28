//===- KernelSplitPass.cpp - Ascend logical kernel outlining -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Common/Attributes.h"
#include "Conversion/Ascend/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

#define GEN_PASS_DECL_ASCENDKERNELSPLITPASS
#define GEN_PASS_DEF_ASCENDKERNELSPLITPASS
#include "Conversion/Ascend/Passes.h.inc"

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kKernelMetadataKernelKey = "kernel";
constexpr llvm::StringLiteral kKernelMetadataSelectedTileShapeKey =
    "selected_tile_shape";
constexpr llvm::StringLiteral kKernelMetadataGuardMarkersKey =
    "guard_markers";
constexpr llvm::StringLiteral kKernelMetadataTailPoliciesKey =
    "tail_policies";
constexpr llvm::StringLiteral kKernelMetadataTailPlanKey = "tail_plan";
constexpr llvm::StringLiteral kKernelMetadataTailMarkersKey = "tail_markers";
constexpr llvm::StringLiteral kKernelMetadataTargetTilePolicyKey =
    "target_tile_policy";

struct KernelSplitPlan {
  std::string kernelId;
  unsigned firstOrdinal = 0;
  llvm::SetVector<Operation *> ops;
  SmallVector<Value> arguments;
  SmallVector<Value> results;
  DictionaryAttr scheduleMetadata;
};

Operation *getTopLevelOpInFunc(Operation *op, func::FuncOp funcOp) {
  while (op && op->getParentOp() && op->getParentOp() != funcOp)
    op = op->getParentOp();
  return op && op->getParentOp() == funcOp ? op : nullptr;
}

bool isFuncEntryBlockArgument(Value value, func::FuncOp funcOp) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  return blockArg && blockArg.getOwner() == &funcOp.getBody().front();
}

std::optional<std::string> getKernelId(Operation *op) {
  if (auto kernelAttr = op->getAttrOfType<StringAttr>(
          ::mlir::ascend::kKernelAttr))
    return kernelAttr.getValue().str();
  return std::nullopt;
}

FailureOr<SmallVector<DictionaryAttr>>
getScheduleMetadata(func::FuncOp funcOp) {
  SmallVector<DictionaryAttr> entries;
  auto metadata = funcOp->getAttrOfType<ArrayAttr>(
      ::mlir::ascend::kScheduleKernelMetadataAttr);
  if (!metadata)
    return entries;

  for (auto [index, rawEntry] : llvm::enumerate(metadata)) {
    auto entry = dyn_cast<DictionaryAttr>(rawEntry);
    if (!entry)
      return funcOp.emitError()
             << ::mlir::ascend::kScheduleKernelMetadataAttr
             << " element " << index << " must be a dictionary attribute";
    auto kernel =
        dyn_cast_or_null<StringAttr>(entry.get(kKernelMetadataKernelKey));
    if (!kernel)
      return funcOp.emitError()
             << ::mlir::ascend::kScheduleKernelMetadataAttr
             << " element " << index
             << " entries must include a string kernel field";
    entries.push_back(entry);
  }
  return entries;
}

DictionaryAttr findScheduleMetadata(ArrayRef<DictionaryAttr> entries,
                                    StringRef kernelId) {
  for (DictionaryAttr entry : entries) {
    auto kernel =
        cast<StringAttr>(entry.get(kKernelMetadataKernelKey));
    if (kernel.getValue() == kernelId)
      return entry;
  }
  return DictionaryAttr();
}

void setFunctionScheduleAttrs(func::FuncOp funcOp,
                              DictionaryAttr scheduleMetadata) {
  Builder builder(funcOp.getContext());
  if (scheduleMetadata)
    funcOp->setAttr(::mlir::ascend::kScheduleKernelMetadataAttr,
                    builder.getArrayAttr({scheduleMetadata}));

  if (!scheduleMetadata)
    return;

  if (Attribute attr =
          scheduleMetadata.get(kKernelMetadataSelectedTileShapeKey))
    funcOp->setAttr(::mlir::ascend::kScheduleSelectedTileShapeAttr,
                    attr);
  if (Attribute attr = scheduleMetadata.get(kKernelMetadataGuardMarkersKey))
    funcOp->setAttr(::mlir::ascend::kScheduleGuardMarkersAttr, attr);
  if (Attribute attr = scheduleMetadata.get(kKernelMetadataTailPoliciesKey))
    funcOp->setAttr(::mlir::ascend::kScheduleTailPoliciesAttr, attr);
  if (Attribute attr = scheduleMetadata.get(kKernelMetadataTailPlanKey))
    funcOp->setAttr(::mlir::ascend::kScheduleTailPlanAttr, attr);
  if (Attribute attr = scheduleMetadata.get(kKernelMetadataTailMarkersKey))
    funcOp->setAttr(::mlir::ascend::kScheduleTailMarkersAttr, attr);
  if (Attribute attr =
          scheduleMetadata.get(kKernelMetadataTargetTilePolicyKey))
    funcOp->setAttr(::mlir::ascend::kScheduleTargetTilePolicyAttr,
                    attr);
}

void includeDependency(Value value, StringRef kernelId, func::FuncOp funcOp,
                       llvm::SetVector<Operation *> &included,
                       llvm::SmallPtrSetImpl<Operation *> &active);

void walkReferencedOperands(Operation *op,
                            llvm::function_ref<void(Value)> callback) {
  for (Value operand : op->getOperands())
    callback(operand);
  op->walk([&](Operation *nested) {
    if (nested == op)
      return;
    for (Value operand : nested->getOperands())
      callback(operand);
  });
}

void includeOp(Operation *op, StringRef kernelId, func::FuncOp funcOp,
               llvm::SetVector<Operation *> &included,
               llvm::SmallPtrSetImpl<Operation *> &active) {
  if (!op || included.contains(op))
    return;
  if (!active.insert(op).second)
    return;

  if (std::optional<std::string> ownerKernel = getKernelId(op);
      ownerKernel && *ownerKernel != kernelId) {
    active.erase(op);
    return;
  }

  walkReferencedOperands(op, [&](Value operand) {
    includeDependency(operand, kernelId, funcOp, included, active);
  });

  included.insert(op);
  active.erase(op);
}

void includeDependency(Value value, StringRef kernelId, func::FuncOp funcOp,
                       llvm::SetVector<Operation *> &included,
                       llvm::SmallPtrSetImpl<Operation *> &active) {
  if (isFuncEntryBlockArgument(value, funcOp))
    return;
  Operation *definingOp = value.getDefiningOp();
  Operation *topLevelDef =
      definingOp ? getTopLevelOpInFunc(definingOp, funcOp) : nullptr;
  if (!topLevelDef)
    return;
  includeOp(topLevelDef, kernelId, funcOp, included, active);
}

bool isExternalUse(Operation *user, func::FuncOp funcOp,
                   const llvm::SetVector<Operation *> &included) {
  Operation *topLevelUser = getTopLevelOpInFunc(user, funcOp);
  if (!topLevelUser)
    return true;
  if (isa<func::ReturnOp>(topLevelUser))
    return true;
  return !included.contains(topLevelUser);
}

void collectArgumentsAndResults(func::FuncOp funcOp, KernelSplitPlan &plan) {
  llvm::MapVector<Value, unsigned> argumentIndex;
  for (Operation *op : plan.ops) {
    walkReferencedOperands(op, [&](Value operand) {
      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        if (!isFuncEntryBlockArgument(blockArg, funcOp))
          return;
      }

      Operation *def = operand.getDefiningOp();
      Operation *topLevelDef =
          def ? getTopLevelOpInFunc(def, funcOp) : nullptr;
      if (isFuncEntryBlockArgument(operand, funcOp) ||
          !topLevelDef || !plan.ops.contains(topLevelDef)) {
        if (!argumentIndex.count(operand)) {
          unsigned index = static_cast<unsigned>(plan.arguments.size());
          argumentIndex.insert({operand, index});
          plan.arguments.push_back(operand);
        }
      }
    });
  }

  llvm::SetVector<Value> results;
  for (Operation *op : plan.ops) {
    std::optional<std::string> kernelId = getKernelId(op);
    if (!kernelId || *kernelId != plan.kernelId)
      continue;
    for (Value result : op->getResults()) {
      for (OpOperand &use : result.getUses()) {
        if (isExternalUse(use.getOwner(), funcOp, plan.ops)) {
          results.insert(result);
          break;
        }
      }
    }
  }
  plan.results.assign(results.begin(), results.end());
}

FailureOr<SmallVector<KernelSplitPlan>>
buildSplitPlans(func::FuncOp funcOp) {
  if (funcOp.getBody().empty())
    return SmallVector<KernelSplitPlan>{};
  if (!llvm::hasSingleElement(funcOp.getBody()))
    return funcOp.emitError()
           << "ascend-kernel-split requires single-block functions";

  FailureOr<SmallVector<DictionaryAttr>> metadata = getScheduleMetadata(funcOp);
  if (failed(metadata))
    return failure();

  llvm::StringMap<unsigned> planIndex;
  SmallVector<KernelSplitPlan> plans;
  unsigned ordinal = 0;
  for (Operation &op : funcOp.getBody().front()) {
    if (isa<func::ReturnOp>(op))
      continue;
    std::optional<std::string> kernelId = getKernelId(&op);
    if (!kernelId) {
      ++ordinal;
      continue;
    }
    auto [it, inserted] = planIndex.try_emplace(*kernelId, plans.size());
    if (inserted) {
      KernelSplitPlan plan;
      plan.kernelId = *kernelId;
      plan.firstOrdinal = ordinal;
      plan.scheduleMetadata = findScheduleMetadata(*metadata, *kernelId);
      plans.push_back(std::move(plan));
    }
    ++ordinal;
  }

  if (plans.size() <= 1)
    return SmallVector<KernelSplitPlan>{};

  for (KernelSplitPlan &plan : plans) {
    llvm::SmallPtrSet<Operation *, 16> active;
    for (Operation &op : funcOp.getBody().front()) {
      std::optional<std::string> kernelId = getKernelId(&op);
      if (kernelId && *kernelId == plan.kernelId)
        includeOp(&op, plan.kernelId, funcOp, plan.ops, active);
    }
    collectArgumentsAndResults(funcOp, plan);
  }

  llvm::sort(plans, [](const KernelSplitPlan &lhs,
                       const KernelSplitPlan &rhs) {
    return lhs.firstOrdinal < rhs.firstOrdinal;
  });
  return plans;
}

LogicalResult verifyNoSymbolConflict(ModuleOp module, func::FuncOp source,
                                     ArrayRef<KernelSplitPlan> plans) {
  for (const KernelSplitPlan &plan : plans) {
    if (auto existing = module.lookupSymbol<func::FuncOp>(plan.kernelId)) {
      if (existing != source)
        return source.emitError()
               << "ascend-kernel-split cannot create function '"
               << plan.kernelId << "' because the symbol already exists";
    }
  }
  return success();
}

func::FuncOp createOutlinedKernel(ModuleOp module, func::FuncOp source,
                                  const KernelSplitPlan &plan) {
  OpBuilder builder(module.getContext());
  builder.setInsertionPoint(source);

  SmallVector<Type> argTypes;
  argTypes.reserve(plan.arguments.size());
  for (Value arg : plan.arguments)
    argTypes.push_back(arg.getType());

  SmallVector<Type> resultTypes;
  resultTypes.reserve(plan.results.size());
  for (Value result : plan.results)
    resultTypes.push_back(result.getType());

  FunctionType funcType =
      builder.getFunctionType(argTypes, resultTypes);
  auto outlined = builder.create<func::FuncOp>(source.getLoc(), plan.kernelId,
                                               funcType);
  if (Attribute normalized =
          source->getAttr(::mlir::ascend::kNormalizedAttr))
    outlined->setAttr(::mlir::ascend::kNormalizedAttr, normalized);
  setFunctionScheduleAttrs(outlined, plan.scheduleMetadata);

  Block *entry = outlined.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  IRMapping mapper;
  for (auto [index, sourceArg] : llvm::enumerate(plan.arguments))
    mapper.map(sourceArg, entry->getArgument(index));

  for (Operation *op : plan.ops)
    builder.clone(*op, mapper);

  SmallVector<Value> mappedResults;
  mappedResults.reserve(plan.results.size());
  for (Value result : plan.results)
    mappedResults.push_back(mapper.lookup(result));
  builder.create<func::ReturnOp>(source.getLoc(), mappedResults);

  return outlined;
}

LogicalResult splitFunction(ModuleOp module, func::FuncOp funcOp) {
  FailureOr<SmallVector<KernelSplitPlan>> plans = buildSplitPlans(funcOp);
  if (failed(plans))
    return failure();
  if (plans->empty())
    return success();

  if (failed(verifyNoSymbolConflict(module, funcOp, *plans)))
    return failure();

  for (const KernelSplitPlan &plan : *plans)
    createOutlinedKernel(module, funcOp, plan);
  funcOp.erase();
  return success();
}

} // namespace

namespace mlir::ascend {

struct AscendKernelSplitPass
    : public ::impl::AscendKernelSplitPassBase<AscendKernelSplitPass> {
  using AscendKernelSplitPassBase::AscendKernelSplitPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    for (func::FuncOp funcOp : functions) {
      if (funcOp->getParentOp() != module)
        continue;
      if (failed(splitFunction(module, funcOp))) {
        signalPassFailure();
        return;
      }
    }
  }
};

std::unique_ptr<::mlir::Pass> createAscendKernelSplitPass() {
  return std::make_unique<AscendKernelSplitPass>();
}

} // namespace mlir::ascend
