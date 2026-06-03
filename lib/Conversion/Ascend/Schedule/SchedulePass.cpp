//===- SchedulePass.cpp - Ascend schedule pass -------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Schedule/SchedulePass.h"

#include "Conversion/Ascend/Debug/DebugOptions.h"
#include "AxisCoalescer.h"
#include "KernelPatternView.h"
#include "ScheduleCache.h"
#include "ScheduleDecision.h"
#include "SchedulePersistentCacheIO.h"
#include "ScheduleProblemBuilder.h"
#include "ScheduleSearch.h"
#include "ScheduleTuningDB.h"
#include "ScheduleTypes.h"
#include "ScheduleContractDriver.h"
#include "TemplateRegistry.h"
#include "Target/Ascend/CannTargetProfileLoader.h"
#include "Target/Ascend/TargetCostModel.h"
#include "Target/Ascend/TargetIntrinsicModel.h"
#include "Target/Ascend/TargetMemoryModel.h"
#include "Target/Ascend/TargetModelVerifier.h"
#include "Target/Ascend/TargetProfile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#define GEN_PASS_DECL_ASCENDSCHEDULEPASS
#define GEN_PASS_DEF_ASCENDSCHEDULEPASS
#include "Conversion/Ascend/Passes.h.inc"

using namespace mlir;
using namespace mlir::ascend::schedule;

namespace {

struct ScheduleReportEntry {
  OpRole opRole = OpRole::Unknown;
  unsigned resultRank = 0;
  SmallVector<int64_t> staticShape;
  std::string scheduleFamily;
  std::string scheduleTemplate;
  std::string decisionId;
};

struct ScheduleDebugEntry {
  ScheduleProblem problem;
  SmallVector<const ScheduleTemplateImplementation *> templateMatches;
  ScheduleSearchResult searchResult;
  ScheduleDecisionSet decisionSet;
  ScheduleContractReport scheduleContractReport;
};

constexpr llvm::StringLiteral kLegacyDefaultTilePolicyMode = "legacy-default";
constexpr llvm::StringLiteral kTargetAwareTilePolicyMode = "target-aware";
constexpr llvm::StringLiteral kRequireExplicitTilePolicyMode =
    "require-explicit";
constexpr llvm::StringLiteral kSchedulePersistentTuningCacheAttr =
    "ascend.schedule.tuning_cache";

struct ScheduleTargetModelContext {
  ::mlir::ascend::TargetProfile profile;
  ::mlir::ascend::TargetMemoryModel memoryModel;
  ::mlir::ascend::TargetIntrinsicModel intrinsicModel;
  ::mlir::ascend::TargetCostModel costModel;
};

static bool isSupportedTargetTilePolicyMode(StringRef mode) {
  return mode == kRequireExplicitTilePolicyMode ||
         mode == kLegacyDefaultTilePolicyMode ||
         mode == kTargetAwareTilePolicyMode;
}

static void clearOwnedScheduleAttrs(Operation *op) {
  op->removeAttr(kScheduleFamilyAttr);
  op->removeAttr(kScheduleTemplateAttr);
  op->removeAttr(kScheduleDecisionIdAttr);
  op->removeAttr(kScheduleRuntimeTopKAttr);
  op->removeAttr(kScheduleContractAttr);
  op->removeAttr(kScheduleTileBindingAttr);
  op->removeAttr(kScheduleTileParamsAttr);
  op->removeAttr(kScheduleGuardMarkersAttr);
  op->removeAttr(kScheduleTailPoliciesAttr);
  op->removeAttr(kScheduleTailPlanAttr);
  op->removeAttr(kScheduleTailMarkersAttr);
  op->removeAttr(kScheduleTargetTilePolicyAttr);
  op->removeAttr(kScheduleKernelMetadataAttr);
}

static void clearOwnedScheduleAttrs(ModuleOp module) {
  module.walk([](Operation *op) { clearOwnedScheduleAttrs(op); });
}

static SmallVector<std::string, 8>
loadPersistentTuningCache(ModuleOp module) {
  SmallVector<std::string, 8> signatures;
  auto cacheAttr =
      module->getAttrOfType<ArrayAttr>(kSchedulePersistentTuningCacheAttr);
  if (!cacheAttr)
    return signatures;

  for (Attribute attr : cacheAttr) {
    auto signature = dyn_cast<StringAttr>(attr);
    if (signature)
      signatures.push_back(signature.getValue().str());
  }
  return signatures;
}

static void storePersistentTuningCache(ModuleOp module,
                                       ScheduleCacheModel &cacheModel) {
  MLIRContext *context = module.getContext();
  SmallVector<Attribute, 8> attrs;
  for (const std::string &signature :
       cacheModel.getPersistentTuningSignatures())
    attrs.push_back(StringAttr::get(context, signature));
  module->setAttr(kSchedulePersistentTuningCacheAttr,
                  ArrayAttr::get(context, attrs));
}

static FailureOr<ScheduleTargetModelContext>
buildScheduleTargetModelContext(ModuleOp module, StringRef cannRoot,
                                StringRef soc) {
  FailureOr<::mlir::ascend::TargetProfile> profile =
      ::mlir::ascend::CannTargetProfileLoader::load(cannRoot, soc);
  if (failed(profile))
    return module.emitError()
           << "ascend-schedule target-aware tiling failed to load target "
              "profile";

  FailureOr<::mlir::ascend::TargetMemoryModel> memoryModel =
      ::mlir::ascend::TargetMemoryModelBuilder().build(*profile,
                                                       llvm::errs());
  if (failed(memoryModel))
    return module.emitError()
           << "ascend-schedule target-aware tiling failed to build target "
              "memory model";

  FailureOr<::mlir::ascend::TargetIntrinsicModel> intrinsicModel =
      ::mlir::ascend::TargetIntrinsicModelBuilder().build(*profile);
  if (failed(intrinsicModel))
    return module.emitError()
           << "ascend-schedule target-aware tiling failed to build target "
              "intrinsic model";

  FailureOr<::mlir::ascend::TargetCostModel> costModel =
      ::mlir::ascend::TargetCostModelBuilder().build(*profile, *memoryModel,
                                                     llvm::errs());
  if (failed(costModel))
    return module.emitError()
           << "ascend-schedule target-aware tiling failed to build target "
              "cost model";

  if (failed(::mlir::ascend::TargetModelVerifier().verify(
          *profile, *memoryModel, *intrinsicModel, *costModel,
          llvm::errs())))
    return module.emitError()
           << "ascend-schedule target-aware tiling failed target model "
              "verification";

  ScheduleTargetModelContext context;
  context.profile = std::move(*profile);
  context.memoryModel = std::move(*memoryModel);
  context.intrinsicModel = std::move(*intrinsicModel);
  context.costModel = std::move(*costModel);
  return context;
}

static FailureOr<int64_t>
getAvailableCapacity(const ::mlir::ascend::TargetMemoryModel &memoryModel,
                     ::mlir::ascend::MemoryPlace place) {
  FailureOr<::mlir::ascend::CapacityRule> capacity =
      memoryModel.getCapacity(place);
  if (failed(capacity))
    return failure();
  return capacity->availableCapacityBytes > 0
             ? capacity->availableCapacityBytes
             : capacity->staticCapacityBytes;
}

static bool hasVectorComputeIntrinsic(
    const ::mlir::ascend::TargetIntrinsicModel &intrinsicModel,
    OpRole dominantRole) {
  if (dominantRole == OpRole::Reduction)
    return !intrinsicModel
                .getIntrinsicsForComputeKind(
                    ::mlir::ascend::ComputeKind::VectorReduce)
                .empty();
  if (dominantRole == OpRole::Vector)
    return !intrinsicModel
                .getIntrinsicsForUnit(::mlir::ascend::ExecutionUnit::Vector)
                .empty();
  if (dominantRole == OpRole::Cube)
    return !intrinsicModel
                .getIntrinsicsForComputeKind(
                    ::mlir::ascend::ComputeKind::Matmul)
                .empty();
  return true;
}

static bool hasPrimitiveUse(ArrayRef<PrimitiveAxisUseKind> primitiveUses,
                            PrimitiveAxisUseKind use) {
  return llvm::is_contained(primitiveUses, use);
}

static bool hasGatherAxis(const ScheduleProblem &problem) {
  return llvm::any_of(
      problem.axes.axisScheduleConstraints,
      [](const AxisScheduleConstraint &constraint) {
        return hasPrimitiveUse(constraint.primitiveUses,
                               PrimitiveAxisUseKind::GatherIndex);
      });
}

static FailureOr<int64_t> deriveSemanticAlignmentGranularity(
    const ScheduleProblem &problem,
    const ::mlir::ascend::TargetMemoryModel &memoryModel) {
  if (!hasGatherAxis(problem))
    return static_cast<int64_t>(0);
  if (problem.resultElementBitWidth == 0)
    return failure();

  FailureOr<::mlir::ascend::AlignmentRule> alignment =
      memoryModel.getAlignment(::mlir::ascend::MemoryPlace::VECIN);
  if (failed(alignment))
    return failure();

  int64_t elementBytes =
      std::max<int64_t>(1, (problem.resultElementBitWidth + 7) / 8);
  int64_t addressAlignedElements = 1;
  if (alignment->addressAlignmentBytes > 0) {
    addressAlignedElements =
        (alignment->addressAlignmentBytes + elementBytes - 1) / elementBytes;
  }
  int64_t tileAlignedElements =
      std::max<int64_t>(1, alignment->tileAlignmentElements);
  return std::max({static_cast<int64_t>(1), addressAlignedElements,
                   tileAlignedElements});
}

static void setDynamicTargetTileFallback(TargetTilePolicy &policy,
                                         StringRef reason) {
  policy.defaultParallelTile =
      std::max<int64_t>(1, policy.defaultParallelTile);
  policy.policyId =
      (llvm::Twine("target_") + reason + "_" +
       llvm::Twine(policy.defaultParallelTile))
          .str();
}

static std::optional<int64_t>
getStaticReductionElementSpan(const ScheduleProblem &problem) {
  int64_t span = 1;
  for (const LogicalAxisInfo &axis : problem.axes.logicalAxes) {
    if (axis.kind != AxisKind::Reduction)
      continue;
    if (ShapedType::isDynamic(axis.staticExtent) || axis.staticExtent <= 0)
      return std::nullopt;
    if (span > std::numeric_limits<int64_t>::max() / axis.staticExtent)
      return std::nullopt;
    span *= axis.staticExtent;
  }
  return span;
}

static void applySemanticAlignmentGranularity(ScheduleProblem &problem) {
  for (AxisScheduleConstraint &constraint :
       problem.axes.axisScheduleConstraints) {
    if (!hasPrimitiveUse(constraint.primitiveUses,
                         PrimitiveAxisUseKind::GatherIndex))
      continue;
    constraint.semanticAlignmentGranularity =
        problem.targetTilePolicy.semanticAlignmentGranularity;
  }
}

static FailureOr<TargetTilePolicy>
deriveTargetTilePolicy(const ScheduleProblem &problem,
                       const ScheduleTargetModelContext &targetContext) {
  TargetTilePolicy policy;
  FailureOr<int64_t> semanticAlignmentGranularity =
      deriveSemanticAlignmentGranularity(problem, targetContext.memoryModel);
  if (failed(semanticAlignmentGranularity))
    return failure();
  if (*semanticAlignmentGranularity > 0)
    policy.semanticAlignmentGranularity = *semanticAlignmentGranularity;

  if (!hasVectorComputeIntrinsic(targetContext.intrinsicModel,
                                 problem.dominantRole))
    return failure();

  FailureOr<int64_t> vecInCapacity = getAvailableCapacity(
      targetContext.memoryModel, ::mlir::ascend::MemoryPlace::VECIN);
  FailureOr<int64_t> vecOutCapacity = getAvailableCapacity(
      targetContext.memoryModel, ::mlir::ascend::MemoryPlace::VECOUT);
  FailureOr<int64_t> vecCalcCapacity = getAvailableCapacity(
      targetContext.memoryModel, ::mlir::ascend::MemoryPlace::VECCALC);
  if (failed(vecInCapacity) || failed(vecOutCapacity) ||
      failed(vecCalcCapacity))
    return failure();

  auto gmToVecIn = targetContext.memoryModel.findDirectPaths(
      ::mlir::ascend::MemoryPlace::GM, ::mlir::ascend::MemoryPlace::VECIN);
  auto vecOutToGm = targetContext.memoryModel.findDirectPaths(
      ::mlir::ascend::MemoryPlace::VECOUT, ::mlir::ascend::MemoryPlace::GM);
  if (gmToVecIn.empty() || vecOutToGm.empty() ||
      failed(targetContext.costModel.getPathCost(gmToVecIn.front())) ||
      failed(targetContext.costModel.getPathCost(vecOutToGm.front())))
    return failure();

  if (problem.resultElementBitWidth == 0)
    return failure();

  int64_t capacityBytes =
      std::min({*vecInCapacity, *vecOutCapacity, *vecCalcCapacity});
  int64_t elementBytes =
      std::max<int64_t>(1, (problem.resultElementBitWidth + 7) / 8);
  int64_t vectorBufferCount =
      std::max<int64_t>(1, policy.vectorBufferCount);

  int64_t parallelElementSpan = 1;
  if (problem.resultShape.size() >= 2) {
    int64_t innerExtent = problem.resultShape.back();
    if (ShapedType::isDynamic(innerExtent) || innerExtent <= 0) {
      setDynamicTargetTileFallback(policy, "dynamic_inner");
      return policy;
    }
    parallelElementSpan = innerExtent;
  }
  if (problem.dominantRole == OpRole::Reduction) {
    std::optional<int64_t> reductionElementSpan =
        getStaticReductionElementSpan(problem);
    if (!reductionElementSpan) {
      setDynamicTargetTileFallback(policy, "dynamic_reduction");
      return policy;
    }
    if (parallelElementSpan >
        std::numeric_limits<int64_t>::max() / *reductionElementSpan)
      return failure();
    parallelElementSpan *= *reductionElementSpan;
  }

  int64_t rowBytes = parallelElementSpan * elementBytes;
  if (rowBytes <= 0)
    return failure();

  int64_t derivedTile = capacityBytes / vectorBufferCount / rowBytes;
  if (derivedTile <= 0)
    derivedTile = 1;
  if (!problem.resultShape.empty() &&
      !ShapedType::isDynamic(problem.resultShape.front()))
    derivedTile = std::min(derivedTile, problem.resultShape.front());

  policy.defaultParallelTile = std::max<int64_t>(1, derivedTile);
  policy.policyId =
      (llvm::Twine("target_ub_") + llvm::Twine(policy.defaultParallelTile))
          .str();
  return policy;
}

void emitScheduleReport(ArrayRef<ScheduleReportEntry> entries,
                        llvm::raw_ostream &os) {
  os << "Schedule report\n";
  for (const ScheduleReportEntry &entry : entries) {
    os << "  op_role = \"" << stringifyOpRole(entry.opRole) << "\"\n";
    os << "  result_rank = " << entry.resultRank << "\n";
    if (!entry.staticShape.empty()) {
      os << "  static_shape = [";
      llvm::interleaveComma(entry.staticShape, os);
      os << "]\n";
    }
    os << "  schedule_family = \"" << entry.scheduleFamily << "\"\n";
    os << "  schedule_template = \"" << entry.scheduleTemplate << "\"\n";
    os << "  schedule_decision_id = \"" << entry.decisionId << "\"\n";
  }
}

} // namespace

namespace mlir::ascend {

struct AscendSchedulePass
    : public ::impl::AscendSchedulePassBase<AscendSchedulePass> {
  using AscendSchedulePassBase::AscendSchedulePassBase;

  void runOnOperation() override {
    if (!isSupportedTargetTilePolicyMode(targetTilePolicy)) {
      getOperation()->emitError()
          << "unsupported ascend-schedule target-tile-policy \""
          << targetTilePolicy << "\"";
      signalPassFailure();
      return;
    }
    if (StringRef(targetTilePolicy) == kRequireExplicitTilePolicyMode) {
      getOperation()->emitError()
          << "ascend-schedule requires explicit target-tile-policy: use "
             "target-tile-policy=legacy-default for compatibility or "
             "target-tile-policy=target-aware with cann-root and soc";
      signalPassFailure();
      return;
    }

    ::mlir::ascend::debug::DebugOptions options{
        ::mlir::ascend::debug::parseDebugStage(debugStage), dumpReport,
        StringRef(debugDumpDir).str()};
    if (::mlir::ascend::debug::shouldDump(
            options, ::mlir::ascend::debug::DebugStage::Schedule))
      ::mlir::ascend::debug::emitStageHeader(
          llvm::errs(), ::mlir::ascend::debug::DebugStage::Schedule,
          getArgument());

    ModuleOp module = getOperation();
    clearOwnedScheduleAttrs(module);
    if (failed(::mlir::ascend::debug::dumpCheckpoint(
            module, options, ::mlir::ascend::debug::DebugStage::Schedule,
            "031-schedule-cleared"))) {
      signalPassFailure();
      return;
    }

    MLIRContext *context = module.getContext();
    ScheduleSearchOptions searchOptions;
    searchOptions.runtimeTopK = runtimeTopK;
    searchOptions.maxAxisProductTileShapes = maxSearchBudget;
    ScheduleCacheModel scheduleCacheModel;
    SmallVector<std::string, 8> seededSignatures =
        loadPersistentTuningCache(module);
    if (!StringRef(tuningCacheIn).empty()) {
      FailureOr<SmallVector<std::string, 8>> fileSignatures =
          loadPersistentTuningCacheFile(tuningCacheIn);
      if (failed(fileSignatures)) {
        module.emitError()
            << "failed to read ascend schedule tuning cache file: "
            << tuningCacheIn;
        signalPassFailure();
        return;
      }
      seededSignatures.append(fileSignatures->begin(), fileSignatures->end());
    }
    ScheduleTuningDatabase tuningDb;
    std::string tuningTarget = StringRef(soc).str();
    std::string tuningPolicy = StringRef(targetTilePolicy).str();
    if (!StringRef(tuningDbIn).empty()) {
      FailureOr<ScheduleTuningDatabase> loadedDb =
          loadScheduleTuningDBFile(tuningDbIn);
      if (failed(loadedDb)) {
        module.emitError()
            << "failed to read ascend schedule tuning database file: "
            << tuningDbIn;
        signalPassFailure();
        return;
      }
      tuningDb = std::move(*loadedDb);
      SmallVector<std::string, 8> dbSignatures =
          collectMatchingTuningSignatures(tuningDb, tuningTarget,
                                          tuningPolicy);
      seededSignatures.append(dbSignatures.begin(), dbSignatures.end());
    }
    scheduleCacheModel.seedPersistentTuningSignatures(seededSignatures);
    std::vector<ScheduleDebugEntry> scheduleDebugEntries;
    SmallVector<ScheduleReportEntry> reportEntries;
    std::optional<ScheduleTargetModelContext> targetModelContext;
    if (StringRef(targetTilePolicy) == kTargetAwareTilePolicyMode) {
      FailureOr<ScheduleTargetModelContext> builtContext =
          buildScheduleTargetModelContext(module, cannRoot, soc);
      if (failed(builtContext)) {
        signalPassFailure();
        return;
      }
      targetModelContext = std::move(*builtContext);
    }

    FailureOr<SmallVector<KernelPatternView>> patternViews =
        buildKernelPatternViews(module);
    if (failed(patternViews)) {
      signalPassFailure();
      return;
    }

    for (const KernelPatternView &pattern : *patternViews) {
      FailureOr<CoalescedAxisInfo> axisInfo = coalesceAxes(pattern);
      if (failed(axisInfo)) {
        signalPassFailure();
        return;
      }

      CoalescedAxisInfo axes = std::move(*axisInfo);
      FailureOr<ScheduleProblem> scheduleProblem =
          buildScheduleProblem(pattern, axes);
      if (failed(scheduleProblem)) {
        signalPassFailure();
        return;
      }
      if (targetModelContext) {
        FailureOr<TargetTilePolicy> policy =
            deriveTargetTilePolicy(*scheduleProblem, *targetModelContext);
        if (failed(policy)) {
          module.emitError()
              << "ascend-schedule target-aware tiling failed to derive target "
                 "tile policy";
          signalPassFailure();
          return;
        }
        scheduleProblem->targetTilePolicy = std::move(*policy);
      }
      applySemanticAlignmentGranularity(*scheduleProblem);

      SmallVector<const ScheduleTemplateImplementation *> templateMatches =
          matchScheduleTemplateImplementations(*scheduleProblem);
      if (templateMatches.empty()) {
        scheduleCacheModel.recordNegativeCacheEntry();
        if (const PatternOpView *primaryOpView =
                selectDominantPrimaryOp(pattern)) {
          primaryOpView->op->emitError()
              << "no schedule template for kernel "
              << scheduleProblem->kernelId << " role "
              << stringifyOpRole(scheduleProblem->dominantRole) << " rank "
              << scheduleProblem->resultRank;
        }
        signalPassFailure();
        return;
      }
      ScheduleSearchResult searchResult = searchScheduleInstancesWithStats(
          *scheduleProblem, templateMatches, searchOptions);
      scheduleCacheModel.recordGuardBudgetPruned(
          searchResult.prunedByGuardBudget);
      if (searchResult.keptInstances.empty()) {
        if (const PatternOpView *primaryOpView =
                selectDominantPrimaryOp(pattern)) {
          primaryOpView->op->emitError()
              << "no schedule instance for kernel "
              << scheduleProblem->kernelId << " role "
              << stringifyOpRole(scheduleProblem->dominantRole) << " rank "
              << scheduleProblem->resultRank;
        }
        signalPassFailure();
        return;
      }
      ScheduleDecisionSet decisionSet = buildScheduleDecisionSet(
          *scheduleProblem, searchResult.keptInstances, searchOptions);
      scheduleCacheModel.recordScheduleDecisionSet(*scheduleProblem,
                                                   decisionSet);
      const ScheduleDecision &selectedDecision = decisionSet.decisions.front();
      const ScheduleInstance &selectedInstance = selectedDecision.instance;

      StringAttr scheduleFamilyAttr =
          StringAttr::get(context, selectedInstance.tmpl.family);
      StringAttr scheduleTemplateAttr =
          StringAttr::get(context, selectedInstance.tmpl.name);
      StringAttr scheduleDecisionIdAttr =
          StringAttr::get(context, selectedDecision.decisionId);
      IntegerAttr runtimeTopKAttr = IntegerAttr::get(
          IntegerType::get(context, 64), decisionSet.runtimeTopK);

      for (const PatternOpView &opView : pattern.ops) {
        Operation *op = opView.op;
        op->setAttr(kScheduleFamilyAttr, scheduleFamilyAttr);
        op->setAttr(kScheduleTemplateAttr, scheduleTemplateAttr);
        op->setAttr(kScheduleDecisionIdAttr, scheduleDecisionIdAttr);
        op->setAttr(kScheduleRuntimeTopKAttr, runtimeTopKAttr);
      }

      ScheduleContractReport scheduleContractReport;
      if (failed(applyScheduleContractMarkers(pattern, *scheduleProblem,
                                              decisionSet,
                                              scheduleContractReport))) {
        signalPassFailure();
        return;
      }

      reportEntries.push_back(ScheduleReportEntry{
          scheduleProblem->dominantRole, scheduleProblem->resultRank,
          scheduleProblem->resultShape, selectedInstance.tmpl.family,
          selectedInstance.tmpl.name, selectedDecision.decisionId});
      scheduleDebugEntries.push_back(ScheduleDebugEntry{
          std::move(*scheduleProblem), std::move(templateMatches),
          std::move(searchResult), std::move(decisionSet),
          std::move(scheduleContractReport)});
    }
    if (failed(::mlir::ascend::debug::dumpCheckpoint(
            module, options, ::mlir::ascend::debug::DebugStage::Schedule,
            "032-schedule-decisions"))) {
      signalPassFailure();
      return;
    }

    if (::mlir::ascend::debug::shouldDump(
            options, ::mlir::ascend::debug::DebugStage::Schedule)) {
      printKernelPatternViews(*patternViews, llvm::errs());
      for (const ScheduleDebugEntry &entry : scheduleDebugEntries) {
        const ScheduleProblem &problem = entry.problem;
        printAxisCoalescingReport(problem.kernelId, problem.axes, llvm::errs());
        printScheduleProblemReport(problem, llvm::errs());
        printTemplateRegistryReport(problem.kernelId, entry.templateMatches,
                                    llvm::errs());
        const ScheduleSearchResult &searchResult = entry.searchResult;
        printScheduleSearchReport(problem.kernelId,
                                  searchResult.generatedCount, searchOptions,
                                  searchResult.keptInstances, llvm::errs());
        printScheduleGuardsReport(problem, searchResult, llvm::errs());
        printScheduleDecisionSetReport(entry.decisionSet, llvm::errs());
        printScheduleContractReport(entry.scheduleContractReport, llvm::errs());
      }
      printScheduleCacheReport(scheduleCacheModel, llvm::errs());
      emitScheduleReport(reportEntries, llvm::errs());
    }
    storePersistentTuningCache(module, scheduleCacheModel);
    if (!StringRef(tuningCacheOut).empty() &&
        failed(writePersistentTuningCacheFile(
            tuningCacheOut,
            scheduleCacheModel.getPersistentTuningSignatures()))) {
      module.emitError() << "failed to write ascend schedule tuning cache file: "
                         << tuningCacheOut;
      signalPassFailure();
      return;
    }
    if (!StringRef(tuningDbOut).empty()) {
      if (failed(appendTuningResultRecords(tuningDb, tuningTarget, tuningPolicy,
                                           scheduleCacheModel.getTuningResultKeys()))) {
        module.emitError() << "invalid ascend schedule tuning database fields"
                           << " (target='" << tuningTarget
                           << "', policy='" << tuningPolicy << "')";
        signalPassFailure();
        return;
      }
      if (failed(writeScheduleTuningDBFile(tuningDbOut, tuningDb))) {
        module.emitError()
            << "failed to write ascend schedule tuning database file: "
            << tuningDbOut;
        signalPassFailure();
        return;
      }
    }
    if (failed(::mlir::ascend::debug::dumpCheckpoint(
            module, options, ::mlir::ascend::debug::DebugStage::Schedule,
            "033-schedule-final"))) {
      signalPassFailure();
      return;
    }
  }
};

std::unique_ptr<Pass> createAscendSchedulePass() {
  return std::make_unique<AscendSchedulePass>();
}

} // namespace mlir::ascend
