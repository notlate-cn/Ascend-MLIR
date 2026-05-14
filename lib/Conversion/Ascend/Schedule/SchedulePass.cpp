//===- SchedulePass.cpp - Ascend schedule pass -------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Schedule/SchedulePass.h"

#include "Conversion/Ascend/Debug/DebugOptions.h"
#include "Conversion/Ascend/Schedule/AxisCoalescer.h"
#include "Conversion/Ascend/Schedule/KernelPatternView.h"
#include "Conversion/Ascend/Schedule/ScheduleCache.h"
#include "Conversion/Ascend/Schedule/ScheduleDecision.h"
#include "Conversion/Ascend/Schedule/ScheduleProblemBuilder.h"
#include "Conversion/Ascend/Schedule/ScheduleSearch.h"
#include "Conversion/Ascend/Schedule/ScheduleTypes.h"
#include "Conversion/Ascend/Schedule/StructuredLoweringDriver.h"
#include "Conversion/Ascend/Schedule/TemplateRegistry.h"
#include "Target/Ascend/CannTargetProfileLoader.h"
#include "Target/Ascend/TargetCostModel.h"
#include "Target/Ascend/TargetIntrinsicModel.h"
#include "Target/Ascend/TargetMemoryModel.h"
#include "Target/Ascend/TargetModelVerifier.h"
#include "Target/Ascend/TargetProfile.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#define GEN_PASS_DECL_ASCENDSCHEDULEPASS
#define GEN_PASS_DEF_ASCENDSCHEDULEPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::afir::ascend::schedule;

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
  SmallVector<ScheduleTemplate> templateMatches;
  ScheduleSearchResult searchResult;
  ScheduleDecisionSet decisionSet;
  StructuredLoweringReport structuredLoweringReport;
};

constexpr llvm::StringLiteral kLegacyDefaultTilePolicyMode = "legacy-default";
constexpr llvm::StringLiteral kTargetAwareTilePolicyMode = "target-aware";

struct ScheduleTargetModelContext {
  ::mlir::ascend::TargetProfile profile;
  ::mlir::ascend::TargetMemoryModel memoryModel;
  ::mlir::ascend::TargetIntrinsicModel intrinsicModel;
  ::mlir::ascend::TargetCostModel costModel;
};

static bool isSupportedTargetTilePolicyMode(StringRef mode) {
  return mode == kLegacyDefaultTilePolicyMode ||
         mode == kTargetAwareTilePolicyMode;
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

static FailureOr<TargetTilePolicy>
deriveTargetTilePolicy(const ScheduleProblem &problem,
                       const ScheduleTargetModelContext &targetContext) {
  TargetTilePolicy policy;
  if (!hasVectorComputeIntrinsic(targetContext.intrinsicModel,
                                 problem.dominantRole))
    return failure();

  if (problem.resultShape.size() < 2 || problem.resultElementBitWidth == 0)
    return policy;

  int64_t innerExtent = problem.resultShape.back();
  if (ShapedType::isDynamic(innerExtent) || innerExtent <= 0)
    return policy;

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

  int64_t capacityBytes =
      std::min({*vecInCapacity, *vecOutCapacity, *vecCalcCapacity});
  int64_t elementBytes =
      std::max<int64_t>(1, (problem.resultElementBitWidth + 7) / 8);
  int64_t rowBytes = innerExtent * elementBytes;
  if (rowBytes <= 0)
    return policy;

  constexpr int64_t kConservativeVectorBufferCount = 4;
  int64_t derivedTile = capacityBytes / kConservativeVectorBufferCount /
                        rowBytes;
  if (derivedTile <= 0)
    derivedTile = 1;
  if (!ShapedType::isDynamic(problem.resultShape.front()))
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

namespace mlir::afir {

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

    ::mlir::afir::ascend::debug::DebugOptions options{
        ::mlir::afir::ascend::debug::parseDebugStage(debugStage), dumpReport};
    if (::mlir::afir::ascend::debug::shouldDump(
            options, ::mlir::afir::ascend::debug::DebugStage::Schedule))
      ::mlir::afir::ascend::debug::emitStageHeader(
          llvm::errs(), ::mlir::afir::ascend::debug::DebugStage::Schedule,
          getArgument());

    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    ScheduleSearchOptions searchOptions;
    searchOptions.runtimeTopK = runtimeTopK;
    ScheduleCacheModel scheduleCacheModel;
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

      SmallVector<ScheduleTemplate> templateMatches =
          matchScheduleTemplates(*scheduleProblem);
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

      StructuredLoweringReport structuredLoweringReport;
      if (failed(applyStructuredLoweringMarkers(pattern, *scheduleProblem,
                                                decisionSet,
                                                structuredLoweringReport))) {
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
          std::move(structuredLoweringReport)});
    }

    if (::mlir::afir::ascend::debug::shouldDump(
            options, ::mlir::afir::ascend::debug::DebugStage::Schedule)) {
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
        printStructuredLoweringReport(entry.structuredLoweringReport,
                                      llvm::errs());
      }
      printScheduleCacheReport(scheduleCacheModel, llvm::errs());
      emitScheduleReport(reportEntries, llvm::errs());
    }
  }
};

std::unique_ptr<Pass> createAscendSchedulePass() {
  return std::make_unique<AscendSchedulePass>();
}

} // namespace mlir::afir
