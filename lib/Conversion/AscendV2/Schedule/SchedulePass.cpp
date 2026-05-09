//===- SchedulePass.cpp - Ascend V2 schedule pass -------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Schedule/SchedulePass.h"

#include "Conversion/AscendV2/Debug/DebugOptions.h"
#include "Conversion/AscendV2/Schedule/AxisCoalescer.h"
#include "Conversion/AscendV2/Schedule/KernelPatternView.h"
#include "Conversion/AscendV2/Schedule/ScheduleCache.h"
#include "Conversion/AscendV2/Schedule/ScheduleDecision.h"
#include "Conversion/AscendV2/Schedule/ScheduleProblemBuilder.h"
#include "Conversion/AscendV2/Schedule/ScheduleSearch.h"
#include "Conversion/AscendV2/Schedule/ScheduleTypes.h"
#include "Conversion/AscendV2/Schedule/StructuredLoweringDriver.h"
#include "Conversion/AscendV2/Schedule/TemplateRegistry.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>
#include <vector>

#define GEN_PASS_DECL_ASCENDSCHEDULEPASS
#define GEN_PASS_DEF_ASCENDSCHEDULEPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::afir::ascend::v2::schedule;

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
    ::mlir::ascend::v2::DebugOptions options{
        ::mlir::ascend::v2::parseDebugStage(debugStage), dumpReport};
    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Schedule))
      ::mlir::ascend::v2::emitStageHeader(
          llvm::errs(), ::mlir::ascend::v2::DebugStage::Schedule,
          getArgument());

    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    ScheduleSearchOptions searchOptions;
    searchOptions.runtimeTopK = runtimeTopK;
    ScheduleCacheModel scheduleCacheModel;
    std::vector<ScheduleDebugEntry> scheduleDebugEntries;
    SmallVector<ScheduleReportEntry> reportEntries;
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
          scheduleProblem->kernelId, searchResult.keptInstances,
          searchOptions);
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

    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Schedule)) {
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
