//===- SchedulePass.cpp - Ascend V2 schedule pass -------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Schedule/SchedulePass.h"

#include "Conversion/AscendV2/Debug/DebugOptions.h"
#include "Conversion/AscendV2/Schedule/KernelPatternView.h"
#include "Conversion/AscendV2/Schedule/ScheduleTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>

#define GEN_PASS_DECL_ASCENDSCHEDULEPASS
#define GEN_PASS_DEF_ASCENDSCHEDULEPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::afir::ascend::v2::schedule;

namespace {

constexpr llvm::StringLiteral kSingleTilePerBlock = "single_tile_per_block";

struct ScheduleProblem {
  OpRole opRole = OpRole::Unknown;
  std::optional<int64_t> resultRank;
  SmallVector<int64_t> staticShape;
  SmallVector<std::string> iteratorTypes;
};

struct ScheduleReportEntry {
  ScheduleProblem problem;
  std::string scheduleFamily;
  std::string scheduleTemplate;
  std::string decisionId;
};

std::string stringifyAttr(Attribute attr) {
  if (auto stringAttr = dyn_cast<StringAttr>(attr))
    return stringAttr.getValue().str();

  SmallString<32> storage;
  llvm::raw_svector_ostream os(storage);
  attr.print(os);
  return std::string(storage.str());
}

std::string stringifyStructuredIteratorType(utils::IteratorType iteratorType) {
  if (iteratorType == utils::IteratorType::parallel)
    return "parallel";
  if (iteratorType == utils::IteratorType::reduction)
    return "reduction";
  return "unknown";
}

ScheduleProblem extractScheduleProblem(Operation *op, OpRole opRole) {
  ScheduleProblem problem;
  problem.opRole = opRole;

  if (op->getNumResults() != 0) {
    if (auto resultType =
            dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
      problem.resultRank = resultType.getRank();
      for (int64_t dim : resultType.getShape())
        problem.staticShape.push_back(dim);
    }
  }

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    for (utils::IteratorType iteratorType : linalgOp.getIteratorTypesArray())
      problem.iteratorTypes.push_back(
          stringifyStructuredIteratorType(iteratorType));
    return problem;
  }

  if (auto iteratorTypes = op->getAttrOfType<ArrayAttr>("iterator_types")) {
    for (Attribute iteratorType : iteratorTypes)
      problem.iteratorTypes.push_back(stringifyAttr(iteratorType));
  }

  return problem;
}

std::optional<StringRef> chooseScheduleFamily(const ScheduleProblem &problem) {
  if (problem.opRole == OpRole::Reduction)
    return StringRef("reduction_static");
  if (problem.opRole == OpRole::Cube)
    return StringRef("cube_static_matmul");
  if (problem.opRole != OpRole::Vector)
    return std::nullopt;

  if (problem.resultRank == 1)
    return StringRef("vector_static_1d");
  if (problem.resultRank == 2)
    return StringRef("vector_static_2d");

  return std::nullopt;
}

const PatternOpView *getFirstPrimaryOpView(const KernelPatternView &pattern) {
  for (const PatternOpView &opView : pattern.ops) {
    if (opView.primary)
      return &opView;
  }
  return nullptr;
}

void emitScheduleReport(ArrayRef<ScheduleReportEntry> entries,
                        llvm::raw_ostream &os) {
  os << "Schedule report\n";
  for (const ScheduleReportEntry &entry : entries) {
    os << "  op_role = \"" << stringifyOpRole(entry.problem.opRole) << "\"\n";
    if (entry.problem.resultRank)
      os << "  result_rank = " << *entry.problem.resultRank << "\n";
    if (!entry.problem.staticShape.empty()) {
      os << "  static_shape = [";
      llvm::interleaveComma(entry.problem.staticShape, os);
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
    SmallVector<ScheduleReportEntry> reportEntries;
    FailureOr<SmallVector<KernelPatternView>> patternViews =
        buildKernelPatternViews(module);
    if (failed(patternViews)) {
      signalPassFailure();
      return;
    }

    unsigned nextDecisionId = 0;

    for (const KernelPatternView &pattern : *patternViews) {
      const PatternOpView *primaryOpView = getFirstPrimaryOpView(pattern);
      if (!primaryOpView) {
        signalPassFailure();
        return;
      }

      ScheduleProblem problem =
          extractScheduleProblem(primaryOpView->op, primaryOpView->role);
      std::optional<StringRef> scheduleFamily = chooseScheduleFamily(problem);
      if (!scheduleFamily) {
        primaryOpView->op->emitError() << "unsupported schedule role";
        signalPassFailure();
        return;
      }

      std::string decisionId =
          (llvm::Twine("decision_") + llvm::Twine(nextDecisionId++)).str();
      StringAttr scheduleFamilyAttr = StringAttr::get(context, *scheduleFamily);
      StringAttr scheduleTemplateAttr =
          StringAttr::get(context, kSingleTilePerBlock);
      StringAttr scheduleDecisionIdAttr = StringAttr::get(context, decisionId);

      for (const PatternOpView &opView : pattern.ops) {
        Operation *op = opView.op;
        op->setAttr(kScheduleFamilyAttr, scheduleFamilyAttr);
        op->setAttr(kScheduleTemplateAttr, scheduleTemplateAttr);
        op->setAttr(kScheduleDecisionIdAttr, scheduleDecisionIdAttr);
      }

      reportEntries.push_back(ScheduleReportEntry{
          std::move(problem), scheduleFamily->str(), kSingleTilePerBlock.str(),
          decisionId});
    }

    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Schedule)) {
      printKernelPatternViews(*patternViews, llvm::errs());
      emitScheduleReport(reportEntries, llvm::errs());
    }
  }
};

std::unique_ptr<Pass> createAscendSchedulePass() {
  return std::make_unique<AscendSchedulePass>();
}

} // namespace mlir::afir
