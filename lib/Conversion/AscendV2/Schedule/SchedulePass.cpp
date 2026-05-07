//===- SchedulePass.cpp - Ascend V2 schedule pass -------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Schedule/SchedulePass.h"

#include "Conversion/AscendV2/Debug/DebugOptions.h"
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

namespace {

constexpr llvm::StringLiteral kKernelAttr = "ascend.v2.kernel";
constexpr llvm::StringLiteral kOpRoleAttr = "ascend.v2.op_role";
constexpr llvm::StringLiteral kScheduleFamilyAttr =
    "ascend.v2.schedule.family";
constexpr llvm::StringLiteral kScheduleTemplateAttr =
    "ascend.v2.schedule.template";
constexpr llvm::StringLiteral kScheduleDecisionIdAttr =
    "ascend.v2.schedule.decision_id";
constexpr llvm::StringLiteral kSingleTilePerBlock = "single_tile_per_block";

struct ScheduleProblem {
  std::string opRole;
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

bool isLinalgStructuredOp(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  return op->getName().getDialectNamespace() == "linalg" &&
         opName != "linalg.yield" && opName != "linalg.index";
}

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

ScheduleProblem extractScheduleProblem(Operation *op, StringRef opRole) {
  ScheduleProblem problem;
  problem.opRole = opRole.str();

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
  StringRef opRole(problem.opRole);
  if (opRole == "reduction")
    return StringRef("reduction_static");
  if (opRole == "cube")
    return StringRef("cube_static_matmul");
  if (opRole != "vector")
    return std::nullopt;

  if (problem.resultRank == 1)
    return StringRef("vector_static_1d");
  if (problem.resultRank == 2)
    return StringRef("vector_static_2d");

  return std::nullopt;
}

void emitScheduleReport(ArrayRef<ScheduleReportEntry> entries) {
  llvm::errs() << "Schedule report\n";
  for (const ScheduleReportEntry &entry : entries) {
    llvm::errs() << "  op_role = \"" << entry.problem.opRole << "\"\n";
    if (entry.problem.resultRank)
      llvm::errs() << "  result_rank = " << *entry.problem.resultRank << "\n";
    if (!entry.problem.staticShape.empty()) {
      llvm::errs() << "  static_shape = [";
      llvm::interleaveComma(entry.problem.staticShape, llvm::errs());
      llvm::errs() << "]\n";
    }
    llvm::errs() << "  schedule_family = \"" << entry.scheduleFamily << "\"\n";
    llvm::errs() << "  schedule_template = \"" << entry.scheduleTemplate
                 << "\"\n";
    llvm::errs() << "  schedule_decision_id = \"" << entry.decisionId
                 << "\"\n";
  }
}

} // namespace

namespace mlir::afir {

struct AscendSchedulePass
    : public ::impl::AscendSchedulePassBase<AscendSchedulePass> {
  using AscendSchedulePassBase::AscendSchedulePassBase;

  void runOnOperation() override {
    ascend::v2::DebugOptions options{
        ascend::v2::parseDebugStage(debugStage), dumpReport};
    if (ascend::v2::shouldDump(options, ascend::v2::DebugStage::Schedule))
      ascend::v2::emitStageHeader(llvm::errs(),
                                  ascend::v2::DebugStage::Schedule,
                                  getArgument());

    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    SmallVector<ScheduleReportEntry> reportEntries;
    unsigned nextDecisionId = 0;

    if (module
            .walk([&](Operation *op) {
              if (!isLinalgStructuredOp(op))
                return WalkResult::advance();

              if (!op->hasAttr(kKernelAttr)) {
                op->emitError() << "requires ascend.v2.kernel";
                return WalkResult::interrupt();
              }

              auto opRoleAttr = op->getAttrOfType<StringAttr>(kOpRoleAttr);
              if (!opRoleAttr) {
                op->emitError() << "unsupported schedule role";
                return WalkResult::interrupt();
              }

              ScheduleProblem problem =
                  extractScheduleProblem(op, opRoleAttr.getValue());
              std::optional<StringRef> scheduleFamily =
                  chooseScheduleFamily(problem);
              if (!scheduleFamily) {
                op->emitError() << "unsupported schedule role";
                return WalkResult::interrupt();
              }

              std::string decisionId =
                  (llvm::Twine("decision_") + llvm::Twine(nextDecisionId++))
                      .str();
              op->setAttr(kScheduleFamilyAttr,
                          StringAttr::get(context, *scheduleFamily));
              op->setAttr(kScheduleTemplateAttr,
                          StringAttr::get(context, kSingleTilePerBlock));
              op->setAttr(kScheduleDecisionIdAttr,
                          StringAttr::get(context, decisionId));
              reportEntries.push_back(ScheduleReportEntry{
                  std::move(problem), scheduleFamily->str(),
                  kSingleTilePerBlock.str(), decisionId});
              return WalkResult::advance();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    if (ascend::v2::shouldDump(options, ascend::v2::DebugStage::Schedule))
      emitScheduleReport(reportEntries);
  }
};

std::unique_ptr<Pass> createAscendSchedulePass() {
  return std::make_unique<AscendSchedulePass>();
}

} // namespace mlir::afir
