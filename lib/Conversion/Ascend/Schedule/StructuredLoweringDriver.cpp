//===- StructuredLoweringDriver.cpp - Ascend structured lowering ------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Schedule/StructuredLoweringDriver.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::afir::ascend::schedule {
namespace {

constexpr llvm::StringLiteral kLoopSkeletonV0 = "loop_skeleton_v0";

Operation *getDiagnosticOp(const KernelPatternView &pattern) {
  if (!pattern.primaryOps.empty())
    return pattern.primaryOps.front();
  if (!pattern.ops.empty())
    return pattern.ops.front().op;
  return nullptr;
}

} // namespace

LogicalResult
applyStructuredLoweringMarkers(const KernelPatternView &pattern,
                               const ScheduleProblem &scheduleProblem,
                               const ScheduleDecisionSet &decisionSet) {
  StructuredLoweringReport report;
  return applyStructuredLoweringMarkers(pattern, scheduleProblem, decisionSet,
                                        report);
}

LogicalResult applyStructuredLoweringMarkers(
    const KernelPatternView &pattern, const ScheduleProblem &scheduleProblem,
    const ScheduleDecisionSet &decisionSet, StructuredLoweringReport &report) {
  report.kernelId = scheduleProblem.kernelId;
  report.skeleton = kLoopSkeletonV0.str();
  report.verifiedOps = 0;

  if (pattern.kernelId != scheduleProblem.kernelId ||
      decisionSet.kernelId != scheduleProblem.kernelId) {
    if (Operation *op = getDiagnosticOp(pattern))
      op->emitError()
          << "structured lowering kernel id mismatch: pattern = "
          << pattern.kernelId << ", problem = " << scheduleProblem.kernelId
          << ", decision set = " << decisionSet.kernelId;
    return failure();
  }

  if (decisionSet.decisions.empty()) {
    if (Operation *op = getDiagnosticOp(pattern))
      op->emitError()
          << "structured lowering requires at least one schedule decision for "
             "kernel "
          << scheduleProblem.kernelId;
    return failure();
  }

  const std::string &selectedDecisionId =
      decisionSet.decisions.front().decisionId;
  for (const PatternOpView &opView : pattern.ops) {
    Operation *op = opView.op;
    auto decisionIdAttr =
        op->getAttrOfType<StringAttr>(kScheduleDecisionIdAttr);
    if (!decisionIdAttr) {
      op->emitError() << "structured lowering requires "
                      << kScheduleDecisionIdAttr << " = \""
                      << selectedDecisionId << "\"";
      return failure();
    }
    if (decisionIdAttr.getValue() != selectedDecisionId) {
      op->emitError() << "structured lowering expected "
                      << kScheduleDecisionIdAttr << " = \""
                      << selectedDecisionId << "\", got \""
                      << decisionIdAttr.getValue() << "\"";
      return failure();
    }
  }

  for (const PatternOpView &opView : pattern.ops) {
    opView.op->setAttr(kStructuredLoweringAttr,
                       StringAttr::get(opView.op->getContext(),
                                       kLoopSkeletonV0));
    ++report.verifiedOps;
  }

  return success();
}

void printStructuredLoweringReport(const StructuredLoweringReport &report,
                                   llvm::raw_ostream &os) {
  os << "StructuredLowering:\n";
  os << "  kernel = " << report.kernelId << "\n";
  os << "  skeleton = " << report.skeleton << "\n";
  os << "  verified_ops = " << report.verifiedOps << "\n";
}

} // namespace mlir::afir::ascend::schedule
