//===- StructuredLoweringDriver.cpp - Ascend structured lowering ------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Schedule/StructuredLoweringDriver.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
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

ArrayAttr buildTailPoliciesAttr(MLIRContext *context,
                                const ScheduleDecision &decision) {
  SmallVector<Attribute> tailPolicies;
  tailPolicies.reserve(decision.tailPlans.size());
  for (const ScheduledAxisTailPlan &tailPlan : decision.tailPlans) {
    tailPolicies.push_back(
        StringAttr::get(context,
                        stringifyAxisTailPolicy(tailPlan.selectedPolicy)));
  }
  return ArrayAttr::get(context, tailPolicies);
}

ArrayAttr buildAffectedPrimitiveUsesAttr(
    Builder &builder, ArrayRef<PrimitiveAxisUseKind> affectedPrimitiveUses) {
  SmallVector<Attribute> affected;
  affected.reserve(affectedPrimitiveUses.size());
  for (PrimitiveAxisUseKind use : affectedPrimitiveUses) {
    affected.push_back(
        builder.getStringAttr(stringifyPrimitiveAxisUseKind(use)));
  }
  return builder.getArrayAttr(affected);
}

void appendGuardMarkerEntries(Builder &builder,
                              SmallVectorImpl<Attribute> &entries,
                              ArrayRef<ScheduleGuard> guards,
                              StringRef scope) {
  for (const ScheduleGuard &guard : guards) {
    entries.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("scope", builder.getStringAttr(scope)),
        builder.getNamedAttr("kind",
                             builder.getStringAttr(
                                 stringifyGuardKind(guard.kind))),
        builder.getNamedAttr(
            "domain",
            builder.getStringAttr(
                stringifyGuardAxisDomain(guard.axisDomain))),
        builder.getNamedAttr("dim",
                             builder.getI64IntegerAttr(
                                 static_cast<int64_t>(guard.dim))),
        builder.getNamedAttr("value",
                             builder.getI64IntegerAttr(guard.value)),
        builder.getNamedAttr("text", builder.getStringAttr(guard.text)),
    }));
  }
}

ArrayAttr buildGuardMarkersAttr(MLIRContext *context,
                                const ScheduleDecision &decision) {
  Builder builder(context);
  SmallVector<Attribute> entries;
  entries.reserve(decision.instance.candidateGuards.size() +
                  decision.instance.decisionGuards.size());
  appendGuardMarkerEntries(builder, entries, decision.instance.candidateGuards,
                           "candidate");
  appendGuardMarkerEntries(builder, entries, decision.instance.decisionGuards,
                           "decision");
  return builder.getArrayAttr(entries);
}

ArrayAttr buildTailPlanAttr(MLIRContext *context,
                            const ScheduleDecision &decision) {
  Builder builder(context);
  SmallVector<Attribute> entries;
  entries.reserve(decision.tailPlans.size());
  for (const ScheduledAxisTailPlan &tailPlan : decision.tailPlans) {
    entries.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("axis",
                             builder.getI64IntegerAttr(
                                 static_cast<int64_t>(
                                     tailPlan.logicalAxisId))),
        builder.getNamedAttr(
            "selected",
            builder.getStringAttr(
                stringifyAxisTailPolicy(tailPlan.selectedPolicy))),
        builder.getNamedAttr(
            "affected",
            buildAffectedPrimitiveUsesAttr(
                builder, tailPlan.affectedPrimitiveUses)),
        builder.getNamedAttr("align",
                             builder.getI64IntegerAttr(
                                 tailPlan.alignmentGranularity)),
        builder.getNamedAttr(
            "buffering",
            builder.getStringAttr(
                stringifyTailBufferingMode(tailPlan.tailBufferingMode))),
    }));
  }
  return builder.getArrayAttr(entries);
}

ArrayAttr buildTailMarkersAttr(MLIRContext *context,
                               const ScheduleDecision &decision) {
  Builder builder(context);
  SmallVector<Attribute> entries;
  entries.reserve(decision.tailPlans.size());
  for (const ScheduledAxisTailPlan &tailPlan : decision.tailPlans) {
    entries.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("axis",
                             builder.getI64IntegerAttr(
                                 static_cast<int64_t>(
                                     tailPlan.logicalAxisId))),
        builder.getNamedAttr(
            "selected",
            builder.getStringAttr(
                stringifyAxisTailPolicy(tailPlan.selectedPolicy))),
        builder.getNamedAttr("align",
                             builder.getI64IntegerAttr(
                                 tailPlan.alignmentGranularity)),
        builder.getNamedAttr(
            "buffering",
            builder.getStringAttr(
                stringifyTailBufferingMode(tailPlan.tailBufferingMode))),
        builder.getNamedAttr("guard",
                             builder.getBoolAttr(
                                 tailPlan.emitsRuntimeGuard)),
    }));
  }
  return builder.getArrayAttr(entries);
}

void setScheduleMetadata(Operation *op, DenseI64ArrayAttr selectedTileShape,
                         ArrayAttr guardMarkers, ArrayAttr tailPolicies,
                         ArrayAttr tailPlan, ArrayAttr tailMarkers,
                         StringAttr targetTilePolicy) {
  op->setAttr(kScheduleSelectedTileShapeAttr, selectedTileShape);
  op->setAttr(kScheduleGuardMarkersAttr, guardMarkers);
  op->setAttr(kScheduleTailPoliciesAttr, tailPolicies);
  op->setAttr(kScheduleTailPlanAttr, tailPlan);
  op->setAttr(kScheduleTailMarkersAttr, tailMarkers);
  op->setAttr(kScheduleTargetTilePolicyAttr, targetTilePolicy);
}

LogicalResult preserveFunctionScheduleMetadata(Operation *op,
                                               DenseI64ArrayAttr selectedTileShape,
                                               ArrayAttr guardMarkers,
                                               ArrayAttr tailPolicies,
                                               ArrayAttr tailPlan,
                                               ArrayAttr tailMarkers,
                                               StringAttr targetTilePolicy) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return success();

  // Phase 5 artifacts currently model one primary global kernel per function.
  // If multiple scheduled kernels exist, keep the first stable traversal result.
  bool hasSelectedTileShape = funcOp->hasAttr(kScheduleSelectedTileShapeAttr);
  bool hasTailPolicies = funcOp->hasAttr(kScheduleTailPoliciesAttr);
  bool hasTailPlan = funcOp->hasAttr(kScheduleTailPlanAttr);
  bool hasAnyMetadata = hasSelectedTileShape || hasTailPolicies || hasTailPlan;
  bool hasAllMetadata = hasSelectedTileShape && hasTailPolicies && hasTailPlan;
  if (hasAnyMetadata && !hasAllMetadata)
    return funcOp.emitError()
           << "function schedule metadata must include "
           << kScheduleSelectedTileShapeAttr << ", "
           << kScheduleTailPoliciesAttr << ", and "
           << kScheduleTailPlanAttr << " together";
  if (hasAllMetadata)
    return success();

  funcOp->setAttr(kScheduleSelectedTileShapeAttr, selectedTileShape);
  funcOp->setAttr(kScheduleGuardMarkersAttr, guardMarkers);
  funcOp->setAttr(kScheduleTailPoliciesAttr, tailPolicies);
  funcOp->setAttr(kScheduleTailPlanAttr, tailPlan);
  funcOp->setAttr(kScheduleTailMarkersAttr, tailMarkers);
  funcOp->setAttr(kScheduleTargetTilePolicyAttr, targetTilePolicy);
  return success();
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
  Operation *metadataOp = getDiagnosticOp(pattern);
  if (!metadataOp)
    return failure();
  MLIRContext *context = metadataOp->getContext();
  DenseI64ArrayAttr selectedTileShape = DenseI64ArrayAttr::get(
      context, decisionSet.decisions.front().instance.tileShape.tileSizes);
  const ScheduleDecision &selectedDecision = decisionSet.decisions.front();
  ArrayAttr guardMarkers = buildGuardMarkersAttr(context, selectedDecision);
  ArrayAttr tailPolicies = buildTailPoliciesAttr(context, selectedDecision);
  ArrayAttr tailPlan = buildTailPlanAttr(context, selectedDecision);
  ArrayAttr tailMarkers = buildTailMarkersAttr(context, selectedDecision);
  StringAttr targetTilePolicy =
      StringAttr::get(context, scheduleProblem.targetTilePolicy.policyId);

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
    setScheduleMetadata(opView.op, selectedTileShape, guardMarkers,
                        tailPolicies, tailPlan, tailMarkers,
                        targetTilePolicy);
    if (failed(preserveFunctionScheduleMetadata(
            opView.op, selectedTileShape, guardMarkers, tailPolicies, tailPlan,
            tailMarkers, targetTilePolicy)))
      return failure();
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
