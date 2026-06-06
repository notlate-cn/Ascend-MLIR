//===- ScheduleContractDriver.cpp - Ascend schedule contract markers ------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ScheduleContractDriver.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::ascend::schedule {
namespace {

constexpr llvm::StringLiteral kGenericTiledLoopContract = "generic_tiled_loop";
constexpr llvm::StringLiteral kKernelMetadataKernelKey = "kernel";
constexpr llvm::StringLiteral kKernelMetadataDecisionIdKey = "decision_id";
constexpr llvm::StringLiteral kKernelMetadataTileBindingKey = "tile_binding";
constexpr llvm::StringLiteral kKernelMetadataTileParamsKey = "tile_params";
constexpr llvm::StringLiteral kKernelMetadataGuardMarkersKey =
    "guard_markers";
constexpr llvm::StringLiteral kKernelMetadataTailPoliciesKey =
    "tail_policies";
constexpr llvm::StringLiteral kKernelMetadataTailPlanKey = "tail_plan";
constexpr llvm::StringLiteral kKernelMetadataTailMarkersKey = "tail_markers";
constexpr llvm::StringLiteral kKernelMetadataTargetTilePolicyKey =
    "target_tile_policy";
constexpr llvm::StringLiteral kKernelMetadataStructuredLoweringKey =
    "structured_lowering";

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

ArrayAttr buildAxisExecutionRolesAttr(Builder &builder,
                                      ArrayRef<AxisExecutionRole> roles) {
  SmallVector<Attribute> roleAttrs;
  roleAttrs.reserve(roles.size());
  for (AxisExecutionRole role : roles)
    roleAttrs.push_back(
        builder.getStringAttr(stringifyAxisExecutionRole(role)));
  return builder.getArrayAttr(roleAttrs);
}

ArrayAttr buildTileParamsAttr(MLIRContext *context,
                              const ScheduleDecision &decision) {
  Builder builder(context);
  SmallVector<Attribute> entries;
  entries.reserve(decision.tileParams.size());
  for (const ScheduleTileParam &param : decision.tileParams) {
    entries.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("name", builder.getStringAttr(param.name)),
        builder.getNamedAttr("axis",
                             builder.getI64IntegerAttr(
                                 static_cast<int64_t>(
                                     param.logicalAxisId))),
        builder.getNamedAttr("axis_kind",
                             builder.getStringAttr(
                                 stringifyAxisKind(param.axisKind))),
        builder.getNamedAttr(
            "binding",
            builder.getStringAttr(
                stringifyTileParamBinding(param.binding))),
        builder.getNamedAttr("default",
                             builder.getI64IntegerAttr(param.defaultValue)),
        builder.getNamedAttr("upper_bound",
                             builder.getI64IntegerAttr(param.upperBound)),
        builder.getNamedAttr("extent",
                             builder.getI64IntegerAttr(param.extent)),
        builder.getNamedAttr("roles",
                             buildAxisExecutionRolesAttr(builder,
                                                         param.roles)),
        builder.getNamedAttr(
            "primitive_uses",
            buildAffectedPrimitiveUsesAttr(builder, param.primitiveUses)),
    }));
  }
  return builder.getArrayAttr(entries);
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

DictionaryAttr buildStructuredLoweringAttr(MLIRContext *context,
                                           const ScheduleDecision &decision) {
  Builder builder(context);
  SmallVector<Attribute> loopAxes;
  loopAxes.reserve(decision.tileParams.size());
  for (const ScheduleTileParam &param : decision.tileParams) {
    loopAxes.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("axis",
                             builder.getI64IntegerAttr(
                                 static_cast<int64_t>(
                                     param.logicalAxisId))),
        builder.getNamedAttr("axis_kind",
                             builder.getStringAttr(
                                 stringifyAxisKind(param.axisKind))),
        builder.getNamedAttr("tile_param",
                             builder.getStringAttr(param.name)),
        builder.getNamedAttr(
            "binding",
            builder.getStringAttr(
                stringifyTileParamBinding(param.binding))),
        builder.getNamedAttr("roles",
                             buildAxisExecutionRolesAttr(builder,
                                                         param.roles)),
        builder.getNamedAttr(
            "primitive_uses",
            buildAffectedPrimitiveUsesAttr(builder,
                                           param.primitiveUses)),
    }));
  }

  int64_t guardMarkerCount =
      static_cast<int64_t>(decision.instance.candidateGuards.size() +
                           decision.instance.decisionGuards.size());
  return builder.getDictionaryAttr({
      builder.getNamedAttr("contract",
                           builder.getStringAttr(kGenericTiledLoopContract)),
      builder.getNamedAttr("representation",
                           builder.getStringAttr(
                               "symbolic_marker_contract")),
      builder.getNamedAttr("loop_axes", builder.getArrayAttr(loopAxes)),
      builder.getNamedAttr("guard_marker_count",
                           builder.getI64IntegerAttr(guardMarkerCount)),
      builder.getNamedAttr("tail_marker_count",
                           builder.getI64IntegerAttr(
                               static_cast<int64_t>(
                                   decision.tailPlans.size()))),
      builder.getNamedAttr("cache_read_marker",
                           builder.getStringAttr("metadata_deferred")),
      builder.getNamedAttr("cache_write_marker",
                           builder.getStringAttr("metadata_deferred")),
      builder.getNamedAttr("pipeline_marker",
                           builder.getStringAttr("none")),
      builder.getNamedAttr("double_buffer_marker",
                           builder.getStringAttr("none")),
  });
}

void setScheduleMetadata(Operation *op, StringAttr tileBinding,
                         ArrayAttr tileParams,
                         ArrayAttr guardMarkers, ArrayAttr tailPolicies,
                         ArrayAttr tailPlan, ArrayAttr tailMarkers,
                         StringAttr targetTilePolicy,
                         DictionaryAttr structuredLowering) {
  op->setAttr(kScheduleTileBindingAttr, tileBinding);
  op->setAttr(kScheduleTileParamsAttr, tileParams);
  op->setAttr(kScheduleGuardMarkersAttr, guardMarkers);
  op->setAttr(kScheduleTailPoliciesAttr, tailPolicies);
  op->setAttr(kScheduleTailPlanAttr, tailPlan);
  op->setAttr(kScheduleTailMarkersAttr, tailMarkers);
  op->setAttr(kScheduleTargetTilePolicyAttr, targetTilePolicy);
  op->setAttr(kScheduleStructuredLoweringAttr, structuredLowering);
}

void clearLegacyFunctionScheduleMetadata(func::FuncOp funcOp) {
  funcOp->removeAttr(kScheduleTileBindingAttr);
  funcOp->removeAttr(kScheduleTileParamsAttr);
  funcOp->removeAttr(kScheduleGuardMarkersAttr);
  funcOp->removeAttr(kScheduleTailPoliciesAttr);
  funcOp->removeAttr(kScheduleTailPlanAttr);
  funcOp->removeAttr(kScheduleTailMarkersAttr);
  funcOp->removeAttr(kScheduleTargetTilePolicyAttr);
  funcOp->removeAttr(kScheduleStructuredLoweringAttr);
}

void setLegacyFunctionScheduleMetadata(func::FuncOp funcOp,
                                       StringAttr tileBinding,
                                       ArrayAttr tileParams,
                                       ArrayAttr guardMarkers,
                                       ArrayAttr tailPolicies,
                                       ArrayAttr tailPlan,
                                       ArrayAttr tailMarkers,
                                       StringAttr targetTilePolicy,
                                       DictionaryAttr structuredLowering) {
  funcOp->setAttr(kScheduleTileBindingAttr, tileBinding);
  funcOp->setAttr(kScheduleTileParamsAttr, tileParams);
  funcOp->setAttr(kScheduleGuardMarkersAttr, guardMarkers);
  funcOp->setAttr(kScheduleTailPoliciesAttr, tailPolicies);
  funcOp->setAttr(kScheduleTailPlanAttr, tailPlan);
  funcOp->setAttr(kScheduleTailMarkersAttr, tailMarkers);
  funcOp->setAttr(kScheduleTargetTilePolicyAttr, targetTilePolicy);
  funcOp->setAttr(kScheduleStructuredLoweringAttr, structuredLowering);
}

DictionaryAttr
buildKernelScheduleMetadataEntry(Builder &builder, StringRef kernelId,
                                 StringRef decisionId,
                                 StringAttr tileBinding,
                                 ArrayAttr tileParams,
                                 ArrayAttr guardMarkers,
                                 ArrayAttr tailPolicies, ArrayAttr tailPlan,
                                 ArrayAttr tailMarkers,
                                 StringAttr targetTilePolicy,
                                 DictionaryAttr structuredLowering) {
  return builder.getDictionaryAttr({
      builder.getNamedAttr(kKernelMetadataKernelKey,
                           builder.getStringAttr(kernelId)),
      builder.getNamedAttr(kKernelMetadataDecisionIdKey,
                           builder.getStringAttr(decisionId)),
      builder.getNamedAttr(kKernelMetadataTileBindingKey, tileBinding),
      builder.getNamedAttr(kKernelMetadataTileParamsKey, tileParams),
      builder.getNamedAttr(kKernelMetadataGuardMarkersKey, guardMarkers),
      builder.getNamedAttr(kKernelMetadataTailPoliciesKey, tailPolicies),
      builder.getNamedAttr(kKernelMetadataTailPlanKey, tailPlan),
      builder.getNamedAttr(kKernelMetadataTailMarkersKey, tailMarkers),
      builder.getNamedAttr(kKernelMetadataTargetTilePolicyKey,
                           targetTilePolicy),
      builder.getNamedAttr(kKernelMetadataStructuredLoweringKey,
                           structuredLowering),
  });
}

bool kernelScheduleMetadataEntryMatches(DictionaryAttr entry,
                                        StringAttr tileBinding,
                                        ArrayAttr tileParams,
                                        ArrayAttr guardMarkers,
                                        ArrayAttr tailPolicies,
                                        ArrayAttr tailPlan,
                                        ArrayAttr tailMarkers,
                                        StringAttr targetTilePolicy,
                                        DictionaryAttr structuredLowering) {
  return entry.get(kKernelMetadataTileBindingKey) == tileBinding &&
         entry.get(kKernelMetadataTileParamsKey) == tileParams &&
         entry.get(kKernelMetadataGuardMarkersKey) == guardMarkers &&
         entry.get(kKernelMetadataTailPoliciesKey) == tailPolicies &&
         entry.get(kKernelMetadataTailPlanKey) == tailPlan &&
         entry.get(kKernelMetadataTailMarkersKey) == tailMarkers &&
         entry.get(kKernelMetadataTargetTilePolicyKey) == targetTilePolicy &&
         entry.get(kKernelMetadataStructuredLoweringKey) ==
             structuredLowering;
}

LogicalResult verifyLegacyFunctionScheduleMetadataShape(func::FuncOp funcOp) {
  bool hasTileBinding = funcOp->hasAttr(kScheduleTileBindingAttr);
  bool hasTileParams = funcOp->hasAttr(kScheduleTileParamsAttr);
  bool hasGuardMarkers = funcOp->hasAttr(kScheduleGuardMarkersAttr);
  bool hasTailPolicies = funcOp->hasAttr(kScheduleTailPoliciesAttr);
  bool hasTailPlan = funcOp->hasAttr(kScheduleTailPlanAttr);
  bool hasTailMarkers = funcOp->hasAttr(kScheduleTailMarkersAttr);
  bool hasTargetTilePolicy = funcOp->hasAttr(kScheduleTargetTilePolicyAttr);
  bool hasStructuredLowering =
      funcOp->hasAttr(kScheduleStructuredLoweringAttr);
  bool hasAnyMetadata = hasTileBinding || hasTileParams || hasGuardMarkers || hasTailPolicies ||
                        hasTailPlan || hasTailMarkers || hasTargetTilePolicy ||
                        hasStructuredLowering;
  bool hasAllCoreMetadata =
      hasTailPolicies && hasTailPlan;
  if (hasAnyMetadata && !hasAllCoreMetadata)
    return funcOp.emitError()
           << "function schedule metadata must include "
           << kScheduleTailPoliciesAttr << " and "
           << kScheduleTailPlanAttr << " together";
  if (hasAnyMetadata && hasTileBinding != hasTileParams)
    return funcOp.emitError()
           << "function schedule metadata must include "
           << kScheduleTileBindingAttr << " and "
           << kScheduleTileParamsAttr << " together";
  return success();
}

FailureOr<ArrayAttr>
upsertKernelScheduleMetadata(func::FuncOp funcOp, StringRef kernelId,
                             StringRef decisionId,
                             StringAttr tileBinding,
                             ArrayAttr tileParams,
                             ArrayAttr guardMarkers, ArrayAttr tailPolicies,
                             ArrayAttr tailPlan, ArrayAttr tailMarkers,
                             StringAttr targetTilePolicy,
                             DictionaryAttr structuredLowering) {
  Builder builder(funcOp.getContext());
  SmallVector<Attribute> entries;
  bool foundKernel = false;

  if (Attribute existing = funcOp->getAttr(kScheduleKernelMetadataAttr)) {
    auto existingEntries = dyn_cast<ArrayAttr>(existing);
    if (!existingEntries)
      return funcOp.emitError()
             << kScheduleKernelMetadataAttr << " must be an array attribute";

    for (Attribute rawEntry : existingEntries) {
      auto entry = dyn_cast<DictionaryAttr>(rawEntry);
      if (!entry)
        return funcOp.emitError()
               << kScheduleKernelMetadataAttr
               << " entries must be dictionary attributes";
      auto entryKernel =
          dyn_cast_or_null<StringAttr>(entry.get(kKernelMetadataKernelKey));
      if (!entryKernel)
        return funcOp.emitError()
               << kScheduleKernelMetadataAttr
               << " entries must include a string kernel field";

      if (entryKernel.getValue() == kernelId) {
        foundKernel = true;
        if (!kernelScheduleMetadataEntryMatches(
                entry, tileBinding, tileParams, guardMarkers, tailPolicies,
                tailPlan, tailMarkers,
                targetTilePolicy, structuredLowering))
          return funcOp.emitError()
                 << "function contains conflicting schedule metadata for "
                    "kernel \""
                 << kernelId << "\"";
      }
      entries.push_back(entry);
    }
  }

  if (!foundKernel)
    entries.push_back(buildKernelScheduleMetadataEntry(
        builder, kernelId, decisionId, tileBinding, tileParams, guardMarkers,
        tailPolicies, tailPlan, tailMarkers, targetTilePolicy,
        structuredLowering));

  return builder.getArrayAttr(entries);
}

void reconcileLegacyFunctionScheduleMetadata(func::FuncOp funcOp,
                                             ArrayAttr kernelMetadata,
                                             StringAttr tileBinding,
                                             ArrayAttr tileParams,
                                             ArrayAttr guardMarkers,
                                             ArrayAttr tailPolicies,
                                             ArrayAttr tailPlan,
                                             ArrayAttr tailMarkers,
                                             StringAttr targetTilePolicy,
                                             DictionaryAttr structuredLowering) {
  bool allEntriesShareMetadata = true;
  for (Attribute rawEntry : kernelMetadata) {
    auto entry = cast<DictionaryAttr>(rawEntry);
    if (!kernelScheduleMetadataEntryMatches(
            entry, tileBinding, tileParams, guardMarkers, tailPolicies,
            tailPlan, tailMarkers, targetTilePolicy,
            structuredLowering)) {
      allEntriesShareMetadata = false;
      break;
    }
  }

  if (allEntriesShareMetadata) {
    setLegacyFunctionScheduleMetadata(funcOp, tileBinding, tileParams,
                                      guardMarkers, tailPolicies, tailPlan,
                                      tailMarkers, targetTilePolicy,
                                      structuredLowering);
    return;
  }

  clearLegacyFunctionScheduleMetadata(funcOp);
}

LogicalResult preserveFunctionScheduleMetadata(Operation *op,
                                               StringAttr tileBinding,
                                               ArrayAttr tileParams,
                                               ArrayAttr guardMarkers,
                                               ArrayAttr tailPolicies,
                                               ArrayAttr tailPlan,
                                               ArrayAttr tailMarkers,
                                               StringAttr targetTilePolicy,
                                               DictionaryAttr structuredLowering) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return success();

  auto kernelAttr = op->getAttrOfType<StringAttr>(kKernelAttr);
  if (!kernelAttr)
    return op->emitError()
           << "schedule contract requires " << kKernelAttr;

  if (failed(verifyLegacyFunctionScheduleMetadataShape(funcOp)))
    return failure();

  auto decisionIdAttr = op->getAttrOfType<StringAttr>(kScheduleDecisionIdAttr);
  if (!decisionIdAttr)
    return op->emitError()
           << "schedule contract requires " << kScheduleDecisionIdAttr;

  FailureOr<ArrayAttr> kernelMetadata = upsertKernelScheduleMetadata(
      funcOp, kernelAttr.getValue(), decisionIdAttr.getValue(),
      tileBinding, tileParams, guardMarkers, tailPolicies, tailPlan,
      tailMarkers, targetTilePolicy, structuredLowering);
  if (failed(kernelMetadata))
    return failure();

  funcOp->setAttr(kScheduleKernelMetadataAttr, *kernelMetadata);
  reconcileLegacyFunctionScheduleMetadata(
      funcOp, *kernelMetadata, tileBinding, tileParams, guardMarkers,
      tailPolicies, tailPlan, tailMarkers, targetTilePolicy,
      structuredLowering);
  return success();
}

} // namespace

LogicalResult
applyScheduleContractMarkers(const KernelPatternView &pattern,
                             const ScheduleProblem &scheduleProblem,
                             const ScheduleDecisionSet &decisionSet) {
  ScheduleContractReport report;
  return applyScheduleContractMarkers(pattern, scheduleProblem, decisionSet,
                                      report);
}

LogicalResult applyScheduleContractMarkers(
    const KernelPatternView &pattern, const ScheduleProblem &scheduleProblem,
    const ScheduleDecisionSet &decisionSet, ScheduleContractReport &report) {
  report.kernelId = scheduleProblem.kernelId;
  report.contract = kGenericTiledLoopContract.str();
  report.verifiedOps = 0;

  if (pattern.kernelId != scheduleProblem.kernelId ||
      decisionSet.kernelId != scheduleProblem.kernelId) {
    if (Operation *op = getDiagnosticOp(pattern))
      op->emitError()
          << "schedule contract kernel id mismatch: pattern = "
          << pattern.kernelId << ", problem = " << scheduleProblem.kernelId
          << ", decision set = " << decisionSet.kernelId;
    return failure();
  }

  if (decisionSet.decisions.empty()) {
    if (Operation *op = getDiagnosticOp(pattern))
      op->emitError()
          << "schedule contract requires at least one schedule decision for "
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
  const ScheduleDecision &selectedDecision = decisionSet.decisions.front();
  StringAttr tileBinding =
      StringAttr::get(context, kScheduleTileBindingSymbolic);
  ArrayAttr tileParams = buildTileParamsAttr(context, selectedDecision);
  ArrayAttr guardMarkers = buildGuardMarkersAttr(context, selectedDecision);
  ArrayAttr tailPolicies = buildTailPoliciesAttr(context, selectedDecision);
  ArrayAttr tailPlan = buildTailPlanAttr(context, selectedDecision);
  ArrayAttr tailMarkers = buildTailMarkersAttr(context, selectedDecision);
  StringAttr targetTilePolicy =
      StringAttr::get(context, scheduleProblem.targetTilePolicy.policyId);
  DictionaryAttr structuredLowering =
      buildStructuredLoweringAttr(context, selectedDecision);

  for (const PatternOpView &opView : pattern.ops) {
    Operation *op = opView.op;
    auto decisionIdAttr =
        op->getAttrOfType<StringAttr>(kScheduleDecisionIdAttr);
    if (!decisionIdAttr) {
      op->emitError() << "schedule contract requires "
                      << kScheduleDecisionIdAttr << " = \""
                      << selectedDecisionId << "\"";
      return failure();
    }
    if (decisionIdAttr.getValue() != selectedDecisionId) {
      op->emitError() << "schedule contract expected "
                      << kScheduleDecisionIdAttr << " = \""
                      << selectedDecisionId << "\", got \""
                      << decisionIdAttr.getValue() << "\"";
      return failure();
    }
  }

  for (const PatternOpView &opView : pattern.ops) {
    opView.op->setAttr(
        kScheduleContractAttr,
        StringAttr::get(opView.op->getContext(), kGenericTiledLoopContract));
    setScheduleMetadata(opView.op, tileBinding, tileParams, guardMarkers,
                        tailPolicies, tailPlan, tailMarkers,
                        targetTilePolicy, structuredLowering);
    if (failed(preserveFunctionScheduleMetadata(
            opView.op, tileBinding, tileParams, guardMarkers, tailPolicies,
            tailPlan, tailMarkers,
            targetTilePolicy, structuredLowering)))
      return failure();
    ++report.verifiedOps;
  }

  return success();
}

void printScheduleContractReport(const ScheduleContractReport &report,
                                 llvm::raw_ostream &os) {
  os << "ScheduleContract:\n";
  os << "  kernel = " << report.kernelId << "\n";
  os << "  contract = " << report.contract << "\n";
  os << "  verified_ops = " << report.verifiedOps << "\n";
}

} // namespace mlir::ascend::schedule
