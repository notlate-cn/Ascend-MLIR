//===- KernelizeOpInterface.cpp - Ascend kernelize op interface ----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/KernelizeOpInterface.h"

#include "Conversion/Ascend/Common/Attributes.h"

namespace mlir::afir::ascend::kernelize {
namespace {

void appendDefaultPreferredTemplateFamilies(KernelizeOpSemanticInfo &info) {
  if (!info.preferredTemplateFamilies.empty())
    return;

  if (info.accessPattern == AccessPatternKind::Contraction) {
    info.preferredTemplateFamilies.push_back(kOpRoleCube.str());
  } else if (info.accessPattern == AccessPatternKind::Reduction) {
    info.preferredTemplateFamilies.push_back(kOpRoleReduction.str());
  } else if (info.accessPattern == AccessPatternKind::Elementwise ||
             info.accessPattern == AccessPatternKind::Broadcast ||
             info.accessPattern == AccessPatternKind::LayoutTransform) {
    info.preferredTemplateFamilies.push_back(kOpRoleVector.str());
  }
}

} // namespace

void KernelizeOpModelRegistry::registerModel(KernelizeOpModel model) {
  models.push_back(model);
}

FailureOr<KernelizeOpSemanticInfo>
KernelizeOpModelRegistry::resolve(Operation *op) const {
  for (const KernelizeOpModel &model : models) {
    if (!model.match || !model.match(op))
      continue;

    KernelizeOpSemanticInfo info;
    info.modelName = model.name;
    if (!model.populate)
      return info;
    if (failed(model.populate(op, info)))
      return failure();
    if (info.modelName == "unknown")
      info.modelName = model.name;
    appendDefaultPreferredTemplateFamilies(info);
    return info;
  }

  KernelizeOpSemanticInfo info;
  info.participation = KernelizeParticipationKind::Unsupported;
  info.accessPattern = AccessPatternKind::Unknown;
  info.modelName = "unknown";
  info.unsupportedReason = "no kernelize semantic model for op " +
                           op->getName().getStringRef().str();
  return info;
}

llvm::StringRef stringifyAccessPattern(AccessPatternKind kind) {
  switch (kind) {
  case AccessPatternKind::NotApplicable:
    return "NotApplicable";
  case AccessPatternKind::Elementwise:
    return "Elementwise";
  case AccessPatternKind::Broadcast:
    return "Broadcast";
  case AccessPatternKind::Reduction:
    return "Reduction";
  case AccessPatternKind::Contraction:
    return "Contraction";
  case AccessPatternKind::Gather:
    return "Gather";
  case AccessPatternKind::Scatter:
    return "Scatter";
  case AccessPatternKind::LayoutTransform:
    return "LayoutTransform";
  case AccessPatternKind::Unknown:
    return "Unknown";
  }
  return "Unknown";
}

llvm::StringRef stringifyIteratorKind(IteratorKind kind) {
  switch (kind) {
  case IteratorKind::Parallel:
    return "parallel";
  case IteratorKind::Reduction:
    return "reduction";
  case IteratorKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

llvm::StringRef
stringifyKernelizeParticipation(KernelizeParticipationKind kind) {
  switch (kind) {
  case KernelizeParticipationKind::Ignore:
    return "ignore";
  case KernelizeParticipationKind::Analyze:
    return "analyze";
  case KernelizeParticipationKind::Transparent:
    return "transparent";
  case KernelizeParticipationKind::Unsupported:
    return "unsupported";
  }
  return "unsupported";
}

llvm::StringRef stringifyKernelizeSemanticTrait(KernelizeSemanticTrait trait) {
  switch (trait) {
  case KernelizeSemanticTrait::Unknown:
    return "unknown";
  case KernelizeSemanticTrait::Structured:
    return "structured";
  case KernelizeSemanticTrait::TensorView:
    return "tensor_view";
  case KernelizeSemanticTrait::LayoutTransform:
    return "layout_transform";
  case KernelizeSemanticTrait::HandwrittenGroup:
    return "handwritten_group";
  }
  return "unknown";
}

llvm::StringRef stringifyKernelizeSeedPolicy(KernelizeSeedPolicy policy) {
  switch (policy) {
  case KernelizeSeedPolicy::MaySeed:
    return "may_seed";
  case KernelizeSeedPolicy::NonSeedWhenFused:
    return "non_seed_when_fused";
  case KernelizeSeedPolicy::NeverSeed:
    return "never_seed";
  }
  return "never_seed";
}

} // namespace mlir::afir::ascend::kernelize
