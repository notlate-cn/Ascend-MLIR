//===- OpRoleClassification.cpp - Ascend V2 op role classification -------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Kernelize/OpRoleClassification.h"

#include "Conversion/AscendV2/Kernelize/KernelizeTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::afir::ascend::v2::kernelize {
namespace {

constexpr OpRole kRolePriority[] = {
    OpRole::Primary,         OpRole::Cube,    OpRole::Vector,
    OpRole::Reduction,       OpRole::Injective,
    OpRole::Indexing,        OpRole::LayoutTransform,
    OpRole::Branch,          OpRole::Merge,
    OpRole::Barrier,         OpRole::Unsupported};

bool hasRole(ArrayRef<OpRole> roles, OpRole role) {
  return llvm::is_contained(roles, role);
}

void appendRole(SmallVectorImpl<OpRole> &roles, OpRole role) {
  if (!hasRole(roles, role))
    roles.push_back(role);
}

void sortByPriority(SmallVectorImpl<OpRole> &roles) {
  OpRoleList sorted;
  for (OpRole role : kRolePriority) {
    if (hasRole(roles, role))
      sorted.push_back(role);
  }
  roles.assign(sorted.begin(), sorted.end());
}

void appendComputeRoles(AccessPatternKind accessPattern,
                        SmallVectorImpl<OpRole> &roles) {
  switch (accessPattern) {
  case AccessPatternKind::Contraction:
    appendRole(roles, OpRole::Primary);
    appendRole(roles, OpRole::Cube);
    return;
  case AccessPatternKind::Reduction:
    appendRole(roles, OpRole::Primary);
    appendRole(roles, OpRole::Reduction);
    return;
  case AccessPatternKind::Elementwise:
  case AccessPatternKind::Broadcast:
    appendRole(roles, OpRole::Primary);
    appendRole(roles, OpRole::Vector);
    appendRole(roles, OpRole::Injective);
    return;
  case AccessPatternKind::Gather:
    appendRole(roles, OpRole::Indexing);
    return;
  case AccessPatternKind::LayoutTransform:
    appendRole(roles, OpRole::LayoutTransform);
    return;
  case AccessPatternKind::Unknown:
  case AccessPatternKind::Scatter:
  case AccessPatternKind::NotApplicable:
    appendRole(roles, OpRole::Unsupported);
    return;
  }
  appendRole(roles, OpRole::Unsupported);
}

bool hasTrueBoolAttr(Operation *op, StringRef attrName) {
  auto attr = op->getAttrOfType<BoolAttr>(attrName);
  return attr && attr.getValue();
}

StringRef getMvpScalarRole(ArrayRef<OpRole> roles) {
  if (hasRole(roles, OpRole::Cube))
    return "cube";
  if (hasRole(roles, OpRole::Reduction))
    return "reduction";
  if (hasRole(roles, OpRole::Vector))
    return "vector";
  return "unsupported";
}

ArrayAttr buildRoleArrayAttr(MLIRContext *context, ArrayRef<OpRole> roles) {
  SmallVector<Attribute> roleAttrs;
  roleAttrs.reserve(roles.size());
  for (OpRole role : roles)
    roleAttrs.push_back(StringAttr::get(context, stringifyOpRole(role)));
  return ArrayAttr::get(context, roleAttrs);
}

void printRoleList(raw_ostream &os, ArrayRef<OpRole> roles) {
  os << "[";
  llvm::interleaveComma(roles, os, [&](OpRole role) {
    os << "\"" << stringifyOpRole(role) << "\"";
  });
  os << "]";
}

} // namespace

FailureOr<OpRoleMap>
OpRoleClassifier::classify(const DependencyAnalysisResult &deps) const {
  OpRoleMap roleMap;
  for (Operation *op : deps.index.orderedOps) {
    auto summaryIt = deps.summaries.find(op);
    if (summaryIt == deps.summaries.end())
      return failure();

    OpRoleList roles;
    appendComputeRoles(summaryIt->second.accessPattern, roles);

    if (hasTrueBoolAttr(op, kBranchRootAttr))
      appendRole(roles, OpRole::Branch);
    if (hasTrueBoolAttr(op, kMergeRootAttr))
      appendRole(roles, OpRole::Merge);

    sortByPriority(roles);
    roleMap.try_emplace(op, roles);
  }
  return roleMap;
}

void attachRoleAttributes(ModuleOp module, const OpRoleMap &roleMap) {
  MLIRContext *context = module.getContext();

  for (const auto &entry : roleMap) {
    Operation *op = entry.first;
    op->removeAttr(kOpRolesAttr);
    op->removeAttr(kOpRoleAttr);
  }

  for (const auto &entry : roleMap) {
    Operation *op = entry.first;
    ArrayRef<OpRole> roles = entry.second;
    op->setAttr(kOpRolesAttr, buildRoleArrayAttr(context, roles));
    op->setAttr(kOpRoleAttr,
                StringAttr::get(context, getMvpScalarRole(roles)));
  }
}

void emitOpRoleClassificationReport(raw_ostream &os,
                                     const DependencyAnalysisResult &deps,
                                     const OpRoleMap &roleMap) {
  os << "OpRoleClassification\n";
  for (Operation *op : deps.index.orderedOps) {
    auto roleIt = roleMap.find(op);
    if (roleIt == roleMap.end())
      continue;

    os << "  op_id = " << deps.index.opIds.lookup(op).value << " roles = ";
    printRoleList(os, roleIt->second);
    os << " op_role = \"" << getMvpScalarRole(roleIt->second) << "\"\n";
  }
}

} // namespace mlir::afir::ascend::v2::kernelize
