//===- ScheduleProblemBuilder.cpp - Ascend schedule problem ------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Schedule/ScheduleProblemBuilder.h"

#include "Conversion/Ascend/Schedule/KernelPatternView.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace mlir::afir::ascend::schedule {
namespace {

bool isDirectProducerConsumer(Operation *producer, Operation *consumer) {
  for (Value operand : consumer->getOperands()) {
    if (operand.getDefiningOp() == producer)
      return true;
  }
  return false;
}

bool hasVectorProducerConsumerChain(const KernelPatternView &pattern) {
  for (const PatternOpView &producer : pattern.ops) {
    if (producer.role != OpRole::Vector)
      continue;
    for (const PatternOpView &consumer : pattern.ops) {
      if (consumer.role != OpRole::Vector || producer.op == consumer.op)
        continue;
      if (isDirectProducerConsumer(producer.op, consumer.op))
        return true;
    }
  }
  return false;
}

void appendTemplateTag(OpRole role, SmallVectorImpl<std::string> &tags) {
  switch (role) {
  case OpRole::Cube:
    tags.push_back("cube");
    return;
  case OpRole::Reduction:
    tags.push_back("reduction");
    return;
  case OpRole::Vector:
    tags.push_back("vector");
    return;
  case OpRole::Memory:
  case OpRole::Unknown:
    return;
  }
}

void appendShapeConstraints(ArrayRef<int64_t> shape,
                            SmallVectorImpl<std::string> &constraints) {
  for (auto [index, dim] : llvm::enumerate(shape)) {
    if (ShapedType::isDynamic(dim)) {
      constraints.push_back((llvm::Twine("d") + llvm::Twine(index) +
                             " dynamic")
                                .str());
      continue;
    }

    constraints.push_back((llvm::Twine("d") + llvm::Twine(index) + " == " +
                           llvm::Twine(dim))
                              .str());
  }
}

void appendStructureConstraints(
    const KernelPatternView &pattern,
    SmallVectorImpl<std::string> &constraints) {
  switch (pattern.dominantRole) {
  case OpRole::Cube:
    constraints.push_back("matmul_contract");
    return;
  case OpRole::Reduction:
    constraints.push_back("single_reduction_region");
    return;
  case OpRole::Vector:
    if (hasVectorProducerConsumerChain(pattern))
      constraints.push_back("elementwise_chain");
    return;
  case OpRole::Memory:
  case OpRole::Unknown:
    return;
  }
}

void printStringList(ArrayRef<std::string> values, llvm::raw_ostream &os) {
  os << "[";
  llvm::interleaveComma(values, os);
  os << "]";
}

void printShape(ArrayRef<int64_t> shape, llvm::raw_ostream &os) {
  os << "[";
  llvm::interleaveComma(shape, os, [&](int64_t dim) {
    if (ShapedType::isDynamic(dim)) {
      os << "?";
      return;
    }
    os << dim;
  });
  os << "]";
}

void printCompactAxisList(ArrayRef<unsigned> axes, llvm::raw_ostream &os) {
  os << "[";
  llvm::interleave(axes, os, [&](unsigned axis) { os << axis; }, ",");
  os << "]";
}

void printAxisExecutionRoles(ArrayRef<AxisExecutionRole> roles,
                             llvm::raw_ostream &os) {
  os << "[";
  llvm::interleave(
      roles, os,
      [&](AxisExecutionRole role) {
        os << stringifyAxisExecutionRole(role);
      },
      ",");
  os << "]";
}

void printAxisTailPolicies(ArrayRef<AxisTailPolicy> policies,
                           llvm::raw_ostream &os) {
  os << "[";
  llvm::interleave(
      policies, os,
      [&](AxisTailPolicy policy) { os << stringifyAxisTailPolicy(policy); },
      ",");
  os << "]";
}

void printPrimitiveUses(ArrayRef<PrimitiveAxisUseKind> uses,
                        llvm::raw_ostream &os) {
  os << "[";
  llvm::interleave(
      uses, os,
      [&](PrimitiveAxisUseKind use) {
        os << stringifyPrimitiveAxisUseKind(use);
      },
      ",");
  os << "]";
}

bool shouldPrintTailContractFields(
    ArrayRef<AxisScheduleConstraint> constraints) {
  for (const AxisScheduleConstraint &constraint : constraints) {
    if (constraint.semanticAlignmentGranularity != 0)
      return true;
    if (llvm::is_contained(constraint.allowedTailPolicies,
                           AxisTailPolicy::PadAndMask))
      return true;
    if (llvm::is_contained(constraint.primitiveUses,
                           PrimitiveAxisUseKind::GatherIndex))
      return true;
  }
  return false;
}

} // namespace

FailureOr<ScheduleProblem>
buildScheduleProblem(const KernelPatternView &pattern,
                     const CoalescedAxisInfo &axes) {
  const PatternOpView *primaryOpView = selectDominantPrimaryOp(pattern);
  if (!primaryOpView || !primaryOpView->op)
    return failure();

  Operation *primaryOp = primaryOpView->op;
  if (primaryOp->getNumResults() == 0) {
    primaryOp->emitError()
        << "ScheduleProblem requires selected dominant primary op with a "
           "ranked shaped first result";
    return failure();
  }

  auto resultType = dyn_cast<ShapedType>(primaryOp->getResult(0).getType());
  if (!resultType || !resultType.hasRank()) {
    primaryOp->emitError()
        << "ScheduleProblem requires selected dominant primary op with a "
           "ranked shaped first result";
    return failure();
  }

  ScheduleProblem problem;
  problem.kernelId = pattern.kernelId;
  problem.dominantRole = pattern.dominantRole;
  problem.resultRank = resultType.getRank();
  Type elementType = resultType.getElementType();
  problem.resultElementBitWidth =
      elementType.isIntOrFloat() ? elementType.getIntOrFloatBitWidth() : 0;
  llvm::append_range(problem.resultShape, resultType.getShape());
  problem.axes = axes;
  problem.guardBudget = 8;
  appendTemplateTag(problem.dominantRole, problem.templateTags);
  appendShapeConstraints(problem.resultShape, problem.shapeConstraints);
  appendStructureConstraints(pattern, problem.structureConstraints);
  return problem;
}

void printScheduleProblemReport(const ScheduleProblem &problem,
                                llvm::raw_ostream &os) {
  os << "ScheduleProblem:\n";
  os << "  kernel = " << problem.kernelId << "\n";
  os << "  role = " << stringifyOpRole(problem.dominantRole) << "\n";
  os << "  result_rank = " << problem.resultRank << "\n";
  os << "  result_shape = ";
  printShape(problem.resultShape, os);
  os << "\n";
  os << "  guard_budget = " << problem.guardBudget << "\n";
  os << "  template_tags = ";
  printStringList(problem.templateTags, os);
  os << "\n";
  os << "  shape_constraints = ";
  printStringList(problem.shapeConstraints, os);
  os << "\n";
  os << "  structure_constraints = ";
  printStringList(problem.structureConstraints, os);
  os << "\n";
  os << "  axis_constraints = [\n";
  bool printTailContract =
      shouldPrintTailContractFields(problem.axes.axisScheduleConstraints);
  for (const AxisScheduleConstraint &constraint :
       problem.axes.axisScheduleConstraints) {
    os << "    axis=" << constraint.logicalAxisId << " roles=";
    printAxisExecutionRoles(constraint.allowedRoles, os);
    os << " tail=" << stringifyAxisTailPolicy(constraint.tailPolicy);
    if (printTailContract) {
      os << " allowed_tail=";
      printAxisTailPolicies(constraint.allowedTailPolicies, os);
      os << " primitive_uses=";
      printPrimitiveUses(constraint.primitiveUses, os);
      os << " semantic_align="
         << constraint.semanticAlignmentGranularity;
    }
    os << "\n";
  }
  os << "  ]\n";
  os << "  coalescing_hints = [\n";
  for (const AxisCoalescingHint &hint : problem.axes.axisCoalescingHints) {
    os << "    group=" << hint.groupId
       << " kind=" << stringifyCoalescingHintKind(hint.kind)
       << " members=";
    printCompactAxisList(hint.memberAxisIds, os);
    os << "\n";
  }
  os << "  ]\n";
}

} // namespace mlir::afir::ascend::schedule
