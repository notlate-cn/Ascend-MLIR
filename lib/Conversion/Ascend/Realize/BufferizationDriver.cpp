//===- BufferizationDriver.cpp - Ascend realize buffer facts -------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/BufferizationDriver.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace mlir::afir::ascend::realize {
namespace {

enum class TensorValueRole { Input, Temporary, Output };

struct KernelTensorFacts {
  llvm::DenseMap<Value, TensorValueRole> roles;
};

static bool isTensorValue(Value value) {
  return llvm::isa<TensorType>(value.getType());
}

static StringRef getKernelId(Operation *op) {
  auto kernelAttr = op->getAttrOfType<StringAttr>(kKernelAttr);
  return kernelAttr ? kernelAttr.getValue() : StringRef();
}

static bool isProducedByKernel(Value value, StringRef kernelId) {
  Operation *def = value.getDefiningOp();
  return def && getKernelId(def) == kernelId;
}

static bool hasUseOutsideKernel(Value value, StringRef kernelId) {
  for (Operation *user : value.getUsers()) {
    if (getKernelId(user) != kernelId)
      return true;
  }
  return false;
}

static bool hasUseInsideKernel(Value value, StringRef kernelId) {
  for (Operation *user : value.getUsers()) {
    if (getKernelId(user) == kernelId)
      return true;
  }
  return false;
}

static bool isVectorTemporary(Value value, StringRef kernelId) {
  Operation *def = value.getDefiningOp();
  if (!def || getKernelId(def) != kernelId)
    return false;

  auto role = def->getAttrOfType<StringAttr>(kOpRoleAttr);
  return role && role.getValue() == kOpRoleVector;
}

static void recordRole(KernelTensorFacts &facts, Value value,
                       TensorValueRole role) {
  auto it = facts.roles.find(value);
  if (it == facts.roles.end() || static_cast<unsigned>(role) >
                                     static_cast<unsigned>(it->second))
    facts.roles[value] = role;
}

static void collectLinalgFacts(linalg::LinalgOp linalgOp, StringRef kernelId,
                               KernelTensorFacts &facts) {
  for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
    Value input = operand->get();
    if (!isTensorValue(input))
      continue;
    if (isProducedByKernel(input, kernelId))
      recordRole(facts, input, TensorValueRole::Temporary);
    else
      recordRole(facts, input, TensorValueRole::Input);
  }

  for (Value result : linalgOp->getResults()) {
    if (!isTensorValue(result))
      continue;
    if (hasUseOutsideKernel(result, kernelId))
      recordRole(facts, result, TensorValueRole::Output);
    else if (hasUseInsideKernel(result, kernelId))
      recordRole(facts, result, TensorValueRole::Temporary);
  }
}

static BufferizedKernelIR buildIR(StringRef kernelId,
                                  const KernelTensorFacts &facts) {
  BufferizedKernelIR ir;
  ir.kernelId = kernelId.str();
  ir.mode = "tensor_facts";
  for (const auto &entry : facts.roles) {
    switch (entry.second) {
    case TensorValueRole::Input:
      ++ir.inputValueCount;
      break;
    case TensorValueRole::Temporary:
      ++ir.temporaryValueCount;
      break;
    case TensorValueRole::Output:
      ++ir.outputValueCount;
      break;
    }

    if (entry.second == TensorValueRole::Temporary &&
        isVectorTemporary(entry.first, kernelId))
      ++ir.vectorTemporaryValueCount;
  }
  ir.bufferValueCount =
      ir.inputValueCount + ir.outputValueCount + ir.temporaryValueCount;
  return ir;
}

} // namespace

FailureOr<SmallVector<BufferizedKernelIR, 4>>
BufferizationDriver::collectTensorFacts(ModuleOp module) const {
  llvm::StringMap<KernelTensorFacts> factsByKernel;

  module.walk([&](Operation *op) {
    StringRef kernelId = getKernelId(op);
    if (kernelId.empty())
      return;

    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
      collectLinalgFacts(linalgOp, kernelId, factsByKernel[kernelId]);
  });

  SmallVector<StringRef, 4> kernelIds;
  for (const auto &entry : factsByKernel)
    kernelIds.push_back(entry.getKey());
  llvm::sort(kernelIds);

  SmallVector<BufferizedKernelIR, 4> result;
  for (StringRef kernelId : kernelIds)
    result.push_back(buildIR(kernelId, factsByKernel[kernelId]));
  return result;
}

LogicalResult BufferizationDriver::runOneShotBufferize(ModuleOp module) const {
  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.allowReturnAllocsFromLoops = true;
  options.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);

  bufferization::BufferizationState state;
  bufferization::BufferizationStatistics statistics;
  return bufferization::runOneShotModuleBufferize(module, options, state,
                                                  &statistics);
}

} // namespace mlir::afir::ascend::realize
