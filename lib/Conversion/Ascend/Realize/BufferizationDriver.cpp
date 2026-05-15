//===- BufferizationDriver.cpp - Ascend realize buffer facts -------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "BufferizationDriver.h"

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

struct KernelTensorFacts {
  llvm::DenseMap<Value, BufferizedValueRole> roles;
  llvm::SmallVector<Value, 8> orderedValues;
};

static bool isTensorValue(Value value) {
  return llvm::isa<TensorType>(value.getType());
}

static FailureOr<uint64_t> getStaticTensorByteSize(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType || !tensorType.hasStaticShape())
    return failure();

  unsigned elementBits = tensorType.getElementTypeBitWidth();
  if (elementBits == 0)
    return failure();

  uint64_t elementCount = static_cast<uint64_t>(tensorType.getNumElements());
  uint64_t totalBits = elementCount * static_cast<uint64_t>(elementBits);
  return (totalBits + 7) / 8;
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

static bool opRolesAttrHasRole(Operation *op, StringRef roleName) {
  auto roles = op->getAttrOfType<ArrayAttr>(kOpRolesAttr);
  if (!roles)
    return false;
  for (Attribute attr : roles) {
    auto role = dyn_cast<StringAttr>(attr);
    if (role && role.getValue() == roleName)
      return true;
  }
  return false;
}

static bool hasVectorRole(Operation *op) {
  if (opRolesAttrHasRole(op, kKernelizeOpRoleVector) ||
      opRolesAttrHasRole(op, kOpRoleVector))
    return true;

  auto role = op->getAttrOfType<StringAttr>(kOpRoleAttr);
  return role && role.getValue() == kOpRoleVector;
}

static bool isVectorTemporary(Value value, StringRef kernelId) {
  Operation *def = value.getDefiningOp();
  if (!def || getKernelId(def) != kernelId)
    return false;

  return hasVectorRole(def);
}

static void recordRole(KernelTensorFacts &facts, Value value,
                       BufferizedValueRole role) {
  auto it = facts.roles.find(value);
  if (it == facts.roles.end()) {
    facts.orderedValues.push_back(value);
    facts.roles[value] = role;
    return;
  }

  if (static_cast<unsigned>(role) > static_cast<unsigned>(it->second))
    it->second = role;
}

static void collectLinalgFacts(linalg::LinalgOp linalgOp, StringRef kernelId,
                               KernelTensorFacts &facts) {
  for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
    Value input = operand->get();
    if (!isTensorValue(input))
      continue;
    if (isProducedByKernel(input, kernelId))
      recordRole(facts, input, BufferizedValueRole::Temporary);
    else
      recordRole(facts, input, BufferizedValueRole::Input);
  }

  for (Value result : linalgOp->getResults()) {
    if (!isTensorValue(result))
      continue;
    if (hasUseOutsideKernel(result, kernelId))
      recordRole(facts, result, BufferizedValueRole::Output);
    else if (hasUseInsideKernel(result, kernelId))
      recordRole(facts, result, BufferizedValueRole::Temporary);
  }
}

static BufferizedKernelIR buildIR(StringRef kernelId,
                                  const KernelTensorFacts &facts) {
  BufferizedKernelIR ir;
  ir.kernelId = kernelId.str();
  ir.mode = "tensor_facts";
  bool allStaticByteSizesKnown = true;
  for (Value value : facts.orderedValues) {
    auto roleIt = facts.roles.find(value);
    if (roleIt == facts.roles.end())
      continue;

    FailureOr<uint64_t> byteSize = getStaticTensorByteSize(value);
    if (failed(byteSize))
      allStaticByteSizesKnown = false;

    bool vectorTemporary =
        roleIt->second == BufferizedValueRole::Temporary &&
        isVectorTemporary(value, kernelId);

    BufferizedValueFact valueFact;
    valueFact.valueId = ir.valueFacts.size();
    valueFact.role = roleIt->second;
    valueFact.isVectorTemporary = vectorTemporary;
    valueFact.staticByteSizeKnown = succeeded(byteSize);
    if (succeeded(byteSize))
      valueFact.byteSize = *byteSize;
    ir.valueFacts.push_back(valueFact);

    switch (roleIt->second) {
    case BufferizedValueRole::Input:
      ++ir.inputValueCount;
      if (succeeded(byteSize))
        ir.inputByteCount += *byteSize;
      break;
    case BufferizedValueRole::Temporary:
      ++ir.temporaryValueCount;
      if (succeeded(byteSize))
        ir.temporaryByteCount += *byteSize;
      break;
    case BufferizedValueRole::Output:
      ++ir.outputValueCount;
      if (succeeded(byteSize))
        ir.outputByteCount += *byteSize;
      break;
    }

    if (vectorTemporary) {
      ++ir.vectorTemporaryValueCount;
      if (succeeded(byteSize))
        ir.vectorTemporaryByteCount += *byteSize;
    }
  }
  ir.bufferValueCount =
      ir.inputValueCount + ir.outputValueCount + ir.temporaryValueCount;
  ir.staticByteSizeKnown =
      !facts.roles.empty() && allStaticByteSizesKnown;
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
