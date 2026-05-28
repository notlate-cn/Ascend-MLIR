//===- BufferizationDriver.cpp - Ascend realize buffer facts -------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "BufferizationDriver.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::ascend::realize {
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

static std::string joinFactors(ArrayRef<std::string> factors) {
  if (factors.empty())
    return "1";
  std::string result;
  llvm::raw_string_ostream os(result);
  llvm::interleave(factors, os,
                   [&os](const std::string &factor) { os << factor; },
                   " * ");
  return result;
}

static std::optional<std::string> getTensorDimExpr(Value value, unsigned dim);

static std::optional<std::string> getIndexExpr(Value value) {
  std::optional<int64_t> constant = getConstantIntValue(value);
  if (constant)
    return std::to_string(*constant);

  if (auto dimOp = value.getDefiningOp<tensor::DimOp>()) {
    std::optional<int64_t> dimIndex = getConstantIntValue(dimOp.getIndex());
    if (!dimIndex || *dimIndex < 0)
      return std::nullopt;
    return getTensorDimExpr(dimOp.getSource(),
                            static_cast<unsigned>(*dimIndex));
  }

  return std::nullopt;
}

static std::optional<std::string> getFunctionArgDimExpr(Value value,
                                                        unsigned dim) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return std::nullopt;
  Operation *parentOp = blockArg.getOwner()->getParentOp();
  if (!isa_and_nonnull<func::FuncOp>(parentOp))
    return std::nullopt;
  return (llvm::Twine("dim_arg") + llvm::Twine(blockArg.getArgNumber()) +
          "_" + llvm::Twine(dim))
      .str();
}

static std::optional<std::string> getEmptyResultDimExpr(tensor::EmptyOp emptyOp,
                                                        unsigned dim) {
  auto tensorType = dyn_cast<RankedTensorType>(emptyOp.getType());
  if (!tensorType || dim >= static_cast<unsigned>(tensorType.getRank()) ||
      !tensorType.isDynamicDim(dim))
    return std::nullopt;

  unsigned dynamicIndex = 0;
  for (unsigned i = 0; i < dim; ++i)
    if (tensorType.isDynamicDim(i))
      ++dynamicIndex;
  if (dynamicIndex >= emptyOp.getDynamicSizes().size())
    return std::nullopt;
  return getIndexExpr(emptyOp.getDynamicSizes()[dynamicIndex]);
}

static std::optional<std::string> getLinalgResultDimExpr(linalg::LinalgOp op,
                                                         OpResult result,
                                                         unsigned dim) {
  OpOperand *init = op.getDpsInitOperand(result.getResultNumber());
  if (!init)
    return std::nullopt;
  return getTensorDimExpr(init->get(), dim);
}

static std::optional<std::string> getTensorDimExpr(Value value, unsigned dim) {
  if (std::optional<std::string> argExpr = getFunctionArgDimExpr(value, dim))
    return argExpr;

  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;

  if (auto emptyOp = dyn_cast<tensor::EmptyOp>(def))
    return getEmptyResultDimExpr(emptyOp, dim);

  if (auto castOp = dyn_cast<tensor::CastOp>(def))
    return getTensorDimExpr(castOp.getSource(), dim);

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(def))
    if (auto result = dyn_cast<OpResult>(value))
      return getLinalgResultDimExpr(linalgOp, result, dim);

  return std::nullopt;
}

static std::optional<std::string> getTensorByteSizeExpr(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return std::nullopt;

  unsigned elementBits = tensorType.getElementTypeBitWidth();
  if (elementBits == 0 || elementBits % 8 != 0)
    return std::nullopt;

  SmallVector<std::string, 4> factors;
  for (auto [index, dim] : llvm::enumerate(tensorType.getShape())) {
    if (ShapedType::isDynamic(dim)) {
      std::optional<std::string> dimExpr =
          getTensorDimExpr(value, static_cast<unsigned>(index));
      if (!dimExpr)
        return std::nullopt;
      if (*dimExpr != "1")
        factors.push_back(*dimExpr);
      continue;
    }
    if (dim != 1)
      factors.push_back(std::to_string(dim));
  }

  uint64_t elementBytes = elementBits / 8;
  if (elementBytes != 1)
    factors.push_back(std::to_string(elementBytes));
  return joinFactors(factors);
}

static StringRef getKernelId(Operation *op) {
  auto kernelAttr = op->getAttrOfType<StringAttr>(kKernelAttr);
  return kernelAttr ? kernelAttr.getValue() : StringRef();
}

static bool isSupportedTensorViewOp(Operation *op) {
  return op->getNumResults() == 1 &&
         isa<tensor::CastOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp,
             tensor::ExtractSliceOp, tensor::ReshapeOp>(op);
}

static std::optional<Value> getTensorViewSource(Operation *op) {
  if (!isSupportedTensorViewOp(op) || op->getNumOperands() == 0)
    return std::nullopt;
  Value source = op->getOperand(0);
  if (!isTensorValue(source))
    return std::nullopt;
  return source;
}

static std::optional<Value> getKernelProducedTensorRoot(Value value,
                                                        StringRef kernelId) {
  llvm::DenseSet<Operation *> visited;
  Value current = value;
  while (Operation *def = current.getDefiningOp()) {
    if (getKernelId(def) == kernelId)
      return current;
    if (!visited.insert(def).second)
      return std::nullopt;

    std::optional<Value> source = getTensorViewSource(def);
    if (!source)
      return std::nullopt;
    current = *source;
  }
  return std::nullopt;
}

static bool hasUseOutsideKernel(Value value, StringRef kernelId,
                                llvm::DenseSet<Operation *> &visited) {
  for (Operation *user : value.getUsers()) {
    if (getKernelId(user) == kernelId)
      continue;
    if (!visited.insert(user).second)
      continue;
    if (isSupportedTensorViewOp(user)) {
      if (hasUseOutsideKernel(user->getResult(0), kernelId, visited))
        return true;
      continue;
    }
    return true;
  }
  return false;
}

static bool hasUseOutsideKernel(Value value, StringRef kernelId) {
  llvm::DenseSet<Operation *> visited;
  return hasUseOutsideKernel(value, kernelId, visited);
}

static bool hasUseInsideKernel(Value value, StringRef kernelId,
                               llvm::DenseSet<Operation *> &visited) {
  for (Operation *user : value.getUsers()) {
    if (getKernelId(user) == kernelId)
      return true;
    if (!visited.insert(user).second)
      continue;
    if (isSupportedTensorViewOp(user) &&
        hasUseInsideKernel(user->getResult(0), kernelId, visited))
      return true;
  }
  return false;
}

static bool hasUseInsideKernel(Value value, StringRef kernelId) {
  llvm::DenseSet<Operation *> visited;
  return hasUseInsideKernel(value, kernelId, visited);
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
    std::optional<Value> root = getKernelProducedTensorRoot(input, kernelId);
    if (root)
      recordRole(facts, *root, BufferizedValueRole::Temporary);
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
    std::optional<std::string> byteSizeExpr = getTensorByteSizeExpr(value);

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
    valueFact.byteSizeExprKnown = byteSizeExpr.has_value();
    if (byteSizeExpr)
      valueFact.byteSizeExpr = *byteSizeExpr;
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

} // namespace mlir::ascend::realize
