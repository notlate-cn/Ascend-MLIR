//===- KernelizeOpRegistry.cpp - Ascend kernelize op registry ------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "KernelizeOpRegistry.h"

#include "KernelizeSemanticUtils.h"

using namespace mlir;

namespace mlir::afir::ascend::kernelize {
namespace {

LogicalResult populateFallbackLinalgSemanticInfo(
    Operation *op, KernelizeOpSemanticInfo &info) {
  return populateLinalgSemanticInfo(op, info, "linalg");
}

LogicalResult populateFallbackArithConstantSemanticInfo(
    Operation *op, KernelizeOpSemanticInfo &info) {
  return populateArithConstantSemanticInfo(op, info, "arith_constant");
}

LogicalResult populateFallbackTensorViewSemanticInfo(
    Operation *op, KernelizeOpSemanticInfo &info) {
  return populateTensorViewSemanticInfo(op, info, "tensor_view");
}

} // namespace

void registerDefaultKernelizeOpModels(KernelizeOpModelRegistry &registry) {
  registry.registerModel({"linalg", matchLinalgSemanticOp,
                          populateFallbackLinalgSemanticInfo});
  registry.registerModel({"arith_constant", matchArithConstantSemanticOp,
                          populateFallbackArithConstantSemanticInfo});
  registry.registerModel({"tensor_view", matchTensorViewSemanticOp,
                          populateFallbackTensorViewSemanticInfo});
}

} // namespace mlir::afir::ascend::kernelize
