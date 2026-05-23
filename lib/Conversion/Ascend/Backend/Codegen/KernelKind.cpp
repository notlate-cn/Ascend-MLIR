//===- KernelKind.cpp - Ascend kernel kind annotation ---------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// Derive a function-level ascendc.kernel_kind attribute from descendant
// ascendc.unit annotations.
//
//===----------------------------------------------------------------------===//

#include "CodegenPasses.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace mlir::afir {

namespace {

static bool classifyKernelKind(llvm::StringRef unit, bool &sawCube,
                               bool &sawVector) {
  if (unit == ascend::kAscendCUnitCube) {
    sawCube = true;
    return true;
  }
  if (unit == ascend::kAscendCUnitVector) {
    sawVector = true;
    return true;
  }
  return false;
}

} // namespace

struct AscendCodegenAnnotateKernelKindPass
    : public PassWrapper<AscendCodegenAnnotateKernelKindPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AscendCodegenAnnotateKernelKindPass)

  StringRef getArgument() const final {
    return "ascend-codegen-annotate-kernel-kind-internal";
  }

  StringRef getDescription() const final {
    return "Run internal Ascend kernel kind annotation";
  }

  void runOnOperation() override {
    if (failed(annotateAscendKernelKind(getOperation())))
      signalPassFailure();
  }
};

LogicalResult annotateAscendKernelKind(func::FuncOp funcOp) {
  bool sawCube = false;
  bool sawVector = false;
  bool sawAny = false;
  bool hadError = false;

  funcOp.walk([&](Operation *op) {
    auto unitAttr = op->getAttrOfType<StringAttr>(ascend::kAscendCUnitAttr);
    if (!unitAttr)
      return;
    sawAny = true;
    if (!classifyKernelKind(unitAttr.getValue(), sawCube, sawVector)) {
      op->emitError("unsupported ascendc.unit value for kernel kind: ")
          << unitAttr.getValue();
      hadError = true;
    }
  });
  if (hadError)
    return failure();
  if (!sawAny)
    return success();

  std::string kernelKind;
  if (sawCube && sawVector)
    kernelKind = ascend::kAscendCKernelKindMix.str();
  else if (sawCube)
    kernelKind = ascend::kAscendCKernelKindCube.str();
  else
    kernelKind = ascend::kAscendCKernelKindVec.str();

  funcOp->setAttr(ascend::kAscendCKernelKindAttr,
                  StringAttr::get(funcOp.getContext(), kernelKind));
  return success();
}

std::unique_ptr<Pass> createAscendCodegenAnnotateKernelKindPass() {
  return std::make_unique<AscendCodegenAnnotateKernelKindPass>();
}

} // namespace mlir::afir
