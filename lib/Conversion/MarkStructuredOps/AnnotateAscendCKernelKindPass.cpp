//===- AnnotateAscendCKernelKindPass.cpp - Kernel kind annotation ---------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// Derive a function-level ascendc.kernel_kind attribute from descendant
// ascendc.unit annotations.
//
//===----------------------------------------------------------------------===//

#include "Conversion/MarkStructuredOps/MarkStructuredOpsPass.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"

#define GEN_PASS_DECL_ANNOTATEASCENDCKERNELKINDPASS
#define GEN_PASS_DEF_ANNOTATEASCENDCKERNELKINDPASS
#include "Conversion/Passes.h.inc"

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

struct AnnotateAscendCKernelKindPass
    : public ::impl::AnnotateAscendCKernelKindPassBase<
          AnnotateAscendCKernelKindPass> {
  using Base =
      ::impl::AnnotateAscendCKernelKindPassBase<AnnotateAscendCKernelKindPass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
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
        signalPassFailure();
        hadError = true;
      }
    });
    if (hadError)
      return;
    if (!sawAny)
      return;

    std::string kernelKind;
    if (sawCube && sawVector)
      kernelKind = ascend::kAscendCKernelKindMix.str();
    else if (sawCube)
      kernelKind = ascend::kAscendCKernelKindCube.str();
    else
      kernelKind = ascend::kAscendCKernelKindVec.str();

    funcOp->setAttr(ascend::kAscendCKernelKindAttr,
                    StringAttr::get(funcOp.getContext(), kernelKind));
  }
};

std::unique_ptr<Pass> createAnnotateAscendCKernelKindPass() {
  return std::make_unique<AnnotateAscendCKernelKindPass>();
}

} // namespace mlir::afir
