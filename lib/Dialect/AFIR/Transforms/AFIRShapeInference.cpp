//===- AFIRShapeInference.cpp - AFIR shape inference pass -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIROps.h"
#include "Dialect/AFIR/Transforms/Passes.h"
#include "Interface/ShapeHelperOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <functional>

namespace mlir {
namespace afir {

#define GEN_PASS_DEF_AFIRSHAPEINFERENCEPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"

namespace {

struct AFIRShapeInferencePass
    : public impl::AFIRShapeInferencePassBase<AFIRShapeInferencePass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Walk all operations and infer shapes
    module.walk([&](Operation *op) {
      if (auto shapeOp = dyn_cast<ShapeHelperOpInterface>(op)) {
        // Infer shapes for operations implementing the interface
        std::function<void(mlir::Region &)> doShapeInference = [&](mlir::Region &region) {
          // Recursively process regions if needed
          region.walk([&](Operation *innerOp) {
            if (auto innerShapeOp = dyn_cast<ShapeHelperOpInterface>(innerOp)) {
              (void)innerShapeOp.inferShapes(doShapeInference);
            }
          });
        };

        if (failed(shapeOp.inferShapes(doShapeInference))) {
          op->emitWarning("Shape inference failed");
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createAFIRShapeInferencePass() {
  return std::make_unique<AFIRShapeInferencePass>();
}

} // namespace afir
} // namespace mlir
