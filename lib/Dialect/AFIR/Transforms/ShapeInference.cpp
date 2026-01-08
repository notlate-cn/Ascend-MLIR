//===- AFIRShapeInference.cpp - AFIR shape inference pass -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/Ops.h"
#include "Dialect/AFIR/Transforms/Passes.h"
#include "Interface/ShapeHelperOpInterface.h"
#include "Interface/ShapeInferenceOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <functional>

namespace mlir {
namespace afir {

#define GEN_PASS_DEF_AFIRSHAPEINFERENCEPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"

namespace {

struct AFIRShapeInferencePass : public impl::AFIRShapeInferencePassBase<AFIRShapeInferencePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Walk all operations and infer shapes
    module.walk([&](Operation *op) {
      if (auto shapeInfOp = dyn_cast<ShapeInferenceOpInterface>(op)) {
        // Infer shapes for operations implementing the interface
        std::function<void(mlir::Region &)> doShapeInference = [&](mlir::Region &region) {
          // Recursively process regions if needed
          region.walk([&](Operation *innerOp) {
            if (auto innerShapeInfOp = dyn_cast<ShapeInferenceOpInterface>(innerOp)) {
              (void)innerShapeInfOp.inferShapes(doShapeInference);
            }
          });
        };

        if (failed(shapeInfOp.inferShapes(doShapeInference))) {
          op->emitWarning("Shape inference failed");
        }
      }
    });

    // Update function signatures to match inferred return types
    module.walk([&](func::FuncOp funcOp) {
      // Collect all return operations
      SmallVector<func::ReturnOp> returnOps;
      funcOp.walk([&](func::ReturnOp returnOp) { returnOps.push_back(returnOp); });

      if (returnOps.empty()) return;

      // Get the types of the values being returned
      auto firstReturn = returnOps[0];
      SmallVector<Type> newResultTypes;
      for (Value operand : firstReturn.getOperands()) {
        newResultTypes.push_back(operand.getType());
      }

      // Update function type
      auto funcType = funcOp.getFunctionType();
      auto newFuncType = FunctionType::get(funcOp.getContext(), funcType.getInputs(), newResultTypes);
      funcOp.setFunctionType(newFuncType);
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRShapeInferencePass() {
  return std::make_unique<AFIRShapeInferencePass>();
}

}  // namespace afir
}  // namespace mlir
