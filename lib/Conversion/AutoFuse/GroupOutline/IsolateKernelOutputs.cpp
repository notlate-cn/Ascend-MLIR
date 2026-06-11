#include "Conversion/AutoFuse/AutoFusePasses.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#define GEN_PASS_DECL_AUTOFUSEISOLATEKERNELOUTPUTS
#define GEN_PASS_DEF_AUTOFUSEISOLATEKERNELOUTPUTS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

static Value peelTensorAliases(Value v) {
  while (Operation *defOp = v.getDefiningOp()) {
    if (auto c = dyn_cast<tensor::CastOp>(defOp))          { v = c.getSource(); continue; }
    if (auto c = dyn_cast<tensor::CollapseShapeOp>(defOp)) { v = c.getSrc();    continue; }
    if (auto c = dyn_cast<tensor::ExpandShapeOp>(defOp))   { v = c.getSrc();    continue; }
    break;
  }
  return v;
}

struct AutoFuseIsolateKernelOutputsPass
    : public ::impl::AutoFuseIsolateKernelOutputsBase<
          AutoFuseIsolateKernelOutputsPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func.isPrivate()) return;
    if (func.getBody().empty()) return;
    Block &entry = func.front();

    auto retOp = dyn_cast<func::ReturnOp>(entry.getTerminator());
    if (!retOp) return;

    OpBuilder builder(func.getContext());

    for (Value ret : retOp.getOperands()) {
      // Peel return-side aliases to reach the producing op.
      Value cur = ret;
      while (Operation *defOp = cur.getDefiningOp()) {
        if (isa<tensor::CastOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp>(defOp))
          cur = defOp->getOperand(0);
        else
          break;
      }
      Operation *defOp = cur.getDefiningOp();
      if (!defOp) continue;
      auto dps = dyn_cast<DestinationStyleOpInterface>(defOp);
      if (!dps) continue;

      auto opResult = dyn_cast<OpResult>(cur);
      if (!opResult) continue;
      unsigned resIdx = opResult.getResultNumber();
      if (resIdx >= dps.getNumDpsInits()) continue;
      OpOperand *initOperand = dps.getDpsInitOperand(resIdx);
      Value initVal = initOperand->get();

      // Only wrap when the init traces back to a func BlockArgument; local
      // intermediates (tensor.empty etc.) already bufferize to a fresh buffer.
      Value rootInit = peelTensorAliases(initVal);
      auto blockArg = dyn_cast<BlockArgument>(rootInit);
      if (!blockArg || blockArg.getOwner() != &entry) continue;

      auto tensorTy = dyn_cast<RankedTensorType>(initVal.getType());
      if (!tensorTy) continue;

      builder.setInsertionPoint(defOp);
      // When a copy source is provided, AllocTensorOp infers dynamic dim
      // sizes from the source, and passing them again is a verifier error
      // ("dynamic sizes not needed when copying a tensor").
      Value fresh = builder.create<bufferization::AllocTensorOp>(
                            defOp->getLoc(), tensorTy, ValueRange{},
                            /*copy=*/initVal)
                        .getResult();
      initOperand->set(fresh);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createAutoFuseIsolateKernelOutputsPass() {
  return std::make_unique<AutoFuseIsolateKernelOutputsPass>();
}

} // namespace mlir::afir
