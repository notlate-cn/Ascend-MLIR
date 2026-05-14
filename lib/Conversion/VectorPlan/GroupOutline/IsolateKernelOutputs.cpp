#include "Conversion/VectorPlan/VectorPlanPasses.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#define GEN_PASS_DECL_VECTORPLANISOLATEKERNELOUTPUTS
#define GEN_PASS_DEF_VECTORPLANISOLATEKERNELOUTPUTS
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

struct VectorPlanIsolateKernelOutputsPass
    : public ::impl::VectorPlanIsolateKernelOutputsBase<
          VectorPlanIsolateKernelOutputsPass> {

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
      SmallVector<Value> dynSizes;
      for (auto [i, d] : llvm::enumerate(tensorTy.getShape())) {
        if (ShapedType::isDynamic(d))
          dynSizes.push_back(
              builder.create<tensor::DimOp>(defOp->getLoc(), initVal, i));
      }
      Value fresh = builder.create<bufferization::AllocTensorOp>(
                            defOp->getLoc(), tensorTy, dynSizes,
                            /*copy=*/initVal)
                        .getResult();
      initOperand->set(fresh);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createVectorPlanIsolateKernelOutputsPass() {
  return std::make_unique<VectorPlanIsolateKernelOutputsPass>();
}

} // namespace mlir::afir
