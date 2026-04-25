#include "Conversion/VectorPlan/VectorPlanPasses.h"
#include "Conversion/VectorPlan/GroupInfo.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL_VECTORPLANTILEFUSE
#define GEN_PASS_DEF_VECTORPLANTILEFUSE
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

namespace {
struct VectorPlanTileFusePass
    : public ::impl::VectorPlanTileFuseBase<VectorPlanTileFusePass> {
  void runOnOperation() override {
    // TODO: impl-03
  }
};
} // namespace

std::unique_ptr<Pass> createVectorPlanTileFusePass() {
  return std::make_unique<VectorPlanTileFusePass>();
}

} // namespace mlir::afir
