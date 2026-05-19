#include "Conversion/VectorPlan/VectorPlanPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DECL_VECTORPLANRESTOREMATMUL
#define GEN_PASS_DEF_VECTORPLANRESTOREMATMUL
#include "Conversion/Passes.h.inc"

#define DEBUG_TYPE "vector-plan-restore-matmul"

using namespace mlir;

namespace mlir::afir {

namespace {

// True iff `gen` matches the signature of a generalized linalg.matmul:
//   - 3-D iter [par, par, red]
//   - 2 inputs + 1 init (DPS)
//   - Body has at least one arith.mulf AND one arith.addf
//   - Indexing maps are the standard matmul affine maps:
//       in0: (M, N, K) -> (M, K)
//       in1: (M, N, K) -> (K, N)
//       out: (M, N, K) -> (M, N)
// Mirrors the `isMatmulGeneric` helper in CubeEmitter.cpp; if either grows a
// new variant, both must be updated.
static bool isMatmulGeneric(linalg::GenericOp gen) {
  auto iter = gen.getIteratorTypesArray();
  if (iter.size() != 3) return false;
  if (iter[0] != utils::IteratorType::parallel ||
      iter[1] != utils::IteratorType::parallel ||
      iter[2] != utils::IteratorType::reduction)
    return false;
  if (gen.getNumDpsInputs() != 2 || gen.getNumDpsInits() != 1) return false;

  bool hasMul = false, hasAdd = false;
  for (Operation &op : gen.getBody()->getOperations()) {
    if (isa<arith::MulFOp>(op)) hasMul = true;
    else if (isa<arith::AddFOp>(op)) hasAdd = true;
  }
  if (!hasMul || !hasAdd) return false;

  // Verify indexing maps match the matmul pattern.
  auto maps = gen.getIndexingMapsArray();
  if (maps.size() != 3) return false;
  MLIRContext *ctx = gen.getContext();
  auto m = getAffineDimExpr(0, ctx);
  auto n = getAffineDimExpr(1, ctx);
  auto k = getAffineDimExpr(2, ctx);
  auto in0Expected = AffineMap::get(3, 0, {m, k}, ctx);
  auto in1Expected = AffineMap::get(3, 0, {k, n}, ctx);
  auto outExpected = AffineMap::get(3, 0, {m, n}, ctx);
  return maps[0] == in0Expected && maps[1] == in1Expected &&
         maps[2] == outExpected;
}

struct VectorPlanRestoreMatmulPass
    : public ::impl::VectorPlanRestoreMatmulBase<
          VectorPlanRestoreMatmulPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());

    SmallVector<linalg::GenericOp> targets;
    func.walk([&](linalg::GenericOp op) {
      if (isMatmulGeneric(op))
        targets.push_back(op);
    });

    for (linalg::GenericOp gen : targets) {
      builder.setInsertionPoint(gen);
      // Rebuild as linalg.matmul.  Preserve all attrs except those that are
      // intrinsic to the generic form (indexing_maps, iterator_types).
      ValueRange inputs = gen.getDpsInputs();
      ValueRange inits  = gen.getDpsInits();
      Type resTy = gen->getResult(0).getType();
      auto matmul = builder.create<linalg::MatmulOp>(
          gen.getLoc(), TypeRange{resTy}, ValueRange{inputs[0], inputs[1]},
          ValueRange{inits[0]});
      // Copy user-level attrs (`ascendc.unit`, etc.) onto the matmul.
      for (NamedAttribute attr : gen->getAttrs()) {
        StringRef name = attr.getName().strref();
        if (name == "indexing_maps" || name == "iterator_types" ||
            name == "operandSegmentSizes")
          continue;
        matmul->setAttr(attr.getName(), attr.getValue());
      }
      gen->getResult(0).replaceAllUsesWith(matmul.getResult(0));
      gen->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createVectorPlanRestoreMatmulPass() {
  return std::make_unique<VectorPlanRestoreMatmulPass>();
}

} // namespace mlir::afir
