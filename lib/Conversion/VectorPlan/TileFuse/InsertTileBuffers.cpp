#include "Conversion/VectorPlan/VectorPlanPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define GEN_PASS_DECL_VECTORPLANINSERTTILEBUFFERS
#define GEN_PASS_DEF_VECTORPLANINSERTTILEBUFFERS
#include "Conversion/Passes.h.inc"

#define DEBUG_TYPE "vector-plan-insert-tile-buffers"

using namespace mlir;

namespace mlir::afir {

namespace {

static int64_t getMemSpace(Type type) {
  auto mrt = dyn_cast<MemRefType>(type);
  if (!mrt)
    return -1;
  Attribute space = mrt.getMemorySpace();
  if (!space)
    return 0;
  if (auto ia = dyn_cast<IntegerAttr>(space))
    return ia.getInt();
  return -1;
}

// Allocate an on-chip memref with the same shape/element type as `src` but
// with memory_space = `memSpace`.  Dynamic dimensions are materialized via
// memref.dim ops inserted at `builder`'s current insertion point.
static memref::AllocOp allocOnChip(OpBuilder &b, Location loc, Value src,
                                    int64_t memSpace) {
  auto mrt = cast<MemRefType>(src.getType());
  Attribute spaceAttr =
      IntegerAttr::get(IntegerType::get(b.getContext(), 64), memSpace);
  auto spacedType = MemRefType::get(mrt.getShape(), mrt.getElementType(),
                                    MemRefLayoutAttrInterface{}, spaceAttr);

  SmallVector<Value> dynSizes;
  for (unsigned d = 0, rank = mrt.getRank(); d < rank; ++d) {
    if (ShapedType::isDynamic(mrt.getShape()[d]))
      dynSizes.push_back(b.create<memref::DimOp>(loc, src, d));
  }
  return b.create<memref::AllocOp>(loc, spacedType, dynSizes);
}

struct VectorPlanInsertTileBuffersPass
    : public ::impl::VectorPlanInsertTileBuffersBase<
          VectorPlanInsertTileBuffersPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());

    // Collect generics whose output is in global memory (memory_space == 0).
    SmallVector<linalg::GenericOp> targets;
    func.walk([&](linalg::GenericOp op) {
      Value out = op.getDpsInitOperand(0)->get();
      if (op.getNumDpsInits() == 1 && getMemSpace(out.getType()) == 0)
        targets.push_back(op);
    });

    for (linalg::GenericOp genOp : targets) {
      Location loc = genOp.getLoc();

      // ---- Promote inputs: GM → VECIN ----
      builder.setInsertionPoint(genOp);
      for (OpOperand *inputOperand : genOp.getDpsInputOperands()) {
        Value inputMem = inputOperand->get();
        if (getMemSpace(inputMem.getType()) != 0)
          continue;
        auto vecin = allocOnChip(builder, loc, inputMem, /*VECIN=*/9);
        builder.create<memref::CopyOp>(loc, inputMem, vecin.getResult());
        inputOperand->set(vecin.getResult());
      }

      // ---- Promote output: GM → VECOUT ----
      OpOperand *outOperand = genOp.getDpsInitOperand(0);
      Value gmOut = outOperand->get();
      auto vecout = allocOnChip(builder, loc, gmOut, /*VECOUT=*/10);
      outOperand->set(vecout.getResult());

      // Insert VECOUT → GM copy after the generic.
      builder.setInsertionPointAfter(genOp);
      builder.create<memref::CopyOp>(loc, vecout.getResult(), gmOut);

      LLVM_DEBUG(llvm::dbgs() << "[insert-tile-buffers] promoted " << genOp
                               << "\n");
    }

    // Erase GM→GM memref.copy ops (bufferization yield-copy artifacts).
    SmallVector<memref::CopyOp> gmGmCopies;
    func.walk([&](memref::CopyOp op) {
      if (getMemSpace(op.getSource().getType()) == 0 &&
          getMemSpace(op.getTarget().getType()) == 0)
        gmGmCopies.push_back(op);
    });
    for (memref::CopyOp op : gmGmCopies)
      op.erase();
  }
};

} // namespace

std::unique_ptr<Pass> createVectorPlanInsertTileBuffersPass() {
  return std::make_unique<VectorPlanInsertTileBuffersPass>();
}

} // namespace mlir::afir
