#include "Conversion/VectorPlan/VectorPlanPasses.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// Allocate an on-chip memref with explicit dynamic size (e.g. the loop step)
// and the given element type and memory space.  The resulting memref is 1-D
// with one dynamic extent backed by `stepSize`.
static memref::AllocOp allocOnChipSized(OpBuilder &b, Location loc,
                                         Value stepSize, Type elemType,
                                         int64_t memSpace) {
  Attribute spaceAttr =
      IntegerAttr::get(IntegerType::get(b.getContext(), 64), memSpace);
  auto spacedType = MemRefType::get({ShapedType::kDynamic}, elemType,
                                    MemRefLayoutAttrInterface{}, spaceAttr);
  return b.create<memref::AllocOp>(loc, spacedType, ValueRange{stepSize});
}

// Allocate an on-chip memref with the same shape/element type as `src` but
// with memory_space = `memSpace`.  Dynamic dimensions are materialized via
// memref.dim ops inserted at `builder`'s current insertion point.
// Used as a fallback when the op is not inside a for loop.
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

    // Track GM values that have pending MTE2 (local→GM) writes.  When a
    // subsequent op reads from the same GM address, we insert a
    // PipeBarrier(MTE2) barrier to prevent the MTE1 load from observing stale
    // data before the MTE2 store has committed.
    llvm::SmallDenseMap<Value, Value> pendingMTE2Writes; // GM value → vecout

    for (linalg::GenericOp genOp : targets) {
      Location loc = genOp.getLoc();

      // Find the innermost enclosing scf::ForOp (the sub-tile loop).
      // Buffer allocs must be hoisted BEFORE this loop so that AscendC's
      // TPipe::InitBuffer is called once per block rather than once per
      // sub-tile.  Calling InitBuffer repeatedly on the same TBuf/TQue handle
      // inside a loop is a bump-allocator leak that causes the 4th and later
      // sub-tile iterations to receive a zero-filled buffer.
      auto innerFor = genOp->getParentOfType<scf::ForOp>();

      // ---- Promote inputs: GM → VECIN ----
      // Allocs go before the inner loop; copies and barriers go before genOp.
      for (OpOperand *inputOperand : genOp.getDpsInputOperands()) {
        Value inputMem = inputOperand->get();
        if (getMemSpace(inputMem.getType()) != 0)
          continue;

        // Insert PipeBarrier inside the loop (before genOp) if there is a
        // pending MTE2 store to this GM buffer.
        builder.setInsertionPoint(genOp);
        if (pendingMTE2Writes.count(inputMem)) {
          auto pipeAttr = ascendc::PipeAttr::get(builder.getContext(),
                                                 ascendc::Pipe::PIPE_MTE2);
          builder.create<ascendc::PipeBarrierOp>(loc, pipeAttr);
          pendingMTE2Writes.erase(inputMem);
        }

        // Allocate the VECIN buffer outside the inner loop so InitBuffer is
        // invoked only once per block.
        Value vecin;
        if (innerFor) {
          builder.setInsertionPoint(innerFor);
          auto elemTy = cast<MemRefType>(inputMem.getType()).getElementType();
          vecin = allocOnChipSized(builder, loc, innerFor.getStep(), elemTy,
                                   /*VECIN=*/9)
                      .getResult();
        } else {
          builder.setInsertionPoint(genOp);
          vecin = allocOnChip(builder, loc, inputMem, /*VECIN=*/9).getResult();
        }

        // Insert the GM→VECIN copy inside the loop (fills current sub-tile).
        builder.setInsertionPoint(genOp);
        builder.create<memref::CopyOp>(loc, inputMem, vecin);
        inputOperand->set(vecin);
      }

      // ---- Promote output: GM → VECOUT ----
      // Alloc outside the inner loop; copy after genOp (inside loop).
      OpOperand *outOperand = genOp.getDpsInitOperand(0);
      Value gmOut = outOperand->get();

      Value vecout;
      if (innerFor) {
        builder.setInsertionPoint(innerFor);
        auto elemTy = cast<MemRefType>(gmOut.getType()).getElementType();
        vecout = allocOnChipSized(builder, loc, innerFor.getStep(), elemTy,
                                  /*VECOUT=*/10)
                     .getResult();
      } else {
        builder.setInsertionPoint(genOp);
        vecout = allocOnChip(builder, loc, gmOut, /*VECOUT=*/10).getResult();
      }
      outOperand->set(vecout);

      // Insert VECOUT → GM copy after the generic (inside the loop).
      builder.setInsertionPointAfter(genOp);
      builder.create<memref::CopyOp>(loc, vecout, gmOut);

      // Record that gmOut now has a pending MTE2 write (local→GM).
      pendingMTE2Writes[gmOut] = vecout;

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
