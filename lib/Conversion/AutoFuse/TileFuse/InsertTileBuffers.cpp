#include "Conversion/AutoFuse/AutoFusePasses.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define GEN_PASS_DECL_AUTOFUSEINSERTTILEBUFFERS
#define GEN_PASS_DEF_AUTOFUSEINSERTTILEBUFFERS
#include "Conversion/Passes.h.inc"

#define DEBUG_TYPE "auto-fuse-insert-tile-buffers"

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

// Allocate an on-chip memref whose shape matches `src`, using the subview's
// explicit size operands.  `hoistPoint` is the operation BEFORE which the
// alloc will be inserted; size operands that do not dominate it are re-derived
// by cloning their defining memref.DimOp at the hoist point (valid because
// the DimOp's source is always a function argument).
static memref::AllocOp allocOnChipMatchingSubview(OpBuilder &b, Location loc,
                                                   Value src, int64_t memSpace,
                                                   Operation *hoistPoint) {
  auto mrt = cast<MemRefType>(src.getType());
  Attribute spaceAttr =
      IntegerAttr::get(IntegerType::get(b.getContext(), 64), memSpace);

  if (auto sv = src.getDefiningOp<memref::SubViewOp>()) {
    DominanceInfo di;
    SmallVector<int64_t> resultShape;
    SmallVector<Value> dynSizes;
    for (OpFoldResult sz : sv.getMixedSizes()) {
      if (auto cst = getConstantIntValue(sz)) {
        resultShape.push_back(*cst);
      } else {
        Value szVal = cast<Value>(sz);
        resultShape.push_back(ShapedType::kDynamic);
        // If szVal is defined inside the loop (doesn't dominate hoistPoint),
        // try to re-derive it.  The common case is a memref.DimOp whose
        // source argument dominates everywhere; the source may also be an
        // scf.for iter_arg, whose shape equals its init operand's (the loop
        // body cannot retype a memref), so we walk the iter_arg chain
        // outward until we hit a value that dominates the hoist point.
        if (!di.dominates(szVal, hoistPoint)) {
          if (auto dimOp = szVal.getDefiningOp<memref::DimOp>()) {
            Value dimSrc = dimOp.getSource();
            while (auto ba = dyn_cast<BlockArgument>(dimSrc)) {
              auto forOp =
                  dyn_cast_or_null<scf::ForOp>(ba.getOwner()->getParentOp());
              if (!forOp || ba.getArgNumber() == 0) break; // 0 = induction var
              dimSrc = forOp.getInitArgs()[ba.getArgNumber() - 1];
            }
            if (di.dominates(dimSrc, hoistPoint) &&
                di.dominates(dimOp.getIndex(), hoistPoint)) {
              szVal = b.create<memref::DimOp>(loc, dimSrc, dimOp.getIndex());
            }
          }
        }
        dynSizes.push_back(szVal);
      }
    }
    auto spacedType = MemRefType::get(resultShape, mrt.getElementType(),
                                      MemRefLayoutAttrInterface{}, spaceAttr);
    return b.create<memref::AllocOp>(loc, spacedType, dynSizes);
  }

  // Fallback: memref.dim at the current insertion point.
  auto spacedType = MemRefType::get(mrt.getShape(), mrt.getElementType(),
                                    MemRefLayoutAttrInterface{}, spaceAttr);
  SmallVector<Value> dynSizes;
  for (unsigned d = 0, rank = mrt.getRank(); d < rank; ++d)
    if (ShapedType::isDynamic(mrt.getShape()[d]))
      dynSizes.push_back(b.create<memref::DimOp>(loc, src, d));
  return b.create<memref::AllocOp>(loc, spacedType, dynSizes);
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

struct AutoFuseInsertTileBuffersPass
    : public ::impl::AutoFuseInsertTileBuffersBase<
          AutoFuseInsertTileBuffersPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());

    // CV-fusion Phase 4: mix kernels go through AscendCBufferPlacement
    // (A1/A2/B1/B2/CO1 memory spaces driven by dataflow annotations) — they
    // must NOT also be wrapped by InsertTileBuffers' vector pipeline
    // VECIN/VECOUT/VECCALC machinery.  Skip the whole func for mix.
    if (auto kk = func->getAttrOfType<StringAttr>("ascendc.kernel_kind"))
      if (kk.getValue() == "mix")
        return;

    // Collect generics whose output is in global memory (memory_space == 0).
    SmallVector<linalg::GenericOp> targets;
    func.walk([&](linalg::GenericOp op) {
      Value out = op.getDpsInitOperand(0)->get();
      if (op.getNumDpsInits() == 1 && getMemSpace(out.getType()) == 0)
        targets.push_back(op);
    });

    // Track GM values that have pending writes (UB→GM, MTE3).  When a
    // subsequent op reads from the same GM address, we insert PipeBarrier(ALL)
    // to ensure both the UB→GM store (MTE3) AND any preceding vector pipeline
    // (PIPE_V) operations (e.g., Broadcast internal PipeBarrier(PIPE_V)) have
    // fully committed before the GM→UB reload.  PIPE_MTE3 alone is insufficient
    // when the store chain includes PIPE_V operations that may not yet be
    // visible to the MTE3 engine.
    llvm::SmallDenseMap<Value, Value> pendingMTE3Writes; // GM value → vecout

    for (linalg::GenericOp genOp : targets) {
      Location loc = genOp.getLoc();

      // Determine where to anchor on-chip allocs.  Two cases:
      //   (1) genOp is inside an scf.for sub-tile loop — hoist allocs BEFORE
      //       that for so AscendC's TPipe::InitBuffer is called once per
      //       block rather than once per sub-tile (bump-allocator leak).
      //   (2) genOp is inside an scf.if then/else block (the Case-C tail
      //       peel block) and not further inside an inner for — the if
      //       runs at most once per kernel, so the alloc lives in the if
      //       block.  It must be placed AFTER any dynamic size operands
      //       (e.g. tailSize) defined inside the if, so we insert right
      //       before genOp.
      auto innerFor = genOp->getParentOfType<scf::ForOp>();
      Operation *allocAnchor = innerFor.getOperation();
      {
        Operation *cur = genOp->getParentOp();
        while (cur) {
          if (isa<scf::ForOp>(cur)) break;
          if (isa<scf::IfOp>(cur)) {
            allocAnchor = genOp.getOperation();
            break;
          }
          cur = cur->getParentOp();
        }
      }

      // ---- Promote inputs: GM → VECIN ----
      // Allocs go before the inner loop; copies and barriers go before genOp.
      for (OpOperand *inputOperand : genOp.getDpsInputOperands()) {
        Value inputMem = inputOperand->get();
        if (getMemSpace(inputMem.getType()) != 0)
          continue;

        // Insert PipeBarrier(ALL) before genOp if there is a pending write
        // (UB→GM) to this GM buffer.  PIPE_ALL ensures both the MTE3 store and
        // any preceding PIPE_V operations complete before the GM→UB reload.
        builder.setInsertionPoint(genOp);
        if (pendingMTE3Writes.count(inputMem)) {
          auto pipeAttr = ascendc::PipeAttr::get(builder.getContext(),
                                                 ascendc::Pipe::PIPE_ALL);
          builder.create<ascendc::PipeBarrierOp>(loc, pipeAttr);
          pendingMTE3Writes.erase(inputMem);
        }

        // Allocate the VECIN buffer outside the inner loop so InitBuffer is
        // invoked only once per block.  Always use the subview-matching
        // allocator so the buffer extent is read from the subview's actual
        // size operands.  The previous rank-1 special case used
        // `innerFor.getStep()`, which is only correct when the rank-1
        // operand's axis is exactly the innermost loop's axis — for
        // multi-axis broadcast operands (e.g. `a[d1]` with d2 as the
        // innermost loop) it would size the tile by the wrong dim.
        Value vecin;
        if (allocAnchor) {
          builder.setInsertionPoint(allocAnchor);
          vecin = allocOnChipMatchingSubview(builder, loc, inputMem,
                                              /*VECIN=*/9, allocAnchor)
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
      if (allocAnchor) {
        builder.setInsertionPoint(allocAnchor);
        vecout = allocOnChipMatchingSubview(builder, loc, gmOut,
                                            /*VECOUT=*/10, allocAnchor)
                     .getResult();
      } else {
        builder.setInsertionPoint(genOp);
        vecout = allocOnChip(builder, loc, gmOut, /*VECOUT=*/10).getResult();
      }
      outOperand->set(vecout);

      // Insert VECOUT → GM copy after the generic (inside the loop).
      builder.setInsertionPointAfter(genOp);
      builder.create<memref::CopyOp>(loc, vecout, gmOut);

      // Record that gmOut now has a pending write (UB→GM).
      pendingMTE3Writes[gmOut] = vecout;

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

std::unique_ptr<Pass> createAutoFuseInsertTileBuffersPass() {
  return std::make_unique<AutoFuseInsertTileBuffersPass>();
}

} // namespace mlir::afir
