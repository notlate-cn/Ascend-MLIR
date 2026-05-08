#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define GEN_PASS_DECL_ASCENDCFLATTENGMPTRPASS
#define GEN_PASS_DEF_ASCENDCFLATTENGMPTRPASS
#include "Conversion/Passes.h.inc"

#define DEBUG_TYPE "ascendc-flatten-gm-ptr"

using namespace mlir;
using namespace mlir::ascendc;
using namespace mlir::emitasc;

namespace mlir::afir {

static constexpr int64_t kGMSpace = 22;

static Value materializeOffset(OpBuilder &b, Location loc, OpFoldResult ofr) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    int64_t v = cast<IntegerAttr>(attr).getValue().getSExtValue();
    return b.create<arith::ConstantIndexOp>(loc, v);
  }
  return ofr.get<Value>();
}

// Walk a value to its root BlockArgument + flat index offset.
// For 2D subviews, uses memref.dim %base, 1 for the column stride.
static std::pair<BlockArgument, Value>
resolveGMChain(Value start, OpBuilder &b, Location loc) {
  Value cur = start;
  Value acc = b.create<arith::ConstantIndexOp>(loc, 0);
  while (true) {
    if (auto sv = cur.getDefiningOp<memref::SubViewOp>()) {
      SmallVector<OpFoldResult> offs = sv.getMixedOffsets();
      if (offs.size() == 1) {
        acc = b.create<arith::AddIOp>(loc, acc,
                                      materializeOffset(b, loc, offs[0]));
        cur = sv.getSource();
        continue;
      }
      if (offs.size() >= 2) {
        Value src = sv.getSource();
        if (auto castOp = src.getDefiningOp<memref::CastOp>())
          src = castOp.getSource();
        auto ba = dyn_cast<BlockArgument>(src);
        if (!ba)
          return {BlockArgument{}, Value{}};
        // Compute flat offset = sum_i(offs[i] * prod_{j=i+1..rank-1} dim(src, j)).
        // offs.size() == rank invariant: InsertTileBuffers always generates
        // full-rank subviews, so this formula is always complete.
        // This correctly handles tensors of any rank (2D, 3D, etc.).
        int64_t rank = cast<MemRefType>(ba.getType()).getRank();
        Value flat = b.create<arith::ConstantIndexOp>(loc, 0);
        for (size_t i = 0; i < offs.size(); ++i) {
          Value stride = b.create<arith::ConstantIndexOp>(loc, 1);
          for (int64_t j = static_cast<int64_t>(i) + 1; j < rank; ++j) {
            Value dimIdx = b.create<arith::ConstantIndexOp>(loc, j);
            stride = b.create<arith::MulIOp>(
                loc, stride, b.create<memref::DimOp>(loc, ba, dimIdx));
          }
          Value off = materializeOffset(b, loc, offs[i]);
          flat = b.create<arith::AddIOp>(
              loc, flat, b.create<arith::MulIOp>(loc, off, stride));
        }
        acc = b.create<arith::AddIOp>(loc, acc, flat);
        return {ba, acc};
      }
      return {BlockArgument{}, Value{}};
    }
    if (auto castOp = cur.getDefiningOp<memref::CastOp>()) {
      cur = castOp.getSource();
      continue;
    }
    // Collapse_shape is a contiguous view: flat index is preserved, look through.
    if (auto colOp = cur.getDefiningOp<memref::CollapseShapeOp>()) {
      cur = colOp.getSrc();
      continue;
    }
    if (auto ba = dyn_cast<BlockArgument>(cur))
      return {ba, acc};
    return {BlockArgument{}, Value{}};
  }
}

// Peel through memref.cast to find the underlying subview (if any).
static memref::SubViewOp peelCastsToSubview(Value v) {
  while (auto castOp = v.getDefiningOp<memref::CastOp>())
    v = castOp.getSource();
  return v.getDefiningOp<memref::SubViewOp>();
}

static void flattenGMPtr(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  Block &entry = func.getBody().front();
  Type i32Ty = IntegerType::get(ctx, 32);

  auto mkFlatTy = [&](Type elem) {
    return MemRefType::get({ShapedType::kDynamic}, elem,
                           MemRefLayoutAttrInterface{},
                           IntegerAttr::get(IntegerType::get(ctx, 32), kGMSpace));
  };

  // ── 7a. Promote top-level GM allocs to func args ──────────────────────────
  DenseMap<unsigned, SmallVector<Value>> promotedArgDynSizes;
  {
    SmallVector<memref::AllocOp> gmAllocs;
    for (Operation &op : entry.without_terminator()) {
      auto allocOp = dyn_cast<memref::AllocOp>(&op);
      if (!allocOp)
        continue;
      if (cast<MemRefType>(allocOp.getResult().getType()).getMemorySpaceAsInt() == 0)
        gmAllocs.push_back(allocOp);
    }
    for (memref::AllocOp allocOp : gmAllocs) {
      auto origTy = cast<MemRefType>(allocOp.getResult().getType());
      // Use C-order strides: innermost is 1, outer dims are dynamic.
      // This lets the runtime correctly infer allocation size = D0*D1*...*Dn.
      // Using all-1 strides would make the runtime compute only D0+D1+...+Dn-1
      // elements (additive), which is far too small for multi-dim allocs.
      SmallVector<int64_t> strides(origTy.getRank(), ShapedType::kDynamic);
      if (!strides.empty()) strides.back() = 1;
      auto stridedLayout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides);
      auto stridedTy = MemRefType::get(origTy.getShape(), origTy.getElementType(),
                                       stridedLayout);
      SmallVector<Value> dynSizes(allocOp.getDynamicSizes());
      BlockArgument newArg = entry.addArgument(stridedTy, allocOp.getLoc());
      promotedArgDynSizes[newArg.getArgNumber()] = dynSizes;
      OpBuilder b(allocOp);
      Value casted = b.create<memref::CastOp>(allocOp.getLoc(), origTy, newArg);
      allocOp.getResult().replaceAllUsesWith(casted);
      allocOp.erase();
    }
  }

  // ── 7b. Flatten subview + set_global_buffer ───────────────────────────────
  // Collect set_global_buffer ops whose buffer leads (through optional casts)
  // to a subview.
  SmallVector<GlobalTensorSetGlobalBufferOp> setGlobalBufferOps;
  func.walk([&](GlobalTensorSetGlobalBufferOp op) {
    if (peelCastsToSubview(op.getBuffer()))
      setGlobalBufferOps.push_back(op);
  });

  for (GlobalTensorSetGlobalBufferOp sgbOp : setGlobalBufferOps) {
    auto subview = peelCastsToSubview(sgbOp.getBuffer());
    if (!subview)
      continue;

    OpBuilder b(sgbOp);
    Location loc = sgbOp.getLoc();
    BlockArgument baseArg;
    Value flatOffset;

    auto [ba, acc] = resolveGMChain(subview.getResult(), b, loc);
    if (!ba)
      continue;
    baseArg = ba;
    flatOffset = acc;

    Value flatOffsetI32 = b.create<arith::IndexCastOp>(loc, i32Ty, flatOffset);
    Type elemTy = cast<MemRefType>(baseArg.getType()).getElementType();
    Value flatBase = b.create<emitasc::ReinterpretCastOp>(loc, mkFlatTy(elemTy), baseArg);
    b.create<GlobalTensorSetGlobalBufferOp>(loc, sgbOp.getTensor(), flatBase,
                                            flatOffsetI32);
    sgbOp.erase();
    if (subview.use_empty())
      subview.erase();
  }

  // Clean up dead subview/cast/collapse/expand chains; repeat until stable.
  {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> dead;
      func.walk([&](Operation *op) {
        if (op->use_empty() &&
            isa<memref::SubViewOp, memref::CastOp,
                memref::CollapseShapeOp, memref::ExpandShapeOp>(op))
          dead.push_back(op);
      });
      for (Operation *op : dead) { op->erase(); changed = true; }
    }
  }

  // ── 7c. GM→GM memref.copy → memmove ─────────────────────────────────────
  {
    SmallVector<memref::CopyOp> gmCopies;
    func.walk([&](memref::CopyOp op) {
      if (cast<MemRefType>(op.getSource().getType()).getMemorySpaceAsInt() == 0 &&
          cast<MemRefType>(op.getTarget().getType()).getMemorySpaceAsInt() == 0)
        gmCopies.push_back(op);
    });
    for (memref::CopyOp copyOp : gmCopies) {
      OpBuilder b(copyOp);
      Location loc = copyOp.getLoc();
      auto [srcArg, srcOff] = resolveGMChain(copyOp.getSource(), b, loc);
      auto [dstArg, dstOff] = resolveGMChain(copyOp.getTarget(), b, loc);
      if (!srcArg || !dstArg) {
        LLVM_DEBUG(llvm::dbgs() << "[flatten-gm-ptr] unresolved GM copy\n");
        continue;
      }
      Type srcElem = cast<MemRefType>(srcArg.getType()).getElementType();
      Type dstElem = cast<MemRefType>(dstArg.getType()).getElementType();
      Value srcBase = b.create<emitasc::ReinterpretCastOp>(loc, mkFlatTy(srcElem), srcArg);
      Value dstBase = b.create<emitasc::ReinterpretCastOp>(loc, mkFlatTy(dstElem), dstArg);
      Value srcPtr = b.create<emitasc::PtrOffsetOp>(
          loc, mkFlatTy(srcElem), srcBase, IntegerAttr{}, srcOff);
      Value dstPtr = b.create<emitasc::PtrOffsetOp>(
          loc, mkFlatTy(dstElem), dstBase, IntegerAttr{}, dstOff);
      // Byte count from promoted alloc's dynamic sizes, or static shape.
      Value byteCount;
      auto dynIt = promotedArgDynSizes.find(srcArg.getArgNumber());
      if (dynIt != promotedArgDynSizes.end() && !dynIt->second.empty()) {
        Value count = b.create<arith::ConstantIndexOp>(loc, 1);
        for (Value sz : dynIt->second)
          count = b.create<arith::MulIOp>(loc, count, sz);
        int64_t elemBytes = srcElem.getIntOrFloatBitWidth() / 8;
        byteCount = b.create<arith::MulIOp>(
            loc, count, b.create<arith::ConstantIndexOp>(loc, elemBytes));
      } else {
        auto mrt = cast<MemRefType>(copyOp.getSource().getType());
        int64_t staticElems = 1;
        bool ok = true;
        for (int64_t d : mrt.getShape()) {
          if (d == ShapedType::kDynamic) {
            ok = false;
            break;
          }
          staticElems *= d;
        }
        if (!ok)
          continue;
        byteCount = b.create<arith::ConstantIndexOp>(
            loc, staticElems * (srcElem.getIntOrFloatBitWidth() / 8));
      }
      b.create<emitasc::VerbatimOp>(loc, b.getStringAttr("memmove($1, $2, $3)"),
                                    ValueRange{dstPtr, srcPtr, byteCount});
      copyOp.erase();
    }
    SmallVector<Operation *> dead;
    func.walk([&](Operation *op) {
      if (op->use_empty() && isa<memref::SubViewOp, memref::CastOp>(op))
        dead.push_back(op);
    });
    for (Operation *op : dead)
      op->erase();
  }

  // Update function type and clear arg_attrs if new args were added (the
  // entry.addArgument calls above do not update FuncOp::arg_attrs, so remove
  // it to avoid a verifier mismatch when intermediate buffers are promoted).
  SmallVector<Type> newArgTypes;
  for (BlockArgument arg : entry.getArguments())
    newArgTypes.push_back(arg.getType());
  func.setFunctionType(FunctionType::get(ctx, newArgTypes,
                                         func.getFunctionType().getResults()));
  if (!promotedArgDynSizes.empty())
    func->removeAttr("arg_attrs");
}

struct AscendCFlattenGMPtrPass
    : public ::impl::AscendCFlattenGMPtrPassBase<AscendCFlattenGMPtrPass> {
  using AscendCFlattenGMPtrPassBase::AscendCFlattenGMPtrPassBase;
  void runOnOperation() override { flattenGMPtr(getOperation()); }
};

std::unique_ptr<Pass> createAscendCFlattenGMPtrPass() {
  return std::make_unique<AscendCFlattenGMPtrPass>();
}

} // namespace mlir::afir
