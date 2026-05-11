#include "Conversion/VectorPlan/VectorPlanPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define GEN_PASS_DECL_VECTORPLANFOLDSHADOWALLOC
#define GEN_PASS_DEF_VECTORPLANFOLDSHADOWALLOC
#include "Conversion/Passes.h.inc"

#define DEBUG_TYPE "vector-plan-fold-shadow-alloc"

using namespace mlir;

namespace mlir::afir {

namespace {

// True when memref has no explicit memory_space attribute or the attribute
// integer-encodes 0 (== GM in our convention).
static bool isGmMemref(Value v) {
  auto mrt = dyn_cast<MemRefType>(v.getType());
  if (!mrt) return false;
  Attribute space = mrt.getMemorySpace();
  if (!space) return true;
  if (auto ia = dyn_cast<IntegerAttr>(space))
    return ia.getInt() == 0;
  return false;
}

// Strip aliasing memref view ops (subview / cast / collapse / expand) to reach
// the "root" memref backing a shadow-copy endpoint.  We need this to verify
// that the pre-copy source and the post-copy destination point at the same
// logical region: bufferize sometimes emits two structurally-identical but
// SSA-distinct subview ops, one as the read source and one as the write
// destination.
//
// `peelSubviews` records the chain of subview ops so we can compare offsets/
// sizes/strides element-wise.
namespace {
struct SubviewView {
  Value root;
  SmallVector<memref::SubViewOp> chain; // outermost-first
};
} // namespace

static SubviewView peelToRoot(Value v) {
  SubviewView out;
  while (true) {
    if (auto sv = v.getDefiningOp<memref::SubViewOp>()) {
      out.chain.push_back(sv);
      v = sv.getSource();
      continue;
    }
    break;
  }
  out.root = v;
  return out;
}

static bool sameOpFoldResult(OpFoldResult a, OpFoldResult b) {
  if (a == b) return true;
  auto va = dyn_cast<Value>(a);
  auto vb = dyn_cast<Value>(b);
  if (va && vb && va == vb) return true;
  auto aa = dyn_cast_or_null<Attribute>(dyn_cast<Attribute>(a));
  auto ab = dyn_cast_or_null<Attribute>(dyn_cast<Attribute>(b));
  if (aa && ab && aa == ab) return true;
  return false;
}

static bool subviewSameRegion(memref::SubViewOp a, memref::SubViewOp b) {
  if (a.getSource() != b.getSource()) return false;
  auto am = a.getMixedOffsets(), bm = b.getMixedOffsets();
  if (am.size() != bm.size()) return false;
  for (auto [x, y] : llvm::zip(am, bm))
    if (!sameOpFoldResult(x, y)) return false;
  auto as = a.getMixedSizes(), bs = b.getMixedSizes();
  if (as.size() != bs.size()) return false;
  for (auto [x, y] : llvm::zip(as, bs))
    if (!sameOpFoldResult(x, y)) return false;
  auto at = a.getMixedStrides(), bt = b.getMixedStrides();
  if (at.size() != bt.size()) return false;
  for (auto [x, y] : llvm::zip(at, bt))
    if (!sameOpFoldResult(x, y)) return false;
  return true;
}

// True if `a` and `b` resolve to the same memref region (same root + same
// subview chain element-wise).  Same SSA value is also handled.
static bool sameMemrefRegion(Value a, Value b) {
  if (a == b) return true;
  auto va = peelToRoot(a);
  auto vb = peelToRoot(b);
  if (va.root != vb.root) return false;
  if (va.chain.size() != vb.chain.size()) return false;
  for (auto [x, y] : llvm::zip(va.chain, vb.chain))
    if (!subviewSameRegion(x, y)) return false;
  return true;
}

// Try to fold one shadow-alloc sandwich.  Returns true on success.
//
// Pattern (inside any block):
//
//   %alloc      = memref.alloc(?) : memref<?x?xT>           (GM, mem_space 0)
//   memref.copy %srcSub,   %alloc                            (pre-copy)
//   linalg.generic ... outs(%alloc) ...                      (one user only)
//   memref.copy %alloc,    %dstSub                            (post-copy)
//
//   srcSub and dstSub must refer to the same memref region.
//
// Rewrite: redirect the linalg outs to %dstSub, erase both copies and %alloc.
static bool tryFoldShadowAlloc(memref::AllocOp alloc) {
  if (!isGmMemref(alloc.getResult())) return false;

  memref::CopyOp preCopy = nullptr;   // src -> alloc
  memref::CopyOp postCopy = nullptr;  // alloc -> dst
  linalg::GenericOp generic = nullptr;

  for (Operation *user : alloc->getUsers()) {
    if (auto cp = dyn_cast<memref::CopyOp>(user)) {
      if (cp.getTarget() == alloc.getResult()) {
        if (preCopy) return false;
        preCopy = cp;
      } else if (cp.getSource() == alloc.getResult()) {
        if (postCopy) return false;
        postCopy = cp;
      } else {
        return false;
      }
    } else if (auto g = dyn_cast<linalg::GenericOp>(user)) {
      if (generic) return false;
      // Must be DPS init operand, not input.
      bool isInit = false;
      for (int i = 0, n = g.getNumDpsInits(); i < n; ++i)
        if (g.getDpsInitOperand(i)->get() == alloc.getResult()) {
          isInit = true;
          break;
        }
      if (!isInit) return false;
      generic = g;
    } else {
      return false;
    }
  }

  if (!preCopy || !postCopy || !generic) return false;

  // Verify ordering inside the parent block.  All four ops must live in the
  // same block (they originated from the same `scf.if` tail branch).
  Block *b = alloc->getBlock();
  if (preCopy->getBlock() != b || postCopy->getBlock() != b ||
      generic->getBlock() != b) return false;
  if (!alloc->isBeforeInBlock(preCopy)) return false;
  if (!preCopy->isBeforeInBlock(generic)) return false;
  if (!generic->isBeforeInBlock(postCopy)) return false;

  if (!sameMemrefRegion(preCopy.getSource(), postCopy.getTarget()))
    return false;

  // Shape compatibility: same element type, rank, shape.  Memref layout
  // (identity vs strided<>) is allowed to differ — linalg.generic is layout-
  // agnostic, so redirecting outs from an identity-layout alloc to a strided
  // subview keeps the loop semantics intact.
  auto aTy = cast<MemRefType>(alloc.getResult().getType());
  auto dTy = cast<MemRefType>(postCopy.getTarget().getType());
  if (aTy.getElementType() != dTy.getElementType()) return false;
  if (aTy.getRank() != dTy.getRank()) return false;
  if (aTy.getShape() != dTy.getShape()) return false;

  Value dst = postCopy.getTarget();
  for (int i = 0, n = generic.getNumDpsInits(); i < n; ++i) {
    OpOperand *o = generic.getDpsInitOperand(i);
    if (o->get() == alloc.getResult())
      o->set(dst);
  }

  preCopy.erase();
  postCopy.erase();
  alloc.erase();
  LLVM_DEBUG(llvm::dbgs() << "[fold-shadow-alloc] folded one sandwich\n");
  return true;
}

struct VectorPlanFoldShadowAllocPass
    : public ::impl::VectorPlanFoldShadowAllocBase<
          VectorPlanFoldShadowAllocPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Iterate to a fixed point: folding one sandwich may expose another.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<memref::AllocOp> allocs;
      func.walk([&](memref::AllocOp op) { allocs.push_back(op); });
      for (memref::AllocOp a : allocs) {
        if (tryFoldShadowAlloc(a)) {
          changed = true;
          break; // restart walk; iterator invalidated
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createVectorPlanFoldShadowAllocPass() {
  return std::make_unique<VectorPlanFoldShadowAllocPass>();
}

} // namespace mlir::afir
