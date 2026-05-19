#include "CubeEmitter.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cube-emitter"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

namespace {

// Identify the body ops of a Cube fused kernel.  Phase 4a assumption: exactly
// one matmul-like op followed by ≥0 trailing pointwise generics on tensors.
// The "last consumer" is the func's return value's defining op.
struct CubeBodyOps {
  linalg::LinalgOp matmul;
  linalg::LinalgOp lastConsumer; // last op in topological / use order
};

static CubeBodyOps findCubeBodyOps(func::FuncOp func) {
  CubeBodyOps r;
  SmallVector<linalg::LinalgOp> allLinalg;
  func.walk([&](linalg::LinalgOp op) {
    if (isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
            linalg::MatmulTransposeBOp, linalg::BatchMatmulOp>(
            op.getOperation())) {
      r.matmul = op;
    }
    allLinalg.push_back(op);
  });
  if (!allLinalg.empty()) r.lastConsumer = allLinalg.back();
  return r;
}

// Read the 5 cube tunables from plan.tileable.  Set the named field if found,
// leave null otherwise (caller checks).
struct CubeTunables {
  Value xblockM, mInner, xblockN, nInner, kInner;
  bool complete() const {
    return xblockM && mInner && xblockN && nInner && kInner;
  }
};
static CubeTunables readCubeTunables(const TilePlan &plan) {
  CubeTunables t;
  for (const auto &grp : plan.tileable) {
    for (const auto &tp : grp) {
      if      (tp.name == "XBLOCK_M") t.xblockM = tp.ssa;
      else if (tp.name == "M_INNER")  t.mInner  = tp.ssa;
      else if (tp.name == "XBLOCK_N") t.xblockN = tp.ssa;
      else if (tp.name == "N_INNER")  t.nInner  = tp.ssa;
      else if (tp.name == "K_INNER")  t.kInner  = tp.ssa;
    }
  }
  return t;
}

// Build a tile-size vector for `op`, with M/N/K slots populated from
// `tunables` if the op has them.  Iteration dim positions for matmul:
//   linalg.matmul        iter = (M, N, K)             → [m, n, k]
//   linalg.generic (vec) iter = (parallel, parallel)  → [m, n]  (M & N)
// Other iter shapes return empty (caller skips).
static SmallVector<OpFoldResult>
buildTileSizes(linalg::LinalgOp op, Value mSize, Value nSize, Value kSize,
                OpBuilder &b) {
  SmallVector<OpFoldResult> sizes;
  OpFoldResult zero = b.getIndexAttr(0);
  unsigned rank = op.getIteratorTypesArray().size();
  if (isa<linalg::MatmulOp>(op.getOperation()) && rank == 3) {
    sizes = {mSize ? OpFoldResult(mSize) : zero,
              nSize ? OpFoldResult(nSize) : zero,
              kSize ? OpFoldResult(kSize) : zero};
  } else if (rank == 2) {
    // Pointwise / elementwise vec epilogue: [M, N] only.
    sizes = {mSize ? OpFoldResult(mSize) : zero,
              nSize ? OpFoldResult(nSize) : zero};
  } else {
    // Unsupported rank / op kind — caller skips.
  }
  return sizes;
}

} // namespace

LogicalResult emitCubeKernel(func::FuncOp func, const TilePlan &plan) {
  if (plan.cubeKind == CubeKind::None)
    return success(); // nothing to do for vector plans

  CubeBodyOps body = findCubeBodyOps(func);
  if (!body.matmul) {
    LLVM_DEBUG(llvm::dbgs() << "[cube-emitter] no matmul op found in "
                            << func.getName() << "\n");
    return failure();
  }
  if (!body.lastConsumer) body.lastConsumer = body.matmul;

  CubeTunables tunables = readCubeTunables(plan);
  if (!tunables.complete()) {
    LLVM_DEBUG(llvm::dbgs() << "[cube-emitter] incomplete cube tunables for "
                            << func.getName() << "\n");
    return failure();
  }

  // Phase 4a: level-1 tile only — tile the lastConsumer on the M/N (outer)
  // parallel axes, fuse the matmul producer in.  Inner M/N + K tiles land in
  // Phase 4b/c.  Result: scf.for M_outer { scf.for N_outer { matmul; vec... } }
  IRRewriter rewriter(func.getContext());
  rewriter.setInsertionPoint(body.lastConsumer);

  OpBuilder b(rewriter.getContext());
  b.setInsertionPoint(body.lastConsumer);
  SmallVector<OpFoldResult> tileSizes =
      buildTileSizes(body.lastConsumer, tunables.xblockM, tunables.xblockN,
                      /*kSize=*/nullptr, b);
  if (tileSizes.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[cube-emitter] unsupported consumer iter\n");
    return failure();
  }

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);
  scf::SCFTileAndFuseOptions tfOptions;
  tfOptions.tilingOptions = tilingOptions;

  auto consumerOp = cast<TilingInterface>(body.lastConsumer.getOperation());
  auto tfResult = scf::tileConsumerAndFuseProducersUsingSCF(
      rewriter, consumerOp, tfOptions);
  if (failed(tfResult)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cube-emitter] tile-and-fuse level-1 failed\n");
    return failure();
  }

  // Replace the original consumer's uses with the tiled result.
  for (auto [orig, repl] : tfResult->replacements)
    rewriter.replaceAllUsesWith(orig, repl);

  // Annotate the 2 outer loops with `ascendc.parallel`.  Phase 4b will add
  // the dataflow prologue/epilogue strings consumed by AscendCBufferPlacement.
  auto unitAttr = rewriter.getUnitAttr();
  for (auto loopLike : tfResult->loops)
    loopLike->setAttr("ascendc.parallel", unitAttr);

  // Phase 4b: level-2 inner M/N tile.  Pick the tiled consumer (the first
  // entry in `tiledAndFusedOps`) — that's the relu/elementwise op inside the
  // 2-level outer scf.for nest.  tileConsumerAndFuseProducersUsingSCF again
  // with [M_INNER, N_INNER] inserts 2 more scf.for loops AND re-fuses the
  // matmul producer (now also tiled twice, with M_INNER × N_INNER × K).
  // Inner loops do NOT get `ascendc.parallel` (intra-block, not multicore).
  Operation *level1Consumer = tfResult->tiledAndFusedOps.empty()
                                  ? nullptr
                                  : tfResult->tiledAndFusedOps.front();
  if (level1Consumer && isa<linalg::LinalgOp>(level1Consumer)) {
    auto level1ConsumerLinalg = cast<linalg::LinalgOp>(level1Consumer);
    SmallVector<OpFoldResult> tileSizes2 =
        buildTileSizes(level1ConsumerLinalg, tunables.mInner, tunables.nInner,
                        /*kSize=*/nullptr, b);
    if (!tileSizes2.empty()) {
      scf::SCFTilingOptions tilingOptions2;
      tilingOptions2.setTileSizes(tileSizes2);
      scf::SCFTileAndFuseOptions tfOptions2;
      tfOptions2.tilingOptions = tilingOptions2;

      rewriter.setInsertionPoint(level1Consumer);
      auto tfResult2 = scf::tileConsumerAndFuseProducersUsingSCF(
          rewriter, cast<TilingInterface>(level1Consumer), tfOptions2);
      if (succeeded(tfResult2)) {
        for (auto [orig, repl] : tfResult2->replacements)
          rewriter.replaceAllUsesWith(orig, repl);
        // Re-annotate ascendc.unit on the level-2 tiled ops (the level-1
        // annotations were on now-replaced ops).
        for (Operation *tiled : tfResult2->tiledAndFusedOps) {
          if (isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
                  linalg::MatmulTransposeBOp, linalg::BatchMatmulOp>(tiled))
            tiled->setAttr("ascendc.unit",
                            rewriter.getStringAttr("AiCore.Cube"));
          else if (isa<linalg::LinalgOp>(tiled))
            tiled->setAttr("ascendc.unit",
                            rewriter.getStringAttr("AiCore.Vector"));
        }
        // tfResult is now stale (level-1 tiled ops replaced by level-2);
        // mark by clearing so any later annotate-by-tfResult is a no-op.
        // (We've already annotated parallel on the level-1 loops above —
        // those loops survive untouched as the outer wrapper of level-2.)
        return success();
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "[cube-emitter] level-2 tile-and-fuse failed; "
                 << "leaving level-1 output\n");
    }
  }

  // Fallback: only level-1 tile.  Annotate `ascendc.unit` on the tiled
  // matmul / vec generic from the level-1 result.
  for (Operation *tiled : tfResult->tiledAndFusedOps) {
    if (isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
            linalg::MatmulTransposeBOp, linalg::BatchMatmulOp>(tiled))
      tiled->setAttr("ascendc.unit",
                     rewriter.getStringAttr("AiCore.Cube"));
    else if (isa<linalg::LinalgOp>(tiled))
      tiled->setAttr("ascendc.unit",
                     rewriter.getStringAttr("AiCore.Vector"));
  }

  return success();
}

} // namespace mlir::afir
