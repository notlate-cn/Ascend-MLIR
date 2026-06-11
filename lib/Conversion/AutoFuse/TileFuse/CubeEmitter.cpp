#include "CubeEmitter.h"
#include "TileFuseUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
using namespace mlir::auto_fuse;

namespace mlir::afir {

namespace {

// Identify the body ops of a Cube fused kernel.  Phase 4a assumption: exactly
// one matmul-like op followed by ≥0 trailing pointwise generics on tensors.
// The "last consumer" is the func's return value's defining op.
struct CubeBodyOps {
  linalg::LinalgOp matmul;
  linalg::LinalgOp lastConsumer; // last op in topological / use order
};

// isMatmulGeneric (loose form: shape + multiply-accumulate body, no map check)
// is shared via TileFuseUtils.h.

static CubeBodyOps findCubeBodyOps(func::FuncOp func) {
  CubeBodyOps r;
  SmallVector<linalg::LinalgOp> allLinalg;
  func.walk([&](linalg::LinalgOp op) {
    Operation *raw = op.getOperation();
    if (isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
            linalg::MatmulTransposeBOp, linalg::BatchMatmulOp>(raw)) {
      r.matmul = op;
    } else if (auto gen = dyn_cast<linalg::GenericOp>(raw)) {
      // After --linalg-generalize-named-ops, the matmul is a generic;
      // recognize by iter-types + body pattern.
      if (!r.matmul && isMatmulGeneric(gen))
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

  // Annotate the 2 outer loops with `ascendc.parallel` + dataflow strings
  // consumed by AscendCBufferPlacement.  The outer loops route:
  //   prologue: GM → A1/B1 (cube inputs), GM → VECIN (vec inputs, e.g. bias)
  //   epilogue: VECOUT → GM (vec output back to global memory)
  // Phase 4c will add the level-3 K loop with `lhs:A1->A2,rhs:B1->B2` and
  // `acc:CO1->VECIN`; for the 2-level (no-K-tile) shape this just routes
  // the level-2-tiled matmul's M_INNER × N_INNER tile straight to/from GM.
  auto unitAttr = rewriter.getUnitAttr();
  // Outermost loop gets the full prologue/epilogue strings; the other parallel
  // loop only carries `ascendc.parallel` (matches matmul-add-leakyrelu shape).
  for (size_t i = 0; i < tfResult->loops.size(); ++i) {
    Operation *loopOp = tfResult->loops[i];
    loopOp->setAttr("ascendc.parallel", unitAttr);
    if (i == 0) {
      loopOp->setAttr(
          "ascendc.prologue",
          rewriter.getStringAttr("lhs:GM->A1,rhs:GM->B1"));
      loopOp->setAttr(
          "ascendc.epilogue",
          rewriter.getStringAttr("result:VECOUT->GM"));
    }
  }

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
        // annotations were on now-replaced ops).  Matmul is detected via
        // named-op kind OR generic-form pattern (post linalg-generalize).
        Operation *level2Matmul = nullptr;
        for (Operation *tiled : tfResult2->tiledAndFusedOps) {
          bool isCube =
              isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
                  linalg::MatmulTransposeBOp, linalg::BatchMatmulOp>(tiled);
          if (!isCube)
            if (auto gen = dyn_cast<linalg::GenericOp>(tiled))
              isCube = isMatmulGeneric(gen);
          if (isCube) {
            tiled->setAttr("ascendc.unit",
                            rewriter.getStringAttr("AiCore.Cube"));
            // Phase 1 CV-fusion groups have exactly one matmul.  If a future
            // multi-matmul group reaches here, the K-tile below would only
            // tile the last one and silently mis-emit the rest — flag it.
            assert(!level2Matmul &&
                   "CubeEmitter: >1 matmul in one cube group is unsupported");
            level2Matmul = tiled;
          } else if (isa<linalg::LinalgOp>(tiled)) {
            tiled->setAttr("ascendc.unit",
                            rewriter.getStringAttr("AiCore.Vector"));
          }
        }

        // Phase 4c: level-3 K tile on the now-doubly-tiled matmul itself.
        // tileUsingSCF (single-op tile, not tile-and-fuse) with [0, 0, K_INNER]
        // wraps the matmul body in an scf.for over K, with the accumulator
        // threaded as iter_args.  Tag the K loop with dataflow
        // `lhs:A1->A2,rhs:B1->B2` / `acc:CO1->VECIN` so AscendCBufferPlacement
        // inserts the A1→A2 / B1→B2 transitions (needed for matmul → mmad
        // lowering in LinalgToAscendC).
        if (level2Matmul && tunables.kInner) {
          rewriter.setInsertionPoint(level2Matmul);
          scf::SCFTilingOptions ktilOptions;
          OpFoldResult zero = b.getIndexAttr(0);
          ktilOptions.setTileSizes(
              {zero, zero, OpFoldResult(tunables.kInner)});
          auto ktilResult = scf::tileUsingSCF(
              rewriter, cast<TilingInterface>(level2Matmul), ktilOptions);
          if (succeeded(ktilResult)) {
            for (auto [orig, repl] : llvm::zip(level2Matmul->getResults(),
                                                ktilResult->replacements))
              rewriter.replaceAllUsesWith(orig, repl);
            // Re-stamp ascendc.unit on the inner-tiled matmul (level-3
            // emits a new matmul inside the K loop).
            for (Operation *tiled : ktilResult->tiledOps) {
              bool isCube =
                  isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
                      linalg::MatmulTransposeBOp, linalg::BatchMatmulOp>(
                      tiled);
              if (!isCube)
                if (auto gen = dyn_cast<linalg::GenericOp>(tiled))
                  isCube = isMatmulGeneric(gen);
              if (isCube)
                tiled->setAttr("ascendc.unit",
                                rewriter.getStringAttr("AiCore.Cube"));
            }
            // Annotate the K loop with the cube-internal dataflow strings.
            // A degenerate K tile (kInner==0 or evenly dividing such that
            // tileUsingSCF emits no loop) leaves `loops` empty — the
            // dataflow annotations would then be silently dropped and
            // AscendCBufferPlacement won't see the A1→A2/B1→B2 transitions.
            if (ktilResult->loops.empty())
              LLVM_DEBUG(llvm::dbgs()
                         << "[cube-emitter] K-tile produced no scf loop "
                            "(degenerate kInner?); cube dataflow annotations "
                            "skipped\n");
            for (auto loopLike : ktilResult->loops) {
              loopLike->setAttr(
                  "ascendc.prologue",
                  rewriter.getStringAttr("lhs:A1->A2,rhs:B1->B2"));
              loopLike->setAttr(
                  "ascendc.epilogue",
                  rewriter.getStringAttr("acc:CO1->VECIN"));
            }
          }
        }
        return success();
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "[cube-emitter] level-2 tile-and-fuse failed; "
                 << "leaving level-1 output\n");
    }
  }

  // Fallback: only level-1 tile.  Annotate `ascendc.unit` on the tiled
  // matmul / vec generic from the level-1 result.  Matmul detected via
  // named-op kind OR generic-form pattern.
  for (Operation *tiled : tfResult->tiledAndFusedOps) {
    bool isCube =
        isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
            linalg::MatmulTransposeBOp, linalg::BatchMatmulOp>(tiled);
    if (!isCube)
      if (auto gen = dyn_cast<linalg::GenericOp>(tiled))
        isCube = isMatmulGeneric(gen);
    if (isCube)
      tiled->setAttr("ascendc.unit",
                     rewriter.getStringAttr("AiCore.Cube"));
    else if (isa<linalg::LinalgOp>(tiled))
      tiled->setAttr("ascendc.unit",
                     rewriter.getStringAttr("AiCore.Vector"));
  }

  return success();
}

} // namespace mlir::afir
