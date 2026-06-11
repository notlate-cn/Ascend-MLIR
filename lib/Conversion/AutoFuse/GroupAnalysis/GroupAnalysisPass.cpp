#include "AxisLattice.h"
#include "CanFuse.h"
#include "FusionGroup.h"
#include "Conversion/AutoFuse/GroupInfo.h"
#include "Conversion/AutoFuse/AutoFusePasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <cstring>

#define GEN_PASS_DECL_AUTOFUSEGROUPANALYSIS
#define GEN_PASS_DEF_AUTOFUSEGROUPANALYSIS
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::auto_fuse;
using namespace mlir::linalg;

namespace mlir::afir {

namespace {

//===----------------------------------------------------------------------===//
// Helper: determine if a linalg op is a matmul-like cube op
//===----------------------------------------------------------------------===//

static bool isCubeOp(linalg::LinalgOp op) {
  // Explicit matmul / conv ops (cube-class; lowered to aclnn fallback).
  if (isa<linalg::MatmulOp, linalg::BatchMatmulOp,
          linalg::Conv2DNchwFchwOp>(op.getOperation()))
    return true;
  // Annotated cube ops
  if (op.getOperation()->hasAttr("ascendc.unit")) {
    // Check if it's a cube unit
    auto attr = op.getOperation()->getAttrOfType<StringAttr>("ascendc.unit");
    if (attr && attr.getValue() == "Cube")
      return true;
  }
  // linalg.generic with reduction and 3 operands shaped as matmul
  if (auto generic = dyn_cast<linalg::GenericOp>(op.getOperation())) {
    // Has exactly one reduction axis and two parallel axes -> likely matmul
    auto iters = op.getIteratorTypesArray();
    if (iters.size() == 3) {
      int redCount = 0, parCount = 0;
      for (auto it : iters) {
        if (it == utils::IteratorType::reduction) ++redCount;
        if (it == utils::IteratorType::parallel)  ++parCount;
      }
      if (redCount == 1 && parCount == 2 && op.getNumDpsInputs() == 2 &&
          op.getNumDpsInits() == 1)
        return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// FusionGroup management
//===----------------------------------------------------------------------===//

static void rebuildGroup(FusionGroup &g, func::FuncOp func) {
  g.canonicalAxes = computeCanonicalAxes(g.members);
  g.boundaryIn    = collectBoundaryIn(g.members, func);
}

static void mergeInto(FusionGroup &g1, FusionGroup &g2, func::FuncOp func) {
  // Move g2 members into g1
  for (auto op : g2.members)
    g1.members.push_back(op);
  // Preserve g1.id, mark g2 as removed
  g2.id = -1;
  // Update kind: if either was Cube, merged is Cube
  if (g2.kind == GroupInfo::Kind::Cube)
    g1.kind = GroupInfo::Kind::Cube;
  // Recompute
  rebuildGroup(g1, func);
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

struct AutoFuseGroupAnalysisPass
    : public ::impl::AutoFuseGroupAnalysisBase<AutoFuseGroupAnalysisPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Step 0-pre: Unshare multi-use tensor.empty ops.
    //
    // torch-mlir export CSE's tensor.empty by type — e.g. 12 layers' QKV
    // projection transposes all output [64x192] and end up sharing one
    // %empty. In SSA tensor semantics this is sound (each linalg op
    // returns a fresh value), but after bufferization the empty becomes
    // one memref with N writes; only the last wins, prior results read
    // garbage (NaN). When CanFuse later allows fusing those N independent
    // ops into one kernel (e.g. horizontal transpose fusion), the bug
    // surfaces as NaN throughout the network.
    //
    // Fix: rematerialize a fresh tensor.empty for every use after the
    // first. tensor.empty is value-less so cloning is correctness-safe;
    // bufferization downstream gets one memref per op, eliminating the
    // shared-write hazard.
    {
      SmallVector<tensor::EmptyOp> shared;
      func.walk([&](tensor::EmptyOp e) {
        if (!e.getResult().use_empty() && !e.getResult().hasOneUse())
          shared.push_back(e);
      });
      for (tensor::EmptyOp e : shared) {
        SmallVector<OpOperand *> uses;
        for (OpOperand &u : e.getResult().getUses())
          uses.push_back(&u);
        OpBuilder b(e);
        b.setInsertionPointAfter(e);
        // Keep the first use bound to the original; clone for the rest.
        for (size_t i = 1; i < uses.size(); ++i) {
          Operation *cloneOp = b.clone(*e.getOperation());
          uses[i]->set(cloneOp->getResult(0));
        }
      }
    }

    // Step 0-pre.2: Constant-fold weight transposes.
    //
    // `linalg.transpose ins(arith.constant<W>) outs(...) perm=P` →
    // `arith.constant<W permuted by P>`. Each transformer layer feeds its
    // Q/K/V/output projection / FFN weights through a transpose before the
    // matmul; those transposes are pure data-shuffles of immutable bytes,
    // perfectly suited to compile-time evaluation. GPT-2 small ⇒ 49 of 97
    // transposes are weight-fed (~50%); folding eliminates one dispatch
    // per matched site at zero runtime risk (no new codegen path).
    //
    // Handles both inline DenseElementsAttr (small constants the parser
    // inlines) and DenseResourceElementsAttr (torch-imported weights via
    // dense_resource blobs). The folded result is always inline dense —
    // downstream emitters (AclnnBackend) bake constant bytes into the
    // generated host source regardless of attr flavor, so the IR-size
    // bloat is washed out by the final binary.
    {
      SmallVector<linalg::TransposeOp> toFold;
      func.walk([&](linalg::TransposeOp tr) {
        Value src = tr.getInput();
        if (!src || !src.getDefiningOp<arith::ConstantOp>())
          return;
        toFold.push_back(tr);
      });
      for (auto tr : toFold) {
        auto cst = cast<arith::ConstantOp>(tr.getInput().getDefiningOp());
        auto srcType = cast<RankedTensorType>(cst.getType());
        auto resType = cast<RankedTensorType>(tr.getResult()[0].getType());
        Type elemType = srcType.getElementType();
        if (!elemType.isIntOrFloat())
          continue;
        unsigned bitWidth = elemType.getIntOrFloatBitWidth();
        if (bitWidth == 0 || bitWidth % 8 != 0)
          continue; // skip non-byte-aligned (i1 packing)
        unsigned elemBytes = bitWidth / 8;

        // Pull raw bytes from either inline-dense or dense_resource.
        ArrayRef<char> srcRaw;
        if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue())) {
          if (dense.isSplat())
            continue; // splat is permutation-invariant
          srcRaw = dense.getRawData();
        } else if (auto resAttr = dyn_cast<DenseResourceElementsAttr>(
                       cst.getValue())) {
          auto *blob = resAttr.getRawHandle().getBlob();
          if (!blob)
            continue;
          srcRaw = blob->getData();
        } else {
          continue;
        }

        int64_t numEl = srcType.getNumElements();
        if (numEl == 0 ||
            int64_t(srcRaw.size()) < numEl * int64_t(elemBytes))
          continue;

        auto perm = tr.getPermutation();
        auto srcShape = srcType.getShape();
        auto outShape = resType.getShape();
        unsigned rank = perm.size();
        if (rank == 0)
          continue;
        SmallVector<int64_t> srcStrides(rank, 1), outStrides(rank, 1);
        for (int d = rank - 2; d >= 0; --d) {
          srcStrides[d] = srcStrides[d + 1] * srcShape[d + 1];
          outStrides[d] = outStrides[d + 1] * outShape[d + 1];
        }
        SmallVector<char> dstRaw(numEl * elemBytes);
        SmallVector<int64_t> srcIdx(rank);
        for (int64_t lin = 0; lin < numEl; ++lin) {
          int64_t r = lin;
          for (unsigned d = 0; d < rank; ++d) {
            srcIdx[d] = r / srcStrides[d];
            r %= srcStrides[d];
          }
          int64_t outLin = 0;
          for (unsigned d = 0; d < rank; ++d)
            outLin += srcIdx[perm[d]] * outStrides[d];
          std::memcpy(&dstRaw[outLin * elemBytes],
                      &srcRaw[lin * elemBytes], elemBytes);
        }

        auto foldedAttr = DenseElementsAttr::getFromRawBuffer(resType, dstRaw);
        OpBuilder b(tr);
        auto newCst =
            b.create<arith::ConstantOp>(tr.getLoc(), resType, foldedAttr);
        tr.getResult()[0].replaceAllUsesWith(newCst.getResult());
        tr.erase();
        if (cst->use_empty())
          cst.erase();
      }
    }

    // Step 0: matmul/bmm/conv/pool go to aclnn single-ops which allocate their
    // own output, so a fill init is redundant.  Detach it (init->bare empty)
    // before grouping so the fill isn't pulled into an unrelated vector group
    // as a dead dual-output (encoder group18 add+fill→two VECOUT deadlock).
    // Conv2D + Pool2D need the same treatment so ResNet's {fill + conv} /
    // {fill + pool} groups reduce to a single-op group that GroupOutline can
    // stamp aclnn.op.
    func.walk([&](linalg::LinalgOp op) {
      if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp,
               linalg::Conv2DNchwFchwOp, linalg::PoolingNchwMaxOp,
               linalg::PoolingNchwSumOp>(op.getOperation()))
        return;
      OpOperand *init = op.getDpsInitOperand(0);
      if (auto fill = init->get().getDefiningOp<linalg::FillOp>()) {
        init->set(fill.getOutputs()[0]);
        if (fill->use_empty())
          fill.erase();
      }
    });

    // Step 1: Collect all linalg ops in program order; each is its own group
    llvm::SmallVector<FusionGroup> groups;
    int32_t nextId = 0;

    func.walk([&](linalg::LinalgOp op) {
      FusionGroup g;
      g.id   = nextId++;
      g.kind = isCubeOp(op) ? GroupInfo::Kind::Cube : GroupInfo::Kind::Vector;
      g.members.push_back(op);
      rebuildGroup(g, func);
      groups.push_back(std::move(g));
    });

    // Step 2: Iterative fusion loop
    CanFuseOptions opts;
    opts.maxReduceEpilogueOps     = this->maxReduceEpilogueOps;
    opts.maxHorizontalExtraInputs = this->maxHorizontalExtraInputs;
    opts.enableReductionSplit     = this->enableReductionSplit;

    struct FusionPair {
      int32_t   g1Idx, g2Idx;
      FusionKind kind;
      int64_t   score;
      int       priority; // 0 = VV, 1 = CV
    };

    bool changed = true;
    while (changed) {
      changed = false;

      llvm::SmallVector<FusionPair> pairs;

      // Generate all valid pairs
      for (int i = 0; i < (int)groups.size(); ++i) {
        if (groups[i].id < 0) continue;
        for (int j = i + 1; j < (int)groups.size(); ++j) {
          if (groups[j].id < 0) continue;

          auto &g1 = groups[i];
          auto &g2 = groups[j];

          FusionKind kind = getFusionKind(g1, g2);
          if (kind == FusionKind::None) continue;

          bool canFuse = false;
          int  priority = 0;

          bool bothVector = (g1.kind == GroupInfo::Kind::Vector &&
                             g2.kind == GroupInfo::Kind::Vector);
          bool oneCube    = (g1.kind == GroupInfo::Kind::Cube ||
                             g2.kind == GroupInfo::Kind::Cube);

          if (bothVector) {
            canFuse  = canFuseVector(g1, g2, kind, groups, opts);
            priority = 0;
          } else if (oneCube) {
            // Cube + Vector epilogue fusion (Phase 1 of [[af-cv-fusion-port]]).
            // Allowed only when exactly one side is Cube and the other is
            // Vector (rule out Cube+Cube — no two-matmul fusion in v1).  The
            // direction matters: `canFuseCubeEpilogue(cube, vec, ...)` assumes
            // cube produces and vec consumes; dispatch on group kinds.
            const FusionGroup *cube = nullptr;
            const FusionGroup *vec  = nullptr;
            if (g1.kind == GroupInfo::Kind::Cube &&
                g2.kind == GroupInfo::Kind::Vector) {
              cube = &g1; vec = &g2;
            } else if (g2.kind == GroupInfo::Kind::Cube &&
                       g1.kind == GroupInfo::Kind::Vector) {
              cube = &g2; vec = &g1;
            }
            if (cube && vec && !this->disableCubeFusion)
              canFuse = canFuseCubeEpilogue(*cube, *vec, groups);
            // Priority 1 = scheduled after vector-vector fusion (priority 0)
            // so CV merges only fire on the residual standalone-Cube groups.
            priority = 1;
          }

          if (!canFuse) continue;

          int64_t score = computeScore(g1, g2);
          pairs.push_back({i, j, kind, score, priority});
        }
      }

      // Sort: priority ASC, then score DESC
      std::stable_sort(pairs.begin(), pairs.end(),
                       [](const FusionPair &a, const FusionPair &b) {
                         if (a.priority != b.priority)
                           return a.priority < b.priority;
                         return a.score > b.score;
                       });

      // Apply pairs
      for (auto &pair : pairs) {
        auto &g1 = groups[pair.g1Idx];
        auto &g2 = groups[pair.g2Idx];

        // Both must still be valid
        if (g1.id < 0 || g2.id < 0) continue;

        // Pairs were scored against a stale snapshot of `groups`; an earlier
        // merge in this same batch may have made this pair cycle-creating.
        // Re-validate against the live state before committing the merge.
        if (wouldCreateCycle(g1, g2, groups))
          continue;

        mergeInto(g1, g2, func);
        changed = true;
      }
    }

    // Step 3: Assign topo_index by walking func in program order
    // Build a map from op -> group id
    llvm::DenseMap<Operation *, int32_t> opToGroupId;
    for (auto &g : groups) {
      if (g.id < 0) continue;
      for (auto op : g.members)
        opToGroupId[op.getOperation()] = g.id;
    }

    // Walk in program order and assign topo indices
    llvm::DenseMap<Operation *, int32_t> opToTopoIndex;
    int32_t topoIdx = 0;
    func.walk([&](linalg::LinalgOp op) {
      opToTopoIndex[op.getOperation()] = topoIdx++;
    });

    // Step 4: Write attributes
    MLIRContext *ctx = func.getContext();
    auto groupIdName   = StringAttr::get(ctx, "auto_fuse.group_id");
    auto topoIndexName = StringAttr::get(ctx, "auto_fuse.topo_index");

    func.walk([&](linalg::LinalgOp op) {
      Operation *rawOp = op.getOperation();

      auto gidIt = opToGroupId.find(rawOp);
      if (gidIt == opToGroupId.end()) return;

      auto tidIt = opToTopoIndex.find(rawOp);
      if (tidIt == opToTopoIndex.end()) return;

      rawOp->setAttr(groupIdName,
                     IntegerAttr::get(IntegerType::get(ctx, 32),
                                      gidIt->second));
      rawOp->setAttr(topoIndexName,
                     IntegerAttr::get(IntegerType::get(ctx, 32),
                                      tidIt->second));
    });
  }
};

} // namespace

std::unique_ptr<Pass> createAutoFuseGroupAnalysisPass() {
  return std::make_unique<AutoFuseGroupAnalysisPass>();
}

} // namespace mlir::afir
