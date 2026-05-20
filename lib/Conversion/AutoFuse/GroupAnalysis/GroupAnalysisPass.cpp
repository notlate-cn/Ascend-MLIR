#include "AxisLattice.h"
#include "CanFuse.h"
#include "FusionGroup.h"
#include "Conversion/AutoFuse/GroupInfo.h"
#include "Conversion/AutoFuse/AutoFusePasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

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
  // Explicit matmul ops
  if (isa<linalg::MatmulOp>(op.getOperation()))
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
            if (cube && vec)
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
