#include "NetworkJsonEmitter.h"
#include "Conversion/VectorPlan/VectorPlanPasses.h"
#include "Conversion/VectorPlan/GroupInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/OwningOpRef.h"
#include <algorithm>
#include <climits>

#define GEN_PASS_DECL_VECTORPLANGROUPOUTLINE
#define GEN_PASS_DEF_VECTORPLANGROUPOUTLINE
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

namespace {

//===----------------------------------------------------------------------===//
// Attribute helpers
//===----------------------------------------------------------------------===//

static int32_t getGroupId(linalg::LinalgOp op) {
  auto attr = op->getAttrOfType<IntegerAttr>("vector_plan.group_id");
  return attr ? (int32_t)attr.getInt() : -1;
}

static int32_t getTopoIndex(linalg::LinalgOp op) {
  auto attr = op->getAttrOfType<IntegerAttr>("vector_plan.topo_index");
  return attr ? (int32_t)attr.getInt() : 0;
}

static int32_t
getMinTopoIndex(const llvm::SmallVector<linalg::LinalgOp> &ops) {
  int32_t minIdx = INT32_MAX;
  for (auto op : ops)
    minIdx = std::min(minIdx, getTopoIndex(op));
  return minIdx;
}

//===----------------------------------------------------------------------===//
// RebuiltGroupInfo — reconstructed per group from def-use analysis
//===----------------------------------------------------------------------===//

struct RebuiltGroupInfo {
  int32_t id;
  llvm::SmallVector<linalg::LinalgOp> topoMembers; // sorted by topo_index
  GroupInfo::Kind kind;
  // All ops in block range [firstGroupOp, lastGroupOp] (block order).
  // Includes linalg group members AND any interstitial non-linalg ops
  // (collapse_shape, expand_shape, etc.) that connect them.
  llvm::SmallVector<Operation *> opsToClone;
  llvm::SmallVector<Value> boundaryIn;  // external inputs to the op range
  llvm::SmallVector<Value> boundaryOut; // range results used outside the range
};

static RebuiltGroupInfo
rebuildGroupInfo(int32_t gid,
                 const llvm::SmallVector<linalg::LinalgOp> &topoMembers) {
  RebuiltGroupInfo info;
  info.id = gid;
  info.topoMembers = topoMembers;

  // kind: any matmul-like op → Cube
  info.kind =
      llvm::any_of(topoMembers,
                   [](linalg::LinalgOp op) {
                     return isa<linalg::MatmulOp, linalg::BatchMatmulOp>(
                         op.getOperation());
                   })
          ? GroupInfo::Kind::Cube
          : GroupInfo::Kind::Vector;

  // Find first and last linalg member ops in block order.
  llvm::DenseSet<Operation *> memberOpsSet;
  for (auto op : topoMembers)
    memberOpsSet.insert(op.getOperation());

  Block *parentBlock = topoMembers.front()->getBlock();
  Operation *firstOp = nullptr, *lastOp = nullptr;
  for (auto &blockOp : *parentBlock) {
    if (memberOpsSet.contains(&blockOp)) {
      if (!firstOp) firstOp = &blockOp;
      lastOp = &blockOp;
    }
  }

  // Collect ALL ops in [firstOp, lastOp] in block order.
  // This includes interstitial non-linalg ops (collapse_shape, etc.).
  bool inRange = false;
  for (auto &blockOp : *parentBlock) {
    if (&blockOp == firstOp) inRange = true;
    if (inRange) info.opsToClone.push_back(&blockOp);
    if (&blockOp == lastOp) break;
  }

  // Values defined inside the op range (outer results only).
  llvm::DenseSet<Value> internalResults;
  for (auto *op : info.opsToClone)
    for (Value r : op->getResults())
      internalResults.insert(r);

  // Helper: is this value "internal" to the op range?
  // Covers: (1) outer op results, (2) block args of nested regions
  // (e.g. linalg body %in/%out), (3) results of ops nested inside cloned ops
  // (e.g. arith.addf inside a linalg body block).
  auto isInternal = [&](Value v) -> bool {
    if (internalResults.count(v)) return true;
    if (auto bArg = dyn_cast<BlockArgument>(v)) {
      for (auto *clonedOp : info.opsToClone)
        for (Region &r : clonedOp->getRegions())
          for (Block &b : r)
            if (&b == bArg.getOwner())
              return true;
    }
    // Result of an op nested inside a cloned op's region (e.g. arith.addf
    // inside linalg.generic body).
    if (auto *defOp = v.getDefiningOp()) {
      for (auto *clonedOp : info.opsToClone) {
        Operation *parent = defOp->getParentOp();
        while (parent) {
          if (parent == clonedOp) return true;
          parent = parent->getParentOp();
        }
      }
    }
    return false;
  };

  // boundaryIn: walk ALL nested regions to catch scalar constants captured
  // inside linalg body blocks (e.g. %cst used in arith.divf inside the body).
  llvm::DenseSet<Value> seen;
  for (auto *op : info.opsToClone) {
    op->walk([&](Operation *innerOp) {
      for (Value operand : innerOp->getOperands()) {
        if (!isInternal(operand) && !seen.count(operand)) {
          info.boundaryIn.push_back(operand);
          seen.insert(operand);
        }
      }
    });
  }

  // boundaryOut: range results that have users OUTSIDE the range.
  llvm::DenseSet<Operation *> opsToCloneSet;
  for (auto *op : info.opsToClone) opsToCloneSet.insert(op);

  for (auto *op : info.opsToClone) {
    for (Value result : op->getResults()) {
      bool usedOutside = llvm::any_of(result.getUsers(), [&](Operation *user) {
        return !opsToCloneSet.count(user);
      });
      if (usedOutside)
        info.boundaryOut.push_back(result);
    }
  }

  return info;
}

//===----------------------------------------------------------------------===//
// outlineGroup — create private kernel func with cloned ops
//===----------------------------------------------------------------------===//

static func::FuncOp outlineGroup(OpBuilder &builder, ModuleOp module,
                                  func::FuncOp coordFunc,
                                  const RebuiltGroupInfo &info,
                                  StringRef funcName) {
  SmallVector<Type> argTypes, resTypes;
  for (Value v : info.boundaryIn)
    argTypes.push_back(v.getType());
  for (Value v : info.boundaryOut)
    resTypes.push_back(v.getType());

  auto funcType = builder.getFunctionType(argTypes, resTypes);

  // Insert kernel func immediately before the coordinator func
  builder.setInsertionPoint(coordFunc);
  auto kernelFunc =
      builder.create<func::FuncOp>(module.getLoc(), funcName, funcType);
  kernelFunc.setPrivate();

  Block *body = kernelFunc.addEntryBlock();

  // Map external inputs → block arguments
  IRMapping mapping;
  for (auto [orig, arg] :
       llvm::zip(info.boundaryIn, body->getArguments()))
    mapping.map(orig, arg);

  // Clone all ops in block range order (linalg + interstitial non-linalg).
  builder.setInsertionPointToEnd(body);
  for (Operation *op : info.opsToClone)
    builder.clone(*op, mapping);

  // Emit return for boundary outputs
  SmallVector<Value> returnVals;
  for (Value v : info.boundaryOut)
    returnVals.push_back(mapping.lookupOrDefault(v));
  builder.create<func::ReturnOp>(module.getLoc(), returnVals);

  return kernelFunc;
}

//===----------------------------------------------------------------------===//
// replaceGroupWithCall — replace group ops in coordinator with func.call
//===----------------------------------------------------------------------===//

static void replaceGroupWithCall(OpBuilder &builder, func::FuncOp coordFunc,
                                  const RebuiltGroupInfo &info,
                                  func::FuncOp kernelFunc) {
  // Insert call before the FIRST op in the range so that all boundaryIn
  // values (defined before the range) are available at the call site.
  builder.setInsertionPoint(info.opsToClone.front());
  auto callOp = builder.create<func::CallOp>(coordFunc.getLoc(), kernelFunc,
                                              info.boundaryIn);

  // Replace all range outputs with call results
  for (auto [origOut, callResult] :
       llvm::zip(info.boundaryOut, callOp.getResults())) {
    Value mutableOut = origOut;
    mutableOut.replaceAllUsesWith(callResult);
  }

  // Erase all range ops (linalg + interstitial) in reverse block order so
  // consumers are erased before producers.
  for (Operation *op : llvm::reverse(info.opsToClone))
    op->erase();
}

//===----------------------------------------------------------------------===//
// stripVectorPlanAttrs — remove all vector_plan.* attributes from linalg ops
//===----------------------------------------------------------------------===//

static void stripVectorPlanAttrs(ModuleOp module) {
  module.walk([](linalg::LinalgOp op) {
    op->removeAttr("vector_plan.group_id");
    op->removeAttr("vector_plan.topo_index");
  });
}

//===----------------------------------------------------------------------===//
// emitFiles — write network.mlir + kernel_groupN.mlir to outputDir
//===----------------------------------------------------------------------===//

static LogicalResult emitFiles(ModuleOp module,
                                ArrayRef<int32_t> sortedGroupIds,
                                StringRef outputDir) {
  if (auto ec = llvm::sys::fs::create_directories(outputDir); ec)
    return module.emitError("cannot create output dir: ") << ec.message();

  OpBuilder b(module.getContext());

  for (int32_t gid : sortedGroupIds) {
    // Find the kernel func for this group
    std::string name = ("kernel_group" + llvm::Twine(gid)).str();
    func::FuncOp kernelFunc;
    module.walk([&](func::FuncOp f) -> WalkResult {
      if (f.getSymName() == name) {
        kernelFunc = f;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!kernelFunc)
      continue;

    // Create a temporary sub-module containing only this kernel func
    OwningOpRef<ModuleOp> subMod =
        ModuleOp::create(module.getLoc());
    b.setInsertionPointToStart(subMod->getBody());
    b.clone(*kernelFunc);

    std::string filename =
        (outputDir + "/kernel_group" + llvm::Twine(gid) + ".mlir").str();
    std::error_code ec;
    llvm::raw_fd_ostream os(filename, ec);
    if (ec)
      return module.emitError("cannot open ") << filename << ": " << ec.message();
    subMod->print(os);
  }

  // Strip kernel func bodies from the main module → coordinator + declarations
  for (int32_t gid : sortedGroupIds) {
    std::string name = ("kernel_group" + llvm::Twine(gid)).str();
    module.walk([&](func::FuncOp f) -> WalkResult {
      if (f.getSymName() == name) {
        f.eraseBody();
        f.setVisibility(SymbolTable::Visibility::Private);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }

  std::string netFile = (outputDir + "/network.mlir").str();
  std::error_code ec2;
  llvm::raw_fd_ostream osNet(netFile, ec2);
  if (ec2)
    return module.emitError("cannot open ") << netFile << ": " << ec2.message();
  module.print(osNet);

  // Also emit network.json describing the coordinator call graph.
  std::string jsonFile = (outputDir + "/network.json").str();
  std::error_code jec;
  llvm::raw_fd_ostream jsonOs(jsonFile, jec);
  if (jec)
    return module.emitError("cannot open network.json: ") << jec.message();
  func::FuncOp coord;
  module.walk([&](func::FuncOp f) -> WalkResult {
    if (!f.isPrivate()) {
      coord = f;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (coord)
    if (auto err = mlir::vector_plan::emitNetworkJson(module, coord, jsonOs))
      return module.emitError("emitNetworkJson: ")
             << llvm::toString(std::move(err));

  return success();
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

struct VectorPlanGroupOutlinePass
    : public ::impl::VectorPlanGroupOutlineBase<VectorPlanGroupOutlinePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Step 1: Bucket annotated linalg ops by group_id
    llvm::DenseMap<int32_t, llvm::SmallVector<linalg::LinalgOp>> buckets;
    module.walk([&](linalg::LinalgOp op) {
      int32_t gid = getGroupId(op);
      if (gid >= 0)
        buckets[gid].push_back(op);
    });

    if (buckets.empty())
      return;

    // Step 2: Sort ops within each bucket by topo_index
    for (auto &[gid, ops] : buckets) {
      llvm::stable_sort(ops, [](linalg::LinalgOp a, linalg::LinalgOp b) {
        return getTopoIndex(a) < getTopoIndex(b);
      });
    }

    // Step 3: Sort groups by min topo_index (producer groups before consumers)
    llvm::SmallVector<int32_t> sortedGroupIds;
    for (auto &[gid, _] : buckets)
      sortedGroupIds.push_back(gid);
    llvm::sort(sortedGroupIds, [&](int32_t a, int32_t b) {
      return getMinTopoIndex(buckets[a]) < getMinTopoIndex(buckets[b]);
    });

    // Find the coordinator func (containing the annotated linalg ops).
    // For now we handle a single func per module; the first annotated op's
    // enclosing FuncOp is the coordinator.
    func::FuncOp coordFunc;
    module.walk([&](linalg::LinalgOp op) -> WalkResult {
      if (getGroupId(op) >= 0) {
        coordFunc = op->getParentOfType<func::FuncOp>();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!coordFunc)
      return;

    OpBuilder builder(module.getContext());

    // Steps 4–6: For each group (in topo order), rebuild boundary info,
    // create kernel func, and replace group ops with a call.
    // rebuildGroupInfo is called INSIDE the loop so it sees the live IR
    // (after previous groups' ops have been erased and their results replaced).
    for (int32_t gid : sortedGroupIds) {
      auto &ops = buckets[gid];
      RebuiltGroupInfo info = rebuildGroupInfo(gid, ops);

      std::string funcName =
          ("kernel_group" + llvm::Twine(gid)).str();
      func::FuncOp kernelFunc =
          outlineGroup(builder, module, coordFunc, info, funcName);
      replaceGroupWithCall(builder, coordFunc, info, kernelFunc);
    }

    // Step 7: Strip all vector_plan.* attributes
    stripVectorPlanAttrs(module);

    // Step 8: Optional file split when outputDir is set
    std::string outDir = outputDir.getValue();
    if (!outDir.empty()) {
      if (failed(emitFiles(module, sortedGroupIds, outDir)))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createVectorPlanGroupOutlinePass() {
  return std::make_unique<VectorPlanGroupOutlinePass>();
}

} // namespace mlir::afir
