#include "Conversion/AutoFuse/GroupOutline/NetworkJsonEmitter.h"
#include "Conversion/AutoFuse/AutoFusePasses.h"
#include "Conversion/AutoFuse/GroupInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include <limits>
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/OwningOpRef.h"
#include <algorithm>
#include <climits>
#include <functional>

#define GEN_PASS_DECL_AUTOFUSEGROUPOUTLINE
#define GEN_PASS_DEF_AUTOFUSEGROUPOUTLINE
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::auto_fuse;

namespace mlir::afir {

namespace {

//===----------------------------------------------------------------------===//
// Attribute helpers
//===----------------------------------------------------------------------===//

static int32_t getGroupId(linalg::LinalgOp op) {
  auto attr = op->getAttrOfType<IntegerAttr>("auto_fuse.group_id");
  return attr ? (int32_t)attr.getInt() : -1;
}

static int32_t getTopoIndex(linalg::LinalgOp op) {
  auto attr = op->getAttrOfType<IntegerAttr>("auto_fuse.topo_index");
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

  // kind: any matmul / conv-like op → Cube (cube-class is routed to aclnn).
  info.kind =
      llvm::any_of(topoMembers,
                   [](linalg::LinalgOp op) {
                     return isa<linalg::MatmulOp, linalg::BatchMatmulOp,
                                linalg::Conv2DNchwFchwOp>(op.getOperation());
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
  // ConstantLike operands (arith.constant fill values / weights) are NOT passed
  // as kernel args — they are rematerialized inside the kernel by outlineGroup.
  // This keeps kernels tensor-arg-only (codegen casts arg types to
  // RankedTensorType) and avoids threading scalar args through the runtime.
  llvm::DenseSet<Value> seen;
  for (auto *op : info.opsToClone) {
    op->walk([&](Operation *innerOp) {
      for (Value operand : innerOp->getOperands()) {
        if (isInternal(operand) || seen.count(operand))
          continue;
        // Rematerialize SCALAR constants inside the kernel (cheap, avoids
        // scalar args that codegen can't type as RankedTensorType).  TENSOR
        // constants (weights) stay as kernel args — they are too large to
        // embed and the host materializes their data.
        if (Operation *def = operand.getDefiningOp();
            def && def->hasTrait<mlir::OpTrait::ConstantLike>() &&
            !mlir::isa<mlir::ShapedType>(operand.getType()))
          continue; // scalar const: rematerialized, not an arg
        info.boundaryIn.push_back(operand);
        seen.insert(operand);
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

  // Stamp `auto_fuse.call_arg_index` on each kernel input arg so downstream
  // consumers (TilePlanGen schema build) don't have to rely on positional
  // equality between coordinator-call operand order and kernel arg order.
  MLIRContext *ctx = kernelFunc.getContext();
  for (unsigned i = 0, e = info.boundaryIn.size(); i < e; ++i) {
    kernelFunc.setArgAttr(i, "auto_fuse.call_arg_index",
                          IntegerAttr::get(IntegerType::get(ctx, 32),
                                           (int32_t)i));
  }

  // CV-fusion Phase 3: stamp `auto_fuse.kind` on the outlined kernel func so
  // downstream passes can dispatch on Cube vs Vector without re-walking ops to
  // find a matmul.  Mirrors the `info.kind` value Phase 1 computed at fusion
  // time.  TileFusePass + Phase-4 LoopNestBuilder can read this directly.
  kernelFunc->setAttr(
      "auto_fuse.kind",
      StringAttr::get(ctx, info.kind == GroupInfo::Kind::Cube
                               ? "Cube"
                               : "Vector"));

  // aclnn fallback: route a standalone cube (matmul / batch_matmul / conv2d)
  // group to the aclnn CPU-reference op, since AscendC cube codegen isn't ready.
  // Only a single-op group maps cleanly to one aclnn op; CV-fused cube groups
  // are left for the (future) cube codegen path.  Stamping `aclnn.op` makes
  // emitNetworkJson tag the kernel kind=aclnn and the host emit run_<Op>.
  if (info.kind == GroupInfo::Kind::Cube && info.topoMembers.size() == 1) {
    linalg::LinalgOp member = info.topoMembers.front();
    Operation *memberOp = member.getOperation();
    if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(memberOp)) {
      kernelFunc->setAttr("aclnn.op", StringAttr::get(ctx, "Matmul"));
    } else if (auto conv = dyn_cast<linalg::Conv2DNchwFchwOp>(memberOp)) {
      // Conv2D in NCHW × FCHW layout.  Pad-as-tensor.pad lives in the
      // coordinator (handled by NetworkJsonEmitter/AclnnBackend), so the
      // kernel sees zero-padding semantics; only strides and dilations need
      // to be threaded through to run_Conv2D.
      kernelFunc->setAttr("aclnn.op", StringAttr::get(ctx, "Conv2D"));
      SmallVector<int64_t> stridesVec(conv.getStrides().getValues<int64_t>());
      SmallVector<int64_t> dilationsVec(conv.getDilations().getValues<int64_t>());
      kernelFunc->setAttr("aclnn.strides",
                          builder.getDenseI64ArrayAttr(stridesVec));
      kernelFunc->setAttr("aclnn.dilations",
                          builder.getDenseI64ArrayAttr(dilationsVec));
    }
  }

  // aclnn fallback: route a standalone transpose to the aclnn CPU-reference
  // permute (the AscendC transpose codegen is unreliable; CanFuse keeps every
  // transpose a singleton group). Stamp aclnn.op="Transpose" + aclnn.perm so
  // emitNetworkJson tags it kind=aclnn and the host emits run_Transpose(perm).
  if (info.topoMembers.size() == 1) {
    linalg::LinalgOp tmember = info.topoMembers.front();
    if (auto t = dyn_cast<linalg::TransposeOp>(tmember.getOperation())) {
      kernelFunc->setAttr("aclnn.op", StringAttr::get(ctx, "Transpose"));
      kernelFunc->setAttr("aclnn.perm",
                          builder.getDenseI64ArrayAttr(t.getPermutation()));
    }
  }

  // Rematerialize ConstantLike operands inside the kernel (they were excluded
  // from boundaryIn).  Clone each referenced constant once and map it so the
  // member clones below pick up the in-kernel constant instead of a dangling
  // cross-region reference.
  builder.setInsertionPointToEnd(body);
  for (Operation *op : info.opsToClone)
    op->walk([&](Operation *innerOp) {
      for (Value operand : innerOp->getOperands()) {
        if (mapping.contains(operand))
          continue;
        Operation *def = operand.getDefiningOp();
        if (def && def->hasTrait<mlir::OpTrait::ConstantLike>() &&
            !mlir::isa<mlir::ShapedType>(operand.getType()))
          builder.clone(*def, mapping); // scalar const only
      }
    });

  // Clone all ops in block range order (linalg + interstitial non-linalg).
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
// stripAutoFuseAttrs — remove all auto_fuse.* attributes from linalg ops
//===----------------------------------------------------------------------===//

static void stripAutoFuseAttrs(ModuleOp module) {
  module.walk([](linalg::LinalgOp op) {
    op->removeAttr("auto_fuse.group_id");
    op->removeAttr("auto_fuse.topo_index");
  });
}

//===----------------------------------------------------------------------===//
// emitFiles — write network.mlir + kernel_groupN.mlir to outputDir
//===----------------------------------------------------------------------===//

static LogicalResult emitFiles(ModuleOp module,
                                StringRef outputDir,
                                func::FuncOp coordFunc) {
  if (auto ec = llvm::sys::fs::create_directories(outputDir); ec)
    return module.emitError("cannot create output dir: ") << ec.message();

  OpBuilder b(module.getContext());

  // Discover private kernel funcs by name prefix.  Covers both the
  // numerically-named groups emitted by this pass (`kernel_groupN`) and any
  // post-split derivatives such as `kernel_groupN_partial` /
  // `kernel_groupN_combine` produced by SplitRCoreGroup.
  SmallVector<func::FuncOp> kernelFuncs;
  module.walk([&](func::FuncOp f) {
    if (f.isPrivate() && !f.getBody().empty() &&
        f.getSymName().starts_with("kernel_group"))
      kernelFuncs.push_back(f);
  });

  for (func::FuncOp kernelFunc : kernelFuncs) {
    OwningOpRef<ModuleOp> subMod =
        ModuleOp::create(module.getLoc());
    b.setInsertionPointToStart(subMod->getBody());
    b.clone(*kernelFunc);

    std::string filename =
        (outputDir + "/" + kernelFunc.getSymName() + ".mlir").str();
    std::error_code ec;
    llvm::raw_fd_ostream os(filename, ec);
    if (ec)
      return module.emitError("cannot open ") << filename << ": " << ec.message();
    subMod->print(os);
  }

  // Provenance sidecar must walk kernel func bodies, so emit BEFORE we strip
  // them. Debug-only — not consumed by the compilation pipeline (debug.md §7.5).
  {
    std::string provFile = (outputDir + "/network.provenance.json").str();
    std::error_code pec;
    llvm::raw_fd_ostream provOs(provFile, pec);
    if (pec)
      return module.emitError("cannot open network.provenance.json: ")
             << pec.message();
    if (auto err = mlir::auto_fuse::emitNetworkProvenanceJson(module, coordFunc,
                                                               provOs))
      return module.emitError("emitNetworkProvenanceJson: ")
             << llvm::toString(std::move(err));
  }

  // Strip kernel func bodies from the main module → coordinator + declarations
  for (func::FuncOp kernelFunc : kernelFuncs) {
    kernelFunc.eraseBody();
    kernelFunc.setVisibility(SymbolTable::Visibility::Private);
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
  if (auto err = mlir::auto_fuse::emitNetworkJson(module, coordFunc, jsonOs))
    return module.emitError("emitNetworkJson: ")
           << llvm::toString(std::move(err));

  return success();
}

//===----------------------------------------------------------------------===//
// reorderGroupsContiguous — schedule ops so each group's members are contiguous
//===----------------------------------------------------------------------===//
//
// Group analysis guarantees groups form a DAG (cycle-checked), hence each group
// is convex (no foreign op both consumes from and feeds into the same group).
// But members can still be *interleaved* in block order, and the range-based
// outliner needs contiguity.  We reorder the coordinator block:
//   - cluster each group's members together, in group topological order;
//   - keep glue used only within one group (internal glue) inside that cluster;
//   - drop cross-group glue (reshapes feeding/fed-by multiple groups) into the
//     GAP between clusters so it stays coordinator-level (never cloned into a
//     kernel).
// Returns false if the group graph is not a DAG (cannot order) — the caller's
// contiguity guard then reports a clean error.
static bool reorderGroupsContiguous(
    func::FuncOp coordFunc,
    const llvm::DenseMap<int32_t, llvm::SmallVector<linalg::LinalgOp>> &buckets) {
  Block &block = coordFunc.getBody().front();

  // member op -> group id
  llvm::DenseMap<Operation *, int32_t> opGroup;
  for (auto &[gid, ops] : buckets)
    for (auto op : ops)
      opGroup[op.getOperation()] = gid;

  // sourceGroups(v): producer groups feeding v, transitively through non-members
  llvm::DenseMap<Value, llvm::SmallDenseSet<int32_t, 2>> srcMemo;
  std::function<llvm::SmallDenseSet<int32_t, 2>(Value)> sourceGroups =
      [&](Value v) -> llvm::SmallDenseSet<int32_t, 2> {
    auto it = srcMemo.find(v);
    if (it != srcMemo.end())
      return it->second;
    llvm::SmallDenseSet<int32_t, 2> out;
    if (Operation *def = v.getDefiningOp()) {
      auto git = opGroup.find(def);
      if (git != opGroup.end())
        out.insert(git->second);
      else
        for (Value o : def->getOperands())
          for (int32_t s : sourceGroups(o))
            out.insert(s);
    }
    srcMemo[v] = out;
    return out;
  };

  // consumerGroups(op): consumer groups of op's results, transitively through
  // non-member ops.
  llvm::DenseMap<Operation *, llvm::SmallDenseSet<int32_t, 2>> consMemo;
  std::function<llvm::SmallDenseSet<int32_t, 2>(Operation *)> consumerGroups =
      [&](Operation *op) -> llvm::SmallDenseSet<int32_t, 2> {
    auto it = consMemo.find(op);
    if (it != consMemo.end())
      return it->second;
    consMemo[op] = {}; // cycle guard (acyclic, but be safe)
    llvm::SmallDenseSet<int32_t, 2> out;
    for (Value r : op->getResults())
      for (Operation *user : r.getUsers()) {
        auto git = opGroup.find(user);
        if (git != opGroup.end())
          out.insert(git->second);
        else
          for (int32_t c : consumerGroups(user))
            out.insert(c);
      }
    consMemo[op] = out;
    return out;
  };

  // Build the glue-aware group DAG and topologically sort it (Kahn).
  llvm::DenseMap<int32_t, llvm::DenseSet<int32_t>> gedges;
  llvm::DenseMap<int32_t, int> indeg;
  for (auto &[gid, ops] : buckets)
    indeg.try_emplace(gid, 0);
  for (auto &[gid, ops] : buckets)
    for (auto op : ops)
      for (Value operand : op.getOperation()->getOperands())
        for (int32_t src : sourceGroups(operand))
          if (src != gid && gedges[src].insert(gid).second)
            ++indeg[gid];

  llvm::SmallVector<int32_t> ready, topo;
  for (auto &[gid, d] : indeg)
    if (d == 0)
      ready.push_back(gid);
  llvm::sort(ready); // determinism
  while (!ready.empty()) {
    int32_t g = ready.front();
    ready.erase(ready.begin());
    topo.push_back(g);
    llvm::SmallVector<int32_t> freed;
    for (int32_t h : gedges.lookup(g))
      if (--indeg[h] == 0)
        freed.push_back(h);
    llvm::sort(freed);
    ready.append(freed.begin(), freed.end());
  }
  if (topo.size() != indeg.size())
    return false; // not a DAG — let the caller's guard report it

  llvm::DenseMap<int32_t, int> groupRank;
  for (int i = 0; i < (int)topo.size(); ++i)
    groupRank[topo[i]] = i;

  // Assign a schedule key per op (excluding terminator).  Even rank 2*r is a
  // group's own slot; odd rank 2*r+1 is the gap after group rank r.
  Operation *term = block.getTerminator();
  llvm::SmallVector<Operation *> ops;
  llvm::DenseMap<Operation *, int> origIndex, keyRank;
  int idx = 0;
  for (Operation &o : block) {
    if (&o == term)
      continue;
    origIndex[&o] = idx++;
    ops.push_back(&o);
  }

  // Lowest possible rank.  Used to hoist constants above every cluster: their
  // only positional requirement is "before all uses", and clustering them with a
  // single consumer group can break SSA dominance when another non-member op
  // captures the constant via a nested region (e.g. tensor.pad's yield).  Region
  // captures are not modeled by the prod/cons sets below; pinning ConstantLike
  // ops to the top of the schedule sidesteps the issue entirely.
  constexpr int kConstantRank = std::numeric_limits<int>::min();

  for (Operation *op : ops) {
    if (op->hasTrait<mlir::OpTrait::ConstantLike>()) {
      keyRank[op] = kConstantRank;
      continue;
    }
    auto git = opGroup.find(op);
    if (git != opGroup.end()) {
      keyRank[op] = 2 * groupRank[git->second]; // member: own cluster
      continue;
    }
    // non-member (glue / constant / etc.)
    llvm::SmallDenseSet<int32_t, 2> prod;
    for (Value v : op->getOperands())
      for (int32_t s : sourceGroups(v))
        prod.insert(s);
    auto cons = consumerGroups(op);
    if (cons.size() == 1) {
      int32_t g = *cons.begin();
      bool prodOk = llvm::all_of(prod, [&](int32_t p) { return p == g; });
      if (prodOk) {
        keyRank[op] = 2 * groupRank[g]; // internal glue: join the cluster
        continue;
      }
    }
    int mp = -1;
    for (int32_t p : prod)
      mp = std::max(mp, groupRank[p]);
    keyRank[op] = 2 * mp + 1; // cross glue: land in the gap after its producer
  }

  llvm::stable_sort(ops, [&](Operation *a, Operation *b) {
    if (keyRank[a] != keyRank[b])
      return keyRank[a] < keyRank[b];
    return origIndex[a] < origIndex[b];
  });

  // Realize the order: move ops, in sorted order, to just before the terminator.
  for (Operation *op : ops)
    op->moveBefore(term);
  return true;
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

struct AutoFuseGroupOutlinePass
    : public ::impl::AutoFuseGroupOutlineBase<AutoFuseGroupOutlinePass> {
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

    // Reorder the coordinator so each group's members are contiguous (analysis
    // may interleave them).  Required because the outliner clones block ranges.
    (void)reorderGroupsContiguous(coordFunc, buckets);

    // Defensive guard: outlineGroup clones each group's entire [firstOp,lastOp]
    // block range (interstitial ops included) and replaceGroupWithCall erases
    // it.  That is sound ONLY when groups are contiguous.  If analysis produced
    // a non-contiguous group whose range straddles another group's ops, the
    // range-clone would erase that foreign group's ops, leaving dangling
    // LinalgOp handles for a later iteration -> use-after-free / SIGSEGV.
    // Detect this and fail cleanly.  (Real networks hit this when over-fusion
    // yields interleaved/cyclic groups — see the transformer-encoder crash.)
    {
      llvm::DenseMap<Operation *, int32_t> opGid;
      for (auto &[gid, ops] : buckets)
        for (auto op : ops)
          opGid[op.getOperation()] = gid;

      for (int32_t gid : sortedGroupIds) {
        auto &ops = buckets[gid];
        llvm::DenseSet<Operation *> memberSet;
        for (auto op : ops)
          memberSet.insert(op.getOperation());
        Block *blk = ops.front()->getBlock();
        Operation *firstOp = nullptr, *lastOp = nullptr;
        for (auto &o : *blk)
          if (memberSet.contains(&o)) {
            if (!firstOp) firstOp = &o;
            lastOp = &o;
          }
        bool inRange = false;
        for (auto &o : *blk) {
          if (&o == firstOp) inRange = true;
          if (inRange && !memberSet.contains(&o)) {
            auto it = opGid.find(&o);
            if (it != opGid.end() && it->second != gid) {
              o.emitError() << "auto-fuse group " << gid << " is not contiguous: "
                            << "its block range straddles an op of group "
                            << it->second
                            << "; group analysis produced a non-outlinable "
                               "(interleaved/cyclic) grouping";
              return signalPassFailure();
            }
          }
          if (&o == lastOp) break;
        }
      }
    }

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

    // Step 7: Strip all auto_fuse.* attributes
    stripAutoFuseAttrs(module);

    // Step 7b: Optionally split full-reduce kernels into partial+combine pairs
    // (RCore template). Models AF's GeneratorRCoreTask / phase_1+phase_2 graph
    // split — see plan docs/superpowers/plans/2026-05-14-p3b-rcore-reduce-
    // multicore.zh.md §5. emitFiles below walks private kernel funcs by name
    // prefix so it picks up both original and post-split variants uniformly.
    if (enableRCoreSplit)
      (void)splitRCoreGroupsInPlace(module, rcoreParallelSlots);

    // Step 8: Optional file split when outputDir is set
    std::string outDir = outputDir.getValue();
    if (!outDir.empty()) {
      if (failed(emitFiles(module, outDir, coordFunc)))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createAutoFuseGroupOutlinePass() {
  return std::make_unique<AutoFuseGroupOutlinePass>();
}

} // namespace mlir::afir
