#include "TilePlanGen.h"
#include "TileFuseUtils.h"
#include "TilePlanGenInternal.h"
#include "Analysis/SymbolicShape/DimSymbolTable.h"
#include "Analysis/SymbolicShape/SymExpr.h"
#include "Conversion/AutoFuse/TilePlan.h"
#include "Target/CannKernel/SocSpec.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <limits>

#define DEBUG_TYPE "tile-plan-gen"

using namespace mlir;
using namespace mlir::auto_fuse;

namespace mlir::afir {

// ===========================================================================
// This file is the "schedule" stage of auto-fuse-tile-fuse.  It mirrors
// AutoFuse's optimize/autoschedule (see
// docs/superpowers/plans/2026-05-11-port-af-scheduler-to-auto-fuse.zh.md):
//
//   genVectorTilePlan                     ≈ AutoSchedule::DoAutoSchedule
//     ├─ (info.grouping)                  ≈ TilingGroup::GenTilingGroup  (X/Y/R/N)
//     ├─ enumerateTilingCases  → [draft]  ≈ GenTilingCase + PruneTilingCase
//     ├─ for each draft: costEstimate     ≈ score_func
//     ├─ pickBest = argmin                ≈ argmin score
//     └─ buildPlan(winner)                ≈ Scheduler::BlockSplit + TileSplit
//                                            (+ computeVectorizedDims, §3.5 row-loop
//                                             degradation)
//
// Status: P1 (refactor) + P3a (per-operand `vectorizedDims` + the §3.5 row-loop
// degradation) + P5a (enumerate/build/cost/pick skeleton) + P5b (real
// reduce-axis enumeration: `enumerateTilingCases` walks the `ubR` candidates
// — R kept whole, or an oversized / `--enable-reduction-split` R axis ub-split,
// plus an ∞-scored RCore variant — `costEstimate` rejects the infeasible ones,
// `buildPlan` consumes `draft.ubTilingAxisR`; the §3.4 change that a non-ub
// parallel axis is fully loaded instead of getting its own `XBLOCK_SUB_n`).
// Still pending: ubY/ubX enumeration over the rest of the y/x groups (needs
// LoopNestBuilder support for a non-block ub axis), the RCore/FullLoad reduce
// templates' codegen (their drafts are enumerated but `costEstimate` keeps them
// ∞ until the codegen lands), a real cost model (§3.6 blockDim / vectorized
// bytes — comes with P6's UB-peak accounting), and the transpose templates
// score (§5).  See the plan doc.
// ===========================================================================




static Value insertFuncArg(func::FuncOp func, OpBuilder &builder,
                            Location loc, int64_t defaultVal,
                            StringRef /*paramName*/) {
  unsigned idx = func.getNumArguments();
  SmallVector<Type> argTypes(func.getFunctionType().getInputs());
  argTypes.push_back(builder.getIndexType());
  func.setType(FunctionType::get(builder.getContext(), argTypes,
                                  func.getFunctionType().getResults()));
  Value newArg =
      func.getBody().front().addArgument(builder.getIndexType(), loc);
  func.setArgAttrs(idx,
                   {NamedAttribute(
                       builder.getStringAttr("auto_fuse.default_tile_size"),
                       builder.getI64IntegerAttr(defaultVal))});
  return newArg;
}


namespace {

// (Axis classification — classifyAxes / transposePerm — moved to
// TileFuseUtils; the Collapse pass computes it and stores it in
// CollapsedGroupInfo::grouping, which buildPlan reads below.)







// ---------------------------------------------------------------------------
// Tensor-IR predicates for classifying the trailing elementwise chain that
// sits between a `linalg.matmul` and the return.  Mirrors the post-bufferize
// helpers in AnnotateMixMatmulSemanticsPass but works on tensor IR (no
// `ascendc.unit` attr yet, ranked tensor types instead of memref).  Used by
// buildCubePlan to stamp `abi_matmul_has_bias` + `abi_matmul_epilogue_kind`
// correctly, replacing the previous "always Relu / never bias" hardcoding.
// Keep narrow: only the AF-canonical bias-add (rank-1 vector, indexing
// map `(d0,d1)->(d1)`) is recognized.  Other bcast forms fall back to
// has_bias=false and are left for Gap-3 follow-up.

static bool isParallelGenericTensor(linalg::GenericOp gen) {
  return llvm::all_of(gen.getIteratorTypesArray(),
                      [](utils::IteratorType t) {
                        return t == utils::IteratorType::parallel;
                      });
}

static bool isRankedTensor(Value v, int64_t rank) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  return t && t.getRank() == rank;
}

// Indexing-map predicates used by the structural bias/identity checks below.
static AffineMap identityMap2D(MLIRContext *ctx) {
  return AffineMap::get(2, 0,
                        {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx)},
                        ctx);
}

// Per-column / per-N broadcast: (d0,d1) -> (d1).  Rejects per-ROW (d0)
// which would mis-fold into mm.SetBias() (broadcasts along N).
static AffineMap colBiasMap2D(MLIRContext *ctx) {
  return AffineMap::get(2, 0, {getAffineDimExpr(1, ctx)}, ctx);
}

// True iff `gen` has the canonical bias-add operand layout for a 2-D matmul
// epilogue: rank-2 identity input + rank-1 column-broadcast input + rank-2
// identity init.
static bool hasCanonicalColumnBiasShape(linalg::GenericOp gen) {
  if (gen.getNumDpsInputs() != 2 || gen.getNumDpsInits() != 1 ||
      !isParallelGenericTensor(gen))
    return false;
  if (!isRankedTensor(gen.getDpsInputOperand(0)->get(), 2) ||
      !isRankedTensor(gen.getDpsInputOperand(1)->get(), 1) ||
      !isRankedTensor(gen.getDpsInitOperand(0)->get(), 2))
    return false;
  auto maps = gen.getIndexingMapsArray();
  if (maps.size() != 3)
    return false;
  MLIRContext *ctx = gen.getContext();
  return maps[0] == identityMap2D(ctx) && maps[1] == colBiasMap2D(ctx) &&
         maps[2] == identityMap2D(ctx);
}

// True iff `gen` is a plain rank-2 pointwise op (1 input, 1 init, identity
// maps), guarding against e.g. transpose-as-relu misclassification.
static bool hasRank2IdentityPointwiseShape(linalg::GenericOp gen) {
  if (gen.getNumDpsInputs() != 1 || gen.getNumDpsInits() != 1 ||
      !isParallelGenericTensor(gen))
    return false;
  if (!isRankedTensor(gen.getDpsInputOperand(0)->get(), 2) ||
      !isRankedTensor(gen.getDpsInitOperand(0)->get(), 2))
    return false;
  auto maps = gen.getIndexingMapsArray();
  if (maps.size() != 2)
    return false;
  MLIRContext *ctx = gen.getContext();
  return maps[0] == identityMap2D(ctx) && maps[1] == identityMap2D(ctx);
}

// Collect the set of arithmetic op kinds in `gen.body` (excluding
// linalg.yield and arith.constant), and require the yield operand to be the
// result of `expectedYield` (a TypeID describing which op produces the
// yielded value).  Returns true if the body is a "clean" expression of
// arithmetic ops with no other operations, mulOps/maxOps duplicated allowed
// (yield must come from expectedYield).
struct GenericBodyShape {
  llvm::SmallDenseSet<TypeID, 4> opKinds;
  TypeID yieldSource;
  bool wellFormed = false;
};

static GenericBodyShape analyzeGenericBody(linalg::GenericOp gen) {
  GenericBodyShape shape;
  Block &body = gen.getRegion().front();
  linalg::YieldOp yieldOp;
  for (Operation &op : body.getOperations()) {
    if (auto y = dyn_cast<linalg::YieldOp>(op)) {
      yieldOp = y;
      continue;
    }
    if (isa<arith::ConstantOp>(op))
      continue;
    // Only single-result arithmetic ops are accepted; anything else
    // (memory ops, control flow, multi-result) bails out.
    if (op.getNumResults() != 1)
      return shape;
    if (!op.getDialect() ||
        op.getDialect()->getNamespace() != "arith")
      return shape;
    shape.opKinds.insert(op.getName().getTypeID());
  }
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return shape;
  Operation *def = yieldOp.getOperand(0).getDefiningOp();
  if (!def)
    return shape;
  shape.yieldSource = def->getName().getTypeID();
  shape.wellFormed = true;
  return shape;
}

// Return true iff the body contains exactly the op kinds in `expected`
// (and yields from `expectedYield`).
static bool bodyHas(const GenericBodyShape &s, TypeID expectedYield,
                    llvm::ArrayRef<TypeID> expected) {
  if (!s.wellFormed || s.yieldSource != expectedYield)
    return false;
  if (s.opKinds.size() != expected.size())
    return false;
  for (TypeID t : expected)
    if (!s.opKinds.count(t))
      return false;
  return true;
}

// Activation-name registry for trailing-elementwise epilogues.
//
// Adding a new activation = one entry here: list the body op kinds + the
// yield-op kind.  Two flavors are registered: with-bias (the fused-form
// generic that consumes matmul result + rank-1 bias) and standalone (a
// chained generic with a single rank-2 input).
//
// Example: to add Gelu = 0.5 * x * (1 + erf(x/sqrt(2))) one would add
//   {{add, mul, erf}, add, "Gelu"} for standalone, and the bias variant.
struct ActivationShape {
  llvm::SmallVector<TypeID, 4> ops;
  TypeID yield;
  StringRef name;
};

static llvm::SmallVector<ActivationShape, 4>
fusedBiasActivationTable(MLIRContext *ctx) {
  // Body kinds for the *fused* (bias + activation) form.  All include
  // arith.addf for the bias.  Yield always comes from the activation tail.
  TypeID addId  = TypeID::get<arith::AddFOp>();
  TypeID mulId  = TypeID::get<arith::MulFOp>();
  TypeID maxId  = TypeID::get<arith::MaximumFOp>();
  (void)ctx;
  return {
    // bias + relu      : addf + maximumf,        yield = maximumf
    {{addId, maxId},        maxId, "BiasAddRelu"},
    // bias + leakyrelu : addf + mulf + maximumf, yield = maximumf
    {{addId, mulId, maxId}, maxId, "BiasAddLeakyRelu"},
  };
}

static llvm::SmallVector<ActivationShape, 4>
standaloneActivationTable(MLIRContext *ctx) {
  // Body kinds for the *standalone* activation (no bias) form.
  TypeID mulId = TypeID::get<arith::MulFOp>();
  TypeID maxId = TypeID::get<arith::MaximumFOp>();
  (void)ctx;
  return {
    // relu      : maximumf,        yield = maximumf
    {{maxId},        maxId, "Relu"},
    // leakyrelu : mulf + maximumf, yield = maximumf
    {{mulId, maxId}, maxId, "LeakyRelu"},
  };
}

// Recognize a *bias-add only* generic (no activation tail).  Used as the
// standalone-form 2-step fallback when the elementwise-fuse pass didn't
// merge bias with the activation generic.
static bool isStandaloneBiasAdd(const GenericBodyShape &s) {
  TypeID addId = TypeID::get<arith::AddFOp>();
  return s.wellFormed && s.yieldSource == addId && s.opKinds.size() == 1 &&
         s.opKinds.count(addId);
}

// Find the unique linalg.generic in `func` whose 1st DPS input is `value`
// and which matches `predicate`.  Returns null on no/multiple matches.
static linalg::GenericOp
findChainedGenericInFunc(func::FuncOp func, Value value,
                         llvm::function_ref<bool(linalg::GenericOp)> pred) {
  linalg::GenericOp matched;
  func.walk([&](linalg::GenericOp gen) {
    if (gen.getNumDpsInputs() < 1) return;
    if (gen.getDpsInputOperand(0)->get() != value) return;
    if (!pred(gen)) return;
    if (matched) { matched = {}; return; }
    matched = gen;
  });
  return matched;
}

// Classify the trailing-elementwise chain on a cube func.  Returns
// {has_bias, epilogue_kind_string} suitable for stamping abi_matmul_* attrs.
//
// Walk strategy (op-set based, no per-pattern matchers):
//   1. Find the linalg.matmul; its result is the chain root.
//   2. Try the *fused* form first: a single generic with bias-add structure
//      whose body op-set matches one of `fusedBiasActivationTable`.  This is
//      the shape `--linalg-fuse-elementwise-ops` produces when bias + act
//      sat in two source generics.
//   3. Otherwise fall back to two-step form: a bias-only generic, then a
//      standalone-activation generic.  Either may be missing (bias-only,
//      activation-only, or neither).
//
// Adding a new activation:
//   - Add one entry to fusedBiasActivationTable (for the bias-fused form)
//   - Add one entry to standaloneActivationTable (for the no-bias form)
//   - Downstream (CannTranslation::emitSupportedMixVectorEpilogue) emits the
//     kernel cpp body; that path has its own pattern walker and stays in
//     sync via its own MixPartitionSummary inference.
static std::pair<bool, StringRef>
classifyCubeEpilogueChain(func::FuncOp func, CubeKind cubeKind) {
  if (cubeKind != CubeKind::MatmulVecFuse)
    return {false, "None"};
  Value cur;
  func.walk([&](linalg::MatmulOp mm) {
    cur = mm.getResult(0);
    return WalkResult::interrupt();
  });
  if (!cur)
    return {false, "None"};

  MLIRContext *ctx = func.getContext();

  // Step 1: fused (bias + activation) single-generic form.
  auto fusedTable = fusedBiasActivationTable(ctx);
  for (const auto &shape : fusedTable) {
    auto matchFused = [&](linalg::GenericOp gen) {
      if (!hasCanonicalColumnBiasShape(gen))
        return false;
      return bodyHas(analyzeGenericBody(gen), shape.yield, shape.ops);
    };
    if (auto found = findChainedGenericInFunc(func, cur, matchFused))
      return {true, shape.name};
  }

  // Step 2: two-step chained form — optional bias-add then optional
  // standalone activation.
  bool hasBias = false;
  if (auto biasGen = findChainedGenericInFunc(
          func, cur, [](linalg::GenericOp gen) {
            if (!hasCanonicalColumnBiasShape(gen))
              return false;
            return isStandaloneBiasAdd(analyzeGenericBody(gen));
          })) {
    hasBias = true;
    cur = biasGen.getResult(0);
  }
  auto standaloneTable = standaloneActivationTable(ctx);
  for (const auto &shape : standaloneTable) {
    auto matchStandalone = [&](linalg::GenericOp gen) {
      if (!hasRank2IdentityPointwiseShape(gen))
        return false;
      return bodyHas(analyzeGenericBody(gen), shape.yield, shape.ops);
    };
    if (auto found = findChainedGenericInFunc(func, cur, matchStandalone)) {
      // Compose "BiasAdd<Act>" if both present, else just "<Act>".
      if (hasBias) {
        if (shape.name == "Relu") return {true, "BiasAddRelu"};
        if (shape.name == "LeakyRelu") return {true, "BiasAddLeakyRelu"};
      }
      return {hasBias, shape.name};
    }
  }
  if (hasBias)
    return {true, "BiasAdd"};
  return {false, "None"};
}

} // namespace

// CV-fusion Phase 2: materialize a Cube template draft.  Emits the 5 cube
// tunable i64 func args (XBLOCK_M, XBLOCK_N, M_INNER, N_INNER, K_INNER) and
// stamps `ascendc.kernel_kind = "mix"` + `afir.cube_kind = "<kind>"` attrs on
// the func.  The TileParam entries land in plan.tileable so emitTilingInfos
// renders them into `auto_fuse.tiling_infos`.  Phase 4 will read these in
// LoopNestBuilder + GroupEmitter to emit the 3-level scf.for nest.
//
// Axis indices are placeholders (0=M, 1=N, 2=K) for now — Phase 4 will line
// them up with the linalg.matmul's iteration dims.
static TilePlan buildCubePlan(func::FuncOp func, const CollapsedGroupInfo &info,
                              const TilePlanDraft &draft,
                              OpBuilder &builder, Location loc) {
  TilePlan plan;
  plan.group    = &info;
  plan.cubeKind = draft.cubeKind;

  // 5 cube tunable args.  Defaults are AF-style for f16 matmul: 128/128/64
  // outer; 32/32 inner-MN; 16 inner-K (one cube fragment).
  auto addInner = [&](StringRef name, int64_t defaultVal, int axisIdx,
                      AxisRole role) {
    Value param = insertFuncArg(func, builder, loc, defaultVal, name);
    SmallVector<TileParam> grp;
    grp.push_back({name.str(), param, OpFoldResult(param), axisIdx,
                    TileLevel::Inner, role});
    plan.tileable.push_back(std::move(grp));
  };
  auto addOuterInner = [&](StringRef outerName, int64_t outerDefault,
                           StringRef innerName, int64_t innerDefault,
                           int axisIdx, AxisRole role) {
    Value outerSsa = insertFuncArg(func, builder, loc, outerDefault, outerName);
    Value innerSsa = insertFuncArg(func, builder, loc, innerDefault, innerName);
    SmallVector<TileParam> grp;
    grp.push_back({outerName.str(), outerSsa, OpFoldResult(outerSsa), axisIdx,
                    TileLevel::Outer, role});
    grp.push_back({innerName.str(), innerSsa, OpFoldResult(innerSsa), axisIdx,
                    TileLevel::Inner, role});
    plan.tileable.push_back(std::move(grp));
  };

  // Order: M outer/inner, N outer/inner, K inner.  Matches AF tile spec
  // (matmul-add-leakyrelu/step5_ascendc.mlir args arg4..arg8).
  addOuterInner("XBLOCK_M", 128, "M_INNER", 32, 0, AxisRole::Parallel);
  addOuterInner("XBLOCK_N", 128, "N_INNER", 32, 1, AxisRole::Parallel);
  addInner     ("K_INNER",        16,                2, AxisRole::Reduction);

  // Mark the func as a mix kernel; AnnotateMixMatmulSemantics + downstream
  // passes use `ascendc.kernel_kind` to gate behavior.
  func->setAttr("ascendc.kernel_kind", builder.getStringAttr("mix"));
  // Record the cube template explicitly so observers (lit / future verifier)
  // can pin the picked path without re-running the analysis.
  StringRef cubeKindName = "None";
  switch (draft.cubeKind) {
  case CubeKind::None:          cubeKindName = "None"; break;
  case CubeKind::MatmulOnly:    cubeKindName = "MatmulOnly"; break;
  case CubeKind::MatmulVecFuse: cubeKindName = "MatmulVecFuse"; break;
  }
  func->setAttr("afir.cube_kind", builder.getStringAttr(cubeKindName));
  // Mirror AnnotateMixMatmulSemantics so MixAbiExtractor's required-attr set
  // is satisfied without re-running that legacy pass on cube-emitted kernels.
  // Phase 1 CV-fusion only supports the canonical matmul shape: no transpose,
  // ND layout on all sides.  bias/epilogue come from a tensor-IR walk of the
  // trailing chain (vs. the prior "always Relu / never bias" hardcoding).
  MLIRContext *ctx = builder.getContext();
  func->setAttr("abi_matmul_op_kind",     StringAttr::get(ctx, "matmul"));
  func->setAttr("abi_matmul_trans_a",     BoolAttr::get(ctx, false));
  func->setAttr("abi_matmul_trans_b",     BoolAttr::get(ctx, false));
  func->setAttr("abi_matmul_layout_a",    StringAttr::get(ctx, "ND"));
  func->setAttr("abi_matmul_layout_b",    StringAttr::get(ctx, "ND"));
  func->setAttr("abi_matmul_layout_c",    StringAttr::get(ctx, "ND"));
  auto [hasBias, epilogueKindName] =
      classifyCubeEpilogueChain(func, draft.cubeKind);
  func->setAttr("abi_matmul_has_bias", BoolAttr::get(ctx, hasBias));
  func->setAttr("abi_matmul_epilogue_kind",
                StringAttr::get(ctx, epilogueKindName));
  return plan;
}

// Materialize one tiling-case draft into a TilePlan: ≈ Scheduler::BlockSplit
// (pickBlockAxis → blockFusedAxes / XBLOCK) + Scheduler::TileSplit (the in-order
// pass below over the collapsed axes, dispatching per axis kind) + the §3.5
// row-loop degradation.  This is the *only* part of TilePlanGen that mutates
// `func` (it appends the tunable tile-size arguments).
static TilePlan buildPlan(func::FuncOp func, const CollapsedGroupInfo &info,
                          const AxisGrouping &g, const TilePlanDraft &draft,
                          OpBuilder &builder, Location loc) {
  // CV-fusion Phase 2: dispatch to the cube branch when the group is a Cube
  // template.  Cube plans don't share the vector axis-grouping pipeline below.
  if (draft.cubeKind != CubeKind::None)
    return buildCubePlan(func, info, draft, builder, loc);
  // P3b-2b: RCore TilePlan structure.  R axis (at draft.ubTilingAxisR) becomes
  // the block axis, dispatched on XBLOCK; an inner RBLOCK loop sweeps the
  // per-block R slice on each core.  LoopNestBuilder reuses its existing
  // Outer-block + Inner-with-parent-step machinery (the per-block inner ub is
  // `parentStep == XBLOCK` via the same code path that Common uses for ubY).
  // GroupEmitter codegen for RCore (per-block reduce + partial-output bump)
  // lands in P3b-2d; until then `genVectorTilePlan` guards with a clear error.
  if (draft.reduceIsBlock) {
    assert(g.yAxes.empty() &&
           "RCore restricted to true full-reduce in P3b-2 (P3b-4 widens)");
    assert(draft.ubTilingAxisR >= 0 && "RCore requires a ub-split R axis");

    TilePlan plan;
    plan.group = &info;
    plan.reduceTemplate = TilePlan::ReduceTemplate::RCore;
    plan.reduceIsBlock = true;
    (void)computeVectorizedDims(info, &plan);

    int rAxis = draft.ubTilingAxisR;
    plan.blockFusedAxes.push_back(rAxis);

    int xsubCount = 0, naxisCount = 0, rblockCount = 0;
    for (int i = 0; i < (int)info.collapsedAxes.size(); ++i) {
      const AxisClass &ax = g.axes[i];
      Value ext = getAxisExtentValue(builder, loc, info, i);

      if (ax.kind == AxisKind::X) {
        // Rare in a full-reduce, but mirror Common's handling.
        std::string name = llvm::formatv("XBLOCK_X_{0}", xsubCount++).str();
        Value param = insertFuncArg(func, builder, loc, 16, name);
        SmallVector<TileParam> group;
        group.push_back({name, param, OpFoldResult(ext), i, TileLevel::Inner,
                          AxisRole::Parallel});
        plan.tileable.push_back(std::move(group));
        if (plan.ubTilingAxisX < 0)
          plan.ubTilingAxisX = i;

      } else if (ax.kind == AxisKind::N) {
        std::string name = llvm::formatv("NAXIS_{0}", naxisCount++).str();
        plan.full.push_back({name, ext, OpFoldResult(ext), i, TileLevel::Full,
                              AxisRole::Parallel});

      } else if (ax.kind == AxisKind::Y) {
        llvm_unreachable("RCore: full-reduce has no Y axes (asserted above)");

      } else { // AxisKind::R
        if (i == rAxis) {
          // R as block axis: XBLOCK (Outer, ascendc.parallel) + RBLOCK (Inner).
          // LoopNestBuilder will give the Inner a ub of parentStep == XBLOCK
          // (per-block R slice), and the composed loopIVs[rAxis] = outer + inner.
          Value xblock = insertFuncArg(func, builder, loc, 128, "XBLOCK");
          std::string innerName =
              llvm::formatv("RBLOCK_{0}", rblockCount++).str();
          Value rblock = insertFuncArg(func, builder, loc, 64, innerName);
          SmallVector<TileParam> group;
          group.push_back({"XBLOCK", xblock, OpFoldResult(ext), i,
                            TileLevel::Outer, AxisRole::Reduction});
          group.push_back({innerName, rblock, OpFoldResult(xblock), i,
                            TileLevel::Inner, AxisRole::Reduction});
          plan.tileable.push_back(std::move(group));
          plan.ubTilingAxisR = i;
          plan.blockDimExprs.push_back(OpFoldResult(
              builder.create<arith::CeilDivSIOp>(loc, ext, xblock)));
        } else {
          // Secondary R axes (full-reduce typically has one) → full-load.
          std::string name =
              llvm::formatv("RBLOCK_{0}", rblockCount++).str();
          plan.full.push_back({name, ext, OpFoldResult(ext), i,
                                TileLevel::Full, AxisRole::Reduction});
        }
      }
    }
    return plan;
  }
  assert(draft.blockTilingId == 0 &&
         "blockTilingId != 0 only valid for RCore");

  TilePlan plan;
  plan.group = &info;
  if (!g.rAxes.empty())
    plan.reduceTemplate = draft.isFullLoad
                              ? TilePlan::ReduceTemplate::FullLoad
                              : TilePlan::ReduceTemplate::Common;
  plan.peelOuterR = draft.peelOuterR;

  DenseSet<int> vecDims = computeVectorizedDims(info, &plan);
  BlockPick     bp      = pickBlockAxis(g, vecDims);
  // The cost model may pick a non-block ub-Y axis; honor it by making *that*
  // axis the block axis (the pickBlockAxis default then becomes an ordinary
  // whole-loaded parallel axis).  When ubY < 0 (the §3.5 row-loop degradation,
  // or a group with no parallel axis) the block axis is the pickBlockAxis result
  // and gets the step-1 row loop instead of XBLOCK_SUB.
  int  blkAxis    = (draft.ubTilingAxisY >= 0) ? draft.ubTilingAxisY : bp.axis;
  bool blkRowLoop = bp.degradeToRowLoop; // implies draft.ubTilingAxisY < 0
  if (blkAxis >= 0)
    plan.blockFusedAxes.push_back(blkAxis);

  // --- ubSplit: one in-order pass over the collapsed axes (≈ TileSplit) ---
  // Counters preserved so func-arg insertion order / `auto_fuse.tiling_infos`
  // numbering match the previous implementation on the shapes that reach here.
  int rblockCount = 0, naxisCount = 0, xsubCount = 0;

  for (int i = 0; i < (int)info.collapsedAxes.size(); ++i) {
    const AxisClass &ax = g.axes[i];
    Value ext = getAxisExtentValue(builder, loc, info, i);

    if (ax.kind == AxisKind::X) {
      // Transpose X axis (input-side divergent): inner-tiled with its own
      // tunable; never the block axis (bindMultiCore == false).  Until the §5
      // transpose schedule generator gives X its own 16-fractal handling, every
      // X axis is an ordinary inner tile (and `draft.ubTilingAxisX` just names
      // the first one for the cost model).
      std::string name = llvm::formatv("XBLOCK_X_{0}", xsubCount++).str();
      Value param = insertFuncArg(func, builder, loc, 16, name);
      SmallVector<TileParam> group;
      group.push_back({name, param, OpFoldResult(ext), i, TileLevel::Inner,
                        AxisRole::Parallel});
      plan.tileable.push_back(std::move(group));
      if (plan.ubTilingAxisX < 0)
        plan.ubTilingAxisX = i;

    } else if (ax.kind == AxisKind::N) {
      // Transpose-N axis (trailing dim the permutation leaves in place): not
      // tiled, not looped — whole-dim slice (≈ AF's n_group).
      std::string name = llvm::formatv("NAXIS_{0}", naxisCount++).str();
      plan.full.push_back({name, ext, OpFoldResult(ext), i, TileLevel::Full,
                            AxisRole::Parallel});

    } else if (ax.kind == AxisKind::Y) {
      // Parallel axis (broadcast axes included — to the scheduler they are
      // ordinary Y axes; the lowering replicates the projecting operand
      // on-chip).
      if (i == blkAxis) {
        // Block axis (= ubY when the draft picked one, else the pickBlockAxis
        // default): XBLOCK (Outer) + inner level (XBLOCK_SUB, or step-1 row loop
        // under the §3.5 degradation).
        Value xblock = insertFuncArg(func, builder, loc, 128, "XBLOCK");
        SmallVector<TileParam> group;
        group.push_back({"XBLOCK", xblock, OpFoldResult(ext), i,
                          TileLevel::Outer, AxisRole::Parallel});
        if (blkRowLoop) {
          // Fixed step 1 (not a tunable func arg): one row per tile body.
          Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
          group.push_back({"XBLOCK_ROW", one, OpFoldResult(xblock), i,
                            TileLevel::Inner, AxisRole::Parallel});
        } else {
          Value xblockSub =
              insertFuncArg(func, builder, loc, 16, "XBLOCK_SUB");
          plan.ubTilingAxisY = i;
          group.push_back({"XBLOCK_SUB", xblockSub, OpFoldResult(xblock), i,
                            TileLevel::Inner, AxisRole::Parallel});
        }
        plan.tileable.push_back(std::move(group));
        plan.blockDimExprs.push_back(
            OpFoldResult(builder.create<arith::CeilDivSIOp>(loc, ext, xblock)));
      } else {
        // §3.4: every non-ub parallel axis is fully loaded — no plan entry, so
        // SliceComputer yields the whole-dim slice and there is no loop over it.
        // (This replaces the old per-axis `XBLOCK_SUB_n` and the broadcast-axis
        // `BCAST_n` step-1 / `BCAST_TILE_n` tunables; also the pre-existing
        // behavior for non-block parallel axes under the §3.5 degradation.)
      }

    } else { // AxisKind::R — reduction axis.
      std::string name = llvm::formatv("RBLOCK_{0}", rblockCount++).str();
      if (i == draft.ubTilingAxisR) {
        Value param = insertFuncArg(func, builder, loc, 64, name);
        SmallVector<TileParam> group;
        group.push_back({name, param, OpFoldResult(ext), i,
                          TileLevel::Inner, AxisRole::Reduction});
        plan.tileable.push_back(std::move(group));
        plan.ubTilingAxisR = i;
      } else {
        plan.full.push_back({name, ext, OpFoldResult(ext), i,
                              TileLevel::Full, AxisRole::Reduction});
      }
    }
  }
  return plan;
}

// Stamp tile-data legality constraints (≈ ATT tiling-data constraints, plan
// §6) onto `plan` for later emission into `auto_fuse.tiling_infos`:
//
//   Divides : XBLOCK_SUB | XBLOCK  (and analogues per tileable group)
//             — required by the existing tail-peel codegen path.
//   LeBytes : conservative on-chip footprint ≤ UB capacity
//             — `numBufs * elemBytes * Π per-axis-tile`.  Each axis's tile
//             is the inner tunable name when tileable, otherwise the static
//             extent / `argN_dimD` shape-key emitted by the SymExpr layer.
//             Skipped (no LeBytes emitted) if any axis fails to resolve;
//             that's also when the runtime UB-aware prune in CannTranslation
//             (`ub_cost_bytes_exprs`) takes over.
//
// These are *recorded* (and surfaced as JSON) — the constraint solver / cost
// model is the autotuner's job; `costEstimate` here just does a coarse static
// check via `staticTileFootprintBytes` already.
static void populateConstraints(TilePlan &plan, const CollapsedGroupInfo &info,
                                 func::FuncOp func, unsigned elemBytes,
                                 SocConstants soc) {
  // Divides: every (Outer, Inner) pair within a tileable group.
  for (auto &group : plan.tileable) {
    const TileParam *outer = nullptr, *inner = nullptr;
    for (auto &tp : group) {
      if (tp.level == TileLevel::Outer)      outer = &tp;
      else if (tp.level == TileLevel::Inner) inner = &tp;
    }
    if (outer && inner)
      plan.constraints.push_back(
          {TileConstraint::Divides, inner->name, outer->name});
  }

  // LeBytes: only when every axis has a usable tile-size expression.
  auto dimSymsAttr = func->getAttrOfType<ArrayAttr>("afir.dim_symbols");
  std::optional<symshape::DimSymbolTable> symTable;
  if (dimSymsAttr)
    symTable = symshape::DimSymbolTable::fromAttr(dimSymsAttr);
  auto nameFor = [&](symshape::SymId id) -> std::string {
    if (!symTable) return "?";
    auto src = symTable->sourceOf(id);
    return "arg" + std::to_string(src.first) + "_dim" +
           std::to_string(src.second);
  };

  // Per-axis tile-size string.
  auto axisTileStr = [&](int axisIdx) -> std::string {
    for (auto &grp : plan.tileable)
      for (auto &tp : grp)
        if (tp.axisIdx == axisIdx && tp.level == TileLevel::Inner)
          return tp.name;
    const auto &ax = info.collapsedAxes[axisIdx];
    if (ax.staticSize != ShapedType::kDynamic)
      return std::to_string(ax.staticSize);
    if (ax.extent.isValid() && symTable)
      return ax.extent.emitC(nameFor);
    return ""; // unknown
  };

  // Tail-offset alignment (≈ AF kAligned default at AlignmentStrategy).  The
  // LoopNestBuilder overlap-tail emits the tail slice at GM offset
  // `extent - innerTileStep` (composed with the outer block IV that lands on a
  // tile boundary).  For the tail's GM DataCopy to be 32-byte aligned, we need
  //   ((extent - INNER_TILE) * elemBytes) % 32 == 0
  // Encoded as a Divides constraint with lhs=32 and rhs the byte offset expr.
  // When the inner tile evenly divides `extent` (no tail) the rhs is
  // `(K-1)*INNER_TILE*elemBytes`; since `INNER_TILE*elemBytes` is itself 32-
  // byte aligned (the AscendC DataCopy size requirement, already implicit in
  // our XBLOCK_SUB choice), this is trivially satisfied.  The constraint only
  // bites when tail fires AND the residual offset is not 32-byte aligned —
  // exactly the f16-reduce-tail bug class (reduce-sum-3d-f16-tail-e2e).
  //
  // Emitted on the innermost Inner-level TileParam per tileable group (the
  // axis that LoopNestBuilder picks for tail-peel: highest-axisIdx Inner).
  // Skipped if the axis extent is unresolvable.  Emitted BEFORE the LeBytes
  // early-return below so a partial per-axis product doesn't suppress this
  // (the alignment expr only needs the one axis's extent, not all).
  const TileParam *innermostInner = nullptr;
  for (auto &grp : plan.tileable)
    for (auto &tp : grp)
      if (tp.level == TileLevel::Inner)
        if (!innermostInner || tp.axisIdx > innermostInner->axisIdx)
          innermostInner = &tp;
  if (innermostInner) {
    std::string extent;
    const auto &ax = info.collapsedAxes[innermostInner->axisIdx];
    if (ax.staticSize != ShapedType::kDynamic)
      extent = std::to_string(ax.staticSize);
    else if (ax.extent.isValid() && symTable)
      extent = ax.extent.emitC(nameFor);
    if (!extent.empty()) {
      std::string rhs = "((" + extent + " - " + innermostInner->name + ") * " +
                        std::to_string(elemBytes) + ")";
      plan.constraints.push_back(
          {TileConstraint::Divides, "32", std::move(rhs)});
    }
  }

  std::string product;
  bool ok = true;
  for (int i = 0; i < (int)info.collapsedAxes.size(); ++i) {
    std::string s = axisTileStr(i);
    if (s.empty()) { ok = false; break; }
    product = product.empty() ? s : ("(" + product + " * " + s + ")");
  }
  if (!ok || product.empty()) return;

  int64_t numBufs = 2;
  if (!info.topoMembers.empty())
    numBufs = (int64_t)info.topoMembers[0]->getNumOperands() + 1;
  std::string footprint = "((" + std::to_string(numBufs * (int64_t)elemBytes) +
                          ") * " + product + ")";
  plan.constraints.push_back(
      {TileConstraint::LeBytes, footprint, std::to_string(soc.ubBytes)});
}

TilePlan genVectorTilePlan(func::FuncOp func,
                            const CollapsedGroupInfo &info,
                            OpBuilder &builder, Location loc,
                            bool enableReductionSplit,
                            llvm::StringRef socName) {
  const AxisGrouping &g = info.grouping; // computed by the Collapse pass
  unsigned elemBytes = operandElemBytes(info);
  DenseSet<int> vecDims = computeVectorizedDims(info);
  SocConstants soc = getSocConstants(socName);

  SmallVector<TilePlanDraft> drafts =
      enumerateTilingCases(g, info, enableReductionSplit, elemBytes);
  assert(!drafts.empty() && "enumerateTilingCases must yield at least one draft");

  // argmin score (≈ score_func selection); ties → enumeration order, which puts
  // the "current scheduler" choice first.  Only the winner is materialized —
  // buildPlan is the one place that mutates `func`.
  const TilePlanDraft *best = &drafts.front();
  double bestScore = costEstimate(g, info, vecDims, *best, elemBytes, soc);
  for (const TilePlanDraft &d : llvm::drop_begin(drafts)) {
    double s = costEstimate(g, info, vecDims, d, elemBytes, soc);
    if (s < bestScore) { bestScore = s; best = &d; }
  }
  // Release-safe barrier (was a debug-only `assert`): if every candidate
  // tiling is infeasible, building a kernel from the best-of-infeasible draft
  // silently produces a UB-overflowing / WIP-codegen kernel.  Fail loud instead
  // of miscompiling.  The passing pipeline always finds a feasible plan, so this
  // never fires for valid kernels.
  if (bestScore >= kInfeasible)
    llvm::report_fatal_error(
        "auto-fuse: no feasible tiling case (all candidate tilings exceed the "
        "UB budget or hit a WIP codegen path); refusing to build an infeasible "
        "kernel");

  TilePlan plan = buildPlan(func, info, g, *best, builder, loc);
  populateConstraints(plan, info, func, elemBytes, soc);
  // P3b-2c/d: LoopNestBuilder is expected to handle RCore via its existing
  // Outer-Inner-on-same-axis path (parentStep mechanism), but the GroupEmitter
  // codegen — per-block R extent on the new rFor + dimension-bumped partial
  // output — lands in P3b-2d.  Surface a clear error until then.
  if (plan.reduceTemplate == TilePlan::ReduceTemplate::RCore) {
    llvm::errs() << "[auto-fuse] RCore TilePlan built (P3b-2b), but "
                    "GroupEmitter codegen WIP (P3b-2d). Plan: "
                    "docs/superpowers/plans/2026-05-14-p3b-rcore-reduce-multicore.zh.md\n";
    /* P3b-2d: guard removed — RCore codegen lives in GroupEmitter now. */
  }
  return plan;
}

// Populate `schema.fields` with Tunable entries for every tile-param block
// argument in `plan.tileable`, in the order they appear (matches the existing
// TilingData struct layout).
static void buildSchemaTunableFields(const TilePlan &plan, func::FuncOp func,
                                     TilingInfoSchema &schema) {
  for (auto &group : plan.tileable) {
    for (const auto &tp : group) {
      auto ba = dyn_cast<BlockArgument>(tp.ssa);
      if (!ba) continue;
      int64_t defaultVal = 0;
      if (auto attr = func.getArgAttrOfType<IntegerAttr>(
              ba.getArgNumber(), "auto_fuse.default_tile_size"))
        defaultVal = attr.getInt();
      int64_t axisSize = -1;
      if (plan.group && tp.axisIdx >= 0 &&
          tp.axisIdx < (int)plan.group->collapsedAxes.size()) {
        int64_t s = plan.group->collapsedAxes[tp.axisIdx].staticSize;
        if (s != ShapedType::kDynamic) axisSize = s;
      }
      assert(ba.getArgNumber() <= (unsigned)INT32_MAX && "arg_index overflow");
      SchemaField f;
      f.name         = tp.name;
      f.kind         = SchemaFieldKind::Tunable;
      f.axisSize     = axisSize;
      f.defaultValue = defaultVal;
      f.argIndex     = (int32_t)ba.getArgNumber();
      schema.fields.push_back(std::move(f));
    }
  }
}

// Populate `schema.args` from the kernel func signature: input / tile-param /
// workspace from block arguments, plus synthetic Output entries from the
// `func.return` operands (pre-bufferize the outputs are not yet args).
static void buildSchemaArgs(func::FuncOp func, TilingInfoSchema &schema) {
  Block &entry = func.getBody().front();
  unsigned numNetworkInputs = 0;
  for (BlockArgument ba : entry.getArguments()) {
    SchemaArg a;
    a.mlirIndex = (int32_t)ba.getArgNumber();
    Type ty = ba.getType();
    if (auto mt = dyn_cast<MemRefType>(ty)) {
      // Workspace heuristic: post-bufferize the workspace memref is always
      // memref<...xi8> (added by canonicalize-cann-signature).  Kernels with
      // genuine i8 user data (quantized inputs) would currently mis-classify
      // here; revisit when such a kernel lands by gating on an explicit arg
      // attribute stamped by the workspace-allocation pass.
      if (mt.getElementType().isInteger(8)) {
        a.role = SchemaArgRole::Workspace;
      } else if (mt.getLayout().isIdentity()) {
        a.role = SchemaArgRole::Input;
        if (auto attr = func.getArgAttrOfType<IntegerAttr>(
                ba.getArgNumber(), "auto_fuse.call_arg_index"))
          a.callArgIndex = (int32_t)attr.getInt();
        else
          a.callArgIndex = (int32_t)numNetworkInputs;
        ++numNetworkInputs;
      } else {
        // TODO multi-result: resultIndex hard-coded to 0; revisit when
        // kernels with >1 DPS output exist.
        a.role = SchemaArgRole::Output;
        a.resultIndex = 0;
      }
    } else if (isa<RankedTensorType>(ty)) {
      // Pre-bufferize: every tensor arg is an input (DPS init tensors are
      // tensor.empty results, not func args).
      a.role = SchemaArgRole::Input;
      if (auto attr = func.getArgAttrOfType<IntegerAttr>(
              ba.getArgNumber(), "auto_fuse.call_arg_index"))
        a.callArgIndex = (int32_t)attr.getInt();
      else
        a.callArgIndex = (int32_t)numNetworkInputs;
      ++numNetworkInputs;
    } else if (isa<IndexType>(ty)) {
      a.role = SchemaArgRole::TileParam;
      for (auto &f : schema.fields)
        if (f.kind == SchemaFieldKind::Tunable &&
            f.argIndex == a.mlirIndex)
          a.tileParamName = f.name;
    } else {
      continue; // unknown arg type — skip
    }
    schema.args.push_back(std::move(a));
  }

  // Synthesize Output SchemaArg entries from func return values.  Pre-
  // bufferize the kernel returns tensors; one-shot-bufferize will later
  // append a corresponding output memref arg per result at position
  // numArgs + i.  We pre-assign that anticipated mlirIndex so post-
  // bufferize consumers can index into the (now-larger) arg list.
  // TODO multi-result: works correctly today (one Output per return).
  if (entry.empty()) return;
  auto retOp = dyn_cast<func::ReturnOp>(entry.getTerminator());
  if (!retOp) return;
  // Skip if we already classified a real Output arg above
  // (post-bufferize path).  Invariant across iterations — the loop only
  // appends new Outputs, so checking once before the loop suffices.
  bool alreadyHaveOutput = llvm::any_of(schema.args, [&](auto &x) {
    return x.role == SchemaArgRole::Output;
  });
  if (alreadyHaveOutput) return;
  unsigned baseIdx = entry.getNumArguments();
  for (auto [i, v] : llvm::enumerate(retOp.getOperands())) {
    if (!isa<RankedTensorType, MemRefType>(v.getType())) continue;
    SchemaArg a;
    a.mlirIndex   = (int32_t)(baseIdx + i);
    a.role        = SchemaArgRole::Output;
    a.resultIndex = (int32_t)i;
    schema.args.push_back(std::move(a));
  }
}

// For each Output SchemaArg, populate `shapeExpr[]` per output dim using the
// symbolic-shape table for real args, or by tracing the synthetic-output
// linalg.generic → tensor.empty → tensor.dim chain back to an Input arg.
static void computeOutputShapeExprs(
    func::FuncOp func, TilingInfoSchema &schema,
    const std::optional<symshape::DimSymbolTable> &symTable) {
  Block &entry = func.getBody().front();

  auto tryEmitShapeExpr = [&](SchemaArg &a, ShapedType st,
                              StringAttr symAttr) {
    auto symList = symAttr ? symshape::parseSymExprList(symAttr.getValue())
                           : std::nullopt;
    for (int64_t d = 0, e = st.getRank(); d < e; ++d) {
      std::string expr;
      if (symList && (size_t)d < symList->size() && symTable) {
        const auto &se = (*symList)[d];
        if (se.getKind() == symshape::SymExpr::Kind::Sym) {
          auto src = symTable->sourceOf(se.getSym());
          for (auto &ia : schema.args)
            if (ia.role == SchemaArgRole::Input &&
                (unsigned)ia.mlirIndex == src.first) {
              expr = "arg" + std::to_string(ia.callArgIndex) +
                     "_dim" + std::to_string(src.second);
              break;
            }
        }
      }
      if (expr.empty() && d < (int64_t)st.getShape().size() &&
          !ShapedType::isDynamic(st.getShape()[d]))
        expr = std::to_string(st.getShape()[d]);
      a.shapeExpr.push_back(expr);
    }
  };

  // Helper for synthetic-output dynamic dims: trace
  //   retOp.getOperand(ri) → linalg.generic → tensor.empty(%d0, %d1, ...)
  //   → tensor.dim %argN, %cIdx  where %argN is an Input schema arg.
  // Render "arg<callArgIndex>_dim<dimIdx>" on success.  The plan spec §6
  // explicitly defers anything more involved (reduce/matmul) to later.
  auto resolveSyntheticDynDim =
      [&](Value retVal, int64_t outDim) -> std::string {
    auto genericOp = retVal.getDefiningOp<linalg::GenericOp>();
    if (!genericOp || genericOp.getOutputs().empty()) {
      LLVM_DEBUG(llvm::dbgs() << "[tiling-info] synthetic output dim "
                              << outDim << ": no defining linalg.generic\n");
      return {};
    }
    auto emptyOp = genericOp.getOutputs()[0].getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp) {
      LLVM_DEBUG(llvm::dbgs() << "[tiling-info] synthetic output dim "
                              << outDim << ": init not tensor.empty\n");
      return {};
    }
    auto mixed = emptyOp.getMixedSizes();
    if ((size_t)outDim >= mixed.size()) return {};
    auto ofr = mixed[outDim];
    auto val = dyn_cast<Value>(ofr);
    if (!val) return {};
    auto dimOp = val.getDefiningOp<tensor::DimOp>();
    if (!dimOp) {
      LLVM_DEBUG(llvm::dbgs() << "[tiling-info] synthetic output dim "
                              << outDim << ": dyn size not tensor.dim\n");
      return {};
    }
    auto ba = dyn_cast<BlockArgument>(dimOp.getSource());
    if (!ba || ba.getOwner() != &entry) return {};
    auto cst = dimOp.getIndex().getDefiningOp<arith::ConstantOp>();
    if (!cst) return {};
    auto intAttr = dyn_cast<IntegerAttr>(cst.getValue());
    if (!intAttr) return {};
    int32_t dimIdx = (int32_t)intAttr.getValue().getSExtValue();
    for (auto &ia : schema.args) {
      if (ia.role == SchemaArgRole::Input &&
          (unsigned)ia.mlirIndex == ba.getArgNumber()) {
        return "arg" + std::to_string(ia.callArgIndex) +
               "_dim" + std::to_string(dimIdx);
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "[tiling-info] synthetic output dim "
                            << outDim << ": no matching Input arg\n");
    return {};
  };

  for (auto &a : schema.args) {
    if (a.role != SchemaArgRole::Output) continue;
    ShapedType st;
    StringAttr symAttr;
    Value retVal;
    bool synthetic = false;
    if ((unsigned)a.mlirIndex < entry.getNumArguments()) {
      // Real arg (post-bufferize path).
      st = cast<ShapedType>(entry.getArgument(a.mlirIndex).getType());
      symAttr = func.getArgAttrOfType<StringAttr>(
          a.mlirIndex, "afir.symbolic_shape");
    } else if (auto retOp =
                   dyn_cast<func::ReturnOp>(entry.getTerminator())) {
      // Synthetic output: use the corresponding return value type.
      unsigned ri = (unsigned)a.resultIndex;
      if (ri < retOp.getNumOperands()) {
        retVal = retOp.getOperand(ri);
        st = dyn_cast<ShapedType>(retVal.getType());
        synthetic = true;
      }
    }
    if (!st) continue;
    if (!synthetic) {
      tryEmitShapeExpr(a, st, symAttr);
      continue;
    }
    // Synthetic-output branch: symAttr is unavailable.  Use static dims
    // directly; for dynamic dims, trace the linalg.generic → tensor.empty
    // → tensor.dim chain back to an Input schema arg.
    for (int64_t d = 0, e = st.getRank(); d < e; ++d) {
      std::string expr;
      if (!ShapedType::isDynamic(st.getShape()[d])) {
        expr = std::to_string(st.getShape()[d]);
      } else {
        expr = resolveSyntheticDynDim(retVal, d);
      }
      a.shapeExpr.push_back(expr);
    }
  }
}

// Walk memref.dim / tensor.dim ops in the kernel body, dedup via (argN, dimIdx),
// and append ShapeDerived SchemaField entries for each unique source dim.
static void collectShapeDerivedFields(func::FuncOp func,
                                      TilingInfoSchema &schema) {
  Block &entry = func.getBody().front();
  llvm::DenseSet<std::pair<int32_t, int32_t>> seenDims;
  auto recordDim = [&](BlockArgument ba, int32_t dimI) {
    if (!ba || ba.getOwner() != &entry) return;
    int32_t argN = (int32_t)ba.getArgNumber();
    // Skip if already a tunable on the same MLIR arg (won't normally
    // collide -- tunables are index-typed, dim sources are shaped -- but
    // dedup is cheap insurance).
    for (auto &f : schema.fields)
      if (f.kind == SchemaFieldKind::Tunable &&
          f.argIndex == argN)
        return;
    if (!seenDims.insert({argN, dimI}).second) return;
    SchemaField f;
    f.name      = "dim_arg" + std::to_string(argN) + "_" + std::to_string(dimI);
    f.kind      = SchemaFieldKind::ShapeDerived;
    f.sourceArg = argN;
    f.sourceDim = dimI;
    schema.fields.push_back(std::move(f));
  };
  func.walk([&](Operation *op) {
    if (auto dimOp = dyn_cast<memref::DimOp>(op)) {
      auto ba = dyn_cast<BlockArgument>(dimOp.getSource());
      if (auto cst = dimOp.getIndex().getDefiningOp<arith::ConstantOp>())
        if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
          recordDim(ba, (int32_t)intAttr.getValue().getSExtValue());
    } else if (auto dimOp = dyn_cast<tensor::DimOp>(op)) {
      auto ba = dyn_cast<BlockArgument>(dimOp.getSource());
      if (auto cst = dimOp.getIndex().getDefiningOp<arith::ConstantOp>())
        if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
          recordDim(ba, (int32_t)intAttr.getValue().getSExtValue());
    }
  });
}

// Group Input args' shape dims by their shared shape-symbol root, so the
// runtime can validate that user-passed input shapes are mutually consistent
// (e.g. for `a + b` with a,b both `?x?x?`, a.dim_i must equal b.dim_i).
// Reads `afir.dim_symbols` (func attr) + per-arg `afir.symbolic_shape`.
// Skips constant entries (broadcast `1`) and unrecognized SymExpr forms.
static void collectShapeEqualities(func::FuncOp func,
                                   TilingInfoSchema &schema) {
  auto dimSymsAttr = func->getAttrOfType<ArrayAttr>("afir.dim_symbols");
  if (!dimSymsAttr) return;
  auto symTable = symshape::DimSymbolTable::fromAttr(dimSymsAttr);
  if (!symTable) return;

  // Group (callArgIndex, dim) by the (rootArg, rootDim) the table resolves
  // their SymId to.  Composite key via DenseMap<pair<unsigned,unsigned>>.
  llvm::DenseMap<std::pair<unsigned, unsigned>,
                 llvm::SmallVector<std::pair<int32_t, int32_t>, 4>>
      groups;
  for (auto &a : schema.args) {
    if (a.role != SchemaArgRole::Input) continue;
    if (a.callArgIndex < 0) continue;
    auto symAttr = func.getArgAttrOfType<StringAttr>(
        (unsigned)a.mlirIndex, "afir.symbolic_shape");
    if (!symAttr) continue;
    auto list = symshape::parseSymExprList(symAttr.getValue());
    if (!list) continue;
    for (int32_t d = 0; d < (int32_t)list->size(); ++d) {
      const auto &e = (*list)[d];
      if (e.getKind() != symshape::SymExpr::Kind::Sym) continue; // skip consts
      if (e.getSym() >= symTable->numRoots()) continue;
      auto src = symTable->sourceOf(e.getSym());
      groups[{src.first, src.second}].push_back({a.callArgIndex, d});
    }
  }

  // Stable output order: sort group keys.
  llvm::SmallVector<std::pair<unsigned, unsigned>, 8> keys;
  for (auto &kv : groups) keys.push_back(kv.first);
  llvm::sort(keys);
  for (auto &k : keys) {
    auto &group = groups[k];
    if (group.size() < 2) continue; // singleton: nothing to validate
    schema.shapeEqualities.push_back(std::move(group));
  }
}

// Serialize TilePlan constraints into the existing {kind, lhs, rhs} dict
// array; returns a null ArrayAttr when there are no constraints.
static ArrayAttr serializeConstraints(MLIRContext *ctx,
                                      ArrayRef<TileConstraint> constraints) {
  if (constraints.empty()) return ArrayAttr();
  SmallVector<Attribute> cs;
  for (auto &c : constraints) {
    NamedAttrList ca;
    ca.append("kind",
              StringAttr::get(ctx, c.kind == TileConstraint::Divides
                                       ? "divides" : "le_bytes"));
    ca.append("lhs", StringAttr::get(ctx, c.lhs));
    ca.append("rhs", StringAttr::get(ctx, c.rhs));
    cs.push_back(ca.getDictionary(ctx));
  }
  return ArrayAttr::get(ctx, cs);
}

// Append the schema dict to the module-level `auto_fuse.tiling_infos`
// ArrayAttr, preserving any existing entries.
static void serializeAndAttach(ModuleOp moduleOp, const TilingInfoSchema &schema,
                               ArrayAttr constraintsAttr) {
  MLIRContext *ctx = moduleOp.getContext();
  DictionaryAttr entryDict =
      serializeTilingInfoSchema(ctx, schema, constraintsAttr);
  StringRef attrName = "auto_fuse.tiling_infos";
  SmallVector<Attribute> infos;
  if (auto existing = moduleOp->getAttrOfType<ArrayAttr>(attrName))
    llvm::append_range(infos, existing.getValue());
  infos.push_back(entryDict);
  moduleOp->setAttr(attrName, ArrayAttr::get(ctx, infos));
}

void emitTilingInfos(func::FuncOp func, const TilePlan &plan) {
  MLIRContext *ctx = func.getContext();
  auto moduleOp = func->getParentOfType<ModuleOp>();
  if (!moduleOp) return;

  // block_dim_expr: a SymExpr string `ceil(<block axis extent>/XBLOCK)` with the
  // block axis extent rendered in `argN_dimD` shape-key terms (the autotuner's
  // var names).  Built from afir.axis_extents / afir.dim_symbols (set by
  // afir-symbolize-shapes + Collapse) -- absent when those aren't available
  // (e.g. multi-op funcs).  Also stamped on the func so afir-translate can lift
  // it into tiling_space.json without re-deriving.
  std::string blockDimExpr;
  {
    auto dimSymsAttr = func->getAttrOfType<ArrayAttr>("afir.dim_symbols");
    auto axisExtAttr = func->getAttrOfType<ArrayAttr>("afir.axis_extents");
    StringRef xblockName;
    for (auto &grp : plan.tileable)
      for (const auto &tp : grp)
        if (tp.level == TileLevel::Outer)
          xblockName = tp.name;
    std::optional<symshape::DimSymbolTable> symTable;
    if (dimSymsAttr)
      symTable = symshape::DimSymbolTable::fromAttr(dimSymsAttr);
    if (axisExtAttr && symTable && !xblockName.empty() &&
        !plan.blockFusedAxes.empty()) {
      symshape::SymExpr ext;
      bool ok = true;
      for (int ax : plan.blockFusedAxes) {
        if (ax < 0 || ax >= (int)axisExtAttr.size()) { ok = false; break; }
        auto s = dyn_cast<StringAttr>(axisExtAttr[ax]);
        auto e = s ? symshape::parseSymExpr(s.getValue()) : std::nullopt;
        if (!e || !e->isValid()) { ok = false; break; }
        ext = ext.isValid() ? symshape::SymExpr::mul(ext, *e) : *e;
      }
      if (ok && ext.isValid()) {
        auto nameFor = [&](symshape::SymId id) -> std::string {
          auto src = symTable->sourceOf(id);
          return "arg" + std::to_string(src.first) + "_dim" +
                 std::to_string(src.second);
        };
        std::string extExpr = ext.emitC(nameFor);
        blockDimExpr = "ceil(" + extExpr + "/" + xblockName.str() + ")";
        // Stamp the bare extent expression too — afir-translate lifts it into
        // tiling_space.json so the runner can evaluate the runtime upper bound
        // (sub product of arg*_dim*) and prune candidates with XBLOCK > extent.
        func->setAttr("afir.axis_extent_expr",
                      StringAttr::get(ctx, extExpr));
      }
    }
  }
  // Static fallback: when symbolic attrs are absent (e.g. an outlined kernel
  // whose static shape never went through afir-symbolize-shapes) but every
  // block-fused axis has a known static extent, emit a literal-int expr. The
  // autotuner's grammar accepts plain integers, so eval just folds.
  if (blockDimExpr.empty() && plan.group && !plan.blockFusedAxes.empty()) {
    StringRef xblockName;
    for (auto &grp : plan.tileable)
      for (const auto &tp : grp)
        if (tp.level == TileLevel::Outer)
          xblockName = tp.name;
    if (!xblockName.empty()) {
      int64_t extent = 1;
      bool allStatic = true;
      for (int ax : plan.blockFusedAxes) {
        if (ax < 0 || ax >= (int)plan.group->collapsedAxes.size()) {
          allStatic = false; break;
        }
        int64_t s = plan.group->collapsedAxes[ax].staticSize;
        if (s == ShapedType::kDynamic) { allStatic = false; break; }
        extent *= s;
      }
      if (allStatic && extent > 0) {
        std::string extExpr = std::to_string(extent);
        blockDimExpr = "ceil(" + extExpr + "/" + xblockName.str() + ")";
        func->setAttr("afir.axis_extent_expr",
                      StringAttr::get(ctx, extExpr));
      }
    }
  }
  if (!blockDimExpr.empty())
    func->setAttr("afir.block_dim_expr", StringAttr::get(ctx, blockDimExpr));

  // Stamp the picked reduce template so downstream passes / lit tests can
  // observe which path was taken without re-running the cost model.  Skipped
  // for non-reduce kernels (ReduceTemplate::None).
  StringRef rtName;
  switch (plan.reduceTemplate) {
  case TilePlan::ReduceTemplate::Common:   rtName = "Common";   break;
  case TilePlan::ReduceTemplate::FullLoad: rtName = "FullLoad"; break;
  case TilePlan::ReduceTemplate::RCore:    rtName = "RCore";    break;
  case TilePlan::ReduceTemplate::None:     break;
  }
  if (!rtName.empty())
    func->setAttr("afir.reduce_template", StringAttr::get(ctx, rtName));

  // ─── Build schema_version=2 TilingInfoSchema ────────────────────────────
  // Authored here (pre-bufferize) so that arg_index references for tunable
  // tile-params land on the index-typed BlockArguments TilePlanGen just
  // injected.  Inputs are tensor-typed at this stage; outputs are not yet
  // appended as args (one-shot-bufferize will add them post-pipeline).  We
  // synthesize Output SchemaArg entries from the func return types using
  // the anticipated post-bufferize positions (numArgs + resultIdx).
  TilingInfoSchema schema;
  schema.kernelId     = func.getName().str();
  schema.blockDimExpr = blockDimExpr;
  if (auto a = func->getAttrOfType<StringAttr>("afir.axis_extent_expr"))
    schema.axisExtentExpr = a.getValue().str();

  buildSchemaTunableFields(plan, func, schema);
  buildSchemaArgs(func, schema);

  // Symbolic-shape table for output shape_expr derivation.  Absent for
  // kernels that never went through afir-symbolize-shapes; computeOutput-
  // ShapeExprs then falls back to literal static dims / synthetic tracing.
  std::optional<symshape::DimSymbolTable> symTable;
  if (auto a = func->getAttrOfType<ArrayAttr>("afir.dim_symbols"))
    symTable = symshape::DimSymbolTable::fromAttr(a);
  computeOutputShapeExprs(func, schema, symTable);

  collectShapeDerivedFields(func, schema);
  collectShapeEqualities(func, schema);

  ArrayAttr constraintsAttr = serializeConstraints(ctx, plan.constraints);
  serializeAndAttach(moduleOp, schema, constraintsAttr);
}

// ===========================================================================
// Public entry points used by AutoFuseTileFusePass for P1b multi-variant
// codegen (one func per feasible TilePlanDraft).
// ===========================================================================
SmallVector<auto_fuse::TilePlanDraft>
enumerateFeasibleDrafts(const auto_fuse::CollapsedGroupInfo &info,
                        bool enableReductionSplit,
                        bool relaxNonBlockUbY,
                        llvm::StringRef socName) {
  // CV-fusion Phase 2: Cube kernels enumerate exactly one cube draft.  Real
  // cube TilingCase enumeration (AF's GenMatmulTilingCase) — picking among
  // multiple tile-shape candidates — is deferred to Phase 2 v2.  For now,
  // hardcoded preset defaults (128/128 outer × 32/32 inner × 16 K).
  if (info.kind == GroupInfo::Kind::Cube) {
    auto_fuse::TilePlanDraft d;
    bool hasTrailingVec = false;
    for (linalg::LinalgOp op : info.topoMembers)
      if (!isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
                linalg::MatmulTransposeBOp, linalg::BatchMatmulOp>(
              op.getOperation())) {
        hasTrailingVec = true;
        break;
      }
    d.cubeKind = hasTrailingVec ? auto_fuse::CubeKind::MatmulVecFuse
                                 : auto_fuse::CubeKind::MatmulOnly;
    return {d};
  }
  const auto_fuse::AxisGrouping &g = info.grouping;
  unsigned elemBytes = operandElemBytes(info);
  DenseSet<int> vecDims = computeVectorizedDims(info);
  SocConstants soc = getSocConstants(socName);
  auto drafts = enumerateTilingCases(g, info, enableReductionSplit, elemBytes);
  SmallVector<auto_fuse::TilePlanDraft> feasible;
  for (const auto &d : drafts) {
    double s = costEstimate(g, info, vecDims, d, elemBytes, soc,
                             relaxNonBlockUbY);
    if (s < kInfeasible) feasible.push_back(d);
  }
  return feasible;
}

auto_fuse::TilePlan
buildPlanForDraft(func::FuncOp func,
                  const auto_fuse::CollapsedGroupInfo &info,
                  const auto_fuse::TilePlanDraft &draft,
                  OpBuilder &builder, Location loc,
                  llvm::StringRef socName) {
  TilePlan plan = buildPlan(func, info, info.grouping, draft, builder, loc);
  // Vector-axis constraints (Divides/LeBytes) don't apply to cube plans —
  // cube tile-data legality is enforced by the existing mix-compiler /
  // matmul tiling library (Phase 6 wires this).  Skip to keep the schema
  // clean.
  if (plan.cubeKind == auto_fuse::CubeKind::None)
    populateConstraints(plan, info, func, operandElemBytes(info),
                        getSocConstants(socName));
  return plan;
}

} // namespace mlir::afir
