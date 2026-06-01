#include "TilePlanGenInternal.h"
#include "TileFuseUtils.h"
#include "Conversion/AutoFuse/TilePlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

using namespace mlir;
using namespace mlir::auto_fuse;

namespace mlir::afir {

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

} // namespace

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
std::pair<bool, StringRef>
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

} // namespace mlir::afir
