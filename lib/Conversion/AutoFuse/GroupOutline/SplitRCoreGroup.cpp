//===- SplitRCoreGroup.cpp - Split full-reduce kernel into partial+combine -===//
//
// Runs after auto-fuse-group-outline.  See the .td description / plan doc
// docs/superpowers/plans/2026-05-14-p3b-rcore-reduce-multicore.zh.md §5.
//
//===---------------------------------------------------------------------===//

#include "Conversion/AutoFuse/AutoFusePasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DECL_AUTOFUSESPLITRCOREGROUP
#define GEN_PASS_DEF_AUTOFUSESPLITRCOREGROUP
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

// True when `fn`'s body is a single linalg.generic with all-reduction iterator
// types, a rank-1 input and a rank-0 init/output.  The canonical R1 / full-
// reduce-to-scalar shape; first-cut SplitRCoreGroup handles only this.
struct ReduceCandidate {
  linalg::GenericOp op;
  int64_t          extent;       // input rank-1 extent (R)
  Type             elemType;
  Value            inputArg;     // func arg passed as ins()
  Value            initArg;      // func arg passed as outs()
};

static std::optional<ReduceCandidate> detect(func::FuncOp fn) {
  if (!fn.isPrivate()) return std::nullopt;
  if (fn.getBody().empty()) return std::nullopt;
  Block &entry = fn.front();

  // Exactly one linalg.generic + one func.return — first-cut restriction.
  linalg::GenericOp gen;
  func::ReturnOp ret;
  for (Operation &op : entry) {
    if (auto g = dyn_cast<linalg::GenericOp>(op)) {
      if (gen) return std::nullopt;
      gen = g;
    } else if (auto r = dyn_cast<func::ReturnOp>(op)) {
      ret = r;
    } else {
      return std::nullopt;
    }
  }
  if (!gen || !ret) return std::nullopt;

  // All iterator types must be reduction.
  for (auto it : gen.getIteratorTypesArray())
    if (it != utils::IteratorType::reduction)
      return std::nullopt;

  if (gen.getNumDpsInputs() != 1 || gen.getNumDpsInits() != 1) return std::nullopt;
  if (gen.getNumResults() != 1) return std::nullopt;

  Value ins  = gen.getDpsInputOperand(0)->get();
  Value outs = gen.getDpsInitOperand(0)->get();

  auto insTy  = dyn_cast<RankedTensorType>(ins.getType());
  auto outsTy = dyn_cast<RankedTensorType>(outs.getType());
  if (!insTy || !outsTy) return std::nullopt;
  if (insTy.getRank() != 1) return std::nullopt;       // first-cut: 1-D input
  if (outsTy.getRank() != 0) return std::nullopt;      // first-cut: scalar output
  if (insTy.getShape()[0] == ShapedType::kDynamic)
    return std::nullopt;                                // first-cut: static R

  // Both operands must be func block args (no upstream reshape chain — yet).
  auto insBA  = dyn_cast<BlockArgument>(ins);
  auto outsBA = dyn_cast<BlockArgument>(outs);
  if (!insBA || !outsBA) return std::nullopt;
  if (insBA.getOwner() != &entry || outsBA.getOwner() != &entry)
    return std::nullopt;

  // Result must flow to func.return.
  if (ret.getNumOperands() != 1 || ret.getOperand(0) != gen.getResult(0))
    return std::nullopt;

  ReduceCandidate c;
  c.op       = gen;
  c.extent   = insTy.getShape()[0];
  c.elemType = insTy.getElementType();
  c.inputArg = ins;
  c.initArg  = outs;
  return c;
}

// Largest N ≤ maxSlots that divides R; falls back to 1 if R is prime
// or has no divisor ≤ maxSlots > 1.
static int64_t pickN(int64_t R, int64_t maxSlots) {
  for (int64_t N = std::min(R, maxSlots); N > 1; --N)
    if (R % N == 0) return N;
  return 1;
}

// Build `<name>_partial(%x: tensor<NxKxT>) -> tensor<NxT>`.  Input is the
// 2-D view of the original 1-D `tensor<RxT>` (R = N*K) — the coordinator
// emits a `tensor.expand_shape` before calling so the SSA value here is
// already shaped [N, K].  Linalg.generic doesn't accept scaling affine maps
// (it requires projected-permutation indexing), which is why we can't keep
// the input rank-1 and use `(d0*K + d1)`; the rank-2 input + identity-with-
// projection maps is the way through.
static func::FuncOp buildPartial(OpBuilder &builder, ModuleOp module,
                                 func::FuncOp original,
                                 ReduceCandidate &c, int64_t N,
                                 int64_t K, StringRef name) {
  MLIRContext *ctx = module.getContext();
  Location loc = original.getLoc();
  Type elemTy = c.elemType;

  auto inTy = RankedTensorType::get({N, K}, elemTy);
  auto wsTy = RankedTensorType::get({N}, elemTy);

  auto fnTy = FunctionType::get(ctx, {inTy}, {wsTy});
  builder.setInsertionPoint(original);
  auto fn = builder.create<func::FuncOp>(loc, name, fnTy);
  fn.setPrivate();

  Block *body = fn.addEntryBlock();
  OpBuilder b(ctx);
  b.setInsertionPointToStart(body);

  Value x = body->getArgument(0);

  // %ws_init = arith.constant dense<0.0> : tensor<NxT>
  TypedAttr zeroSplat = SplatElementsAttr::get(wsTy, b.getZeroAttr(elemTy));
  Value wsInit = b.create<arith::ConstantOp>(loc, wsTy, zeroSplat);

  // Standard axis-1 reduce: identity-with-projection maps.
  AffineExpr d0 = b.getAffineDimExpr(0), d1 = b.getAffineDimExpr(1);
  AffineMap mapIn  = AffineMap::get(2, 0, {d0, d1}, ctx);
  AffineMap mapOut = AffineMap::get(2, 0, {d0},     ctx);
  SmallVector<utils::IteratorType> iters{utils::IteratorType::parallel,
                                          utils::IteratorType::reduction};
  auto newGen = b.create<linalg::GenericOp>(
      loc, /*resultTensorTypes=*/TypeRange{wsTy},
      /*inputs=*/ValueRange{x}, /*outputs=*/ValueRange{wsInit},
      /*indexingMaps=*/ArrayRef<AffineMap>{mapIn, mapOut},
      /*iteratorTypes=*/iters);
  // Clone the original generic's region body (one block, scalar yield).
  // Use a separate builder so cloning into newGen's region doesn't disturb
  // `b`'s insertion point (which we still need for the trailing func.return).
  IRMapping bodyMap;
  newGen.getRegion().getBlocks().clear();
  Block &origBody = c.op.getRegion().front();
  Block *newBody = new Block();
  newBody->addArguments({elemTy, elemTy}, {loc, loc});
  newGen.getRegion().push_back(newBody);
  bodyMap.map(origBody.getArgument(0), newBody->getArgument(0));
  bodyMap.map(origBody.getArgument(1), newBody->getArgument(1));
  OpBuilder bb(newBody, newBody->end());
  for (Operation &op : origBody)
    bb.clone(op, bodyMap);

  b.create<func::ReturnOp>(loc, newGen.getResult(0));
  return fn;
}

// Build `<name>_combine(%ws: tensor<NxT>, %init: tensor<T>) -> tensor<T>`.
static func::FuncOp buildCombine(OpBuilder &builder, ModuleOp module,
                                 func::FuncOp original,
                                 ReduceCandidate &c, int64_t N,
                                 StringRef name) {
  MLIRContext *ctx = module.getContext();
  Location loc = original.getLoc();
  Type elemTy = c.elemType;

  auto wsTy   = RankedTensorType::get({N}, elemTy);
  auto initTy = RankedTensorType::get({}, elemTy);

  auto fnTy = FunctionType::get(ctx, {wsTy, initTy}, {initTy});
  builder.setInsertionPoint(original);
  auto fn = builder.create<func::FuncOp>(loc, name, fnTy);
  fn.setPrivate();

  Block *body = fn.addEntryBlock();
  OpBuilder b(ctx);
  b.setInsertionPointToStart(body);

  Value ws   = body->getArgument(0);
  Value init = body->getArgument(1);

  AffineExpr d0 = b.getAffineDimExpr(0);
  AffineMap mapIn  = AffineMap::get(1, 0, {d0}, ctx);
  AffineMap mapOut = AffineMap::get(1, 0, {}, ctx);
  SmallVector<utils::IteratorType> iters{utils::IteratorType::reduction};

  auto newGen = b.create<linalg::GenericOp>(
      loc, /*resultTensorTypes=*/TypeRange{initTy},
      /*inputs=*/ValueRange{ws}, /*outputs=*/ValueRange{init},
      /*indexingMaps=*/ArrayRef<AffineMap>{mapIn, mapOut},
      /*iteratorTypes=*/iters);
  IRMapping bodyMap;
  newGen.getRegion().getBlocks().clear();
  Block &origBody = c.op.getRegion().front();
  Block *newBody = new Block();
  newBody->addArguments({elemTy, elemTy}, {loc, loc});
  newGen.getRegion().push_back(newBody);
  bodyMap.map(origBody.getArgument(0), newBody->getArgument(0));
  bodyMap.map(origBody.getArgument(1), newBody->getArgument(1));
  OpBuilder bb(newBody, newBody->end());
  for (Operation &op : origBody)
    bb.clone(op, bodyMap);

  b.create<func::ReturnOp>(loc, newGen.getResult(0));
  return fn;
}

// Rewrite every `call @original` in the module to `call partial` → `call combine`.
static void rewriteCallSites(ModuleOp module, func::FuncOp original,
                              func::FuncOp partial, func::FuncOp combine) {
  SmallVector<func::CallOp> calls;
  module.walk([&](func::CallOp call) {
    if (call.getCallee() == original.getSymName())
      calls.push_back(call);
  });
  for (func::CallOp call : calls) {
    OpBuilder b(call);
    Location loc = call.getLoc();
    Value x    = call.getOperand(0);
    Value init = call.getOperand(1);

    // The partial kernel takes the 2D [N, K] view; reshape x in the
    // coordinator via tensor.expand_shape.  NetworkJsonEmitter treats this
    // as an alias (same buffer descriptor, different rank metadata) and
    // AclnnBackend already supports it.
    auto xTy = cast<RankedTensorType>(x.getType());
    int64_t R = xTy.getShape()[0];
    int64_t N = cast<RankedTensorType>(partial.getFunctionType().getResult(0))
                    .getShape()[0];
    int64_t K = R / N;
    auto expTy =
        RankedTensorType::get({N, K}, xTy.getElementType());
    SmallVector<ReassociationIndices> reassoc = {{0, 1}};
    SmallVector<OpFoldResult> outputShape{b.getIndexAttr(N), b.getIndexAttr(K)};
    Value x2 =
        b.create<tensor::ExpandShapeOp>(loc, expTy, x, reassoc, outputShape);

    auto partialCall =
        b.create<func::CallOp>(loc, partial, ValueRange{x2});
    auto combineCall = b.create<func::CallOp>(
        loc, combine, ValueRange{partialCall.getResult(0), init});
    call.getResult(0).replaceAllUsesWith(combineCall.getResult(0));
    call.erase();
  }
}

// Shared transform body — used by both the standalone pass and
// GroupOutlinePass (via splitRCoreGroupsInPlace).
static unsigned runSplitRCoreGroups(ModuleOp module, int64_t parallelSlots) {
  OpBuilder builder(module.getContext());

  SmallVector<std::pair<func::FuncOp, ReduceCandidate>> targets;
  module.walk([&](func::FuncOp fn) {
    if (auto c = detect(fn))
      targets.push_back({fn, *c});
  });

  unsigned splits = 0;
  for (auto &kv : targets) {
    func::FuncOp fn = kv.first;
    ReduceCandidate &c = kv.second;
    // Skip when P3b-2's single-block path is sufficient: small R fits in one
    // block (autotune picks XBLOCK ≥ R) and split would just add overhead /
    // hit the tile-peel underflow when K = R/N < default XBLOCK_SUB.
    // Threshold matches the autotuner single-block cap from
    // 2026-05-14-p3b-2-rcore-single-block-boundary.md.
    if (c.extent <= 256)
      continue;
    int64_t N = pickN(c.extent, parallelSlots);
    if (N <= 1)
      continue;
    int64_t K = c.extent / N;

    std::string baseName = fn.getSymName().str();
    auto partial =
        buildPartial(builder, module, fn, c, N, K, baseName + "_partial");
    auto combine =
        buildCombine(builder, module, fn, c, N, baseName + "_combine");

    rewriteCallSites(module, fn, partial, combine);
    fn.erase();
    ++splits;
  }
  return splits;
}

struct AutoFuseSplitRCoreGroupPass
    : public ::impl::AutoFuseSplitRCoreGroupBase<
          AutoFuseSplitRCoreGroupPass> {
  using ::impl::AutoFuseSplitRCoreGroupBase<
      AutoFuseSplitRCoreGroupPass>::AutoFuseSplitRCoreGroupBase;

  void runOnOperation() override {
    (void)runSplitRCoreGroups(getOperation(), parallelSlots);
  }
};

} // namespace

std::unique_ptr<Pass> createAutoFuseSplitRCoreGroupPass() {
  return std::make_unique<AutoFuseSplitRCoreGroupPass>();
}

unsigned splitRCoreGroupsInPlace(ModuleOp module, int64_t parallelSlots) {
  return runSplitRCoreGroups(module, parallelSlots);
}

} // namespace mlir::afir
