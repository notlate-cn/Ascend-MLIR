//===- SplitRCoreGroup.cpp - Split full-reduce kernel into partial+combine -===//
//
// Runs after vector-plan-group-outline.  See the .td description / plan doc
// docs/superpowers/plans/2026-05-14-p3b-rcore-reduce-multicore.zh.md §5.
//
//===---------------------------------------------------------------------===//

#include "Conversion/VectorPlan/VectorPlanPasses.h"

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

#define GEN_PASS_DECL_VECTORPLANSPLITRCOREGROUP
#define GEN_PASS_DEF_VECTORPLANSPLITRCOREGROUP
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

// Build `<name>_partial(%x: tensor<RxT>) -> tensor<NxT>`.
static func::FuncOp buildPartial(OpBuilder &builder, ModuleOp module,
                                 func::FuncOp original,
                                 ReduceCandidate &c, int64_t N,
                                 int64_t K, StringRef name) {
  MLIRContext *ctx = module.getContext();
  Location loc = original.getLoc();
  Type elemTy = c.elemType;

  auto inTy  = RankedTensorType::get({c.extent}, elemTy);
  auto wsTy  = RankedTensorType::get({N}, elemTy);
  auto expTy = RankedTensorType::get({N, K}, elemTy);

  auto fnTy = FunctionType::get(ctx, {inTy}, {wsTy});
  builder.setInsertionPoint(original);
  auto fn = builder.create<func::FuncOp>(loc, name, fnTy);
  fn.setPrivate();

  Block *body = fn.addEntryBlock();
  OpBuilder b(ctx);
  b.setInsertionPointToStart(body);

  Value x = body->getArgument(0);

  // %x2 = tensor.expand_shape %x [[0, 1]] output_shape [N, K] : tensor<R> into tensor<NxK>
  SmallVector<ReassociationIndices> reassoc = {{0, 1}};
  SmallVector<OpFoldResult> outputShape{b.getIndexAttr(N), b.getIndexAttr(K)};
  Value x2 =
      b.create<tensor::ExpandShapeOp>(loc, expTy, x, reassoc, outputShape);

  // %ws_init = arith.constant dense<0.0> : tensor<NxT>
  // Constant tensor avoids introducing a sibling linalg op that the tile-fuse
  // group-emitter would have to wire as an iter_arg alongside the reduce.
  TypedAttr zeroSplat = SplatElementsAttr::get(wsTy, b.getZeroAttr(elemTy));
  Value wsInit = b.create<arith::ConstantOp>(loc, wsTy, zeroSplat);

  // %r = linalg.generic axis-1 reduce of %x2 into %ws_init
  AffineExpr d0 = b.getAffineDimExpr(0), d1 = b.getAffineDimExpr(1);
  AffineMap mapIn  = AffineMap::get(2, 0, {d0, d1}, ctx);
  AffineMap mapOut = AffineMap::get(2, 0, {d0},     ctx);
  SmallVector<utils::IteratorType> iters{utils::IteratorType::parallel,
                                          utils::IteratorType::reduction};
  // Clone the inner accumulator body from the original generic to preserve
  // arith.addf / yield (and the original element type semantics).
  auto newGen = b.create<linalg::GenericOp>(
      loc, /*resultTensorTypes=*/TypeRange{wsTy},
      /*inputs=*/ValueRange{x2}, /*outputs=*/ValueRange{wsInit},
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
    auto partialCall =
        b.create<func::CallOp>(loc, partial, ValueRange{x});
    auto combineCall = b.create<func::CallOp>(
        loc, combine, ValueRange{partialCall.getResult(0), init});
    call.getResult(0).replaceAllUsesWith(combineCall.getResult(0));
    call.erase();
  }
}

struct VectorPlanSplitRCoreGroupPass
    : public ::impl::VectorPlanSplitRCoreGroupBase<
          VectorPlanSplitRCoreGroupPass> {
  using ::impl::VectorPlanSplitRCoreGroupBase<
      VectorPlanSplitRCoreGroupPass>::VectorPlanSplitRCoreGroupBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    SmallVector<std::pair<func::FuncOp, ReduceCandidate>> targets;
    module.walk([&](func::FuncOp fn) {
      if (auto c = detect(fn))
        targets.push_back({fn, *c});
    });

    for (auto &kv : targets) {
      func::FuncOp fn = kv.first;
      ReduceCandidate &c = kv.second;
      int64_t N = pickN(c.extent, parallelSlots);
      if (N <= 1)
        continue; // no parallelism possible — leave original as-is (P3b-2 single block)
      int64_t K = c.extent / N;

      std::string baseName = fn.getSymName().str();
      auto partial =
          buildPartial(builder, module, fn, c, N, K, baseName + "_partial");
      auto combine =
          buildCombine(builder, module, fn, c, N, baseName + "_combine");

      rewriteCallSites(module, fn, partial, combine);
      fn.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createVectorPlanSplitRCoreGroupPass() {
  return std::make_unique<VectorPlanSplitRCoreGroupPass>();
}

} // namespace mlir::afir
