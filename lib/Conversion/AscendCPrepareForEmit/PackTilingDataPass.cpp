#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"

#include "Analysis/SymbolicShape/DimSymbolTable.h"
#include "Analysis/SymbolicShape/SymExpr.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL_ASCENDCPACKTILINGDATAPASS
#define GEN_PASS_DEF_ASCENDCPACKTILINGDATAPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::emitasc;

namespace mlir::afir {

static constexpr int64_t kGMSpace = 22;

struct DimKey {
  unsigned argNumber;
  int64_t dimIndex;
  bool operator==(const DimKey &o) const {
    return argNumber == o.argNumber && dimIndex == o.dimIndex;
  }
};

static emitasc::PyStructType buildTilingDataType(MLIRContext *ctx,
                                                 ArrayRef<StringRef> names) {
  SmallVector<Attribute> typeAttrs, nameAttrs;
  Type i64 = IntegerType::get(ctx, 64);
  for (auto name : names) {
    typeAttrs.push_back(TypeAttr::get(i64));
    nameAttrs.push_back(StringAttr::get(ctx, name));
  }
  return emitasc::PyStructType::get(ctx, StringAttr::get(ctx, "TilingData"),
                                    ArrayAttr::get(ctx, typeAttrs),
                                    ArrayAttr::get(ctx, nameAttrs));
}

static LogicalResult packTilingData(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  Block &entry = func.getBody().front();
  Type indexTy = IndexType::get(ctx);
  Type i64Ty = IntegerType::get(ctx, 64);

  // ── 1. Read tiling args from vector_plan.tiling_infos ────────────────────
  SmallVector<BlockArgument> tilingArgs;
  SmallVector<std::string> tilingArgNames;

  auto moduleOp = func->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return func.emitError("PackTilingData: func not inside a module");

  auto tilingInfosAttr =
      moduleOp->getAttrOfType<ArrayAttr>("vector_plan.tiling_infos");
  if (!tilingInfosAttr)
    return func.emitError("PackTilingData: vector_plan.tiling_infos not found "
                          "on module; ensure TilePlanGen ran before this pass");

  for (Attribute infoAttr : tilingInfosAttr) {
    auto info = dyn_cast<DictionaryAttr>(infoAttr);
    if (!info) continue;
    auto kid = dyn_cast_or_null<StringAttr>(info.get("kernel_id"));
    if (!kid || kid.getValue() != func.getName()) continue;
    auto fieldsAttr = dyn_cast_or_null<ArrayAttr>(info.get("fields"));
    if (!fieldsAttr) break;
    for (Attribute fa : fieldsAttr) {
      auto field = dyn_cast<DictionaryAttr>(fa);
      if (!field) continue;
      unsigned argIdx = (unsigned)cast<IntegerAttr>(field.get("arg_index"))
                            .getValue().getSExtValue();
      if (argIdx >= entry.getNumArguments())
        return func.emitError("PackTilingData: arg_index ")
               << argIdx << " out of range";
      tilingArgs.push_back(entry.getArgument(argIdx));
      tilingArgNames.push_back(
          cast<StringAttr>(field.get("name")).getValue().str());
    }
    break;
  }

  // ── 2. Collect memref.dim uses on block args ──────────────────────────────
  SmallVector<DimKey> dimKeys;        // unique canonical keys
  SmallVector<memref::DimOp> dimOps;

  // afir-symbolize-shapes (when it ran) recorded, per block arg, which symbol
  // each dim is (`afir.symbolic_shape` arg-attr, serialized ids) and which
  // (arg,dim) each root symbol originates from (`afir.dim_symbols` func attr).
  // Use that to fold dims the linalg op proved equal onto one TilingData field
  // (e.g. dim_arg3_0 == dim_arg0_0).  Identity fallback when absent.
  auto dimSymsAttr = func->getAttrOfType<ArrayAttr>("afir.dim_symbols");
  std::optional<mlir::afir::symshape::DimSymbolTable> symTable;
  if (dimSymsAttr)
    symTable = mlir::afir::symshape::DimSymbolTable::fromAttr(dimSymsAttr);
  auto canonicalize = [&](unsigned argN, int64_t dimIdx) -> DimKey {
    if (symTable && dimIdx >= 0) {
      if (auto a =
              func.getArgAttrOfType<StringAttr>(argN, "afir.symbolic_shape")) {
        if (auto list = mlir::afir::symshape::parseSymExprList(a.getValue())) {
          if ((size_t)dimIdx < list->size()) {
            const auto &e = (*list)[dimIdx];
            if (e.getKind() == mlir::afir::symshape::SymExpr::Kind::Sym &&
                e.getSym() < symTable->numRoots()) {
              auto src = symTable->sourceOf(e.getSym());
              return DimKey{src.first, (int64_t)src.second};
            }
          }
        }
      }
    }
    return DimKey{argN, dimIdx};
  };

  auto addDimKey = [&](unsigned argNum, int64_t dimIdx) {
    DimKey key = canonicalize(argNum, dimIdx);
    if (llvm::none_of(dimKeys, [&](const DimKey &k) { return k == key; }))
      dimKeys.push_back(key);
  };

  func.walk([&](memref::DimOp dimOp) {
    auto arg = dyn_cast<BlockArgument>(dimOp.getSource());
    if (!arg) return;
    auto constOp = dimOp.getIndex().getDefiningOp<arith::ConstantOp>();
    if (!constOp) return;
    auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
    if (!intAttr) return;
    addDimKey(arg.getArgNumber(), intAttr.getValue().getSExtValue());
    dimOps.push_back(dimOp);
  });

  // ── 3. Build TilingData field names ──────────────────────────────────────
  SmallVector<std::string> tilingNameStorage;
  for (const std::string &n : tilingArgNames) tilingNameStorage.push_back(n);
  for (const DimKey &k : dimKeys)
    tilingNameStorage.push_back("dim_arg" + std::to_string(k.argNumber) +
                                "_" + std::to_string(k.dimIndex));
  SmallVector<StringRef> tilingNames;
  for (const std::string &s : tilingNameStorage) tilingNames.push_back(s);

  if (tilingNames.empty())
    return success(); // nothing to pack

  // ── 4. Build TilingData type and GM pointer arg type ─────────────────────
  auto tilingStructTy = buildTilingDataType(ctx, tilingNames);
  auto tilingArgTy = MemRefType::get(
      {ShapedType::kDynamic}, tilingStructTy, MemRefLayoutAttrInterface{},
      IntegerAttr::get(IntegerType::get(ctx, 32), kGMSpace));

  // ── 5. Inject TilingData arg and extract fields ───────────────────────────
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(&entry);
  BlockArgument tilingArg = entry.addArgument(tilingArgTy, func.getLoc());
  Value localStruct = builder.create<emitasc::CopyStructOp>(
      func.getLoc(), tilingStructTy, tilingArg);

  SmallVector<Value> tilingFieldVals;
  for (unsigned i = 0; i < tilingNames.size(); ++i)
    tilingFieldVals.push_back(builder.create<emitasc::MemberOp>(
        func.getLoc(), i64Ty, localStruct,
        builder.getStringAttr(tilingNames[i])));

  // ── 6. Replace tiling arg uses (index-cast i64 → index) ──────────────────
  for (unsigned i = 0; i < tilingArgs.size(); ++i) {
    Value val = tilingFieldVals[i];
    if (tilingArgs[i].getType().isIndex())
      val = builder.create<arith::IndexCastOp>(func.getLoc(), indexTy, val);
    tilingArgs[i].replaceAllUsesWith(val);
  }

  // ── 7. Replace memref.dim uses ────────────────────────────────────────────
  unsigned dimFieldBase = tilingArgs.size();
  for (memref::DimOp dimOp : dimOps) {
    auto arg = cast<BlockArgument>(dimOp.getSource());
    auto constOp = dimOp.getIndex().getDefiningOp<arith::ConstantOp>();
    int64_t dimIdxVal =
        cast<IntegerAttr>(constOp.getValue()).getValue().getSExtValue();
    DimKey key = canonicalize(arg.getArgNumber(), dimIdxVal);
    unsigned k = llvm::find_if(dimKeys, [&](const DimKey &d) {
                   return d == key;
                 }) - dimKeys.begin();
    Value i64Val = tilingFieldVals[dimFieldBase + k];
    OpBuilder b(dimOp);
    Value idxVal = b.create<arith::IndexCastOp>(dimOp.getLoc(), indexTy, i64Val);
    dimOp.replaceAllUsesWith(idxVal);
    dimOp.erase();
  }

  // ── 8. Erase tiling block args (reverse order) ────────────────────────────
  SmallVector<unsigned> toErase;
  for (BlockArgument a : tilingArgs) toErase.push_back(a.getArgNumber());
  llvm::sort(toErase, std::greater<unsigned>());
  for (unsigned idx : toErase) entry.eraseArgument(idx);
  if (func->getAttr("arg_attrs"))
    func->removeAttr("arg_attrs");

  // ── 9. Update function type ───────────────────────────────────────────────
  SmallVector<Type> newArgTypes;
  for (BlockArgument a : entry.getArguments()) newArgTypes.push_back(a.getType());
  func.setFunctionType(FunctionType::get(ctx, newArgTypes,
                                         func.getFunctionType().getResults()));

  // ── 10. Declare TilingData struct at module level ─────────────────────────
  OpBuilder modBuilder(ctx);
  modBuilder.setInsertionPointToStart(moduleOp.getBody());
  modBuilder.create<emitasc::DeclarePyStructOp>(func.getLoc(),
                                                TypeAttr::get(tilingStructTy));

  return success();
}

struct AscendCPackTilingDataPass
    : public ::impl::AscendCPackTilingDataPassBase<AscendCPackTilingDataPass> {
  using AscendCPackTilingDataPassBase::AscendCPackTilingDataPassBase;
  void runOnOperation() override {
    if (failed(packTilingData(getOperation())))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createAscendCPackTilingDataPass() {
  return std::make_unique<AscendCPackTilingDataPass>();
}

} // namespace mlir::afir
