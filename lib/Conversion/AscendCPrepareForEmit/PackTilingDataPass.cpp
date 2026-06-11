#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"

#include "Analysis/SymbolicShape/DimSymbolTable.h"
#include "Analysis/SymbolicShape/SymExpr.h"
#include "Conversion/AutoFuse/TilePlan.h"
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

static LogicalResult
packTilingDataFromSchema(func::FuncOp func,
                          const mlir::auto_fuse::TilingInfoSchema &schema) {
  using namespace mlir::auto_fuse;
  MLIRContext *ctx = func.getContext();
  Block &entry = func.getBody().front();
  Type i64Ty = IntegerType::get(ctx, 64);
  Type indexTy = IndexType::get(ctx);

  // Build TilingData struct from schema.fields in order.
  SmallVector<std::string> nameStorage;
  for (auto &f : schema.fields) nameStorage.push_back(f.name);
  SmallVector<StringRef> tilingNames;
  for (auto &s : nameStorage) tilingNames.push_back(s);
  if (tilingNames.empty()) return success();

  auto tilingStructTy = buildTilingDataType(ctx, tilingNames);
  auto tilingArgTy = MemRefType::get(
      {ShapedType::kDynamic}, tilingStructTy, MemRefLayoutAttrInterface{},
      IntegerAttr::get(IntegerType::get(ctx, 32), kGMSpace));

  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(&entry);
  BlockArgument tilingArg = entry.addArgument(tilingArgTy, func.getLoc());
  Value localStruct = builder.create<emitasc::CopyStructOp>(
      func.getLoc(), tilingStructTy, tilingArg);

  SmallVector<Value> fieldVals;
  for (auto &n : nameStorage)
    fieldVals.push_back(builder.create<emitasc::MemberOp>(
        func.getLoc(), i64Ty, localStruct, builder.getStringAttr(n)));

  // Replace tunable tile-param args with i64→index cast of the field value.
  SmallVector<unsigned> argsToErase;
  for (size_t i = 0; i < schema.fields.size(); ++i) {
    auto &f = schema.fields[i];
    if (f.kind != SchemaFieldKind::Tunable) continue;
    if (f.argIndex < 0 ||
        (unsigned)f.argIndex >= entry.getNumArguments())
      return func.emitError("PackTilingData: schema tunable arg_index ")
             << f.argIndex << " out of range";
    BlockArgument tileArg = entry.getArgument(f.argIndex);
    Value val = builder.create<arith::IndexCastOp>(
        func.getLoc(), indexTy, fieldVals[i]);
    tileArg.replaceAllUsesWith(val);
    argsToErase.push_back((unsigned)f.argIndex);
  }

  // Symbol-shape canonicalization: if afir.dim_symbols + per-arg
  // afir.symbolic_shape are present, dim ops on derived buffers (e.g.
  // arg1/arg2 sharing a symbol with arg0) map back onto the canonical
  // (rootArg, rootDim) the schema field was emitted for.  Without this,
  // a kernel func that takes multiple same-shape inputs would have
  // memref.dim ops on args that the schema has no field for.
  auto dimSymsAttr = func->getAttrOfType<ArrayAttr>("afir.dim_symbols");
  std::optional<mlir::afir::symshape::DimSymbolTable> symTable;
  if (dimSymsAttr)
    symTable = mlir::afir::symshape::DimSymbolTable::fromAttr(dimSymsAttr);
  auto parseArgDimExpr =
      [](StringRef s) -> std::optional<std::pair<int32_t, int32_t>> {
    if (!s.consume_front("arg")) return std::nullopt;
    auto sep = s.find("_dim");
    if (sep == StringRef::npos) return std::nullopt;
    int32_t a, d;
    if (s.substr(0, sep).getAsInteger(10, a)) return std::nullopt;
    if (s.substr(sep + 4).getAsInteger(10, d)) return std::nullopt;
    return std::make_pair(a, d);
  };

  auto canonicalize = [&](unsigned argN,
                          int64_t dimIdx) -> std::pair<unsigned, int64_t> {
    if (symTable && dimIdx >= 0) {
      if (auto a =
              func.getArgAttrOfType<StringAttr>(argN, "afir.symbolic_shape")) {
        if (auto list = mlir::afir::symshape::parseSymExprList(a.getValue())) {
          if ((size_t)dimIdx < list->size()) {
            const auto &e = (*list)[dimIdx];
            if (e.getKind() == mlir::afir::symshape::SymExpr::Kind::Sym &&
                e.getSym() < symTable->numRoots()) {
              auto src = symTable->sourceOf(e.getSym());
              return {src.first, (int64_t)src.second};
            }
          }
        }
      }
    }
    // Schema shape_expr path: for Output args, the precomputed shape_expr
    // tells us which (input arg, dim) this output dim depends on.  Covers
    // cases where afir.symbolic_shape didn't survive to the output arg
    // (output memref arg is created post-bufferize, after
    // afir-symbolize-shapes ran).
    for (auto &a : schema.args) {
      if (a.role != mlir::auto_fuse::SchemaArgRole::Output) continue;
      if ((unsigned)a.mlirIndex != argN) continue;
      if (dimIdx < 0 || (size_t)dimIdx >= a.shapeExpr.size()) break;
      auto parsed = parseArgDimExpr(a.shapeExpr[dimIdx]);
      if (!parsed) break;
      for (auto &ia : schema.args) {
        if (ia.role != mlir::auto_fuse::SchemaArgRole::Input) continue;
        if (ia.callArgIndex != parsed->first) continue;
        return {(unsigned)ia.mlirIndex, (int64_t)parsed->second};
      }
      break;
    }
    return {argN, dimIdx};
  };

  // For shape-derived fields, rewrite memref.dim %argN, %cD where
  // (sourceArg, sourceDim) matches (after symbol-shape canonicalization).
  SmallVector<memref::DimOp> dimOps;
  func.walk([&](memref::DimOp d) { dimOps.push_back(d); });
  for (memref::DimOp dimOp : dimOps) {
    auto ba = dyn_cast<BlockArgument>(dimOp.getSource());
    if (!ba || ba.getOwner() != &entry) continue;
    auto cst = dimOp.getIndex().getDefiningOp<arith::ConstantOp>();
    if (!cst) continue;
    int64_t d = cast<IntegerAttr>(cst.getValue()).getInt();
    auto [cArg, cDim] = canonicalize(ba.getArgNumber(), d);
    for (size_t i = 0; i < schema.fields.size(); ++i) {
      auto &f = schema.fields[i];
      if (f.kind != SchemaFieldKind::ShapeDerived) continue;
      // Canonicalize the FIELD's (sourceArg, sourceDim) the same way, so a dim
      // query that maps to the symbol's representative arg (e.g. arg0.dim0)
      // still matches a field keyed on a shape-equal arg (e.g. dim_arg3_1):
      // both reduce to the same root.  Without this, dynamic dims shared across
      // args (residual collapse_shape: arg3.dim1 == arg0.dim0) leave an
      // unresolved memref.dim that ascir-translate cannot emit.
      auto [fArg, fDim] = canonicalize(f.sourceArg, f.sourceDim);
      if ((int64_t)fArg != (int64_t)cArg) continue;
      if (fDim != cDim) continue;
      OpBuilder b(dimOp);
      Value idxVal = b.create<arith::IndexCastOp>(
          dimOp.getLoc(), indexTy, fieldVals[i]);
      dimOp.replaceAllUsesWith(idxVal);
      dimOp.erase();
      break;
    }
  }

  // Erase tile-param args in reverse order.
  llvm::sort(argsToErase, std::greater<unsigned>());
  for (unsigned idx : argsToErase) entry.eraseArgument(idx);
  if (func->getAttr("arg_attrs")) func->removeAttr("arg_attrs");

  SmallVector<Type> newArgTypes;
  for (BlockArgument a : entry.getArguments())
    newArgTypes.push_back(a.getType());
  func.setFunctionType(FunctionType::get(
      ctx, newArgTypes, func.getFunctionType().getResults()));

  OpBuilder modBuilder(ctx);
  modBuilder.setInsertionPointToStart(
      func->getParentOfType<ModuleOp>().getBody());
  modBuilder.create<emitasc::DeclarePyStructOp>(
      func.getLoc(), TypeAttr::get(tilingStructTy));
  return success();
}

static LogicalResult packTilingData(func::FuncOp func) {
  auto moduleOp = func->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return func.emitError("PackTilingData: func not inside a module");
  auto schema =
      mlir::auto_fuse::lookupTilingInfoSchema(moduleOp, func.getName());
  if (!schema)
    return func.emitError("PackTilingData: missing schema_version=2 "
                          "auto_fuse.tiling_infos entry for kernel ")
           << func.getName();
  return packTilingDataFromSchema(func, *schema);
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
