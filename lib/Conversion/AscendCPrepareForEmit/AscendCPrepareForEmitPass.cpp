//===- AscendCPrepareForEmitPass.cpp - Prepare func for ascir-translate ===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This pass transforms a func.func produced by --linalg-to-ascendc /
// --ascendc-parallelize into the form expected by ascir-translate:
//
//   Input signature:
//     @fc_relu(%A: memref<?x?xf32, strided<[?,?], offset:?>>,
//              %B: memref<?x?xf32, strided<[?,?], offset:?>>,
//              ...
//              %TB_M: i64, %TB_N: i64, %Tb_M: i64, %Tb_N: i64, %t_K: i64)
//       -> memref<...>
//
//   Output signature:
//     @fc_relu(%A:       memref<?xf32, 22>,      // __gm__ pointer
//              %B:       memref<?xf32, 22>,
//              ...
//              %tiling:  memref<?x!emitasc.py_struct<"TilingData",...>, 22>)
//       attributes {ascendc.aicore, ascendc.global}
//
// Transformation steps:
//   1. Collect all memref args (tensor data) and all i64 args (tiling params).
//   2. Replace each memref arg type with a flat GM memref: memref<?xElem, 22>.
//   3. Replace all i64 args with a single TilingData GM pointer arg.
//   4. At function entry, emit emitasc.copy_struct to read TilingData from GM,
//      then emitasc.member_ref to extract each tiling field.
//   5. Drop the function return value (kernel returns void).
//   6. Set {ascendc.aicore, ascendc.global} attributes on the function.
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#define GEN_PASS_DECL_ASCENDCPREPAREFOREMITPASS
#define GEN_PASS_DEF_ASCENDCPREPAREFOREMITPASS
#include "Conversion/Passes.h.inc"

#define DEBUG_TYPE "ascendc-prepare-for-emit"

using namespace mlir;
using namespace mlir::ascendc;
using namespace mlir::emitasc;

namespace mlir::afir {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// GM memory_space value used by ascir-translate for __gm__ pointers.
static constexpr int64_t kGMSpace = 22;

/// Build the TilingData PyStruct type with the given field names and i64 types.
static emitasc::PyStructType buildTilingDataType(MLIRContext *ctx,
                                                 ArrayRef<StringRef> names) {
  SmallVector<Attribute> typeAttrs, nameAttrs;
  Type i64 = IntegerType::get(ctx, 64);
  for (auto name : names) {
    typeAttrs.push_back(TypeAttr::get(i64));
    nameAttrs.push_back(StringAttr::get(ctx, name));
  }
  return emitasc::PyStructType::get(
      ctx, StringAttr::get(ctx, "TilingData"),
      ArrayAttr::get(ctx, typeAttrs), ArrayAttr::get(ctx, nameAttrs));
}

//===----------------------------------------------------------------------===//
// Main transformation
//===----------------------------------------------------------------------===//

static LogicalResult prepareFunc(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  Block &entry = func.getBody().front();

  // ── 1. Collect i64 tiling args ───────────────────────────────────────────
  // Memref/index args (tensor data) are kept as-is.
  // i64 args → consolidated into a single TilingData struct GM pointer.
  SmallVector<BlockArgument> i64Args;
  for (BlockArgument arg : entry.getArguments()) {
    if (arg.getType().isInteger(64))
      i64Args.push_back(arg);
  }

  // ── 2. Build TilingData struct type ─────────────────────────────────────
  static const StringRef kDefaultNames[] = {"TB_M", "TB_N", "Tb_M", "Tb_N", "t_K"};
  static const unsigned kDefaultCount =
      sizeof(kDefaultNames) / sizeof(kDefaultNames[0]);

  SmallVector<StringRef> tilingNames;
  for (unsigned i = 0; i < i64Args.size(); ++i)
    tilingNames.push_back(i < kDefaultCount ? kDefaultNames[i] : StringRef("field"));

  emitasc::PyStructType tilingStructTy;
  Type tilingArgTy;
  if (!i64Args.empty()) {
    tilingStructTy = buildTilingDataType(ctx, tilingNames);
    tilingArgTy = MemRefType::get(
        {ShapedType::kDynamic}, tilingStructTy, MemRefLayoutAttrInterface{},
        IntegerAttr::get(IntegerType::get(ctx, 32), kGMSpace));
  }

  // ── 3. Insert TilingData block arg and extract fields FIRST ─────────────
  // (Before erasing i64 args, so we can replace their uses.)
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(&entry);

  SmallVector<Value> tilingFieldVals;
  if (!i64Args.empty()) {
    BlockArgument tilingArg = entry.addArgument(tilingArgTy, func.getLoc());
    Value localStruct =
        builder.create<emitasc::CopyStructOp>(func.getLoc(), tilingStructTy, tilingArg);
    Type i64Ty = builder.getI64Type();
    for (unsigned i = 0; i < i64Args.size(); ++i) {
      tilingFieldVals.push_back(builder.create<emitasc::MemberOp>(
          func.getLoc(), i64Ty, localStruct,
          builder.getStringAttr(tilingNames[i])));
    }
  }

  // ── 4. Replace old i64 uses and erase old i64 block args ────────────────
  for (unsigned i = 0; i < i64Args.size(); ++i)
    i64Args[i].replaceAllUsesWith(tilingFieldVals[i]);

  // Erase in reverse index order so indices stay valid.
  SmallVector<unsigned> toErase;
  for (BlockArgument arg : i64Args)
    toErase.push_back(arg.getArgNumber());
  llvm::sort(toErase, std::greater<unsigned>());
  for (unsigned idx : toErase)
    entry.eraseArgument(idx);

  // ── 5. Update function type to match the new block arg layout ───────────
  // Collect the current block arg types (after erasing i64 args and adding
  // the tilingArg), then set the function type with void return.
  SmallVector<Type> newArgTypes;
  for (BlockArgument arg : entry.getArguments())
    newArgTypes.push_back(arg.getType());

  func.setFunctionType(FunctionType::get(ctx, newArgTypes, /*results=*/{}));

  // ── 6. Add {ascendc.aicore, ascendc.global} attributes ──────────────────
  func->setAttr("ascendc.aicore", UnitAttr::get(ctx));
  func->setAttr("ascendc.global", UnitAttr::get(ctx));

  // ── 7. Remove return values (kernel returns void) ───────────────────────
  func.walk([&](func::ReturnOp ret) {
    if (ret.getNumOperands() > 0) {
      OpBuilder b(ret);
      b.create<func::ReturnOp>(ret.getLoc());
      ret.erase();
    }
  });

  // ── 8. Declare TilingData struct at module level ─────────────────────────
  if (!i64Args.empty()) {
    if (auto moduleOp = func->getParentOfType<ModuleOp>()) {
      OpBuilder modBuilder(ctx);
      modBuilder.setInsertionPointToStart(moduleOp.getBody());
      modBuilder.create<emitasc::DeclarePyStructOp>(
          func.getLoc(), TypeAttr::get(tilingStructTy));
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct AscendCPrepareForEmitPass
    : public ::impl::AscendCPrepareForEmitPassBase<AscendCPrepareForEmitPass> {
  using AscendCPrepareForEmitPassBase::AscendCPrepareForEmitPassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (failed(prepareFunc(func)))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createAscendCPrepareForEmitPass() {
  return std::make_unique<AscendCPrepareForEmitPass>();
}

} // namespace mlir::afir
