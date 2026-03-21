//===- CanonicalizeCannSignaturePass.cpp - CANN signature canonicalization -===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// Transforms aicore kernel func.func from PyAsc internal format:
//   (inputs..., memref<?x!emitasc.py_struct<...>, 22:i32>, outputs...)
// to CANN standard format:
//   (inputs..., outputs..., memref<ui8>, !emitasc.py_struct<...>)
//
// Also removes emitasc.copy_struct and adds cann.num_inputs attribute.
//
//===----------------------------------------------------------------------===//

#include "Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.h"

#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir::afir {

// Declare the pass base template from the generated inc, within the correct namespace.
#define GEN_PASS_DECL_CANONICALIZECANNSIGNATUREPASS
#define GEN_PASS_DEF_CANONICALIZECANNSIGNATUREPASS
#include "Conversion/Passes.h.inc"

namespace {

// Returns true if `type` is the tiling memref: memref<?x!emitasc.py_struct<...>, 22:i32>
static bool isTilingMemref(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType)
    return false;
  if (!isa<emitasc::PyStructType>(memrefType.getElementType()))
    return false;
  auto memSpace = memrefType.getMemorySpace();
  if (!memSpace)
    return false;
  auto intAttr = dyn_cast<IntegerAttr>(memSpace);
  return intAttr && intAttr.getInt() == 22;
}

// Returns the PyStructType extracted from a tiling memref argument.
static emitasc::PyStructType getTilingStructType(Type tilingMemrefType) {
  return cast<emitasc::PyStructType>(
      cast<MemRefType>(tilingMemrefType).getElementType());
}

static LogicalResult canonicalizeFuncOp(func::FuncOp funcOp,
                                        IRRewriter &rewriter) {
  // Only process aicore global kernels.
  if (!funcOp->hasAttr(ascendc::attr::global) ||
      !funcOp->hasAttr(ascendc::attr::aicore))
    return success();

  auto args = funcOp.getArguments();
  int tilingIdx = -1;
  for (int i = 0, e = args.size(); i < e; ++i) {
    if (isTilingMemref(args[i].getType())) {
      if (tilingIdx != -1)
        return funcOp.emitOpError("has multiple tiling memref arguments");
      tilingIdx = i;
    }
  }
  if (tilingIdx == -1)
    return funcOp.emitOpError("has no tiling memref argument "
                              "(memref<?x!emitasc.py_struct<...>, 22:i32>)");

  // Find the copy_struct op that uses the tiling arg.
  emitasc::CopyStructOp copyOp;
  for (Operation *user : args[tilingIdx].getUsers()) {
    if (auto op = dyn_cast<emitasc::CopyStructOp>(user)) {
      if (copyOp)
        return funcOp.emitOpError("tiling arg has multiple copy_struct users");
      copyOp = op;
    }
  }
  if (!copyOp)
    return funcOp.emitOpError("tiling arg has no emitasc.copy_struct user");

  // Verify copy_struct result is only used by emitasc.member ops.
  for (Operation *user : copyOp.getResult().getUsers()) {
    if (!isa<emitasc::MemberOp>(user))
      return funcOp.emitOpError("copy_struct result has non-member user: ")
             << user->getName();
  }

  int numInputs = tilingIdx;
  emitasc::PyStructType tilingStructType =
      getTilingStructType(args[tilingIdx].getType());
  MLIRContext *ctx = funcOp.getContext();

  // Build new arg types: inputs..., outputs..., memref<ui8>, PyStructType
  SmallVector<Type> newArgTypes;
  for (int i = 0; i < tilingIdx; ++i)
    newArgTypes.push_back(args[i].getType());
  for (int i = tilingIdx + 1, e = args.size(); i < e; ++i)
    newArgTypes.push_back(args[i].getType());
  Type workspaceType =
      MemRefType::get({}, IntegerType::get(ctx, 8, IntegerType::Unsigned));
  newArgTypes.push_back(workspaceType);
  newArgTypes.push_back(tilingStructType);

  auto newFuncType = FunctionType::get(ctx, newArgTypes, {});

  Block &entryBlock = funcOp.getBody().front();

  // Add workspace and tiling as new block args.
  BlockArgument wsArg =
      entryBlock.addArgument(workspaceType, funcOp.getLoc());
  (void)wsArg;
  BlockArgument tilingArg =
      entryBlock.addArgument(tilingStructType, funcOp.getLoc());

  // Replace copy_struct result uses with new tiling arg.
  rewriter.replaceAllUsesWith(copyOp.getResult(), tilingArg);
  rewriter.eraseOp(copyOp);

  // Erase the old tiling memref arg (its only user was copy_struct, now erased).
  entryBlock.eraseArgument(tilingIdx);

  // Update function type and add cann.num_inputs attribute.
  funcOp.setType(newFuncType);
  funcOp->setAttr("cann.num_inputs",
                  IntegerAttr::get(IntegerType::get(ctx, 32), numInputs));

  return success();
}

struct CanonicalizeCannSignaturePass
    : public impl::CanonicalizeCannSignaturePassBase<
          CanonicalizeCannSignaturePass> {
  using CanonicalizeCannSignaturePassBase::CanonicalizeCannSignaturePassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(module.getContext());
    WalkResult result = module.walk([&](func::FuncOp funcOp) {
      if (failed(canonicalizeFuncOp(funcOp, rewriter)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::afir
