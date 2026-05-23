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

#include "Conversion/Ascend/Backend/Codegen/CanonicalizeCannSignaturePass.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

#define GEN_PASS_DEF_CANONICALIZECANNSIGNATUREPASS
#include "Conversion/Ascend/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

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

struct PromotedGlobalInput {
  FlatSymbolRefAttr name;
  MemRefType type;
  SmallVector<memref::GetGlobalOp> getOps;
};

static FailureOr<SmallVector<PromotedGlobalInput>>
collectPromotedGlobalInputs(func::FuncOp funcOp) {
  SmallVector<PromotedGlobalInput> globals;
  WalkResult result = funcOp.walk([&](memref::GetGlobalOp getGlobalOp) {
    auto memrefType = dyn_cast<MemRefType>(getGlobalOp.getType());
    if (!memrefType) {
      getGlobalOp.emitOpError("must produce a memref to be promoted to a "
                              "CANN GM input");
      return WalkResult::interrupt();
    }

    FlatSymbolRefAttr name = getGlobalOp.getNameAttr();
    for (PromotedGlobalInput &global : globals) {
      if (global.name != name)
        continue;
      if (global.type != memrefType) {
        getGlobalOp.emitOpError("has type ")
            << memrefType << " but previous use of " << name.getValue()
            << " has type " << global.type;
        return WalkResult::interrupt();
      }
      global.getOps.push_back(getGlobalOp);
      return WalkResult::advance();
    }

    PromotedGlobalInput global;
    global.name = name;
    global.type = memrefType;
    global.getOps.push_back(getGlobalOp);
    globals.push_back(std::move(global));
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return globals;
}

static LogicalResult canonicalizeFuncOp(func::FuncOp funcOp,
                                        IRRewriter &rewriter) {
  // Only process aicore global kernels.
  if (!funcOp->hasAttr(ascendc::attr::global) ||
      !funcOp->hasAttr(ascendc::attr::aicore))
    return success();

  auto args = funcOp.getArguments();
  int tilingIdx = -1;
  int numArgs = static_cast<int>(args.size());
  for (int i = 0; i < numArgs; ++i) {
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

  FailureOr<SmallVector<PromotedGlobalInput>> promotedGlobalsOr =
      collectPromotedGlobalInputs(funcOp);
  if (failed(promotedGlobalsOr))
    return failure();
  SmallVector<PromotedGlobalInput> promotedGlobals =
      std::move(*promotedGlobalsOr);

  int numInputs = tilingIdx + static_cast<int>(promotedGlobals.size());
  emitasc::PyStructType tilingStructType =
      getTilingStructType(args[tilingIdx].getType());
  MLIRContext *ctx = funcOp.getContext();

  // Build new arg types:
  // inputs..., promoted global weights..., outputs..., memref<ui8>, PyStructType
  SmallVector<Type> newArgTypes;
  for (int i = 0; i < tilingIdx; ++i)
    newArgTypes.push_back(args[i].getType());
  for (const PromotedGlobalInput &global : promotedGlobals)
    newArgTypes.push_back(global.type);
  for (int i = tilingIdx + 1, e = args.size(); i < e; ++i)
    newArgTypes.push_back(args[i].getType());
  Type workspaceType =
      MemRefType::get({}, IntegerType::get(ctx, 8, IntegerType::Unsigned));
  newArgTypes.push_back(workspaceType);
  newArgTypes.push_back(tilingStructType);

  auto newFuncType = FunctionType::get(ctx, newArgTypes, {});

  Block &entryBlock = funcOp.getBody().front();

  // Promote module-level memref globals used by the kernel body to explicit
  // GM inputs. CANN kernels receive runtime buffers through GM_ADDR ABI
  // arguments; keeping memref.get_global in the body has no PyAsc printer and
  // also hides the weight dependency from runtime manifests.
  for (auto [index, global] : llvm::enumerate(promotedGlobals)) {
    BlockArgument globalArg = entryBlock.insertArgument(
        tilingIdx + static_cast<int>(index), global.type, funcOp.getLoc());
    for (memref::GetGlobalOp getGlobalOp : global.getOps) {
      rewriter.replaceAllUsesWith(getGlobalOp.getResult(), globalArg);
      rewriter.eraseOp(getGlobalOp);
    }
  }

  // Add workspace and tiling as new block args.
  BlockArgument wsArg =
      entryBlock.addArgument(workspaceType, funcOp.getLoc());
  // workspace arg is required by the CANN ABI but unused in the kernel body.
  (void)wsArg;
  BlockArgument tilingArg =
      entryBlock.addArgument(tilingStructType, funcOp.getLoc());

  // Replace copy_struct result uses with new tiling arg.
  rewriter.replaceAllUsesWith(copyOp.getResult(), tilingArg);
  rewriter.eraseOp(copyOp);

  // Erase the old tiling memref arg (its only user was copy_struct, now erased).
  // New args were appended at the tail; tilingIdx still refers to the
  // original tiling memref arg position, which is unchanged.
  entryBlock.eraseArgument(tilingIdx + promotedGlobals.size());

  // Update function type and add cann.num_inputs attribute.
  funcOp.setType(newFuncType);
  funcOp->setAttr("cann.num_inputs",
                  IntegerAttr::get(IntegerType::get(ctx, 32), numInputs));

  return success();
}

static bool shouldEraseModuleChild(Operation &child) {
  if (isa<func::FuncOp>(child))
    return false;
  return child.getName().getDialectNamespace() == "transform";
}

} // namespace

struct CanonicalizeCannSignaturePass
    : public ::impl::CanonicalizeCannSignaturePassBase<
          CanonicalizeCannSignaturePass> {
  using Base = ::impl::CanonicalizeCannSignaturePassBase<CanonicalizeCannSignaturePass>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(module.getContext());

    // Canonicalize aicore kernel signatures.
    WalkResult result = module.walk([&](func::FuncOp funcOp) {
      if (failed(canonicalizeFuncOp(funcOp, rewriter)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Erase transform sequences so that the output can be parsed by tools that
    // don't register transform dialects. Other module-level ops may carry
    // resources or symbols referenced by the kernel body and must be preserved.
    SmallVector<Operation *> toErase;
    for (Operation &child : module.getBody()->getOperations()) {
      if (shouldEraseModuleChild(child))
        toErase.push_back(&child);
    }
    for (Operation *op : toErase)
      rewriter.eraseOp(op);
  }
};

std::unique_ptr<Pass> createCanonicalizeCannSignaturePass() {
  return std::make_unique<CanonicalizeCannSignaturePass>();
}

} // namespace mlir::afir
