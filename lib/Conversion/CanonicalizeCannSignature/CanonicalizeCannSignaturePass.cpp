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
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

#define GEN_PASS_DEF_CANONICALIZECANNSIGNATUREPASS
#include "Conversion/Passes.h.inc"

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

  // Promoted intermediate GM buffers have StridedLayoutAttr with dynamic offset.
  // Count only real I/O memrefs (no strided dynamic offset) to determine inputs.
  auto isRealIO = [](Type t) {
    auto mrt = dyn_cast<MemRefType>(t);
    if (!mrt)
      return false;
    auto strided = dyn_cast<StridedLayoutAttr>(mrt.getLayout());
    return !(strided && ShapedType::isDynamic(strided.getOffset()));
  };
  int numRealBeforeTiling = 0;
  for (int i = 0; i < tilingIdx; ++i)
    if (isRealIO(args[i].getType()))
      numRealBeforeTiling++;
  int numRealAfterTiling = 0;
  for (int i = tilingIdx + 1, e = static_cast<int>(args.size()); i < e; ++i)
    if (isRealIO(args[i].getType()))
      numRealAfterTiling++;
  // When outputs appear after the tiling arg (PyAsc layout), all real args
  // before tiling are inputs. When outputs appear before the tiling arg
  // (bufferized layout with promoted intermediates), subtract 1 for the output.
  int numInputs;
  if (numRealAfterTiling > 0)
    numInputs = numRealBeforeTiling;
  else
    numInputs = numRealBeforeTiling > 0 ? numRealBeforeTiling - 1 : tilingIdx;
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
  entryBlock.eraseArgument(tilingIdx);

  // Update function type and add cann.num_inputs attribute.
  funcOp.setType(newFuncType);
  funcOp->setAttr("cann.num_inputs",
                  IntegerAttr::get(IntegerType::get(ctx, 32), numInputs));

  return success();
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

    // Erase non-func module-level ops (e.g., transform sequences) so that the
    // output can be parsed by tools that don't register transform dialects.
    SmallVector<Operation *> toErase;
    for (Operation &child : module.getBody()->getOperations()) {
      if (!isa<func::FuncOp>(child))
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
