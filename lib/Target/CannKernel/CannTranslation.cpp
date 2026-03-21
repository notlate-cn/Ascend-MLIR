//===- CannTranslation.cpp - CANN kernel C++ translation --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/CannKernel/CannTranslation.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Target/Asc/CodeEmitter.h"
#include "ascir/Target/Asc/Common.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace {

/// Emit the TilingData struct declaration from a PyStructType.
static LogicalResult emitTilingStructDecl(CodeEmitter &emitter, Location loc,
                                          emitasc::PyStructType pyType) {
  auto &os = emitter.ostream();
  StringRef structName = pyType.getNameAttr().getValue();
  os << "struct " << structName << " {\n";
  os.indent();

  auto typesAttr = pyType.getTypesAttr();
  auto namesAttr = pyType.getNamesAttr();
  auto types = typesAttr.getValue();
  auto names = namesAttr.getValue();

  if (types.size() != names.size())
    return emitError(loc, "PyStructType types/names size mismatch");

  for (size_t i = 0; i < types.size(); ++i) {
    Type fieldType = cast<TypeAttr>(types[i]).getValue();
    StringRef fieldName = cast<StringAttr>(names[i]).getValue();
    if (failed(emitter.emitType(loc, fieldType)))
      return failure();
    os << " " << fieldName << ";\n";
  }

  os.unindent() << "};\n\n";
  return success();
}

/// Emit the CANN-standard function signature and body.
static LogicalResult printCannFuncOp(CodeEmitter &emitter,
                                     func::FuncOp funcOp) {
  CodeEmitter::Scope scope(emitter);
  auto &os = emitter.ostream();

  // cann.num_inputs must be present (set by CanonicalizeCannSignaturePass).
  if (!funcOp->hasAttr("cann.num_inputs"))
    return funcOp.emitOpError("missing cann.num_inputs attribute; "
                               "run --canonicalize-cann-signature first");

  auto args = funcOp.getArguments();
  int numArgs = (int)args.size();

  // Layout: [0..N-3] = inputs+outputs (all GM_ADDR), [N-2] = workspace
  // (memref<ui8>), [N-1] = tiling (!emitasc.py_struct).
  if (numArgs < 4)
    return funcOp.emitOpError(
        "CANN function must have at least 4 args "
        "(inputs, outputs, workspace, tiling)");

  BlockArgument tilingArg = args[numArgs - 1];
  auto tilingType = dyn_cast<emitasc::PyStructType>(tilingArg.getType());
  if (!tilingType)
    return funcOp.emitOpError("last argument must be !emitasc.py_struct");

  // Validate workspace arg is memref<ui8>.
  auto wsType = dyn_cast<MemRefType>(args[numArgs - 2].getType());
  if (!wsType || !wsType.getElementType().isUnsignedInteger(8))
    return funcOp.emitOpError(
        "second-to-last argument must be memref<ui8> workspace");

  // Emit function header
  os << "extern \"C\" __global__ __aicore__ void " << funcOp.getName() << "(\n";
  os.indent();

  // Emit input/output GM_ADDR args (all memref args)
  for (int i = 0; i < numArgs - 1; ++i) {
    os << "GM_ADDR " << emitter.getOrCreateName(args[i]);
    os << ",\n";
  }

  // Emit tiling arg as struct by value
  StringRef tilingStructName = tilingType.getNameAttr().getValue();
  os << tilingStructName << " " << emitter.getOrCreateName(tilingArg) << "\n";

  os.unindent() << ") {\n";
  os.indent();

  // Emit body ops
  for (Block &block : funcOp.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      // Skip func.return (void kernel, no return value needed)
      if (isa<func::ReturnOp>(op)) {
        os << "return;\n";
        continue;
      }
      if (failed(emitOperation(emitter, op, needsSemicolon(&op))))
        return failure();
    }
  }

  os.unindent() << "}\n";
  return success();
}

} // namespace

LogicalResult mlir::translateToCannKernel(Operation *op, raw_ostream &os) {
  auto moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp)
    return op->emitOpError("expected a module op");

  CodeEmitter emitter(os);
  CodeEmitter::Scope scope(emitter);

  os << "#include \"kernel_operator.h\"\n\n";

  // First pass: emit TilingData struct declarations from aicore funcs
  for (Operation &child : moduleOp.getBody()->getOperations()) {
    auto funcOp = dyn_cast<func::FuncOp>(child);
    if (!funcOp)
      continue;
    if (!funcOp->hasAttr(ascendc::attr::global))
      continue;

    auto args = funcOp.getArguments();
    if (args.empty())
      continue;
    auto tilingType =
        dyn_cast<emitasc::PyStructType>(args.back().getType());
    if (!tilingType)
      continue;

    if (failed(emitTilingStructDecl(emitter, funcOp.getLoc(), tilingType)))
      return failure();
  }

  // Second pass: emit functions
  for (Operation &child : moduleOp.getBody()->getOperations()) {
    auto funcOp = dyn_cast<func::FuncOp>(child);
    if (funcOp && funcOp->hasAttr(ascendc::attr::global)) {
      if (failed(printCannFuncOp(emitter, funcOp)))
        return failure();
    } else {
      if (failed(emitOperation(emitter, child, /*trailingSemicolon=*/false)))
        return failure();
    }
  }

  return success();
}
