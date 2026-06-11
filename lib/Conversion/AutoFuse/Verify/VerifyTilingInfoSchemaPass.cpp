#include "Conversion/AutoFuse/AutoFusePasses.h"
#include "Conversion/AutoFuse/TilePlan.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

#define GEN_PASS_DECL_AUTOFUSEVERIFYTILINGINFOSCHEMA
#define GEN_PASS_DEF_AUTOFUSEVERIFYTILINGINFOSCHEMA
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::auto_fuse;

namespace mlir::afir {

namespace {

//===----------------------------------------------------------------------===//
// The 5 schema↔IR consistency checks.  All read-only: inspect + emit error.
// Returns failure when any check fails (caller signals pass failure).
//===----------------------------------------------------------------------===//

static LogicalResult verifySchema(func::FuncOp funcOp,
                                  const TilingInfoSchema &schema) {
  bool ok = true;
  unsigned numArgs = funcOp.getNumArguments();

  // ---- Check 1: TileParam arg index in range + index-type -----------------
  //
  // Only the TileParam role's mlir_index is checked against the bufferized
  // signature, because that is the only role whose mlir_index PackTilingData
  // dereferences post-bufferize (`entry.getArgument(f.argIndex)`).
  //
  // Input/Output mlir_index describe the *pre-bufferize* kernel ABI (inputs +
  // appended strided output) authored by TilePlanGen.  IsolateKernelOutputs +
  // one-shot-bufferize may collapse an in-place-writeback output back into an
  // input arg (e.g. reduce kernels that return %argN), so for those roles the
  // schema mlir_index legitimately no longer matches the live signature.
  // Checking them here would reject valid kernels; their post-bufferize
  // consistency is instead covered by Check 5 (sourceArg / callArgIndex), which
  // is what PackTilingData actually relies on.  (This is the Check-1/Check-2
  // relaxation noted in the design: a real identity-layout in-place output.)
  for (const SchemaArg &a : schema.args) {
    if (a.role != SchemaArgRole::TileParam)
      continue;
    if (a.mlirIndex < 0 || (unsigned)a.mlirIndex >= numArgs) {
      funcOp.emitError("schema tile_param arg mlir_index ")
          << a.mlirIndex << " out of range (func has " << numArgs << " args)";
      ok = false;
      continue;
    }
    Type argTy = funcOp.getArgument(a.mlirIndex).getType();
    if (!isa<IndexType>(argTy)) {
      funcOp.emitError("schema arg mlir_index ")
          << a.mlirIndex << " (role tile_param) expected index, got " << argTy;
      ok = false;
    }
  }

  // ---- Check 3: tunable field ↔ TileParam arg consistency (both ways) ------
  for (const SchemaField &f : schema.fields) {
    if (f.kind != SchemaFieldKind::Tunable)
      continue;
    bool found = false;
    for (const SchemaArg &a : schema.args) {
      if (a.role == SchemaArgRole::TileParam && a.mlirIndex == f.argIndex &&
          a.tileParamName == f.name) {
        found = true;
        break;
      }
    }
    if (!found) {
      funcOp.emitError("tunable field '")
          << f.name << "' (arg_index " << f.argIndex
          << ") has no matching tile_param SchemaArg";
      ok = false;
    }
  }
  for (const SchemaArg &a : schema.args) {
    if (a.role != SchemaArgRole::TileParam)
      continue;
    bool found = false;
    for (const SchemaField &f : schema.fields) {
      if (f.kind == SchemaFieldKind::Tunable && f.argIndex == a.mlirIndex &&
          f.name == a.tileParamName) {
        found = true;
        break;
      }
    }
    if (!found) {
      funcOp.emitError("tile_param arg mlir_index ")
          << a.mlirIndex << " (name '" << a.tileParamName
          << "') has no matching tunable field";
      ok = false;
    }
  }

  // ---- Check 4: Output resultIndex values unique 0..K-1 --------------------
  {
    SmallVector<int32_t> resultIdxs;
    for (const SchemaArg &a : schema.args)
      if (a.role == SchemaArgRole::Output)
        resultIdxs.push_back(a.resultIndex);
    int K = (int)resultIdxs.size();
    SmallVector<int32_t> sorted(resultIdxs);
    llvm::sort(sorted);
    bool valid = true;
    for (int i = 0; i < K; ++i)
      if (sorted[i] != i)
        valid = false;
    if (!valid) {
      auto err = funcOp.emitError(
          "Output resultIndex values must be unique 0..K-1, got [");
      for (int i = 0; i < K; ++i)
        err << (i ? ", " : "") << resultIdxs[i];
      err << "]";
      ok = false;
    }
  }

  // ---- Check 5: shape_derived + callArgIndex + shape_equalities validity ---
  for (const SchemaField &f : schema.fields) {
    if (f.kind != SchemaFieldKind::ShapeDerived)
      continue;
    if (f.sourceArg < 0 || (unsigned)f.sourceArg >= numArgs) {
      funcOp.emitError("shape_derived field '")
          << f.name << "' source_arg " << f.sourceArg
          << " out of range (func has " << numArgs << " args)";
      ok = false;
    }
  }
  {
    llvm::DenseSet<int32_t> seenCallIdx;
    for (const SchemaArg &a : schema.args) {
      if (a.role != SchemaArgRole::Input)
        continue;
      if (a.callArgIndex < 0 || !seenCallIdx.insert(a.callArgIndex).second) {
        funcOp.emitError("Input arg ")
            << a.mlirIndex << " has invalid/duplicate call_arg_index "
            << a.callArgIndex;
        ok = false;
      }
    }
    for (const auto &group : schema.shapeEqualities) {
      for (const auto &p : group) {
        if (!seenCallIdx.count(p.first)) {
          funcOp.emitError("shape_equalities references call_arg_index ")
              << p.first << " that no Input arg declares";
          ok = false;
        }
      }
    }
  }

  return success(ok);
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

struct AutoFuseVerifyTilingInfoSchemaPass
    : public ::impl::AutoFuseVerifyTilingInfoSchemaBase<
          AutoFuseVerifyTilingInfoSchemaPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto moduleOp = func->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return; // nothing to verify against

    auto schema = lookupTilingInfoSchema(moduleOp, func.getName());
    if (!schema)
      return; // no v2 schema for this func — skip (coordinator funcs, etc.)

    if (failed(verifySchema(func, *schema)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createAutoFuseVerifyTilingInfoSchemaPass() {
  return std::make_unique<AutoFuseVerifyTilingInfoSchemaPass>();
}

} // namespace mlir::afir
