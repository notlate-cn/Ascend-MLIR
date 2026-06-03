//===- EntryNormalizationVerifier.cpp - Normalize verifier ---------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "EntryNormalizationVerifier.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "Conversion/Ascend/Common/SymbolConstraints.h"
#include "SymbolEquivalenceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace mlir::ascend::normalize {
namespace {

bool isStaticRankedTensorDim(symbol::DimRef ref) {
  auto type = dyn_cast<RankedTensorType>(ref.value.getType());
  return type && !type.isDynamicDim(ref.dim);
}

LogicalResult verifyFuncSymbolConstraints(func::FuncOp func) {
  if (!func->getAttr(kSymbolConstraintsAttr))
    return func.emitError() << kSymbolConstraintsAttr << " missing";

  FailureOr<symbol::SymbolConstraintTable> table =
      symbol::parseSymbolConstraintAttr(func);
  if (failed(table))
    return failure();

  FailureOr<SymbolEquivalenceResult> expected =
      analyzeSymbolEquivalence(func);
  if (failed(expected))
    return func.emitError("failed to recompute symbol equivalence proofs");

  LogicalResult result = success();
  for (const SymbolEqualityProof &proof : expected->proofs) {
    if (isStaticRankedTensorDim(proof.lhs) ||
        isStaticRankedTensorDim(proof.rhs))
      continue;
    if (table->areEquivalent(proof.lhs, proof.rhs))
      continue;

    func.emitError() << "symbol constraints missing " << proof.rule
                     << " equality proof";
    result = failure();
  }
  return result;
}

} // namespace

LogicalResult verifyEntryNormalization(ModuleOp module) {
  LogicalResult result = success();
  module.walk([&](func::FuncOp func) {
    if (failed(verifyFuncSymbolConstraints(func)))
      result = failure();
  });
  return result;
}

} // namespace mlir::ascend::normalize
