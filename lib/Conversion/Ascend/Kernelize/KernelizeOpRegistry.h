//===- KernelizeOpRegistry.h - Ascend kernelize op registry -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPREGISTRY_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPREGISTRY_H

#include "DependencyAnalysis.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::afir::ascend::kernelize {

enum class KernelizeOpTraitKind {
  Unknown,
  StructuredLinalg,
  TensorView,
};

llvm::StringRef stringifyKernelizeOpTrait(KernelizeOpTraitKind kind);

struct KernelizeRegistryModel {
  using MatchFn = bool (*)(Operation *op);
  using PopulateFn = void (*)(Operation *op, OpSemanticSummary &summary);

  llvm::StringRef name = "unknown";
  KernelizeOpTraitKind trait = KernelizeOpTraitKind::Unknown;
  MatchFn match = nullptr;
  PopulateFn populate = nullptr;
};

class KernelizeOpRegistry {
public:
  static KernelizeOpRegistry buildDefault();

  void registerModel(KernelizeRegistryModel model);
  bool isTargetOp(Operation *op) const;
  OpSemanticSummary summarize(Operation *op, OperationId opId) const;

private:
  const KernelizeRegistryModel *lookupModel(Operation *op) const;

  SmallVector<KernelizeRegistryModel, 4> models;
};

} // namespace mlir::afir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPREGISTRY_H
