//===- KernelizeOpInterface.h - Ascend kernelize op interface -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPINTERFACE_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPINTERFACE_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::afir::ascend::kernelize {

enum class AccessPatternKind {
  NotApplicable,
  Elementwise,
  Broadcast,
  Reduction,
  Contraction,
  Gather,
  Scatter,
  LayoutTransform,
  Unknown
};

enum class IteratorKind {
  Parallel,
  Reduction,
  Unknown
};

enum class KernelizeParticipationKind {
  Ignore,
  Analyze,
  Transparent,
  Unsupported,
};

enum class KernelizeSemanticTrait {
  Unknown,
  Structured,
  TensorView,
  LayoutTransform,
  HandwrittenGroup,
};

struct KernelizeOpSemanticInfo {
  KernelizeParticipationKind participation =
      KernelizeParticipationKind::Unsupported;
  AccessPatternKind accessPattern = AccessPatternKind::Unknown;
  SmallVector<IteratorKind, 4> iteratorKinds;
  SmallVector<AffineMap, 4> indexingMaps;
  SmallVector<unsigned, 2> resultRanks;
  SmallVector<KernelizeSemanticTrait, 4> traits;
  std::string modelName = "unknown";
  std::string unsupportedReason;
};

struct KernelizeOpModel {
  using MatchFn = bool (*)(Operation *op);
  using PopulateFn =
      LogicalResult (*)(Operation *op, KernelizeOpSemanticInfo &info);

  llvm::StringRef name = "unknown";
  MatchFn match = nullptr;
  PopulateFn populate = nullptr;
};

class KernelizeOpModelRegistry {
public:
  void registerModel(KernelizeOpModel model);
  FailureOr<KernelizeOpSemanticInfo> resolve(Operation *op) const;

private:
  SmallVector<KernelizeOpModel, 8> models;
};

llvm::StringRef stringifyAccessPattern(AccessPatternKind kind);
llvm::StringRef stringifyIteratorKind(IteratorKind kind);
llvm::StringRef
stringifyKernelizeParticipation(KernelizeParticipationKind kind);
llvm::StringRef stringifyKernelizeSemanticTrait(KernelizeSemanticTrait trait);

} // namespace mlir::afir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPINTERFACE_H
