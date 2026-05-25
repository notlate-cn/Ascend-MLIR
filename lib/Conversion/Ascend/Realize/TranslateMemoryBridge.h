//===- TranslateMemoryBridge.h - Ascend Translate memory bridge -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_TRANSLATEMEMORYBRIDGE_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_TRANSLATEMEMORYBRIDGE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::afir::ascend::realize {

struct TranslateBridgeMaterializationCounts {
  unsigned materializedAllocCount = 0;
  unsigned materializedCopyCount = 0;
};

class TranslateMemoryBridge {
public:
  virtual ~TranslateMemoryBridge() = default;

  virtual FailureOr<llvm::StringMap<TranslateBridgeMaterializationCounts>>
  materialize(ModuleOp module) const = 0;
};

FailureOr<llvm::StringMap<TranslateBridgeMaterializationCounts>>
materializeTranslateMemoryBridge(ModuleOp module);

} // namespace mlir::afir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_TRANSLATEMEMORYBRIDGE_H
