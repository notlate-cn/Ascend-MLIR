//===- DebugOptions.h - Ascend debug options ----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_DEBUG_DEBUGOPTIONS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_DEBUG_DEBUGOPTIONS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::afir::ascend::debug {

enum class DebugStage { None, Normalize, Kernelize, Schedule, Realize, All };

struct DebugOptions {
  DebugStage stage = DebugStage::None;
  bool dumpReport = false;
};

DebugStage parseDebugStage(StringRef value);
bool shouldDump(DebugOptions options, DebugStage stage);
void emitStageHeader(raw_ostream &os, DebugStage stage, StringRef passName);

} // namespace mlir::afir::ascend::debug

#endif // ASCEND_MLIR_CONVERSION_ASCEND_DEBUG_DEBUGOPTIONS_H
