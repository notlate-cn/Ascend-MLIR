//===- DebugOptions.h - Ascend debug options ----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_DEBUG_DEBUGOPTIONS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_DEBUG_DEBUGOPTIONS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace mlir::ascend::debug {

enum class DebugStage { None, Normalize, Kernelize, Schedule, Realize, All };

struct DebugOptions {
  DebugStage stage = DebugStage::None;
  bool dumpReport = false;
  std::string checkpointDumpDir;
};

struct DebugStepInfo {
  StringRef checkpoint;
  StringRef id;
  StringRef title;
  StringRef purpose;
  StringRef inputs;
  StringRef outputs;
  StringRef inspectHint;
  StringRef commonFailures;
};

DebugStage parseDebugStage(StringRef value);
bool shouldDump(DebugOptions options, DebugStage stage);
bool shouldDumpCheckpoint(const DebugOptions &options, DebugStage stage);
void emitStageHeader(raw_ostream &os, DebugStage stage, StringRef passName);
ArrayRef<DebugStepInfo> getDebugStepCatalog(DebugStage stage);
const DebugStepInfo *lookupDebugStepInfo(StringRef checkpoint);
void emitDebugStepCatalog(raw_ostream &os, DebugStage stage);
LogicalResult dumpCheckpoint(ModuleOp module, const DebugOptions &options,
                             DebugStage stage, StringRef basename);

} // namespace mlir::ascend::debug

#endif // ASCEND_MLIR_CONVERSION_ASCEND_DEBUG_DEBUGOPTIONS_H
