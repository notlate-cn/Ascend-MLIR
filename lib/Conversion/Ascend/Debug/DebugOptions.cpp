//===- DebugOptions.cpp - Ascend debug options -------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Debug/DebugOptions.h"

namespace mlir::ascend::debug {

DebugStage parseDebugStage(StringRef value) {
  StringRef trimmed = value.trim();
  if (trimmed.equals_insensitive("normalize"))
    return DebugStage::Normalize;
  if (trimmed.equals_insensitive("kernelize"))
    return DebugStage::Kernelize;
  if (trimmed.equals_insensitive("schedule"))
    return DebugStage::Schedule;
  if (trimmed.equals_insensitive("realize"))
    return DebugStage::Realize;
  if (trimmed.equals_insensitive("all"))
    return DebugStage::All;
  return DebugStage::None;
}

bool shouldDump(DebugOptions options, DebugStage stage) {
  if (!options.dumpReport)
    return false;
  return options.stage == DebugStage::All || options.stage == stage;
}

void emitStageHeader(raw_ostream &os, DebugStage stage, StringRef passName) {
  StringRef stageName = "none";
  switch (stage) {
  case DebugStage::None:
    stageName = "none";
    break;
  case DebugStage::Normalize:
    stageName = "normalize";
    break;
  case DebugStage::Kernelize:
    stageName = "kernelize";
    break;
  case DebugStage::Schedule:
    stageName = "schedule";
    break;
  case DebugStage::Realize:
    stageName = "realize";
    break;
  case DebugStage::All:
    stageName = "all";
    break;
  }
  os << "Ascend " << stageName << " report";
  if (!passName.empty())
    os << " (" << passName << ")";
  os << "\n";
}

} // namespace mlir::ascend::debug
