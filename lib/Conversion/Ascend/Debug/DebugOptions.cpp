//===- DebugOptions.cpp - Ascend debug options -------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Debug/DebugOptions.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

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

bool shouldDumpCheckpoint(const DebugOptions &options, DebugStage stage) {
  if (options.checkpointDumpDir.empty())
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

LogicalResult dumpCheckpoint(ModuleOp module, const DebugOptions &options,
                             DebugStage stage, StringRef basename) {
  if (!shouldDumpCheckpoint(options, stage))
    return success();
  if (basename.empty()) {
    module.emitError() << "empty Ascend debug checkpoint name";
    return failure();
  }

  std::error_code ec =
      llvm::sys::fs::create_directories(options.checkpointDumpDir);
  if (ec) {
    module.emitError() << "failed to create Ascend debug checkpoint directory '"
                       << options.checkpointDumpDir << "': " << ec.message();
    return failure();
  }

  llvm::SmallString<256> path(options.checkpointDumpDir);
  llvm::sys::path::append(path, (basename + ".mlir").str());

  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    module.emitError() << "failed to open Ascend debug checkpoint '" << path
                       << "': " << ec.message();
    return failure();
  }

  module.print(os);
  os << "\n";
  return success();
}

} // namespace mlir::ascend::debug
