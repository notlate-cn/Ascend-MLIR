//===- SchedulePersistentCacheIO.cpp - Schedule cache file IO -------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "SchedulePersistentCacheIO.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::ascend::schedule {

FailureOr<llvm::SmallVector<std::string, 8>>
loadPersistentTuningCacheFile(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return failure();

  llvm::SmallVector<std::string, 8> signatures;
  llvm::SmallVector<llvm::StringRef, 16> lines;
  llvm::StringRef(buffer.get()->getBuffer()).split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (line.empty() || line.starts_with("#"))
      continue;
    signatures.push_back(line.str());
  }
  return signatures;
}

LogicalResult writePersistentTuningCacheFile(
    llvm::StringRef path, llvm::ArrayRef<std::string> signatures) {
  std::error_code ec;
  llvm::ToolOutputFile output(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return failure();
  for (const std::string &signature : signatures)
    output.os() << signature << "\n";
  output.keep();
  return success();
}

} // namespace mlir::ascend::schedule
