#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace mlir::runtime {

struct LaunchTraceTensor {
  size_t bytes = 0;
  uintptr_t ptr = 0;
};

struct LaunchTraceRecord {
  std::string kernelName;
  std::string binaryPath;
  uint32_t magic = 0;
  int32_t deviceId = 0;
  int blockDim = 0;
  llvm::ArrayRef<LaunchTraceTensor> inputs;
  llvm::ArrayRef<LaunchTraceTensor> outputs;
  size_t workspaceSize = 0;
  uintptr_t workspacePtr = 0;
  llvm::ArrayRef<uint8_t> tilingBytes;
  llvm::ArrayRef<uint64_t> launchArgs;
  uint32_t argsSize = 0;
};

bool isNativeLaunchTraceEnabled();
void printNativeLaunchTrace(const LaunchTraceRecord &record,
                            llvm::raw_ostream &os);
void printNativeLaunchBufferSample(llvm::StringRef label,
                                   llvm::ArrayRef<uint8_t> bytes,
                                   size_t totalBytes, llvm::raw_ostream &os);

} // namespace mlir::runtime
