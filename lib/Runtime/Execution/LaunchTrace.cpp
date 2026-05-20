#include "LaunchTrace.h"

#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace mlir::runtime {

namespace {

constexpr size_t kAlignment = 512;
constexpr size_t kMaxTilingWordsToPrint = 8;

void printHex(llvm::raw_ostream &os, uint64_t value) {
  os << "0x";
  os.write_hex(value);
}

void printAlignedPtr(llvm::raw_ostream &os, uintptr_t ptr) {
  printHex(os, ptr);
  os << " align512=" << ((ptr % kAlignment) == 0 ? "yes" : "no");
}

uint64_t readLittleEndianWord(llvm::ArrayRef<uint8_t> bytes, size_t offset) {
  uint64_t word = 0;
  const size_t n = std::min<size_t>(sizeof(word), bytes.size() - offset);
  std::memcpy(&word, bytes.data() + offset, n);
  return word;
}

void printHexByte(llvm::raw_ostream &os, uint8_t value) {
  constexpr char kHex[] = "0123456789abcdef";
  os << kHex[(value >> 4) & 0x0f] << kHex[value & 0x0f];
}

} // namespace

bool isNativeLaunchTraceEnabled() {
  const char *raw = std::getenv("ASCEND_RUNTIME_TRACE_LAUNCH");
  if (!raw || !*raw)
    return false;

  llvm::StringRef value(raw);
  value = value.trim();
  return !value.empty() && value != "0" && !value.equals_insensitive("false") &&
         !value.equals_insensitive("off") && !value.equals_insensitive("no");
}

void printNativeLaunchTrace(const LaunchTraceRecord &record,
                            llvm::raw_ostream &os) {
  os << "[npu-launch] kernel=" << record.kernelName
     << " binary=" << record.binaryPath << " magic=";
  printHex(os, record.magic);
  os << " device_id=" << record.deviceId
     << " block_dim=" << record.blockDim
     << " args_count=" << record.launchArgs.size()
     << " args_size=" << record.argsSize << "\n";

  os << "[npu-launch] inputs=" << record.inputs.size()
     << " outputs=" << record.outputs.size()
     << " workspace_size=" << record.workspaceSize
     << " tiling_bytes=" << record.tilingBytes.size()
     << " tiling_words=" << ((record.tilingBytes.size() + 7) / 8) << "\n";

  for (size_t i = 0; i < record.inputs.size(); ++i) {
    os << "[npu-launch] input[" << i << "] bytes=" << record.inputs[i].bytes
       << " ptr=";
    printAlignedPtr(os, record.inputs[i].ptr);
    os << "\n";
  }

  for (size_t i = 0; i < record.outputs.size(); ++i) {
    os << "[npu-launch] output[" << i << "] bytes=" << record.outputs[i].bytes
       << " ptr=";
    printAlignedPtr(os, record.outputs[i].ptr);
    os << "\n";
  }

  os << "[npu-launch] workspace bytes=" << record.workspaceSize << " ptr=";
  printAlignedPtr(os, record.workspacePtr);
  os << "\n";

  const size_t tilingWords = (record.tilingBytes.size() + 7) / 8;
  const size_t wordsToPrint = std::min(tilingWords, kMaxTilingWordsToPrint);
  for (size_t i = 0; i < wordsToPrint; ++i) {
    os << "[npu-launch] tiling.word[" << i
       << "]=" << readLittleEndianWord(record.tilingBytes, i * 8) << "\n";
  }
}

void printNativeLaunchBufferSample(llvm::StringRef label,
                                   llvm::ArrayRef<uint8_t> bytes,
                                   size_t totalBytes, llvm::raw_ostream &os) {
  os << "[npu-launch] " << label << " bytes=" << totalBytes
     << " sample_bytes=" << bytes.size() << " sample_hex=";
  for (uint8_t byte : bytes)
    printHexByte(os, byte);
  os << "\n";
}

} // namespace mlir::runtime
