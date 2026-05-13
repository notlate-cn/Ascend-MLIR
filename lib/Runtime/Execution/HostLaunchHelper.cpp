//===- HostLaunchHelper.cpp - generated-host AscendC kernel launcher --===//
//
// Implements `hostLaunchAscendCKernel`, the C-callable wrapper around
// NativeExecutionRunner (Simulation mode) used by AclnnBackend-emitted
// network_host.cpp. See HostLaunchHelper.h for the contract.
//
//===----------------------------------------------------------------------===//

#include "Runtime/Execution/HostLaunchHelper.h"
#include "Runtime/Execution/NativeExecutionRunner.h"
#include "Runtime/Support/NpyIO.h"
#include "Runtime/Support/Types.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir::runtime {

namespace {

// aclDataType -> runtime::DType. Returns true on success.
bool aclDtypeToDType(int aclDtype, DType &out) {
  switch (aclDtype) {
  case 0: out = DType::F32;   return true;  // ACL_FLOAT
  case 1: out = DType::F16;   return true;  // ACL_FLOAT16
  case 2: out = DType::INT8;  return true;  // ACL_INT8
  case 3: out = DType::INT32; return true;  // ACL_INT32
  case 9: out = DType::INT64; return true;  // ACL_INT64
  case 27: out = DType::BF16; return true;  // ACL_BF16
  default: return false;
  }
}

struct KernelTiling {
  int blockDim = 1;
  std::vector<uint8_t> packed;  // raw little-endian int64 words for non-meta params
};

class HelperState {
public:
  static HelperState &instance() {
    static HelperState inst;
    return inst;
  }

  std::mutex mu;
  std::unique_ptr<NativeExecutionRunner> runner;
  bool runnerInitialized = false;

  // Cache: parsed tilings keyed by file path.
  std::string loadedTilingsPath;
  std::unordered_map<std::string, KernelTiling> tilingsByKernel;
  bool tilingsLoaded = false;

  std::string dumpDir;
};

// Lazily create the simulation-mode runner. Caller holds the mutex.
int ensureRunner(HelperState &st) {
  if (st.runnerInitialized) return 0;
  st.runner =
      std::make_unique<NativeExecutionRunner>(ExecutionRunnerMode::Simulation);
  if (auto err = st.runner->initialize(/*deviceId=*/0)) {
    llvm::errs() << "hostLaunchAscendCKernel: runner initialize failed: "
                 << llvm::toString(std::move(err)) << "\n";
    st.runner.reset();
    return 1;
  }
  st.runnerInitialized = true;
  return 0;
}

// Load tilings JSON if not already loaded (or path changed). Returns 0 on
// success; missing/empty file is tolerated (no tilings is valid).
int loadTilingsIfNeeded(HelperState &st, const std::string &path) {
  if (st.tilingsLoaded && st.loadedTilingsPath == path)
    return 0;
  st.tilingsByKernel.clear();
  st.loadedTilingsPath = path;
  st.tilingsLoaded = true;
  if (path.empty()) return 0;

  auto bufOr = llvm::MemoryBuffer::getFile(path, /*IsText=*/true);
  if (!bufOr) {
    // Missing tilings file is non-fatal for kernels that need no params.
    return 0;
  }
  auto parsed = llvm::json::parse((*bufOr)->getBuffer());
  if (!parsed) {
    llvm::errs() << "hostLaunchAscendCKernel: tilings JSON parse failed ("
                 << path << "): " << llvm::toString(parsed.takeError())
                 << "\n";
    return 0;  // keep going with empty tilings
  }
  auto *topObj = parsed->getAsObject();
  if (!topObj) return 0;

  for (auto &kv : *topObj) {
    auto *kernelObj = kv.second.getAsObject();
    if (!kernelObj) continue;
    KernelTiling t;
    for (auto &pkv : *kernelObj) {
      llvm::StringRef pname = pkv.first;
      auto valOpt = pkv.second.getAsInteger();
      if (!valOpt) continue;
      int64_t v = *valOpt;
      if (pname == "_block_dim") {
        t.blockDim = static_cast<int>(v);
        continue;
      }
      size_t off = t.packed.size();
      t.packed.resize(off + sizeof(int64_t));
      std::memcpy(t.packed.data() + off, &v, sizeof(int64_t));
    }
    st.tilingsByKernel.emplace(kv.first.str(), std::move(t));
  }
  return 0;
}

// Build an NDArray view over an externally-owned host buffer.
bool wrapTensorAsNDArray(const aclnn::TensorInfo &t, NDArray &out) {
  DType dt;
  if (!aclDtypeToDType(t.dtype, dt)) {
    llvm::errs() << "hostLaunchAscendCKernel: unsupported aclDataType="
                 << t.dtype << "\n";
    return false;
  }
  out.dtype = dt;
  out.shape.assign(t.shape, t.shape + t.rank);
  out.setExternal(t.data);
  return true;
}

void dumpTensorIfEnabled(const std::string &dir, const std::string &kernel,
                          const char *role, int idx,
                          const aclnn::TensorInfo &t) {
  if (dir.empty()) return;
  DType dt;
  if (!aclDtypeToDType(t.dtype, dt)) return;
  NDArray arr;
  arr.dtype = dt;
  arr.shape.assign(t.shape, t.shape + t.rank);
  arr.setExternal(t.data);
  std::string path = dir + "/" + kernel + "_" + role + "_" +
                     std::to_string(idx) + ".npy";
  if (auto err = SaveNpy(path, arr)) {
    llvm::errs() << "hostLaunchAscendCKernel: SaveNpy failed (" << path
                 << "): " << llvm::toString(std::move(err)) << "\n";
  }
}

} // namespace

extern "C" void hostLaunchSetDumpIntermediatesDir(const char *dir) {
  HelperState &st = HelperState::instance();
  std::lock_guard<std::mutex> lk(st.mu);
  st.dumpDir = dir ? std::string(dir) : std::string();
}

extern "C" int hostLaunchAscendCKernel(
    const char *kernelName,
    const char *kernelBinariesDir,
    const char *tilingsPath,
    aclnn::TensorInfo *inputs, int numInputs,
    aclnn::TensorInfo *outputs, int numOutputs) {
  if (!kernelName || !kernelBinariesDir) {
    llvm::errs() << "hostLaunchAscendCKernel: kernelName/kernelBinariesDir "
                    "must not be null\n";
    return 1;
  }

  HelperState &st = HelperState::instance();
  std::lock_guard<std::mutex> lk(st.mu);

  if (int rc = ensureRunner(st)) return rc;
  loadTilingsIfNeeded(st, tilingsPath ? std::string(tilingsPath)
                                       : std::string());

  // Build RunArgs from caller-provided TensorInfos (host buffers).
  RunArgs args;
  args.inputs.reserve(numInputs);
  for (int i = 0; i < numInputs; ++i) {
    NDArray a;
    if (!wrapTensorAsNDArray(inputs[i], a)) return 2;
    args.inputs.push_back(std::move(a));
  }
  args.outputs.reserve(numOutputs);
  for (int i = 0; i < numOutputs; ++i) {
    NDArray a;
    if (!wrapTensorAsNDArray(outputs[i], a)) return 2;
    args.outputs.push_back(std::move(a));
  }

  // Tilings + block dim.
  auto tIt = st.tilingsByKernel.find(kernelName);
  if (tIt != st.tilingsByKernel.end()) {
    args.tiling = tIt->second.packed;
    args.block_dim = tIt->second.blockDim > 0 ? tIt->second.blockDim : 1;
  }

  // Dump inputs.
  for (int i = 0; i < numInputs; ++i)
    dumpTensorIfEnabled(st.dumpDir, kernelName, "in", i, inputs[i]);

  // Launch.
  FileExecutionLaunch launch;
  launch.binaryPath = std::string(kernelBinariesDir) + "/" + kernelName + ".bin";
  launch.kernelName = kernelName;
  // 'Vec' magic matches both Vec and Mix kernel kinds (see SimBackend).
  launch.magic = 0x41415246u;  // kMagicElfAiVec

  if (auto err = st.runner->runFile(launch, args)) {
    llvm::errs() << "hostLaunchAscendCKernel(" << kernelName
                 << ") runFile failed: " << llvm::toString(std::move(err))
                 << "\n";
    return 3;
  }

  // Output bytes already landed in the caller-provided host buffers because
  // wrapTensorAsNDArray used setExternal — runFile's deviceToHost wrote into
  // the same memory. Nothing to copy back.

  // Dump outputs.
  for (int i = 0; i < numOutputs; ++i)
    dumpTensorIfEnabled(st.dumpDir, kernelName, "out", i, outputs[i]);

  return 0;
}

} // namespace mlir::runtime
