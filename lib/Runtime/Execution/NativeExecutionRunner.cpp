#include "Runtime/Execution/NativeExecutionRunner.h"

#include "Runtime/PathUtils.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <type_traits>

namespace mlir::runtime {

struct DevBinary {
  uint32_t magic;
  uint32_t version;
  const char *data;
  uint64_t length;
};

namespace {

static std::string getRuntimeLibPath(ExecutionRunnerMode mode) {
  const std::string home = findAscendHome();
  if (mode == ExecutionRunnerMode::RealDevice)
    return findAscendRuntimeLibPath(home);
  return findAscendRuntimeCamodelPath(home,
                                      resolveSocVersion("", "Ascend910B1"));
}

static std::string getAclLibPath() {
  const std::string home = findAscendHome();
  return findAscendAclLibPath(home);
}

static void prependEnvPath(const char *name, const std::string &prefix) {
  if (prefix.empty())
    return;
  const char *current = std::getenv(name);
  std::string value = prefix;
  if (current && *current) {
    value.push_back(':');
    value += current;
  }
  ::setenv(name, value.c_str(), 1);
}

static std::string buildRegisteredFunctionKey(const FileExecutionLaunch &launch) {
  std::string key = launch.binaryPath;
  key.push_back('\n');
  key += launch.kernelName;
  key.push_back('\n');
  key += std::to_string(launch.magic);
  return key;
}

} // namespace

NativeExecutionRunner::NativeExecutionRunner(ExecutionRunnerMode mode)
    : mode_(mode) {}

NativeExecutionRunner::~NativeExecutionRunner() {
  if (mode_ == ExecutionRunnerMode::Simulation)
    return;

  freeAll();
  if (stream_ && rtStreamDestroy_)
    rtStreamDestroy_(stream_);
  stream_ = nullptr;
  if (aclHandle_)
    dlclose(aclHandle_);
  aclHandle_ = nullptr;
  if (libHandle_)
    dlclose(libHandle_);
  libHandle_ = nullptr;
}

llvm::Error NativeExecutionRunner::loadRuntimeLibraries() {
  auto ascendHomeOr = requireAscendHome();
  if (!ascendHomeOr)
    return ascendHomeOr.takeError();

  const std::string ascendHome = *ascendHomeOr;
  static std::once_flag runtimeEnvOnce;
  std::call_once(runtimeEnvOnce, [&] {
    prependEnvPath("LD_LIBRARY_PATH", findAscendLib64Dir(ascendHome));
    prependEnvPath("LD_LIBRARY_PATH", findAscendDeviceLibDir(ascendHome));
  });

  const std::string runtimeLib = getRuntimeLibPath(mode_);
  libHandle_ = dlopen(runtimeLib.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (!libHandle_)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "dlopen failed (%s): %s",
                                   runtimeLib.c_str(), dlerror());

#define LOAD_RT(name)                                                            \
  name##_ = reinterpret_cast<decltype(name##_)>(dlsym(libHandle_, #name));      \
  if (!name##_)                                                                  \
    return llvm::createStringError(llvm::inconvertibleErrorCode(),               \
                                   "dlsym " #name " failed: %s", dlerror())
  LOAD_RT(rtSetDevice);
  LOAD_RT(rtDevBinaryRegister);
  LOAD_RT(rtFunctionRegister);
  LOAD_RT(rtMalloc);
  LOAD_RT(rtFree);
  LOAD_RT(rtMemcpy);
  LOAD_RT(rtStreamCreate);
  LOAD_RT(rtStreamDestroy);
  LOAD_RT(rtKernelLaunch);
  LOAD_RT(rtStreamSynchronize);
  LOAD_RT(rtDeviceSynchronize);
#undef LOAD_RT

  return llvm::Error::success();
}

llvm::Error NativeExecutionRunner::initialize(int deviceId) {
  if (initialized_)
    return llvm::Error::success();

  if (auto err = loadRuntimeLibraries())
    return err;

  static std::atomic<bool> deviceInitialized{false};
  bool expected = false;
  if (deviceInitialized.compare_exchange_strong(expected, true)) {
    int rc = rtSetDevice_(static_cast<int32_t>(deviceId));
    if (rc != 0) {
      deviceInitialized.store(false);
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "rtSetDevice failed: rc=%d", rc);
    }
  }

  deviceId_ = static_cast<int32_t>(deviceId);
  if (!stream_) {
    int rc = rtStreamCreate_(&stream_, 0);
    if (rc != 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "rtStreamCreate failed: rc=%d", rc);
  }

  initialized_ = true;
  return llvm::Error::success();
}

llvm::Expected<void *> NativeExecutionRunner::alloc(size_t nbytes) {
  void *raw = nullptr;
  int rc = rtMalloc_(&raw, static_cast<uint64_t>(nbytes + 512),
                     static_cast<uint32_t>(0), static_cast<uint16_t>(33));
  if (rc != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtMalloc failed: rc=%d", rc);

  uintptr_t aligned = (reinterpret_cast<uintptr_t>(raw) + 511) & ~511ULL;
  allocMap_.push_back({raw, reinterpret_cast<void *>(aligned)});
  return reinterpret_cast<void *>(aligned);
}

void NativeExecutionRunner::freeAll() {
  if (!rtFree_) {
    allocMap_.clear();
    return;
  }
  for (auto &alloc : allocMap_)
    rtFree_(alloc.raw);
  allocMap_.clear();
}

llvm::Error NativeExecutionRunner::hostToDevice(void *dst, const void *src,
                                                size_t nbytes) {
  static constexpr size_t kChunk = 4096;
  const uint8_t *s = static_cast<const uint8_t *>(src);
  uint8_t *d = static_cast<uint8_t *>(dst);
  for (size_t off = 0; off < nbytes; off += kChunk) {
    size_t chunk = std::min<size_t>(kChunk, nbytes - off);
    int rc = rtMemcpy_(d + off, chunk, s + off, chunk, /*H2D=*/1);
    if (rc != 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "rtMemcpy H2D failed rc=%d at offset %zu",
                                     rc, off);
  }
  return llvm::Error::success();
}

llvm::Error NativeExecutionRunner::deviceToHost(void *dst, const void *src,
                                                size_t nbytes) {
  uint8_t *d = static_cast<uint8_t *>(dst);
  const uint8_t *s = static_cast<const uint8_t *>(src);
  static constexpr size_t kChunk = 4;
  for (size_t off = 0; off < nbytes; off += kChunk) {
    uint8_t buf[kChunk] = {};
    int rc = rtMemcpy_(buf, kChunk, s + off, kChunk, /*D2H=*/2);
    if (rc != 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "rtMemcpy D2H failed rc=%d at offset %zu",
                                     rc, off);
    size_t chunk = std::min<size_t>(kChunk, nbytes - off);
    std::memcpy(d + off, buf, chunk);
  }
  return llvm::Error::success();
}

llvm::Error NativeExecutionRunner::runBinary(
    const std::vector<uint8_t> &binaryData, const std::string &functionName,
    RunArgs &args, uint32_t magic) {
  DevBinary devBin;
  devBin.magic = magic;
  devBin.version = 0;
  devBin.data = reinterpret_cast<const char *>(binaryData.data());
  devBin.length = static_cast<uint64_t>(binaryData.size());

  void *binHandle = nullptr;
  int rc = rtDevBinaryRegister_(&devBin, &binHandle);
  if (rc != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtDevBinaryRegister failed: rc=%d", rc);

  const char *fnName = functionName.c_str();
  void *fnNameVoid = const_cast<char *>(fnName);
  rc = rtFunctionRegister_(binHandle, fnNameVoid, fnName, fnNameVoid, 0);
  if (rc != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtFunctionRegister failed: rc=%d", rc);

  std::vector<void *> inputGm;
  for (auto &input : args.inputs) {
    auto ptrOr = alloc(input.nbytes());
    if (!ptrOr) {
      freeAll();
      return ptrOr.takeError();
    }
    inputGm.push_back(*ptrOr);
    if (auto err = hostToDevice(*ptrOr, input.data, input.nbytes())) {
      freeAll();
      return err;
    }
  }

  std::vector<void *> outputGm;
  for (auto &output : args.outputs) {
    auto ptrOr = alloc(output.nbytes());
    if (!ptrOr) {
      freeAll();
      return ptrOr.takeError();
    }
    outputGm.push_back(*ptrOr);
  }

  auto workspaceOr = alloc(args.workspace_size);
  if (!workspaceOr) {
    freeAll();
    return workspaceOr.takeError();
  }

  std::vector<uint64_t> launchArgs;
  for (auto *ptr : inputGm)
    launchArgs.push_back(reinterpret_cast<uint64_t>(ptr));
  for (auto *ptr : outputGm)
    launchArgs.push_back(reinterpret_cast<uint64_t>(ptr));
  launchArgs.push_back(reinterpret_cast<uint64_t>(*workspaceOr));

  const auto &tiling = args.tiling;
  for (size_t i = 0; i < tiling.size(); i += 8) {
    uint64_t word = 0;
    std::memcpy(&word, tiling.data() + i,
                std::min<size_t>(8, tiling.size() - i));
    launchArgs.push_back(word);
  }

  uint32_t argsSize =
      static_cast<uint32_t>(launchArgs.size() * sizeof(uint64_t));
  rc = rtKernelLaunch_(fnNameVoid, static_cast<uint32_t>(args.block_dim),
                       launchArgs.data(), argsSize, nullptr, stream_);
  if (rc != 0) {
    freeAll();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtKernelLaunch failed: rc=%d", rc);
  }

  rtStreamSynchronize_(stream_);

  for (size_t i = 0; i < args.outputs.size(); ++i) {
    if (auto err =
            deviceToHost(args.outputs[i].data, outputGm[i],
                         args.outputs[i].nbytes())) {
      freeAll();
      return err;
    }
  }

  freeAll();
  return llvm::Error::success();
}

llvm::Expected<void *> NativeExecutionRunner::registerBinary(
    const std::string &binaryPath, const std::string &functionName,
    uint32_t magic) {
  auto buf = llvm::MemoryBuffer::getFile(binaryPath, /*IsText=*/false);
  if (!buf)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot read binary: %s", binaryPath.c_str());

  registeredBinaries_.emplace_back(
      reinterpret_cast<const uint8_t *>((*buf)->getBufferStart()),
      reinterpret_cast<const uint8_t *>((*buf)->getBufferStart()) +
          (*buf)->getBufferSize());
  registeredNames_.push_back(functionName);

  const std::vector<uint8_t> &bytes = registeredBinaries_.back();
  const std::string &stableName = registeredNames_.back();

  DevBinary devBin;
  devBin.magic = magic;
  devBin.version = 0;
  devBin.data = reinterpret_cast<const char *>(bytes.data());
  devBin.length = static_cast<uint64_t>(bytes.size());

  void *binHandle = nullptr;
  int rc = rtDevBinaryRegister_(&devBin, &binHandle);
  if (rc != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtDevBinaryRegister failed: rc=%d", rc);

  const char *fnName = stableName.c_str();
  void *fnNameVoid = const_cast<char *>(fnName);
  rc = rtFunctionRegister_(binHandle, fnNameVoid, fnName, fnNameVoid, 0);
  if (rc != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtFunctionRegister failed: rc=%d", rc);

  return fnNameVoid;
}

llvm::Error NativeExecutionRunner::runWithHandle(void *funcHandle,
                                                  RunArgs &args) {
  std::vector<void *> inputGm;
  for (auto &input : args.inputs) {
    auto ptrOr = alloc(input.nbytes());
    if (!ptrOr) {
      freeAll();
      return ptrOr.takeError();
    }
    inputGm.push_back(*ptrOr);
    if (auto err = hostToDevice(*ptrOr, input.data, input.nbytes())) {
      freeAll();
      return err;
    }
  }

  std::vector<void *> outputGm;
  for (auto &output : args.outputs) {
    auto ptrOr = alloc(output.nbytes());
    if (!ptrOr) {
      freeAll();
      return ptrOr.takeError();
    }
    outputGm.push_back(*ptrOr);
  }

  auto workspaceOr = alloc(args.workspace_size);
  if (!workspaceOr) {
    freeAll();
    return workspaceOr.takeError();
  }

  std::vector<uint64_t> launchArgs;
  for (auto *ptr : inputGm)
    launchArgs.push_back(reinterpret_cast<uint64_t>(ptr));
  for (auto *ptr : outputGm)
    launchArgs.push_back(reinterpret_cast<uint64_t>(ptr));
  launchArgs.push_back(reinterpret_cast<uint64_t>(*workspaceOr));

  const auto &tiling = args.tiling;
  for (size_t i = 0; i < tiling.size(); i += 8) {
    uint64_t word = 0;
    std::memcpy(&word, tiling.data() + i,
                std::min<size_t>(8, tiling.size() - i));
    launchArgs.push_back(word);
  }

  uint32_t argsSize =
      static_cast<uint32_t>(launchArgs.size() * sizeof(uint64_t));
  int rc = rtKernelLaunch_(funcHandle, static_cast<uint32_t>(args.block_dim),
                           launchArgs.data(), argsSize, nullptr, stream_);
  if (rc != 0) {
    freeAll();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtKernelLaunch failed: rc=%d", rc);
  }

  rtStreamSynchronize_(stream_);

  for (size_t i = 0; i < args.outputs.size(); ++i) {
    if (auto err =
            deviceToHost(args.outputs[i].data, outputGm[i],
                         args.outputs[i].nbytes())) {
      freeAll();
      return err;
    }
  }

  freeAll();
  return llvm::Error::success();
}

llvm::Error NativeExecutionRunner::runFile(const FileExecutionLaunch &launch,
                                           RunArgs &args) {
  if (auto err = initialize())
    return err;

  const std::string key = buildRegisteredFunctionKey(launch);
  auto handleIt = registeredFunctionHandles_.find(key);
  if (handleIt == registeredFunctionHandles_.end()) {
    auto handleOr =
        registerBinary(launch.binaryPath, launch.kernelName, launch.magic);
    if (!handleOr)
      return handleOr.takeError();
    handleIt = registeredFunctionHandles_.emplace(key, *handleOr).first;
  }
  return runWithHandle(handleIt->second, args);
}

llvm::Error NativeExecutionRunner::runDynamicLibraryArtifact(
    const DynamicLibraryExecutionLaunch &launch, RunArgs &args) {
  if (auto err = initialize())
    return err;

  if (!libHandle_ || !rtSetDevice_ || !rtStreamCreate_ || !rtStreamDestroy_ ||
      !rtMalloc_ || !rtFree_ || !rtMemcpy_ || !rtStreamSynchronize_)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "NativeExecutionRunner::initialize() must succeed before "
        "runDynamicLibraryArtifact()");

  if (!aclHandle_) {
    const std::string aclLib = getAclLibPath();
    void *newAclHandle = dlopen(aclLib.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!newAclHandle)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "dlopen failed (%s): %s", aclLib.c_str(),
                                     dlerror());

    auto loadAclSymbol = [&](auto &fn, const char *symbolName) -> llvm::Error {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(
          dlsym(newAclHandle, symbolName));
      if (!fn) {
        const char *err = dlerror();
        dlclose(newAclHandle);
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "dlsym %s failed: %s", symbolName,
                                       err ? err : "unknown");
      }
      return llvm::Error::success();
    };

    decltype(aclInit_) newAclInit = nullptr;
    decltype(aclFinalize_) newAclFinalize = nullptr;
    decltype(aclrtSetDevice_) newAclrtSetDevice = nullptr;
    decltype(aclrtResetDevice_) newAclrtResetDevice = nullptr;
    decltype(aclrtCreateStream_) newAclrtCreateStream = nullptr;
    decltype(aclrtDestroyStream_) newAclrtDestroyStream = nullptr;
    decltype(aclrtMalloc_) newAclrtMalloc = nullptr;
    decltype(aclrtFree_) newAclrtFree = nullptr;
    decltype(aclrtMallocHost_) newAclrtMallocHost = nullptr;
    decltype(aclrtFreeHost_) newAclrtFreeHost = nullptr;
    decltype(aclrtMemcpy_) newAclrtMemcpy = nullptr;
    decltype(aclrtSynchronizeStream_) newAclrtSynchronizeStream = nullptr;
    if (auto err = loadAclSymbol(newAclInit, "aclInit"))
      return err;
    if (auto err = loadAclSymbol(newAclFinalize, "aclFinalize"))
      return err;
    if (auto err = loadAclSymbol(newAclrtSetDevice, "aclrtSetDevice"))
      return err;
    if (auto err = loadAclSymbol(newAclrtResetDevice, "aclrtResetDevice"))
      return err;
    if (auto err = loadAclSymbol(newAclrtCreateStream, "aclrtCreateStream"))
      return err;
    if (auto err = loadAclSymbol(newAclrtDestroyStream, "aclrtDestroyStream"))
      return err;
    if (auto err = loadAclSymbol(newAclrtMalloc, "aclrtMalloc"))
      return err;
    if (auto err = loadAclSymbol(newAclrtFree, "aclrtFree"))
      return err;
    if (auto err = loadAclSymbol(newAclrtMallocHost, "aclrtMallocHost"))
      return err;
    if (auto err = loadAclSymbol(newAclrtFreeHost, "aclrtFreeHost"))
      return err;
    if (auto err = loadAclSymbol(newAclrtMemcpy, "aclrtMemcpy"))
      return err;
    if (auto err =
            loadAclSymbol(newAclrtSynchronizeStream, "aclrtSynchronizeStream"))
      return err;

    aclHandle_ = newAclHandle;
    aclInit_ = newAclInit;
    aclFinalize_ = newAclFinalize;
    aclrtSetDevice_ = newAclrtSetDevice;
    aclrtResetDevice_ = newAclrtResetDevice;
    aclrtCreateStream_ = newAclrtCreateStream;
    aclrtDestroyStream_ = newAclrtDestroyStream;
    aclrtMalloc_ = newAclrtMalloc;
    aclrtFree_ = newAclrtFree;
    aclrtMallocHost_ = newAclrtMallocHost;
    aclrtFreeHost_ = newAclrtFreeHost;
    aclrtMemcpy_ = newAclrtMemcpy;
    aclrtSynchronizeStream_ = newAclrtSynchronizeStream;
  }

  void *dynamicLib =
      dlopen(launch.sharedLibraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!dynamicLib)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "dlopen dynamic library artifact failed (%s): %s",
                                   launch.sharedLibraryPath.c_str(), dlerror());

  using LaunchFnMax =
      uint32_t (*)(uint32_t, void *, void *, void *, void *, void *, void *,
                   void *, void *, void *, void *, void *, void *, void *,
                   void *, void *, void *, void *);
  auto *launchFn =
      reinterpret_cast<LaunchFnMax>(dlsym(dynamicLib, launch.symbolName.c_str()));
  if (!launchFn) {
    const char *err = dlerror();
    dlclose(dynamicLib);
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "dlsym %s failed: %s",
                                   launch.symbolName.c_str(),
                                   err ? err : "unknown");
  }

  static constexpr uint32_t kAclMemMallocHugeFirst = 0;
  static constexpr int32_t kAclMemcpyHostToDevice = 1;
  static constexpr int32_t kAclMemcpyDeviceToHost = 2;

  bool aclInitialized = false;
  bool deviceSet = false;
  void *aclStream = nullptr;
  std::vector<void *> aclAllocs;
  auto cleanup = [&]() {
    for (auto it = aclAllocs.rbegin(); it != aclAllocs.rend(); ++it) {
      if (*it)
        (void)aclrtFree_(*it);
    }
    aclAllocs.clear();
    if (aclStream)
      (void)aclrtDestroyStream_(aclStream);
    if (deviceSet)
      (void)aclrtResetDevice_(deviceId_);
    if (aclInitialized)
      (void)aclFinalize_();
    dlclose(dynamicLib);
  };

  int rc = aclInit_(nullptr);
  if (rc != 0) {
    dlclose(dynamicLib);
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "aclInit failed: rc=%d", rc);
  }
  aclInitialized = true;

  rc = aclrtSetDevice_(deviceId_);
  if (rc != 0) {
    cleanup();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "aclrtSetDevice failed: rc=%d", rc);
  }
  deviceSet = true;

  rc = aclrtCreateStream_(&aclStream);
  if (rc != 0) {
    cleanup();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "aclrtCreateStream failed: rc=%d", rc);
  }

  std::vector<void *> inputGm;
  inputGm.reserve(args.inputs.size());
  for (auto &input : args.inputs) {
    void *dev = nullptr;
    rc = aclrtMalloc_(&dev, static_cast<uint64_t>(input.nbytes()),
                      kAclMemMallocHugeFirst);
    if (rc != 0) {
      cleanup();
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "aclrtMalloc input failed: rc=%d", rc);
    }
    aclAllocs.push_back(dev);
    inputGm.push_back(dev);
    rc = aclrtMemcpy_(dev, static_cast<uint64_t>(input.nbytes()), input.data,
                      static_cast<uint64_t>(input.nbytes()),
                      kAclMemcpyHostToDevice);
    if (rc != 0) {
      cleanup();
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "aclrtMemcpy input failed: rc=%d", rc);
    }
  }

  std::vector<void *> outputGm;
  outputGm.reserve(args.outputs.size());
  for (auto &output : args.outputs) {
    void *dev = nullptr;
    rc = aclrtMalloc_(&dev, static_cast<uint64_t>(output.nbytes()),
                      kAclMemMallocHugeFirst);
    if (rc != 0) {
      cleanup();
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "aclrtMalloc output failed: rc=%d", rc);
    }
    aclAllocs.push_back(dev);
    outputGm.push_back(dev);
  }

  void *workspace = nullptr;
  rc = aclrtMalloc_(&workspace, static_cast<uint64_t>(args.workspace_size),
                    kAclMemMallocHugeFirst);
  if (rc != 0) {
    cleanup();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "aclrtMalloc workspace failed: rc=%d", rc);
  }
  aclAllocs.push_back(workspace);

  void *tilingGm = nullptr;
  if (!args.tiling.empty()) {
    rc = aclrtMalloc_(&tilingGm, static_cast<uint64_t>(args.tiling.size()),
                      kAclMemMallocHugeFirst);
    if (rc != 0) {
      cleanup();
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "aclrtMalloc tiling failed: rc=%d", rc);
    }
    aclAllocs.push_back(tilingGm);
    rc = aclrtMemcpy_(tilingGm, static_cast<uint64_t>(args.tiling.size()),
                      args.tiling.data(),
                      static_cast<uint64_t>(args.tiling.size()),
                      kAclMemcpyHostToDevice);
    if (rc != 0) {
      cleanup();
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "aclrtMemcpy tiling failed: rc=%d", rc);
    }
  }

  void *gmArgs[16] = {};
  size_t idx = 0;
  for (auto *p : inputGm) {
    if (idx < 16)
      gmArgs[idx++] = p;
  }
  for (auto *p : outputGm) {
    if (idx < 16)
      gmArgs[idx++] = p;
  }
  if (idx < 16)
    gmArgs[idx++] = workspace;
  if (idx < 16)
    gmArgs[idx++] = tilingGm;

  uint32_t launchRc = launchFn(static_cast<uint32_t>(args.block_dim), aclStream,
                               gmArgs[0], gmArgs[1], gmArgs[2], gmArgs[3],
                               gmArgs[4], gmArgs[5], gmArgs[6], gmArgs[7],
                               gmArgs[8], gmArgs[9], gmArgs[10], gmArgs[11],
                               gmArgs[12], gmArgs[13], gmArgs[14], gmArgs[15]);
  if (launchRc != 0) {
    cleanup();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "dynamic library artifact launch failed: rc=%u",
                                   launchRc);
  }

  rc = aclrtSynchronizeStream_(aclStream);
  if (rc != 0) {
    cleanup();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "aclrtSynchronizeStream failed: rc=%d", rc);
  }

  for (size_t i = 0; i < args.outputs.size(); ++i) {
    rc = aclrtMemcpy_(args.outputs[i].data,
                      static_cast<uint64_t>(args.outputs[i].nbytes()),
                      outputGm[i],
                      static_cast<uint64_t>(args.outputs[i].nbytes()),
                      kAclMemcpyDeviceToHost);
    if (rc != 0) {
      cleanup();
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "aclrtMemcpy output failed: rc=%d", rc);
    }
  }

  cleanup();
  return llvm::Error::success();
}

} // namespace mlir::runtime
