// lib/Runtime/Executor.cpp
#include "Runtime/Executor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <sys/stat.h>

namespace mlir::runtime {

Executor::Executor(BackendMode mode) : mode_(mode) {}

Executor::~Executor() {
  FreeAll();
  if (stream_ && rtStreamDestroy_)
    rtStreamDestroy_(stream_);
  stream_ = nullptr;
  if (acl_handle_)
    dlclose(acl_handle_);
  acl_handle_ = nullptr;
  if (lib_handle_)
    dlclose(lib_handle_);
  lib_handle_ = nullptr;
}

static std::string getAscendHomePath() {
  const char* envs[] = {
      std::getenv("ASCEND_HOME_PATH"),
      std::getenv("ASCEND_TOOLKIT_HOME"),
      "/home/niu/Ascend/latest",
      "/usr/local/Ascend/latest",
      "/usr/local/Ascend/ascend-toolkit/latest",
  };
  for (const char* home : envs) {
    if (home && *home)
      return home;
  }
  return "/usr/local/Ascend/ascend-toolkit/latest";
}

static std::string getLibPath() {
  const std::string home = getAscendHomePath();
  const char* soc = std::getenv("SOC_VERSION");
  if (!soc) soc = "Ascend910B1";
#if defined(__x86_64__) || defined(_M_X64)
  const char* cann_arch = "x86_64-linux";
#elif defined(__aarch64__) || defined(_M_ARM64)
  const char* cann_arch = "aarch64-linux";
#else
  const char* cann_arch = "aarch64-linux";
#endif
  std::vector<std::string> candidates = {
      home + "/" + cann_arch + "/simulator/" + std::string(soc) + "/lib/libruntime_camodel.so",
      home + "/tools/simulator/" + std::string(soc) + "/lib/libruntime_camodel.so",
      home + "/runtime/lib64/libruntime_camodel.so",
  };
  for (const auto& path : candidates) {
    struct stat st;
    if (::stat(path.c_str(), &st) == 0)
      return path;
  }
  return candidates.front();
}

static std::string getAclLibPath() {
  const std::string home = getAscendHomePath();
  // Try common paths for libascendcl.so
  std::vector<std::string> candidates = {
      home + "/lib64/libascendcl.so",
      home + "/aarch64-linux/lib64/libascendcl.so",
      home + "/x86_64-linux/lib64/libascendcl.so",
  };
  for (const auto& path : candidates) {
    struct stat st;
    if (::stat(path.c_str(), &st) == 0)
      return path;
  }
  return home + "/lib64/libascendcl.so";
}

llvm::Error Executor::LoadLib() {
  if (mode_ == BackendMode::RealDevice)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "RealDevice mode not implemented");
  std::string lib = getLibPath();
  lib_handle_ = dlopen(lib.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (!lib_handle_)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "dlopen failed (%s): %s", lib.c_str(),
                                   dlerror());

  std::string aclLib = getAclLibPath();
  acl_handle_ = dlopen(aclLib.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (!acl_handle_)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "dlopen failed (%s): %s", aclLib.c_str(),
                                   dlerror());

#define LOAD_RT(name)                                                            \
  name##_ = reinterpret_cast<decltype(name##_)>(dlsym(lib_handle_, #name));      \
  if (!name##_)                                                                  \
    return llvm::createStringError(llvm::inconvertibleErrorCode(),               \
                                   "dlsym " #name " failed: %s", dlerror())
#define LOAD_ACL(name)                                                           \
  name##_ = reinterpret_cast<decltype(name##_)>(dlsym(acl_handle_, #name));      \
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
  LOAD_ACL(aclInit);
  LOAD_ACL(aclFinalize);
  LOAD_ACL(aclrtSetDevice);
  LOAD_ACL(aclrtResetDevice);
  LOAD_ACL(aclrtCreateStream);
  LOAD_ACL(aclrtDestroyStream);
  LOAD_ACL(aclrtMalloc);
  LOAD_ACL(aclrtFree);
  LOAD_ACL(aclrtMallocHost);
  LOAD_ACL(aclrtFreeHost);
  LOAD_ACL(aclrtMemcpy);
  LOAD_ACL(aclrtSynchronizeStream);
#undef LOAD_RT
#undef LOAD_ACL
  return llvm::Error::success();
}

llvm::Error Executor::Initialize(int device_id) {
  if (auto err = LoadLib())
    return err;
  static std::atomic<bool> device_initialized{false};
  bool expected = false;
  if (device_initialized.compare_exchange_strong(expected, true)) {
    int rc = rtSetDevice_(static_cast<int32_t>(device_id));
    if (rc != 0) {
      device_initialized.store(false);
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "rtSetDevice failed: rc=%d", rc);
    }
  }
  device_id_ = static_cast<int32_t>(device_id);
  if (!stream_) {
    int rc = rtStreamCreate_(&stream_, 0);
    if (rc != 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "rtStreamCreate failed: rc=%d", rc);
  }
  return llvm::Error::success();
}

llvm::Expected<void*> Executor::Alloc(size_t nbytes) {
  void* raw = nullptr;
  int rc = rtMalloc_(&raw, static_cast<uint64_t>(nbytes + 512),
                     static_cast<uint32_t>(0), static_cast<uint16_t>(33));
  if (rc != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtMalloc failed: rc=%d", rc);
  uintptr_t aligned = (reinterpret_cast<uintptr_t>(raw) + 511) & ~511ULL;
  alloc_map_.push_back({raw, reinterpret_cast<void*>(aligned)});
  return reinterpret_cast<void*>(aligned);
}

void Executor::FreeAll() {
  if (!rtFree_) { alloc_map_.clear(); return; }
  for (auto& a : alloc_map_) rtFree_(a.raw);
  alloc_map_.clear();
}

llvm::Error Executor::H2D(void* dst, const void* src, size_t n) {
  static constexpr size_t kH2DChunk = 4096;
  const uint8_t* s = static_cast<const uint8_t*>(src);
  uint8_t* d = static_cast<uint8_t*>(dst);
  for (size_t off = 0; off < n; off += kH2DChunk) {
    size_t chunk = std::min<size_t>(kH2DChunk, n - off);
    int rc = rtMemcpy_(d + off, chunk, s + off, chunk, /*H2D=*/1);
    if (rc != 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "rtMemcpy H2D failed rc=%d at offset %zu",
                                     rc, off);
  }
  return llvm::Error::success();
}

llvm::Error Executor::D2H(void* dst, const void* src, size_t n) {
  uint8_t* d = static_cast<uint8_t*>(dst);
  const uint8_t* s = static_cast<const uint8_t*>(src);
  static constexpr size_t kD2HChunk = 4;
  for (size_t off = 0; off < n; off += kD2HChunk) {
    uint8_t buf[kD2HChunk] = {};
    int rc = rtMemcpy_(buf, kD2HChunk, s + off, kD2HChunk, /*D2H=*/2);
    if (rc != 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "rtMemcpy D2H failed rc=%d at offset %zu",
                                     rc, off);
    size_t chunk = std::min<size_t>(kD2HChunk, n - off);
    std::memcpy(d + off, buf, chunk);
  }
  return llvm::Error::success();
}

llvm::Error Executor::Run(const std::vector<uint8_t>& binary_data,
                          const std::string& function_name, RunArgs& args,
                          uint32_t magic) {
  DevBinary dev_bin;
  dev_bin.magic   = magic;
  dev_bin.version = 0;
  dev_bin.data    = reinterpret_cast<const char*>(binary_data.data());
  dev_bin.length  = static_cast<uint64_t>(binary_data.size());

  void* bin_handle = nullptr;
  int rc = rtDevBinaryRegister_(&dev_bin, &bin_handle);
  if (rc != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtDevBinaryRegister failed: rc=%d", rc);

  const char* fn_name      = function_name.c_str();
  void*       fn_name_void = const_cast<char*>(fn_name);
  rc = rtFunctionRegister_(bin_handle, fn_name_void, fn_name, fn_name_void, 0);
  if (rc != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtFunctionRegister failed: rc=%d", rc);
  void* func_handle = fn_name_void;

  std::vector<void*> input_gm;
  for (auto& inp : args.inputs) {
    auto p = Alloc(inp.nbytes());
    if (!p) { FreeAll(); return p.takeError(); }
    input_gm.push_back(*p);
    if (auto err = H2D(*p, inp.data, inp.nbytes())) { FreeAll(); return err; }
  }

  std::vector<void*> output_gm;
  for (auto& out : args.outputs) {
    auto p = Alloc(out.nbytes());
    if (!p) { FreeAll(); return p.takeError(); }
    output_gm.push_back(*p);
  }

  auto ws = Alloc(args.workspace_size);
  if (!ws) { FreeAll(); return ws.takeError(); }

  std::vector<uint64_t> launch_args;
  for (auto* p : input_gm)  launch_args.push_back(reinterpret_cast<uint64_t>(p));
  for (auto* p : output_gm) launch_args.push_back(reinterpret_cast<uint64_t>(p));
  launch_args.push_back(reinterpret_cast<uint64_t>(*ws));

  const auto& t = args.tiling;
  for (size_t i = 0; i < t.size(); i += 8) {
    uint64_t w = 0;
    std::memcpy(&w, t.data() + i, std::min<size_t>(8, t.size() - i));
    launch_args.push_back(w);
  }

  uint32_t args_size = static_cast<uint32_t>(launch_args.size() * sizeof(uint64_t));
  rc = rtKernelLaunch_(func_handle, static_cast<uint32_t>(args.block_dim),
                       launch_args.data(), args_size, nullptr, stream_);
  if (rc != 0) {
    FreeAll();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtKernelLaunch failed: rc=%d", rc);
  }

  rtStreamSynchronize_(stream_);

  for (size_t i = 0; i < args.outputs.size(); ++i) {
    if (auto err = D2H(args.outputs[i].data, output_gm[i],
                       args.outputs[i].nbytes())) {
      FreeAll(); return err;
    }
  }

  FreeAll();
  return llvm::Error::success();
}

llvm::Error Executor::RunFile(const std::string& binary_path,
                              const std::string& function_name, RunArgs& args,
                              uint32_t magic) {
  auto buf = llvm::MemoryBuffer::getFile(binary_path, /*IsText=*/false);
  if (!buf)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot read binary: %s",
                                   binary_path.c_str());
  const uint8_t* data =
      reinterpret_cast<const uint8_t*>((*buf)->getBufferStart());
  std::vector<uint8_t> bytes(data, data + (*buf)->getBufferSize());
  return Run(bytes, function_name, args, magic);
}

llvm::Error Executor::RunPackedMixFile(const std::string& shared_lib_path,
                                       const std::string& kernel_name,
                                       RunArgs& args) {
  if (!lib_handle_ || !aclInit_ || !aclFinalize_ || !aclrtSetDevice_ ||
      !aclrtResetDevice_ || !aclrtCreateStream_ || !aclrtDestroyStream_ ||
      !aclrtMalloc_ || !aclrtFree_ || !aclrtMemcpy_ ||
      !aclrtSynchronizeStream_)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Executor::Initialize() must succeed before RunPackedMixFile()");

  void* mix_lib = dlopen(shared_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!mix_lib)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "dlopen packed mix failed (%s): %s",
                                   shared_lib_path.c_str(), dlerror());

  std::string launch_name = "aclrtlaunch_" + kernel_name;
  // Generic launch function: (numBlocks, stream, inputs..., outputs..., workspace, tiling)
  // We use void* variadic approach via a function pointer that accepts void* array.
  // The actual calling convention uses a fixed-arity stub generated by the host stub.
  // We build an argv-style array and call via dlsym with the correct arity.
  // Since we don't know arity at compile time, we look up the symbol and call
  // with the maximum supported args (all void*).
  using LaunchFnMax = uint32_t (*)(uint32_t, void*,
                                   void*, void*, void*, void*, void*, void*, void*, void*,
                                   void*, void*, void*, void*, void*, void*, void*, void*);
  auto* launch = reinterpret_cast<LaunchFnMax>(dlsym(mix_lib, launch_name.c_str()));
  if (!launch) {
    const char* err = dlerror();
    dlclose(mix_lib);
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "dlsym %s failed: %s",
                                   launch_name.c_str(), err ? err : "unknown");
  }

  static constexpr uint32_t kAclMemMallocHugeFirst = 0;
  static constexpr int32_t kAclMemcpyHostToDevice = 1;
  static constexpr int32_t kAclMemcpyDeviceToHost = 2;

  bool aclInitialized = false;
  bool deviceSet = false;
  void* aclStream = nullptr;
  std::vector<void*> aclAllocs;
  auto cleanup = [&]() {
    for (auto it = aclAllocs.rbegin(); it != aclAllocs.rend(); ++it) {
      if (*it)
        (void)aclrtFree_(*it);
    }
    aclAllocs.clear();
    if (aclStream)
      (void)aclrtDestroyStream_(aclStream);
    if (deviceSet)
      (void)aclrtResetDevice_(device_id_);
    if (aclInitialized)
      (void)aclFinalize_();
    dlclose(mix_lib);
  };

  int rc = aclInit_(nullptr);
  if (rc != 0) {
    dlclose(mix_lib);
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "aclInit failed: rc=%d", rc);
  }
  aclInitialized = true;

  rc = aclrtSetDevice_(device_id_);
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

  std::vector<void*> input_gm;
  input_gm.reserve(args.inputs.size());
  for (auto& inp : args.inputs) {
    void* dev = nullptr;
    rc = aclrtMalloc_(&dev, static_cast<uint64_t>(inp.nbytes()),
                      kAclMemMallocHugeFirst);
    if (rc != 0) {
      cleanup();
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "aclrtMalloc input failed: rc=%d", rc);
    }
    aclAllocs.push_back(dev);
    input_gm.push_back(dev);
    rc = aclrtMemcpy_(dev, static_cast<uint64_t>(inp.nbytes()), inp.data,
                      static_cast<uint64_t>(inp.nbytes()),
                      kAclMemcpyHostToDevice);
    if (rc != 0) {
      cleanup();
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "aclrtMemcpy input failed: rc=%d", rc);
    }
  }

  std::vector<void*> output_gm;
  output_gm.reserve(args.outputs.size());
  for (auto& out : args.outputs) {
    void* dev = nullptr;
    rc = aclrtMalloc_(&dev, static_cast<uint64_t>(out.nbytes()),
                      kAclMemMallocHugeFirst);
    if (rc != 0) {
      cleanup();
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "aclrtMalloc output failed: rc=%d", rc);
    }
    aclAllocs.push_back(dev);
    output_gm.push_back(dev);
  }

  void* workspace = nullptr;
  rc = aclrtMalloc_(&workspace, static_cast<uint64_t>(args.workspace_size),
                    kAclMemMallocHugeFirst);
  if (rc != 0) {
    cleanup();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "aclrtMalloc workspace failed: rc=%d", rc);
  }
  aclAllocs.push_back(workspace);

  void* tiling_gm = nullptr;
  if (!args.tiling.empty()) {
    rc = aclrtMalloc_(&tiling_gm, static_cast<uint64_t>(args.tiling.size()),
                      kAclMemMallocHugeFirst);
    if (rc != 0) {
      cleanup();
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "aclrtMalloc tiling failed: rc=%d", rc);
    }
    aclAllocs.push_back(tiling_gm);
    rc = aclrtMemcpy_(tiling_gm, static_cast<uint64_t>(args.tiling.size()),
                      args.tiling.data(),
                      static_cast<uint64_t>(args.tiling.size()),
                      kAclMemcpyHostToDevice);
    if (rc != 0) {
      cleanup();
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "aclrtMemcpy tiling failed: rc=%d", rc);
    }
  }

  // Build the GM args array: [inputs..., outputs..., workspace, tiling_gm]
  // Pad to 16 slots with nullptr (the generated stub only reads the arity it expects)
  void* gm_args[16] = {};
  size_t idx = 0;
  for (auto* p : input_gm)  { if (idx < 16) gm_args[idx++] = p; }
  for (auto* p : output_gm) { if (idx < 16) gm_args[idx++] = p; }
  if (idx < 16) gm_args[idx++] = workspace;
  if (idx < 16) gm_args[idx++] = tiling_gm;

  uint32_t launchRc = launch(
      static_cast<uint32_t>(args.block_dim), aclStream,
      gm_args[0], gm_args[1], gm_args[2], gm_args[3],
      gm_args[4], gm_args[5], gm_args[6], gm_args[7],
      gm_args[8], gm_args[9], gm_args[10], gm_args[11],
      gm_args[12], gm_args[13], gm_args[14], gm_args[15]);
  if (launchRc != 0) {
    cleanup();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "packed mix launch failed: rc=%u", launchRc);
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
                      output_gm[i],
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

llvm::Expected<void*> Executor::RegisterBinary(const std::string& binary_path,
                                               const std::string& function_name,
                                               uint32_t magic) {
  auto buf = llvm::MemoryBuffer::getFile(binary_path, /*IsText=*/false);
  if (!buf)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot read binary: %s",
                                   binary_path.c_str());

  registered_binaries_.emplace_back(
      reinterpret_cast<const uint8_t*>((*buf)->getBufferStart()),
      reinterpret_cast<const uint8_t*>((*buf)->getBufferStart()) +
          (*buf)->getBufferSize());
  registered_names_.push_back(function_name);

  const std::vector<uint8_t>& bytes = registered_binaries_.back();
  const std::string& stable_name    = registered_names_.back();

  DevBinary dev_bin;
  dev_bin.magic   = magic;
  dev_bin.version = 0;
  dev_bin.data    = reinterpret_cast<const char*>(bytes.data());
  dev_bin.length  = static_cast<uint64_t>(bytes.size());

  void* bin_handle = nullptr;
  int rc = rtDevBinaryRegister_(&dev_bin, &bin_handle);
  if (rc != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtDevBinaryRegister failed: rc=%d", rc);

  const char* fn_name      = stable_name.c_str();
  void*       fn_name_void = const_cast<char*>(fn_name);
  rc = rtFunctionRegister_(bin_handle, fn_name_void, fn_name, fn_name_void, 0);
  if (rc != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtFunctionRegister failed: rc=%d", rc);

  return fn_name_void;
}

llvm::Error Executor::RunWithHandle(void* func_handle, RunArgs& args) {
  std::vector<void*> input_gm;
  for (auto& inp : args.inputs) {
    auto p = Alloc(inp.nbytes());
    if (!p) { FreeAll(); return p.takeError(); }
    input_gm.push_back(*p);
    if (auto err = H2D(*p, inp.data, inp.nbytes())) { FreeAll(); return err; }
  }

  std::vector<void*> output_gm;
  for (auto& out : args.outputs) {
    auto p = Alloc(out.nbytes());
    if (!p) { FreeAll(); return p.takeError(); }
    output_gm.push_back(*p);
  }

  auto ws = Alloc(args.workspace_size);
  if (!ws) { FreeAll(); return ws.takeError(); }

  std::vector<uint64_t> launch_args;
  for (auto* p : input_gm)  launch_args.push_back(reinterpret_cast<uint64_t>(p));
  for (auto* p : output_gm) launch_args.push_back(reinterpret_cast<uint64_t>(p));
  launch_args.push_back(reinterpret_cast<uint64_t>(*ws));

  const auto& t = args.tiling;
  for (size_t i = 0; i < t.size(); i += 8) {
    uint64_t w = 0;
    std::memcpy(&w, t.data() + i, std::min<size_t>(8, t.size() - i));
    launch_args.push_back(w);
  }

  uint32_t args_size = static_cast<uint32_t>(launch_args.size() * sizeof(uint64_t));
  int rc = rtKernelLaunch_(func_handle, static_cast<uint32_t>(args.block_dim),
                           launch_args.data(), args_size, nullptr, stream_);
  if (rc != 0) {
    FreeAll();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtKernelLaunch failed: rc=%d", rc);
  }

  rtStreamSynchronize_(stream_);

  for (size_t i = 0; i < args.outputs.size(); ++i) {
    if (auto err = D2H(args.outputs[i].data, output_gm[i],
                       args.outputs[i].nbytes())) {
      FreeAll(); return err;
    }
  }

  FreeAll();
  return llvm::Error::success();
}

} // namespace mlir::runtime
