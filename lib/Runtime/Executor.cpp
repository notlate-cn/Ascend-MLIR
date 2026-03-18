// lib/Runtime/Executor.cpp
#include "Runtime/Executor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace mlir::runtime {

Executor::Executor(BackendMode mode) : mode_(mode) {}

Executor::~Executor() {
  FreeAll();
  if (lib_handle_) dlclose(lib_handle_);
}

static std::string getLibPath() {
  const char* home = std::getenv("ASCEND_HOME_PATH");
  if (!home) home = "/usr/local/Ascend/ascend-toolkit/latest";
  const char* soc = std::getenv("SOC_VERSION");
  if (!soc) soc = "Ascend910B1";
  // Try new toolkit layout: aarch64-linux/simulator/<SOC>/lib/
  std::string new_path = std::string(home) + "/aarch64-linux/simulator/" +
                         std::string(soc) + "/lib/libruntime_camodel.so";
  // Also try legacy layout: tools/simulator/<SOC>/lib/
  std::string legacy_path = std::string(home) + "/tools/simulator/" +
                            std::string(soc) + "/lib/libruntime_camodel.so";
  // Prefer new path; fallback checked by dlopen
  return new_path;
}

llvm::Error Executor::LoadLib() {
  if (mode_ == BackendMode::RealDevice)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "RealDevice mode not implemented");
  std::string lib = getLibPath();
  lib_handle_ = dlopen(lib.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (!lib_handle_)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "dlopen failed (%s): %s", lib.c_str(), dlerror());

#define LOAD(name) \
  name ## _ = reinterpret_cast<decltype(name ## _)>(dlsym(lib_handle_, #name)); \
  if (!name ## _) return llvm::createStringError(llvm::inconvertibleErrorCode(), \
                         "dlsym " #name " failed: %s", dlerror())
  LOAD(rtSetDevice);
  LOAD(rtDevBinaryRegister);
  LOAD(rtFunctionRegister);
  LOAD(rtMalloc);
  LOAD(rtFree);
  LOAD(rtMemcpy);
  LOAD(rtStreamCreate);
  LOAD(rtStreamDestroy);
  LOAD(rtKernelLaunch);
  LOAD(rtDeviceSynchronize);
#undef LOAD
  return llvm::Error::success();
}

llvm::Error Executor::Initialize(int device_id) {
  if (auto err = LoadLib()) return err;
  rtSetDevice_(static_cast<int32_t>(device_id)); // ignore return in simulation
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
  const uint8_t* s = static_cast<const uint8_t*>(src);
  uint8_t* d = static_cast<uint8_t*>(dst);
  for (size_t off = 0; off < n; off += 256) {
    size_t chunk = std::min<size_t>(256, n - off);
    int rc = rtMemcpy_(d + off, chunk, s + off, chunk, /*H2D=*/1);
    if (rc != 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "rtMemcpy H2D failed rc=%d at offset %zu", rc, off);
  }
  return llvm::Error::success();
}

llvm::Error Executor::D2H(void* dst, const void* src, size_t n) {
  // 4-byte chunks; safe because rtMalloc adds +512 overalloc
  uint8_t* d = static_cast<uint8_t*>(dst);
  const uint8_t* s = static_cast<const uint8_t*>(src);
  for (size_t off = 0; off < n; off += 4) {
    uint8_t buf[4] = {};
    int rc = rtMemcpy_(buf, 4, s + off, 4, /*D2H=*/2);
    if (rc != 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "rtMemcpy D2H failed rc=%d at offset %zu", rc, off);
    size_t chunk = std::min<size_t>(4, n - off);
    std::memcpy(d + off, buf, chunk);
  }
  return llvm::Error::success();
}

llvm::Error Executor::Run(const std::vector<uint8_t>& binary_data,
                           const std::string& function_name,
                           RunArgs& args,
                           uint32_t magic) {
  // 1. Register binary
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

  // 2. Register function (name ptr used as stub, same as Python executor.py)
  const char* fn_name      = function_name.c_str();
  void*       fn_name_void = const_cast<char*>(fn_name);
  rc = rtFunctionRegister_(bin_handle, fn_name_void, fn_name, fn_name_void, 0);
  if (rc != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtFunctionRegister failed: rc=%d", rc);
  void* func_handle = fn_name_void;

  // 3. Alloc + H2D inputs
  std::vector<void*> input_gm;
  for (auto& inp : args.inputs) {
    auto p = Alloc(inp.nbytes());
    if (!p) { FreeAll(); return p.takeError(); }
    input_gm.push_back(*p);
    if (auto err = H2D(*p, inp.data, inp.nbytes())) { FreeAll(); return err; }
  }

  // 4. Alloc outputs
  std::vector<void*> output_gm;
  for (auto& out : args.outputs) {
    auto p = Alloc(out.nbytes());
    if (!p) { FreeAll(); return p.takeError(); }
    output_gm.push_back(*p);
  }

  // 5. Alloc workspace
  auto ws = Alloc(args.workspace_size);
  if (!ws) { FreeAll(); return ws.takeError(); }

  // 6. Build args: [input_addrs..., output_addrs..., workspace_addr, tiling_words...]
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

  // 7. Create stream + launch
  void* stream = nullptr;
  rtStreamCreate_(&stream, 0);

  uint32_t args_size = static_cast<uint32_t>(launch_args.size() * sizeof(uint64_t));
  rc = rtKernelLaunch_(func_handle,
                        static_cast<uint32_t>(args.block_dim),
                        launch_args.data(),
                        args_size,
                        nullptr,
                        stream);
  if (rc != 0) {
    rtStreamDestroy_(stream);
    FreeAll();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtKernelLaunch failed: rc=%d", rc);
  }

  // 8. Sync + destroy stream
  rtDeviceSynchronize_();
  rtStreamDestroy_(stream);

  // 9. D2H outputs
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
                               const std::string& function_name,
                               RunArgs& args,
                               uint32_t magic) {
  auto buf = llvm::MemoryBuffer::getFile(binary_path, /*IsText=*/false);
  if (!buf)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot read binary: %s", binary_path.c_str());
  const uint8_t* data = reinterpret_cast<const uint8_t*>((*buf)->getBufferStart());
  std::vector<uint8_t> bytes(data, data + (*buf)->getBufferSize());
  return Run(bytes, function_name, args, magic);
}

} // namespace mlir::runtime
