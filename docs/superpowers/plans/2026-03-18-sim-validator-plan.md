# SimValidator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a C++ compile+execute+verify tool for AscendC kernels that mirrors `python/runtime/` behavior and can run `examples/broadcast-add-reduce/step8_kernel.cpp` end-to-end with CPU simulation.

**Architecture:** Three layered classes (`Compiler` → `Executor` → `SimValidator`) plus a `sim-validator` CLI tool. Mirrors `python/runtime/compiler.py` and `python/runtime/executor.py` exactly — same subprocess invocations, same memory layout, same chunk sizes. No MLIR dependencies; pure C++17 + LLVM Support.

**Tech Stack:** C++17, LLVM Support (llvm::Expected/llvm::Error, llvm::sys::ExecuteAndWait), CMake `add_mlir_library`, npy file I/O (hand-rolled), libruntime_camodel.so (dlopen at runtime).

---

## File Map

| File | Responsibility |
|------|---------------|
| `include/Runtime/Types.h` | NDArray, DType, RunArgs structs |
| `include/Runtime/Compiler.h` | Compiler class declaration |
| `include/Runtime/Executor.h` | Executor class, BackendMode enum, DevBinary struct |
| `include/Runtime/SimValidator.h` | SimValidator class declaration |
| `include/Runtime/NpyIO.h` | NpyIO load/save declarations |
| `lib/Runtime/Compiler.cpp` | bisheng compile + ld.lld link (exact flags from python/runtime/compiler.py) |
| `lib/Runtime/Executor.cpp` | dlopen libruntime_camodel + DevBinary+rtDevBinaryRegister+rtFunctionRegister+rtKernelLaunch |
| `lib/Runtime/SimValidator.cpp` | compile + run + compare |
| `lib/Runtime/NpyIO.cpp` | Minimal .npy read/write (float16, float32, int32) |
| `lib/Runtime/CMakeLists.txt` | Build rules for Runtime library |
| `tools/sim-validator/sim_validator_main.cpp` | CLI: parse args, load .npy, call SimValidator |
| `tools/sim-validator/CMakeLists.txt` | Build rules for sim-validator executable |

---

## Task 1: Types header

**Files:**
- Create: `include/Runtime/Types.h`

- [ ] **Step 1: Write `include/Runtime/Types.h`**

```cpp
// include/Runtime/Types.h
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlir::runtime {

enum class DType { F16, F32, INT32 };

inline size_t dtypeBytes(DType d) {
  switch (d) {
    case DType::F16:   return 2;
    case DType::F32:   return 4;
    case DType::INT32: return 4;
  }
  return 0;
}

struct NDArray {
  void*                data   = nullptr;
  std::vector<int64_t> shape;
  DType                dtype  = DType::F16;

  size_t numElements() const {
    size_t n = 1;
    for (auto s : shape) n *= static_cast<size_t>(s);
    return n;
  }
  size_t nbytes() const { return numElements() * dtypeBytes(dtype); }
};

struct RunArgs {
  std::vector<NDArray> inputs;
  std::vector<NDArray> outputs;      // pre-allocated, filled after Run()
  std::vector<uint8_t> tiling;       // packed TilingData bytes (little-endian)
  int                  block_dim     = 1;
  size_t               workspace_size = 8192;
};

} // namespace mlir::runtime
```

- [ ] **Step 2: Commit**

```bash
git add include/Runtime/Types.h
git commit -m "feat(runtime): add NDArray/RunArgs types header"
```

---

## Task 2: Compiler class

**Files:**
- Create: `include/Runtime/Compiler.h`
- Create: `lib/Runtime/Compiler.cpp`
- Create: `lib/Runtime/CMakeLists.txt` (stub, extended in later tasks)

Background: `python/runtime/compiler.py` runs (see `_get_common_options()` and `compile()`/`link()`):
1. `bisheng -c -x cce -O3 src.cpp --cce-aicore-arch=dav-c220-vec --cce-aicore-only -std=c++17 --cce-disable-kernel-global-attr-check -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false -I tikcpp/tikcfw -I tikcpp/tikcfw/impl -I tikcpp/tikcfw/interface -DASCENDC_DUMP=0 -D__NPU_TILING__ -DTILING_KEY_VAR=0 -o kernel.o`
2. `ld.lld -m aicorelinux -Ttext=0 kernel.o -static -o kernel.bin`

Note the compile command uses `src.cpp` (filename only, with `cwd=src_file.parent`) not the full path.

- [ ] **Step 1: Write `include/Runtime/Compiler.h`**

```cpp
// include/Runtime/Compiler.h
#pragma once
#include "llvm/Support/Error.h"
#include <string>

namespace mlir::runtime {

class Compiler {
public:
  struct Config {
    std::string soc_version = "Ascend910B1";
    std::string arch        = "dav-c220-vec";
    int         opt_level   = 3;
    bool        verbose     = false;
  };

  explicit Compiler(const Config& cfg = {});

  // Compiles src_file → output_dir/kernel_name.bin; returns binary path.
  llvm::Expected<std::string> Compile(const std::string& src_file,
                                      const std::string& output_dir,
                                      const std::string& kernel_name);

private:
  Config cfg_;

  // Run a subprocess synchronously; return non-success Error if exit code != 0.
  llvm::Error RunProcess(const std::vector<std::string>& args,
                         const std::string& cwd = "");
};

} // namespace mlir::runtime
```

- [ ] **Step 2: Write `lib/Runtime/Compiler.cpp`**

```cpp
// lib/Runtime/Compiler.cpp
#include "Runtime/Compiler.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

namespace mlir::runtime {

Compiler::Compiler(const Config& cfg) : cfg_(cfg) {}

static std::string getAscendHome() {
  const char* home = std::getenv("ASCEND_HOME_PATH");
  return home ? home : "/usr/local/Ascend/ascend-toolkit/latest";
}

llvm::Error Compiler::RunProcess(const std::vector<std::string>& args,
                                  const std::string& cwd) {
  std::vector<llvm::StringRef> argv;
  argv.reserve(args.size());
  for (auto& a : args) argv.push_back(a);

  std::string err_msg;
  // redirects: nullopt = inherit parent's stdout/stderr
  std::optional<llvm::StringRef> redirects[3];

  // Set cwd by chdir if needed (not supported by ExecuteAndWait directly)
  // We use environment variable approach via PATH-style workaround, or
  // just use absolute paths — the compiler cwd issue is handled by caller.
  int ret = llvm::sys::ExecuteAndWait(argv[0], argv,
                                      /*env=*/std::nullopt,
                                      redirects,
                                      /*secondsToWait=*/300,
                                      /*memoryLimit=*/0,
                                      &err_msg);
  if (ret != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Process failed (exit %d): %s", ret, err_msg.c_str());
  return llvm::Error::success();
}

llvm::Expected<std::string> Compiler::Compile(const std::string& src_file,
                                               const std::string& output_dir,
                                               const std::string& kernel_name) {
  ::setenv("SOC_VERSION", cfg_.soc_version.c_str(), 1);

  std::string ascend_home = getAscendHome();
  std::string bisheng     = ascend_home + "/compiler/ccec_compiler/bin/bisheng";
  std::string lld         = ascend_home + "/compiler/ccec_compiler/bin/ld.lld";
  std::string tikcpp      = ascend_home + "/compiler/tikcpp";

  std::string obj_file = output_dir + "/" + kernel_name + ".o";
  std::string bin_file = output_dir + "/" + kernel_name + ".bin";

  // Get src directory and filename (bisheng requires cwd=src_dir, filename only)
  llvm::SmallString<256> src_path(src_file);
  llvm::sys::fs::make_absolute(src_path);
  std::string src_dir  = llvm::sys::path::parent_path(src_path).str();
  std::string src_name = llvm::sys::path::filename(src_path).str();

  // Step 1: compile to .o
  // IMPORTANT: bisheng must be invoked with cwd=src_dir and src_name (not full path)
  // We achieve this by wrapping in a shell command.
  std::string compile_cmd =
      bisheng +
      " -c -x cce"
      " -O" + std::to_string(cfg_.opt_level) +
      " " + src_name +
      " --cce-aicore-arch=" + cfg_.arch +
      " --cce-aicore-only"
      " -std=c++17"
      " --cce-disable-kernel-global-attr-check"
      " -mllvm -cce-aicore-stack-size=0x8000"
      " -mllvm -cce-aicore-function-stack-size=0x8000"
      " -mllvm -cce-aicore-dcci-insert-for-scalar=false"
      " -I " + tikcpp + "/tikcfw"
      " -I " + tikcpp + "/tikcfw/impl"
      " -I " + tikcpp + "/tikcfw/interface"
      " -DASCENDC_DUMP=0"
      " -D__NPU_TILING__"
      " -DTILING_KEY_VAR=0"
      " -o " + obj_file;

  // Run via sh -c so we can set the cwd easily
  std::string wrapped = "cd " + src_dir + " && " + compile_cmd;
  std::vector<std::string> sh_compile = {"/bin/sh", "-c", wrapped};
  if (auto err = RunProcess(sh_compile)) return std::move(err);

  // Step 2: link to .bin
  std::string link_cmd =
      lld + " -m aicorelinux -Ttext=0 " + obj_file + " -static -o " + bin_file;
  std::vector<std::string> sh_link = {"/bin/sh", "-c", link_cmd};
  if (auto err = RunProcess(sh_link)) return std::move(err);

  return bin_file;
}

} // namespace mlir::runtime
```

- [ ] **Step 3: Create stub `lib/Runtime/CMakeLists.txt`**

```cmake
# lib/Runtime/CMakeLists.txt
add_mlir_library(AscendCRuntime
  Compiler.cpp
  Executor.cpp
  SimValidator.cpp
  NpyIO.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Runtime

  LINK_LIBS PUBLIC
  LLVMSupport
)
```

Create empty placeholder files:
```bash
touch lib/Runtime/Executor.cpp
touch lib/Runtime/SimValidator.cpp
touch lib/Runtime/NpyIO.cpp
```

- [ ] **Step 4: Wire into parent CMakeLists**

In `lib/CMakeLists.txt`, add:
```cmake
add_subdirectory(Runtime)
```

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Compiler.h lib/Runtime/Compiler.cpp \
        lib/Runtime/CMakeLists.txt lib/Runtime/Executor.cpp \
        lib/Runtime/SimValidator.cpp lib/Runtime/NpyIO.cpp
git add lib/CMakeLists.txt
git commit -m "feat(runtime): add Compiler class (bisheng + ld.lld wrapper)"
```

---

## Task 3: NpyIO — read/write .npy files

**Files:**
- Create: `include/Runtime/NpyIO.h`
- Modify: `lib/Runtime/NpyIO.cpp`

Background: `.npy` format: magic `\x93NUMPY` (6 bytes) + major/minor version (2 bytes) + header_len (2 bytes LE for v1, 4 bytes LE for v2) + ASCII dict. Dtype: `'<f2'`=float16, `'<f4'`=float32, `'<i4'`=int32. Data immediately follows header.

- [ ] **Step 1: Write `include/Runtime/NpyIO.h`**

```cpp
// include/Runtime/NpyIO.h
#pragma once
#include "Runtime/Types.h"
#include "llvm/Support/Error.h"
#include <string>

namespace mlir::runtime {

// Load a .npy file. Allocates NDArray::data with new uint8_t[].
// Caller must delete[].
llvm::Expected<NDArray> LoadNpy(const std::string& path);

// Save NDArray to .npy file.
llvm::Error SaveNpy(const std::string& path, const NDArray& arr);

} // namespace mlir::runtime
```

- [ ] **Step 2: Write `lib/Runtime/NpyIO.cpp`**

```cpp
// lib/Runtime/NpyIO.cpp
#include "Runtime/NpyIO.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

namespace mlir::runtime {

llvm::Expected<NDArray> LoadNpy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot open: %s", path.c_str());
  char magic[6];
  f.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Not a .npy file: %s", path.c_str());

  uint8_t major, minor;
  f.read(reinterpret_cast<char*>(&major), 1);
  f.read(reinterpret_cast<char*>(&minor), 1);

  uint32_t hlen = 0;
  if (major == 1) {
    uint16_t h16;
    f.read(reinterpret_cast<char*>(&h16), 2);
    hlen = h16;
  } else {
    f.read(reinterpret_cast<char*>(&hlen), 4);
  }

  std::string header(hlen, '\0');
  f.read(header.data(), hlen);

  NDArray arr;

  // Parse shape
  {
    std::regex re(R"('shape'\s*:\s*\(([^)]*)\))");
    std::smatch m;
    if (!std::regex_search(header, m, re))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Cannot parse shape: %s", path.c_str());
    std::istringstream ss(m[1].str());
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      tok.erase(0, tok.find_first_not_of(" \t"));
      tok.erase(tok.find_last_not_of(" \t,") + 1);
      if (!tok.empty()) arr.shape.push_back(std::stoll(tok));
    }
  }

  // Parse dtype
  {
    std::regex re(R"('descr'\s*:\s*'([^']+)')");
    std::smatch m;
    if (!std::regex_search(header, m, re))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Cannot parse dtype: %s", path.c_str());
    std::string descr = m[1].str();
    if (descr == "<f2" || descr == "=f2") arr.dtype = DType::F16;
    else if (descr == "<f4" || descr == "=f4") arr.dtype = DType::F32;
    else if (descr == "<i4" || descr == "=i4") arr.dtype = DType::INT32;
    else return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                        "Unsupported dtype '%s': %s",
                                        descr.c_str(), path.c_str());
  }

  size_t nbytes = arr.nbytes();
  arr.data = new uint8_t[nbytes];
  f.read(reinterpret_cast<char*>(arr.data),
         static_cast<std::streamsize>(nbytes));
  return arr;
}

llvm::Error SaveNpy(const std::string& path, const NDArray& arr) {
  std::ofstream f(path, std::ios::binary);
  if (!f)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot write: %s", path.c_str());

  const char* descr = nullptr;
  switch (arr.dtype) {
    case DType::F16:   descr = "<f2"; break;
    case DType::F32:   descr = "<f4"; break;
    case DType::INT32: descr = "<i4"; break;
  }
  std::string shape_str = "(";
  for (size_t i = 0; i < arr.shape.size(); ++i) {
    shape_str += std::to_string(arr.shape[i]);
    if (arr.shape.size() == 1 || i + 1 < arr.shape.size()) shape_str += ",";
  }
  shape_str += ")";

  std::string dict = "{'descr': '" + std::string(descr) +
                     "', 'fortran_order': False, 'shape': " + shape_str + ", }";
  // Pad to multiple of 64 after 10-byte prefix (magic + ver + hlen)
  size_t dict_len = dict.size() + 1; // +1 for '\n'
  size_t total    = 10 + dict_len;
  size_t pad      = (64 - total % 64) % 64;
  dict.append(pad, ' ');
  dict += '\n';

  uint16_t hlen = static_cast<uint16_t>(dict.size());
  f.write("\x93NUMPY", 6);
  f.put(1); f.put(0);
  f.write(reinterpret_cast<char*>(&hlen), 2);
  f.write(dict.data(), dict.size());
  f.write(reinterpret_cast<const char*>(arr.data), arr.nbytes());
  return llvm::Error::success();
}

} // namespace mlir::runtime
```

- [ ] **Step 3: Commit**

```bash
git add include/Runtime/NpyIO.h lib/Runtime/NpyIO.cpp
git commit -m "feat(runtime): add .npy read/write (f16/f32/i32, v1+v2)"
```

---

## Task 4: Executor class

**Files:**
- Create: `include/Runtime/Executor.h`
- Modify: `lib/Runtime/Executor.cpp`

Background — exact Python flow from `executor.py`:
1. `rtSetDevice(device_id)` — ignore return value in simulation
2. `rtDevBinaryRegister(DevBinary*, &handle)` — register ELF bytes; `DevBinary` = `{magic:u32, version:u32, data:char*, length:u64}`; magic for vec = `0x41415246`
3. `rtFunctionRegister(bin_handle, name_void_ptr, name_char_ptr, name_void_ptr, mode=0)` — register function by name; returns function stub pointer (cast name ptr as stub)
4. `rtMalloc(void**, size+512, uint32_t=0, uint16_t=33)` → align returned ptr to 512-byte boundary, save original for rtFree
5. `rtMemcpy(dst, size, src, size, kind)` — H2D kind=1, D2H kind=2; H2D in 256-byte chunks; D2H in 4-byte chunks
6. Build args: `[input_addrs..., output_addrs..., workspace_addr, tiling_word_0, ..., tiling_word_n]` (each uint64)
7. `rtKernelLaunch(func_handle, block_dim, args_ptr, args_size_bytes, nullptr, stream)`
8. `rtDeviceSynchronize()`
9. D2H copy outputs; `rtFree` all allocations

- [ ] **Step 1: Write `include/Runtime/Executor.h`**

```cpp
// include/Runtime/Executor.h
#pragma once
#include "Runtime/Types.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>
#include <vector>

namespace mlir::runtime {

enum class BackendMode {
  Simulation,  // libruntime_camodel.so (CPU simulation)
  RealDevice,  // reserved, not implemented
};

// Mirrors Python DevBinary struct (field order MUST match runtime ABI)
struct DevBinary {
  uint32_t magic;    // 0x41415246 for vec kernel
  uint32_t version;  // 0
  const char* data;  // ELF bytes pointer
  uint64_t length;   // ELF byte count
};

class Executor {
public:
  static constexpr uint32_t MAGIC_ELF_AIVEC  = 0x41415246u;
  static constexpr uint32_t MAGIC_ELF_AICUBE = 0x41494343u;

  explicit Executor(BackendMode mode = BackendMode::Simulation);
  ~Executor();

  llvm::Error Initialize(int device_id = 0);

  // binary_data: raw ELF bytes from .bin file
  llvm::Error Run(const std::vector<uint8_t>& binary_data,
                  const std::string& function_name,
                  RunArgs& args,
                  uint32_t magic = MAGIC_ELF_AIVEC);

  // Convenience: read binary from file, then call Run
  llvm::Error RunFile(const std::string& binary_path,
                      const std::string& function_name,
                      RunArgs& args,
                      uint32_t magic = MAGIC_ELF_AIVEC);

private:
  BackendMode mode_;
  void*       lib_handle_ = nullptr;

  // Runtime API function pointers (exact signatures from Python executor.py)
  int (*rtSetDevice_)(int32_t)                                          = nullptr;
  int (*rtDevBinaryRegister_)(const DevBinary*, void**)                 = nullptr;
  int (*rtFunctionRegister_)(void*, void*, const char*, void*, uint32_t)= nullptr;
  int (*rtMalloc_)(void**, uint64_t, uint32_t, uint16_t)                = nullptr;
  int (*rtFree_)(void*)                                                  = nullptr;
  int (*rtMemcpy_)(void*, uint64_t, const void*, uint64_t, uint32_t)    = nullptr;
  int (*rtStreamCreate_)(void**, int32_t)                                = nullptr;
  int (*rtStreamDestroy_)(void*)                                         = nullptr;
  int (*rtKernelLaunch_)(void*, uint32_t, void*, uint32_t, void*, void*)= nullptr;
  int (*rtDeviceSynchronize_)()                                          = nullptr;

  llvm::Error LoadLib();

  // Alloc: rtMalloc(size+512, 0, 33) → align to 512-byte boundary
  // Returns aligned ptr; saves raw ptr internally for rtFree
  struct AllocInfo { void* raw; void* aligned; };
  std::vector<AllocInfo> alloc_map_;

  llvm::Expected<void*> Alloc(size_t nbytes);
  void FreeAll();

  // H2D in 256-byte chunks (kind=1); D2H in 4-byte chunks (kind=2)
  // D2H safety: rtMalloc +512 overalloc ensures 4-byte chunk read never OOBs
  llvm::Error H2D(void* dst, const void* src, size_t n);
  llvm::Error D2H(void* dst, const void* src, size_t n);
};

} // namespace mlir::runtime
```

- [ ] **Step 2: Write `lib/Runtime/Executor.cpp`**

```cpp
// lib/Runtime/Executor.cpp
#include "Runtime/Executor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
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
  return std::string(home) + "/runtime/lib64/libruntime_camodel.so";
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
                                     "rtMemcpy H2D failed: rc=%d at offset %zu", rc, off);
  }
  return llvm::Error::success();
}

llvm::Error Executor::D2H(void* dst, const void* src, size_t n) {
  // 4-byte chunks; safe because rtMalloc adds +512 overalloc
  uint8_t* d = static_cast<uint8_t*>(dst);
  const uint8_t* s = static_cast<const uint8_t*>(src);
  for (size_t off = 0; off < n; off += 4) {
    uint8_t buf[4] = {};
    size_t chunk = std::min<size_t>(4, n - off);
    int rc = rtMemcpy_(buf, 4, s + off, 4, /*D2H=*/2);
    if (rc != 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "rtMemcpy D2H failed: rc=%d at offset %zu", rc, off);
    std::memcpy(d + off, buf, chunk); // only copy valid bytes to dst
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

  // 2. Register function (name pointer used as stub, same as Python)
  const char* fn_name = function_name.c_str();
  void*       fn_name_void = const_cast<char*>(fn_name);
  rc = rtFunctionRegister_(bin_handle, fn_name_void, fn_name, fn_name_void, 0);
  if (rc != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtFunctionRegister failed: rc=%d", rc);
  void* func_handle = fn_name_void; // handle = name pointer

  // 3. Alloc + H2D copy inputs
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
                        nullptr,  // smDesc
                        stream);
  if (rc != 0) {
    rtStreamDestroy_(stream);
    FreeAll();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "rtKernelLaunch failed: rc=%d", rc);
  }

  // 8. Sync
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
```

- [ ] **Step 3: Commit**

```bash
git add include/Runtime/Executor.h lib/Runtime/Executor.cpp
git commit -m "feat(runtime): add Executor (DevBinary+rtKernelLaunch wrapper)"
```

---

## Task 5: SimValidator class

**Files:**
- Create: `include/Runtime/SimValidator.h`
- Modify: `lib/Runtime/SimValidator.cpp`

- [ ] **Step 1: Write `include/Runtime/SimValidator.h`**

```cpp
// include/Runtime/SimValidator.h
#pragma once
#include "Runtime/Compiler.h"
#include "Runtime/Executor.h"
#include "Runtime/Types.h"
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace mlir::runtime {

class SimValidator {
public:
  struct Result {
    bool        passed       = false;
    double      max_abs_diff  = 0.0;
    double      mean_abs_diff = 0.0;
    std::string error_msg;
  };

  Result Validate(const std::string&          kernel_src,
                  const std::string&          kernel_name,
                  RunArgs&                    args,
                  const std::vector<NDArray>& expected,
                  double                      atol = 1.0,
                  double                      rtol = 1e-2,
                  const Compiler::Config&     compiler_cfg = {});
};

} // namespace mlir::runtime
```

- [ ] **Step 2: Write `lib/Runtime/SimValidator.cpp`**

```cpp
// lib/Runtime/SimValidator.cpp
#include "Runtime/SimValidator.h"
#include "llvm/Support/FileSystem.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace mlir::runtime {

static float toFloat(const void* base, size_t idx, DType dtype) {
  switch (dtype) {
    case DType::F16: {
      uint16_t h;
      std::memcpy(&h, static_cast<const uint16_t*>(base) + idx, 2);
      uint32_t sign = (h >> 15) & 1;
      uint32_t exp  = (h >> 10) & 0x1f;
      uint32_t frac = h & 0x3ff;
      uint32_t f;
      if (exp == 0)      f = (sign << 31) | (frac << 13);
      else if (exp == 31) f = (sign << 31) | 0x7f800000u | (frac << 13);
      else                f = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
      float result; std::memcpy(&result, &f, 4); return result;
    }
    case DType::F32: return static_cast<const float*>(base)[idx];
    case DType::INT32: return static_cast<float>(static_cast<const int32_t*>(base)[idx]);
  }
  return 0.f;
}

SimValidator::Result SimValidator::Validate(
    const std::string& kernel_src,
    const std::string& kernel_name,
    RunArgs& args,
    const std::vector<NDArray>& expected,
    double atol, double rtol,
    const Compiler::Config& compiler_cfg) {

  Result r;

  // Temp build dir
  llvm::SmallString<256> build_dir;
  if (llvm::sys::fs::createUniqueDirectory("sim_validator_build", build_dir)) {
    r.error_msg = "Cannot create temp dir"; return r;
  }

  // Compile
  Compiler compiler(compiler_cfg);
  auto bin_or = compiler.Compile(kernel_src, build_dir.str().str(), kernel_name);
  if (!bin_or) {
    r.error_msg = "Compile failed: " + llvm::toString(bin_or.takeError());
    return r;
  }

  // Execute
  Executor executor;
  if (auto err = executor.Initialize()) {
    r.error_msg = "Init failed: " + llvm::toString(std::move(err));
    return r;
  }
  if (auto err = executor.RunFile(*bin_or, kernel_name, args)) {
    r.error_msg = "Run failed: " + llvm::toString(std::move(err));
    return r;
  }

  // Compare
  assert(args.outputs.size() == expected.size());
  double sum_diff = 0.0, max_diff = 0.0;
  size_t total = 0;
  bool all_close = true;

  for (size_t oi = 0; oi < args.outputs.size(); ++oi) {
    const NDArray& act = args.outputs[oi];
    const NDArray& exp = expected[oi];
    size_t n = act.numElements();
    total += n;
    for (size_t i = 0; i < n; ++i) {
      double a = toFloat(act.data, i, act.dtype);
      double e = toFloat(exp.data, i, exp.dtype);
      double diff = std::abs(a - e);
      sum_diff += diff;
      if (diff > max_diff) max_diff = diff;
      if (diff > atol + rtol * std::abs(e)) all_close = false;
    }
  }

  r.max_abs_diff  = max_diff;
  r.mean_abs_diff = total > 0 ? sum_diff / total : 0.0;
  r.passed        = all_close;
  return r;
}

} // namespace mlir::runtime
```

- [ ] **Step 3: Commit**

```bash
git add include/Runtime/SimValidator.h lib/Runtime/SimValidator.cpp
git commit -m "feat(runtime): add SimValidator (compile+execute+compare)"
```

---

## Task 6: sim-validator CLI tool

**Files:**
- Create: `tools/sim-validator/sim_validator_main.cpp`
- Create: `tools/sim-validator/CMakeLists.txt`
- Modify: `tools/CMakeLists.txt`

- [ ] **Step 1: Write `tools/sim-validator/CMakeLists.txt`**

```cmake
# tools/sim-validator/CMakeLists.txt
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(sim-validator
  sim_validator_main.cpp
)

target_link_libraries(sim-validator PRIVATE
  AscendCRuntime
)

target_include_directories(sim-validator PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
```

- [ ] **Step 2: Write `tools/sim-validator/sim_validator_main.cpp`**

```cpp
// tools/sim-validator/sim_validator_main.cpp
#include "Runtime/NpyIO.h"
#include "Runtime/SimValidator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <fstream>
#include <sstream>

using namespace mlir::runtime;
using namespace llvm;

static cl::opt<std::string> KernelSrc("kernel",       cl::desc("Kernel .cpp source file"), cl::Required);
static cl::opt<std::string> KernelName("name",         cl::desc("Kernel function name"), cl::Required);
static cl::opt<std::string> TilingParams("tiling-params",
    cl::desc("Comma-separated KEY=VALUE pairs: TB_M=16,TB_N=4,..."), cl::init(""));
static cl::opt<std::string> TilingLayout("tiling-layout",
    cl::desc("Comma-separated types for --tiling-params: int64,int64,..."), cl::init(""));
static cl::opt<std::string> TilingBin("tiling",        cl::desc("Pre-packed tiling .bin file"), cl::init(""));
static cl::opt<std::string> Inputs("inputs",           cl::desc("Comma-separated input .npy files"), cl::Required);
static cl::opt<std::string> Expected("expected",       cl::desc("Expected output .npy file"), cl::Required);
static cl::opt<int>         BlockDim("block-dim",      cl::desc("Number of AiCore blocks"), cl::init(1));
static cl::opt<std::string> SocVersion("soc",          cl::desc("SoC version (default: Ascend910B1)"), cl::init("Ascend910B1"));

static std::vector<std::string> splitComma(const std::string& s) {
  std::vector<std::string> parts;
  std::istringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) parts.push_back(tok);
  return parts;
}

static std::vector<uint8_t> buildTiling(const std::string& params,
                                         const std::string& layout) {
  std::vector<uint8_t> bytes;
  auto pvec = splitComma(params);
  auto lvec = splitComma(layout);
  for (size_t i = 0; i < pvec.size(); ++i) {
    auto eq = pvec[i].find('=');
    int64_t val = std::stoll(pvec[i].substr(eq + 1));
    std::string type = i < lvec.size() ? lvec[i] : "int64";
    if (type == "int64" || type == "int64_t") {
      uint8_t buf[8]; std::memcpy(buf, &val, 8);
      bytes.insert(bytes.end(), buf, buf + 8);
    } else if (type == "int32" || type == "int32_t") {
      int32_t v = static_cast<int32_t>(val);
      uint8_t buf[4]; std::memcpy(buf, &v, 4);
      bytes.insert(bytes.end(), buf, buf + 4);
    }
  }
  return bytes;
}

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv, "AscendC Kernel SimValidator\n");

  // Build tiling bytes
  std::vector<uint8_t> tiling;
  if (!TilingBin.empty()) {
    std::ifstream f(TilingBin.getValue(), std::ios::binary);
    if (!f) { llvm::errs() << "Error: cannot open " << TilingBin << "\n"; return 1; }
    tiling.assign(std::istreambuf_iterator<char>(f), {});
  } else if (!TilingParams.empty()) {
    tiling = buildTiling(TilingParams, TilingLayout);
  } else {
    llvm::errs() << "Error: provide --tiling or --tiling-params\n"; return 1;
  }

  // Load inputs
  RunArgs args;
  args.tiling    = tiling;
  args.block_dim = BlockDim;

  for (auto& path : splitComma(Inputs)) {
    auto arr_or = LoadNpy(path);
    if (!arr_or) {
      llvm::errs() << "Error loading input " << path << ": "
                   << llvm::toString(arr_or.takeError()) << "\n";
      return 1;
    }
    args.inputs.push_back(*arr_or);
  }

  // Load expected
  auto exp_or = LoadNpy(Expected);
  if (!exp_or) {
    llvm::errs() << "Error loading expected: " << llvm::toString(exp_or.takeError()) << "\n";
    return 1;
  }
  std::vector<NDArray> expected_arrs = {*exp_or};

  // Pre-alloc output (same shape/dtype as expected)
  NDArray out_buf;
  out_buf.shape = expected_arrs[0].shape;
  out_buf.dtype = expected_arrs[0].dtype;
  out_buf.data  = new uint8_t[out_buf.nbytes()]();
  args.outputs  = {out_buf};

  Compiler::Config cc;
  cc.soc_version = SocVersion;

  SimValidator validator;
  auto result = validator.Validate(KernelSrc, KernelName, args, expected_arrs,
                                   /*atol=*/1.0, /*rtol=*/1e-2, cc);

  // Free allocations
  delete[] static_cast<uint8_t*>(out_buf.data);
  for (auto& inp : args.inputs) delete[] static_cast<uint8_t*>(inp.data);
  delete[] static_cast<uint8_t*>(expected_arrs[0].data);

  llvm::outs() << "max_abs_diff:  " << result.max_abs_diff  << "\n"
               << "mean_abs_diff: " << result.mean_abs_diff << "\n";
  if (!result.error_msg.empty())
    llvm::errs() << "Error: " << result.error_msg << "\n";

  if (result.passed) { llvm::outs() << "PASS\n"; return 0; }
  else               { llvm::outs() << "FAIL\n"; return 1; }
}
```

- [ ] **Step 3: Add to `tools/CMakeLists.txt`**

```cmake
add_subdirectory(sim-validator)
```

- [ ] **Step 4: Commit**

```bash
git add tools/sim-validator/
git add tools/CMakeLists.txt
git commit -m "feat(runtime): add sim-validator CLI tool"
```

---

## Task 7: Build and integration test

- [ ] **Step 1: Build (in xvm container)**

```bash
ssh xvm@orb
cd /home/niu/code/Ascend-MLIR
cd sim
# Wait 1 second for file sync after any local edits
../scripts/build.sh --build-project --llvm-build-dir ~/code/llvm-project/build
```

Expected: clean build. Common fix: if `llvm::MemoryBuffer` not found, add `#include "llvm/Support/MemoryBuffer.h"`.

- [ ] **Step 2: Generate test inputs**

```bash
python3 - << 'EOF'
import numpy as np
M, N = 32, 32
np.random.seed(42)
a = np.random.rand(M).astype(np.float16)
b = np.random.rand(M, N).astype(np.float16)
exp = (a.astype(np.float32)[:, None] + b.astype(np.float32)).sum(axis=1).astype(np.float16)
np.save('/tmp/input_a.npy', a)
np.save('/tmp/input_b.npy', b)
np.save('/tmp/expected.npy', exp)
print("Saved to /tmp/")
EOF
```

- [ ] **Step 3: Run sim-validator**

```bash
source ../python/test/env.sh
source ../examples/env.sh
sim-validator \
  --kernel   ../examples/broadcast-add-reduce/step8_kernel-adjust.cpp \
  --name     broadcast_add_reducesum \
  --tiling-params "TB_M=16,TB_N=4,dim_arg0_0=32,dim_arg1_1=32" \
  --tiling-layout "int64,int64,int64,int64" \
  --inputs   /tmp/input_a.npy,/tmp/input_b.npy \
  --expected /tmp/expected.npy \
  --block-dim 2 \
  --soc Ascend910B1
```

Expected:
```
max_abs_diff:  <value < 1.0>
mean_abs_diff: <small value>
PASS
```

- [ ] **Step 4: Add tiling_space.json (needed by AutoTuner plan)**

```bash
cat > examples/broadcast-add-reduce/tiling_space.json << 'EOF'
{
  "params": [
    {"name": "TB_M", "min": 16, "max": 128, "step": 16, "alignment": 16},
    {"name": "TB_N", "min": 4,  "max": 32,  "step": 4,  "alignment": 4}
  ],
  "constraints": [
    "TB_N <= 32"
  ]
}
EOF
git add examples/broadcast-add-reduce/tiling_space.json
#git commit -m "chore: add tiling_space.json for broadcast-add-reduce"
```

---

## Acceptance Criteria

- `sim-validator` builds and `--help` runs without error
- Running `sim-validator` on `broadcast-add-reduce/step8_kernel.cpp` with M=32, N=32 produces `PASS`
- `max_abs_diff < 1.0` (f16 tolerance)
- Results match `python3 examples/broadcast-add-reduce/test_e2e.py`

---

## 实现状态与使用方法（2026-03-19 更新）

> 计划已全部实现，并在实现过程中做了若干架构调整，记录如下。

### 架构调整说明

| 计划设计 | 实际实现 | 原因 |
|---------|---------|------|
| `Executor::Run()` 每次调用注册 binary + function | `RegisterBinary()` / `RunWithHandle()` 分离 | 模拟器禁止对同一 stub 指针重复注册（rc=507000）；编译一次、注册一次、多 config 复用 handle |
| 每次 `Run()` 创建/销毁 stream | `Initialize()` 创建持久化 stream，`~Executor()` 销毁 | 模拟器在第 2+ 次 launch 时复用 stream ID 会丢失完成状态导致挂起 |
| `SimValidator::Validate()` 内部自行编译+运行 | 新增 `SimValidator::ValidateBinary(func_handle, executor, args, expected, atol, rtol)` | autotuner 需要在搜索循环外编译一次，循环内只换 tiling args 反复跑 |
| pipeline-analyzer 独立工具 | 删除，改为 autotuner `--sim-report` 选项调用 msopgen | `msopgen sim` 是 CANN 内置工具，功能完全覆盖；避免重复实现 |

### sim-validator 使用

```bash
# 在 xvm 容器内
source /home/niu/code/Ascend-MLIR/examples/env.sh
source /home/niu/code/Ascend-MLIR/python/test/env.sh   # 导出 libruntime_camodel.so 路径

sim-validator \
  --kernel   examples/broadcast-add-reduce/step8_kernel-adjust.cpp \
  --name     broadcast_add_reducesum \
  --tiling-params "TB_M=16,TB_N=4,dim_arg0_0=64,dim_arg1_1=64" \
  --tiling-layout "int64,int64,int64,int64" \
  --inputs   examples/broadcast-add-reduce/input_a.npy,examples/broadcast-add-reduce/input_b.npy \
  --expected examples/broadcast-add-reduce/output_c.npy \
  --block-dim 1 \
  --soc Ascend910B1
```

### autotuner 使用

```bash
source /home/niu/code/Ascend-MLIR/examples/env.sh
source /home/niu/code/Ascend-MLIR/python/test/env.sh

# 基本用法：全空间搜索，输出最优 tiling_func.cpp
autotuner \
  --space   examples/broadcast-add-reduce/tiling_space.json \
  --shape   "M=64,N=64" \
  --inputs  examples/broadcast-add-reduce/input_a.npy,examples/broadcast-add-reduce/input_b.npy \
  --expected examples/broadcast-add-reduce/output_c.npy \
  --output  tiling_func.cpp

# 附带流水图分析（需要先跑过一次，simulator dump 文件在 cwd）
autotuner \
  --space   examples/broadcast-add-reduce/tiling_space.json \
  --shape   "M=64,N=64" \
  --inputs  examples/broadcast-add-reduce/input_a.npy,examples/broadcast-add-reduce/input_b.npy \
  --expected examples/broadcast-add-reduce/output_c.npy \
  --sim-report trace \
  --sim-report-out ./sim_report

# 同时生成流水图 + 代码行热点 CSV（需要 .o 文件与 .bin 同目录）
autotuner ... --sim-report trace,codeline --sim-report-out ./sim_report
```

#### --sim-report 选项说明

| 值 | 输出文件 | 工具 | 用途 |
|----|---------|------|------|
| `trace` | `dump2trace_core*.json` | msopgen sim | chrome://tracing 查看 PIPE 流水，识别 MTE2/VEC/Cube 瓶颈 |
| `codeline` | `code_exe_prof.csv`<br>`instr_exe_prof.csv` | msopgen sim -reloc | 源码行级别热点分析；需 .o 文件（自动从 .bin 路径推导） |

组合示例：`--sim-report trace,codeline`

msopgen 路径优先级：`--msopgen` > 环境变量 `ASCEND_HOME_PATH/tools/msopgen` > `/usr/local/Ascend/ascend-toolkit/latest/tools/msopgen`

### tiling_space.json 格式

```json
{
  "kernel": "broadcast_add_reducesum",
  "kernel_file": "step8_kernel-adjust.cpp",
  "soc": "Ascend910B1",
  "block_dim_expr": "ceil(M/TB_M)",
  "tiling_params": [
    {"name": "TB_M",       "min": 16, "max": 64, "step": 16},
    {"name": "TB_N",       "min": 4,  "max": 64, "step": 4},
    {"name": "dim_arg0_0", "fixed": true, "shape_key": "M"},
    {"name": "dim_arg1_1", "fixed": true, "shape_key": "N"}
  ]
}
```

字段说明：
- `fixed: true` + `shape_key`：从 `--shape` 取值，不搜索
- `values: [...]`：枚举候选值（优先于 min/max/step）
- `block_dim_expr`：支持 `ceil(X/Y)` 和 `X/Y`，变量从 `--shape` 和搜索变量取值

### 环境依赖

```bash
# xvm 容器内需要的环境变量
ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest   # CANN 工具包路径
# env.sh 会设置 LD_LIBRARY_PATH 包含 libruntime_camodel.so
# python/test/env.sh 会设置模拟器 .so 路径
ASCEND_CPU_SIMULATION=1   # 启用 CPU 仿真模式（自动由 env.sh 设置）
```
