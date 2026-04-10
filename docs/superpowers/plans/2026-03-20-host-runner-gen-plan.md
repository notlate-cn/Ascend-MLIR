# HostRunnerGen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `HostRunnerGen` to the Runtime library — generates a Host-side `runner.cpp` and compiles it into a standalone executable that `msprof op [simulator]` can wrap to collect performance data.

**Architecture:** `HostRunnerGen::Generate(cfg, output_dir)` writes a self-contained `runner.cpp` into `output_dir`, then compiles it with `g++` (Host compiler, not bisheng). The runner accepts `--bin`, `--tiling-params`, `--tiling-layout`, `--inputs`, `--output`, `--block-dim` at runtime so it can be moved independently of the `.bin` file. The emitted code inlines all logic (no project headers) and tracks raw `rtMalloc` pointers to call `rtFree` on exit.

**Tech Stack:** C++17, LLVM Support (`llvm::Expected`, `llvm::Error`, `llvm::sys::ExecuteAndWait`), CMake `add_mlir_library`. No new dependencies. Build and test inside `xvm` Docker container via `ssh xvm@orb`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `include/Runtime/HostRunnerGen.h` | Create | `HostRunnerGen` class declaration + `Config` struct |
| `lib/Runtime/HostRunnerGen.cpp` | Create | `Generate()`: emit runner.cpp text + compile with g++ |
| `lib/Runtime/CMakeLists.txt` | Modify | Add `HostRunnerGen.cpp` to `AscendCRuntime` sources |
| `test/tools/runner/test_runner_gen.cpp` | Create | Test driver: call Generate(), verify runner compiles and validates --bin required |
| `test/tools/runner/run_runner_gen.sh` | Create | Shell wrapper: build test driver, run it, check exit code |

---

## Task 1: Header

**Files:**
- Create: `include/Runtime/HostRunnerGen.h`

- [ ] **Step 1: Write `include/Runtime/HostRunnerGen.h`**

```cpp
// include/Runtime/HostRunnerGen.h
#pragma once
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace mlir::runtime {

class HostRunnerGen {
public:
  struct Config {
    std::string kernel_name;
    std::string kernel_type;  // "vec" | "cube" | "mix"
    // kernel_type → DevBinary magic:
    //   "vec"  → 0x41415246 (MAGIC_ELF_AIVEC)
    //   "cube" → 0x41494343 (MAGIC_ELF_AICUBE)
    //   "mix"  → 0x41415246 (same as vec, conservative default)
    std::string soc_version = "Ascend910B1";
    int         num_inputs  = 1;
    int         num_outputs = 1;  // currently only 1 is supported
    std::vector<std::string> tiling_layout;  // e.g. {"int64","int64","int32"}
    bool        verbose = false;
    // Note: arch is NOT needed here. runner is a Host (x86/aarch64) executable
    // compiled with g++. Only the kernel .bin is bisheng-compiled for the NPU arch.
  };

  // Generate runner.cpp and compile it to output_dir/runner.
  // Returns path to the runner executable on success.
  // Returns error if num_outputs != 1 (not yet supported).
  // The runner accepts at runtime:
  //   --bin <kernel.bin>           (required)
  //   --tiling-params "K1=V1,..."  (optional)
  //   --tiling-layout "int64,..."  (optional, overrides compiled-in layout)
  //   --inputs a.npy,b.npy         (required)
  //   --output out.npy             (optional, default /dev/null for perf-only runs)
  //   --block-dim N                (optional, default 1)
  llvm::Expected<std::string> Generate(const Config& cfg,
                                        const std::string& output_dir);
};

} // namespace mlir::runtime
```

- [ ] **Step 2: Commit**

```bash
git add include/Runtime/HostRunnerGen.h
git commit -m "feat(runtime): add HostRunnerGen header"
```

---

## Task 2: Implementation

**Files:**
- Create: `lib/Runtime/HostRunnerGen.cpp`

The `Generate()` function does two things:
1. Emit a self-contained `runner.cpp` into `output_dir/runner.cpp`. The emitted code is a complete C++ program with all logic inline — no `#include` from this project.
2. Compile `runner.cpp` with `g++ -O2 -std=c++17 runner.cpp -ldl -o runner`.

The emitted runner.cpp:
- Parses `--bin`, `--tiling-params`, `--tiling-layout`, `--inputs`, `--output`, `--block-dim`
- Reads input `.npy` files (inline minimal reader: f16/f32/i32)
- dlopen `libruntime_camodel.so` (same pattern as `lib/Runtime/Executor.cpp`)
- Calls `rtSetDevice`, `rtDevBinaryRegister`, `rtFunctionRegister`, `rtMalloc`, `rtMemcpy`, `rtKernelLaunch`, `rtDeviceSynchronize`, then `rtFree` on all allocations
- Writes output to `--output` (skips write when path is `/dev/null`)
- Exits 0 on success, 3 on runtime error, 4 on input error

**String-splicing note:** `emitRunnerCpp()` builds the source by concatenating raw string literals with computed parts. Three splice points:
1. `layout_init` — lines of the form `  tiling_layout.push_back("int64");\n` (variable name is `tiling_layout`, matching the surrounding emitted code)
2. `magic_buf` — hex literal string e.g. `0x41415246`
3. `cfg.kernel_name` — alphanumeric function name

- [ ] **Step 1: Write `lib/Runtime/HostRunnerGen.cpp`**

```cpp
// lib/Runtime/HostRunnerGen.cpp
#include "Runtime/HostRunnerGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <fstream>

namespace mlir::runtime {

static uint32_t magicForType(const std::string& kernel_type) {
  if (kernel_type == "cube") return 0x41494343u;
  return 0x41415246u; // "vec" and "mix" both use AIVEC
}

// Returns the complete runner.cpp source as a string.
// Splice points:
//   layout_init  : push_back lines for the default tiling_layout vector
//   magic_buf    : hex magic constant for DevBinary
//   kernel_name  : function name string literal
static std::string emitRunnerCpp(const HostRunnerGen::Config& cfg) {
  // Build push_back lines. Variable name in emitted code is "tiling_layout"
  // (matches the else-branch vector declaration in the emitted main()).
  std::string layout_init;
  for (size_t i = 0; i < cfg.tiling_layout.size(); ++i)
    layout_init += "    tiling_layout.push_back(\"" + cfg.tiling_layout[i] + "\");\n";

  char magic_buf[32];
  std::snprintf(magic_buf, sizeof(magic_buf), "0x%08X",
                magicForType(cfg.kernel_type));
  // Note: the 'u' unsigned suffix is appended by the raw string splice site below.

  // clang-format off
  return std::string(R"cpp(
// Auto-generated by HostRunnerGen. DO NOT EDIT.
// Compile: g++ -O2 -std=c++17 runner.cpp -ldl -o runner
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

// ── Minimal .npy I/O ──────────────────────────────────────────────────────────

struct NDArray {
  void*                data  = nullptr;
  std::vector<int64_t> shape;
  int                  dtype = 0; // 0=f16 1=f32 2=i32
  size_t numElements() const {
    size_t n = 1;
    for (auto s : shape) n *= (size_t)s;
    return n;
  }
  size_t dtypeBytes() const { return dtype == 0 ? 2 : 4; }
  size_t nbytes()     const { return numElements() * dtypeBytes(); }
};

static bool loadNpy(const std::string& path, NDArray& arr) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::cerr << "Cannot open: " << path << "\n"; return false; }
  char magic[6]; f.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    std::cerr << "Not a .npy file: " << path << "\n"; return false;
  }
  uint8_t major, minor;
  f.read((char*)&major, 1); f.read((char*)&minor, 1);
  uint32_t hlen = 0;
  if (major == 1) { uint16_t h; f.read((char*)&h, 2); hlen = h; }
  else { f.read((char*)&hlen, 4); }
  std::string header(hlen, '\0');
  f.read(header.data(), hlen);
  std::regex re_shape(R"('shape'\s*:\s*\(([^)]*)\))");
  std::smatch m;
  if (!std::regex_search(header, m, re_shape)) {
    std::cerr << "Cannot parse shape in: " << path << "\n"; return false;
  }
  std::istringstream ss(m[1].str()); std::string tok;
  while (std::getline(ss, tok, ',')) {
    tok.erase(0, tok.find_first_not_of(" \t"));
    tok.erase(tok.find_last_not_of(" \t,") + 1);
    if (!tok.empty()) arr.shape.push_back(std::stoll(tok));
  }
  std::regex re_dtype(R"('descr'\s*:\s*'([^']+)')");
  if (!std::regex_search(header, m, re_dtype)) {
    std::cerr << "Cannot parse dtype in: " << path << "\n"; return false;
  }
  std::string descr = m[1].str();
  if      (descr=="<f2"||descr=="=f2") arr.dtype = 0;
  else if (descr=="<f4"||descr=="=f4") arr.dtype = 1;
  else if (descr=="<i4"||descr=="=i4") arr.dtype = 2;
  else { std::cerr << "Unsupported dtype: " << descr << "\n"; return false; }
  arr.data = new uint8_t[arr.nbytes()];
  f.read((char*)arr.data, (std::streamsize)arr.nbytes());
  return true;
}

static bool saveNpy(const std::string& path, const NDArray& arr) {
  if (path == "/dev/null") return true;
  std::ofstream f(path, std::ios::binary);
  if (!f) { std::cerr << "Cannot write: " << path << "\n"; return false; }
  const char* descr = arr.dtype==0 ? "<f2" : (arr.dtype==1 ? "<f4" : "<i4");
  std::string shape_str = "(";
  for (size_t i = 0; i < arr.shape.size(); ++i) {
    shape_str += std::to_string(arr.shape[i]);
    if (arr.shape.size()==1 || i+1 < arr.shape.size()) shape_str += ",";
  }
  shape_str += ")";
  std::string dict = "{'descr': '" + std::string(descr) +
                     "', 'fortran_order': False, 'shape': " + shape_str + ", }";
  size_t total = 10 + dict.size() + 1;
  size_t pad   = (64 - total % 64) % 64;
  dict.append(pad, ' '); dict += '\n';
  uint16_t hlen = (uint16_t)dict.size();
  f.write("\x93NUMPY", 6); f.put(1); f.put(0);
  f.write((char*)&hlen, 2); f.write(dict.data(), dict.size());
  f.write((char*)arr.data, arr.nbytes());
  return true;
}

// ── Runtime API ───────────────────────────────────────────────────────────────

struct DevBinary {
  uint32_t magic; uint32_t version; const char* data; uint64_t length;
};

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::vector<std::string> splitComma(const std::string& s) {
  std::vector<std::string> v;
  std::istringstream ss(s); std::string t;
  while (std::getline(ss, t, ',')) v.push_back(t);
  return v;
}

static std::vector<uint8_t> buildTiling(const std::string& params,
                                         const std::vector<std::string>& layout) {
  std::vector<uint8_t> bytes;
  auto pvec = splitComma(params);
  for (size_t i = 0; i < pvec.size(); ++i) {
    auto eq = pvec[i].find('=');
    int64_t val = std::stoll(pvec[i].substr(eq + 1));
    std::string type = i < layout.size() ? layout[i] : "int64";
    if (type == "int32" || type == "int32_t") {
      int32_t v = (int32_t)val; uint8_t buf[4];
      std::memcpy(buf, &v, 4); bytes.insert(bytes.end(), buf, buf+4);
    } else {
      uint8_t buf[8]; std::memcpy(buf, &val, 8);
      bytes.insert(bytes.end(), buf, buf+8);
    }
  }
  return bytes;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  std::string bin_path, tiling_params, tiling_layout_str, inputs_str,
              output_path = "/dev/null";
  int block_dim = 1;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i+1 >= argc) { std::cerr << "Missing arg after " << a << "\n"; exit(4); }
      return argv[++i];
    };
    if      (a == "--bin")           bin_path          = next();
    else if (a == "--tiling-params") tiling_params     = next();
    else if (a == "--tiling-layout") tiling_layout_str = next();
    else if (a == "--inputs")        inputs_str        = next();
    else if (a == "--output")        output_path       = next();
    else if (a == "--block-dim")     block_dim         = std::stoi(next());
  }
  if (bin_path.empty())  { std::cerr << "--bin required\n";    return 4; }
  if (inputs_str.empty()){ std::cerr << "--inputs required\n"; return 4; }

  // tiling layout: command-line overrides compiled-in default
  std::vector<std::string> tiling_layout;
  if (!tiling_layout_str.empty())
    tiling_layout = splitComma(tiling_layout_str);
  else {
)cpp") + layout_init + R"cpp(  }

  std::vector<uint8_t> tiling_bytes;
  if (!tiling_params.empty())
    tiling_bytes = buildTiling(tiling_params, tiling_layout);

  // Load inputs
  auto input_paths = splitComma(inputs_str);
  std::vector<NDArray> inputs(input_paths.size());
  for (size_t i = 0; i < input_paths.size(); ++i)
    if (!loadNpy(input_paths[i], inputs[i])) return 4;

  // Allocate output (single output; same shape/dtype as inputs[0])
  NDArray output;
  output.shape = inputs[0].shape;
  output.dtype = inputs[0].dtype;
  output.data  = new uint8_t[output.nbytes()]();

  // dlopen camodel
  const char* home = std::getenv("ASCEND_HOME_PATH");
  std::string lib_path = std::string(home)
      + "/runtime/lib64/libruntime_camodel.so";
  void* lib = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (!lib) { std::cerr << "dlopen failed: " << dlerror() << "\n"; return 3; }

#define LOAD(name, T) \
  auto name = reinterpret_cast<T>(dlsym(lib, #name)); \
  if (!name) { std::cerr << "dlsym " #name " failed\n"; dlclose(lib); return 3; }
  LOAD(rtSetDevice,         int(*)(int32_t))
  LOAD(rtDevBinaryRegister, int(*)(const DevBinary*, void**))
  LOAD(rtFunctionRegister,  int(*)(void*, void*, const char*, void*, uint32_t))
  LOAD(rtMalloc,            int(*)(void**, uint64_t, uint32_t, uint16_t))
  LOAD(rtFree,              int(*)(void*))
  LOAD(rtMemcpy,            int(*)(void*, uint64_t, const void*, uint64_t, uint32_t))
  LOAD(rtStreamCreate,      int(*)(void**, int32_t))
  LOAD(rtStreamDestroy,     int(*)(void*))
  LOAD(rtKernelLaunch,      int(*)(void*, uint32_t, void*, uint32_t, void*, void*))
  LOAD(rtDeviceSynchronize, int(*)())
#undef LOAD

  rtSetDevice(0);

  // Read binary
  std::ifstream bf(bin_path, std::ios::binary);
  if (!bf) { std::cerr << "Cannot open bin: " << bin_path << "\n"; return 4; }
  std::vector<uint8_t> bin_data((std::istreambuf_iterator<char>(bf)), {});

  // Register binary + function
  DevBinary dev_bin;
  dev_bin.magic   = )cpp") + std::string(magic_buf) + R"cpp(u;
  dev_bin.version = 0;
  dev_bin.data    = (const char*)bin_data.data();
  dev_bin.length  = bin_data.size();
  void* bin_handle = nullptr;
  if (rtDevBinaryRegister(&dev_bin, &bin_handle) != 0) {
    std::cerr << "rtDevBinaryRegister failed\n"; dlclose(lib); return 3;
  }
  const char* fn_name = ")cpp") + cfg.kernel_name + R"cpp(";
  void* fn_ptr = const_cast<char*>(fn_name);
  if (rtFunctionRegister(bin_handle, fn_ptr, fn_name, fn_ptr, 0) != 0) {
    std::cerr << "rtFunctionRegister failed\n"; dlclose(lib); return 3;
  }

  // Alloc helper: rtMalloc(n+512), align to 512 bytes, track raw ptr for rtFree
  struct AllocRec { void* raw; void* aligned; };
  std::vector<AllocRec> allocs;
  auto doAlloc = [&](size_t n) -> void* {
    void* raw = nullptr;
    rtMalloc(&raw, (uint64_t)(n + 512), 0u, (uint16_t)33u);
    uintptr_t a = ((uintptr_t)raw + 511) & ~511ULL;
    allocs.push_back({raw, (void*)a});
    return (void*)a;
  };
  auto freeAll = [&]() {
    for (auto& r : allocs) rtFree(r.raw);
    allocs.clear();
  };

  // H2D inputs
  std::vector<void*> in_ptrs, out_ptrs;
  for (auto& inp : inputs) {
    void* p = doAlloc(inp.nbytes());
    in_ptrs.push_back(p);
    const uint8_t* src = (const uint8_t*)inp.data;
    for (size_t off = 0; off < inp.nbytes(); off += 256) {
      size_t chunk = std::min<size_t>(256, inp.nbytes() - off);
      rtMemcpy((uint8_t*)p + off, chunk, src + off, chunk, 1);
    }
  }
  void* out_ptr = doAlloc(output.nbytes());
  out_ptrs.push_back(out_ptr);
  void* ws_ptr = doAlloc(8192);

  // Build launch args: [inputs..., outputs..., workspace, tiling_words...]
  std::vector<uint64_t> launch_args;
  for (auto* p : in_ptrs)  launch_args.push_back((uint64_t)p);
  for (auto* p : out_ptrs) launch_args.push_back((uint64_t)p);
  launch_args.push_back((uint64_t)ws_ptr);
  for (size_t i = 0; i < tiling_bytes.size(); i += 8) {
    uint64_t w = 0;
    std::memcpy(&w, tiling_bytes.data() + i,
                std::min<size_t>(8, tiling_bytes.size() - i));
    launch_args.push_back(w);
  }

  void* stream = nullptr;
  rtStreamCreate(&stream, 0);
  int rc = rtKernelLaunch(fn_ptr, (uint32_t)block_dim,
                          launch_args.data(),
                          (uint32_t)(launch_args.size() * 8),
                          nullptr, stream);
  if (rc != 0) {
    std::cerr << "rtKernelLaunch failed: rc=" << rc << "\n";
    rtStreamDestroy(stream); freeAll(); dlclose(lib); return 3;
  }
  rtDeviceSynchronize();
  rtStreamDestroy(stream);

  // D2H output (4-byte chunks; safe due to +512 overalloc)
  const uint8_t* src_d = (const uint8_t*)out_ptr;
  uint8_t* dst_d = (uint8_t*)output.data;
  for (size_t off = 0; off < output.nbytes(); off += 4) {
    uint8_t buf[4] = {};
    rtMemcpy(buf, 4, src_d + off, 4, 2);
    size_t chunk = std::min<size_t>(4, output.nbytes() - off);
    std::memcpy(dst_d + off, buf, chunk);
  }

  freeAll();
  saveNpy(output_path, output);
  dlclose(lib);
  return 0;
}
)cpp";
  // clang-format on
}

llvm::Expected<std::string> HostRunnerGen::Generate(const Config& cfg,
                                                     const std::string& output_dir) {
  // Guard: only num_outputs == 1 is supported
  if (cfg.num_outputs != 1)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "HostRunnerGen: num_outputs=%d not supported "
                                   "(only 1 is currently implemented)",
                                   cfg.num_outputs);

  // Ensure output dir exists
  if (auto ec = llvm::sys::fs::create_directories(output_dir))
    return llvm::createStringError(ec, "Cannot create output dir: %s",
                                   output_dir.c_str());

  std::string src_path = output_dir + "/runner.cpp";
  std::string exe_path = output_dir + "/runner";

  // Write runner.cpp
  {
    std::ofstream f(src_path);
    if (!f)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Cannot write runner.cpp to: %s",
                                     src_path.c_str());
    f << emitRunnerCpp(cfg);
  }

  if (cfg.verbose)
    llvm::errs() << "[HostRunnerGen] Written: " << src_path << "\n";

  // Compile with g++ via /bin/sh -c (inherits PATH, which must include g++).
  // Use 300 s timeout consistent with Compiler.cpp.
  // If g++ is not on PATH, the error message will read "g++: not found".
  // In that case, add g++ to PATH or set its full path in this command.
  std::string cmd = "g++ -O2 -std=c++17 " + src_path + " -ldl -o " + exe_path;
  std::vector<std::string> args = {"/bin/sh", "-c", cmd};
  std::vector<llvm::StringRef> argv;
  for (auto& a : args) argv.push_back(a);

  std::string err_msg;
  std::optional<llvm::StringRef> redirects[3];
  int ret = llvm::sys::ExecuteAndWait(argv[0], argv, std::nullopt, redirects,
                                      /*secondsToWait=*/300, /*memoryLimit=*/0,
                                      &err_msg);
  if (ret != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "g++ failed (exit %d): %s\n"
                                   "  cmd: %s",
                                   ret, err_msg.c_str(), cmd.c_str());

  if (cfg.verbose)
    llvm::errs() << "[HostRunnerGen] Compiled: " << exe_path << "\n";

  return exe_path;
}

} // namespace mlir::runtime
```

- [ ] **Step 2: Add to `lib/Runtime/CMakeLists.txt`**

```cmake
add_mlir_library(AscendCRuntime
  Compiler.cpp
  Executor.cpp
  HostRunnerGen.cpp
  SimValidator.cpp
  NpyIO.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Runtime

  LINK_LIBS PUBLIC
  LLVMSupport
)
```

- [ ] **Step 3: Commit**

```bash
git add lib/Runtime/HostRunnerGen.cpp lib/Runtime/CMakeLists.txt
git commit -m "feat(runtime): add HostRunnerGen (generates + compiles runner executable)"
```

---

## Task 3: Build verification

- [ ] **Step 1: Build inside xvm**

```bash
ssh xvm@orb
cd /home/niu/code/Ascend-MLIR
sleep 1   # wait for file sync after local edits
bash scripts/build.sh --build-project --llvm-build-dir ~/code/llvm-project/llvm/build
```

Expected: clean build, no errors. `AscendCRuntime` library rebuilt with `HostRunnerGen.cpp`.

- [ ] **Step 2: Commit any build fixes**

```bash
git add -p
git commit -m "fix(runtime): HostRunnerGen build fixes"
```

---

## Task 4: Integration test

**Files:**
- Create: `test/tools/runner/test_runner_gen.cpp`
- Create: `test/tools/runner/run_runner_gen.sh`

- [ ] **Step 1: Write `test/tools/runner/test_runner_gen.cpp`**

```cpp
// test/tools/runner/test_runner_gen.cpp
// Tests that HostRunnerGen::Generate() emits and compiles a valid runner,
// and that the runner enforces its required --bin argument.
#include "Runtime/HostRunnerGen.h"
#include "llvm/Support/raw_ostream.h"

int main() {
  mlir::runtime::HostRunnerGen gen;
  mlir::runtime::HostRunnerGen::Config cfg;
  cfg.kernel_name   = "broadcast_add_reducesum";
  cfg.kernel_type   = "vec";
  cfg.soc_version   = "Ascend910B1";
  cfg.num_inputs    = 2;
  cfg.num_outputs   = 1;
  cfg.tiling_layout = {"int64", "int64", "int64", "int64"};
  cfg.verbose       = true;

  // Test 1: Generate succeeds
  auto result = gen.Generate(cfg, "/tmp/runner_gen_test");
  if (!result) {
    llvm::errs() << "FAIL Generate: " << llvm::toString(result.takeError()) << "\n";
    return 1;
  }
  llvm::outs() << "PASS Generate: runner at " << *result << "\n";

  // Test 2: num_outputs > 1 returns error
  cfg.num_outputs = 2;
  auto bad = gen.Generate(cfg, "/tmp/runner_gen_test2");
  if (bad) {
    llvm::errs() << "FAIL: expected error for num_outputs=2 but got success\n";
    return 1;
  }
  llvm::consumeError(bad.takeError());
  llvm::outs() << "PASS: num_outputs=2 correctly rejected\n";

  return 0;
}
```

- [ ] **Step 2: Write `test/tools/runner/run_runner_gen.sh`**

```bash
#!/usr/bin/env bash
set -e
cd /home/niu/code/Ascend-MLIR
source examples/env.sh

LLVM_BUILD=~/code/llvm-project/llvm/build

# Build AscendCRuntime
cd build && cmake --build . --target AscendCRuntime -j4 && cd ..

# Compile test driver
g++ -std=c++17 \
    -I include/ \
    test/tools/runner/test_runner_gen.cpp \
    build/lib/libAscendCRuntime.a \
    $($LLVM_BUILD/bin/llvm-config --ldflags --libs support) \
    -ldl \
    -o /tmp/test_runner_gen

# Run test driver
/tmp/test_runner_gen

# Verify runner enforces --bin required
echo "--- Testing --bin required ---"
if /tmp/runner_gen_test/runner --inputs /dev/null 2>&1 | grep -q "\-\-bin required"; then
  echo "PASS: --bin required check"
else
  echo "FAIL: --bin required check"
  exit 1
fi

echo "ALL TESTS PASSED"
```

- [ ] **Step 3: Run the test inside xvm**

```bash
ssh xvm@orb
bash /home/niu/code/Ascend-MLIR/test/tools/runner/run_runner_gen.sh
```

Expected output:
```
[HostRunnerGen] Written: /tmp/runner_gen_test/runner.cpp
[HostRunnerGen] Compiled: /tmp/runner_gen_test/runner
PASS Generate: runner at /tmp/runner_gen_test/runner
PASS: num_outputs=2 correctly rejected
--- Testing --bin required ---
PASS: --bin required check
ALL TESTS PASSED
```

- [ ] **Step 4: Commit**

```bash
git add test/tools/runner/test_runner_gen.cpp test/tools/runner/run_runner_gen.sh
git commit -m "test(runtime): add HostRunnerGen integration test"
```

---

## Notes for Implementors

- **Raw string delimiter:** All three raw string literals use `R"cpp(...)cpp"` as delimiter. The spliced values (`layout_init`, `magic_buf`, `cfg.kernel_name`) are all alphanumeric or contain only `"`, `\n`, `.`, `_` — none can contain `)cpp"` so the delimiter is safe.
- **`tiling_layout` variable name:** In the emitted runner, the vector holding the default layout is named `tiling_layout` (not `default_layout`). The `layout_init` string in `HostRunnerGen.cpp` must emit `tiling_layout.push_back(...)` lines to match.
- **`rtFree` cleanup:** `freeAll()` is called on all error paths after allocations begin. The aligned pointer returned to callers is NOT the pointer passed to `rtFree` — the `AllocRec` struct tracks the original `raw` pointer.
- **`--output /dev/null`:** `saveNpy()` short-circuits on this path. `msprof` wraps the runner purely for performance data and passes `/dev/null` (or omits `--output`, defaulting to `/dev/null`).
- **`g++` on PATH:** The xvm container has `g++` available via the system. If it is missing, add it to the error message guidance in `Generate()` — the command string is included in the error for diagnosis.
