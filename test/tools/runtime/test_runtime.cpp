// test/tools/runtime/test_runtime.cpp
//
// Unit tests for lib/Runtime: Types, NpyIO.
// Most coverage does not require a simulator or .bin file; the runtime-native
// packed mix error path is xvm/Ascend-environment-specific and self-skips when
// that environment is unavailable.
//
// Covers:
//   - DType byte sizes (including new BF16, INT8, INT64)
//   - NDArray RAII (allocate / setExternal / move semantics)
//   - NpyIO round-trip for all supported dtypes (F16, BF16, F32, INT8, INT32, INT64)
//   - NpyIO error: unsupported dtype
//   - NpyIO error: truncated file
//
// Build (on xvm):
//   cd /path/to/Ascend-MLIR
//   LLVM_BUILD=~/code/llvm-project/llvm/build
//   cd build && cmake --build . --target AscendCRuntime -j4 && cd ..
//   g++ -std=c++17 -I include/ -I $LLVM_BUILD/include \
//       -I ~/code/llvm-project/llvm/include \
//       test/tools/runtime/test_runtime.cpp \
//       build/lib/libAscendCRuntime.a \
//       $($LLVM_BUILD/bin/llvm-config --ldflags --libs support) \
//       -ldl -o /tmp/test_runtime
//   /tmp/test_runtime

#include "Runtime/Execution/NativeExecutionRunner.h"
#include "Runtime/NpyIO.h"
#include "Runtime/PathUtils.h"
#include "Runtime/Types.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace mlir::runtime;

// ── helpers ──────────────────────────────────────────────────────────────────

static int g_pass = 0, g_fail = 0;
#define EXPECT(cond, msg)                                              \
  do {                                                                 \
    if (cond) {                                                        \
      llvm::outs() << "  PASS: " << (msg) << "\n";                    \
      ++g_pass;                                                        \
    } else {                                                           \
      llvm::errs() << "  FAIL: " << (msg) << "\n";                    \
      ++g_fail;                                                        \
    }                                                                  \
  } while (0)

// Write a minimal .npy v1 file for the given dtype descriptor and raw bytes.
static bool writeNpy(const std::string& path, const std::string& descr,
                     const std::vector<int64_t>& shape,
                     const void* data, size_t nbytes) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  std::string shape_str = "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_str += std::to_string(shape[i]);
    if (shape.size() == 1 || i + 1 < shape.size()) shape_str += ",";
  }
  shape_str += ")";
  std::string dict = "{'descr': '" + descr +
                     "', 'fortran_order': False, 'shape': " + shape_str + ", }";
  size_t total = 10 + dict.size() + 1;
  size_t pad   = (64 - total % 64) % 64;
  dict.append(pad, ' ');
  dict += '\n';
  uint16_t hlen = static_cast<uint16_t>(dict.size());
  f.write("\x93NUMPY", 6);
  f.put(1); f.put(0);
  f.write(reinterpret_cast<const char*>(&hlen), 2);
  f.write(dict.data(), dict.size());
  f.write(reinterpret_cast<const char*>(data), nbytes);
  return true;
}

// ── test sections ─────────────────────────────────────────────────────────────

static void testDtypeSizes() {
  llvm::outs() << "\n[DType byte sizes]\n";
  EXPECT(dtypeBytes(DType::F16)   == 2, "F16 = 2 bytes");
  EXPECT(dtypeBytes(DType::BF16)  == 2, "BF16 = 2 bytes");
  EXPECT(dtypeBytes(DType::F32)   == 4, "F32 = 4 bytes");
  EXPECT(dtypeBytes(DType::INT8)  == 1, "INT8 = 1 byte");
  EXPECT(dtypeBytes(DType::INT32) == 4, "INT32 = 4 bytes");
  EXPECT(dtypeBytes(DType::INT64) == 8, "INT64 = 8 bytes");
}

static void testNDArrayRAII() {
  llvm::outs() << "\n[NDArray RAII]\n";

  // allocate() sets owned_data and data
  {
    NDArray a;
    a.shape = {4};
    a.dtype = DType::F32;
    a.allocate();
    EXPECT(a.data != nullptr, "allocate(): data non-null");
    EXPECT(a.owned_data != nullptr, "allocate(): owned_data non-null");
    EXPECT(a.data == a.owned_data.get(), "allocate(): data == owned_data.get()");
    EXPECT(a.nbytes() == 16, "allocate(): nbytes() = 16");
  }

  // setExternal() clears owned_data, sets data to external ptr
  {
    uint8_t buf[8] = {};
    NDArray a;
    a.shape = {2};
    a.dtype = DType::INT32;
    a.setExternal(buf);
    EXPECT(a.data == buf, "setExternal(): data = buf");
    EXPECT(a.owned_data == nullptr, "setExternal(): owned_data = null");
  }

  // Move constructor transfers ownership
  {
    NDArray a;
    a.shape = {3};
    a.dtype = DType::F16;
    a.allocate();
    void* raw_ptr = a.data;

    NDArray b = std::move(a);
    EXPECT(b.data == raw_ptr, "move ctor: b.data = original ptr");
    EXPECT(b.owned_data != nullptr, "move ctor: b owns data");
    EXPECT(a.data == nullptr, "move ctor: a.data = null after move");
    EXPECT(a.owned_data == nullptr, "move ctor: a.owned_data = null after move");
  }

  // Move assignment
  {
    NDArray a;
    a.shape = {2};
    a.dtype = DType::INT64;
    a.allocate();
    void* raw_ptr = a.data;

    NDArray b;
    b = std::move(a);
    EXPECT(b.data == raw_ptr, "move assign: b.data = original ptr");
    EXPECT(a.data == nullptr, "move assign: a.data = null");
  }

  // numElements() with empty shape = scalar (1)
  {
    NDArray a;
    a.dtype = DType::F32;
    EXPECT(a.numElements() == 1, "empty shape → scalar (1 element)");
    EXPECT(a.nbytes() == 4, "scalar F32 → 4 bytes");
  }
}

static void testNpyIORoundTrip() {
  llvm::outs() << "\n[NpyIO round-trip]\n";

  // F16
  {
    // float16 bit pattern for 1.0 = 0x3C00
    std::vector<uint16_t> orig = {0x3C00, 0x4000, 0x0000};
    writeNpy("/tmp/rt_test_f16.npy", "<f2", {3},
             orig.data(), orig.size() * 2);
    auto r = LoadNpy("/tmp/rt_test_f16.npy");
    EXPECT((bool)r, "F16 LoadNpy succeeds");
    if (r) {
      EXPECT(r->dtype == DType::F16, "F16 dtype");
      EXPECT(r->shape == std::vector<int64_t>{3}, "F16 shape");
      uint16_t v; std::memcpy(&v, r->data, 2);
      EXPECT(v == 0x3C00, "F16 data[0] = 0x3C00");
      auto err = SaveNpy("/tmp/rt_test_f16_out.npy", *r);
      EXPECT(!err, "F16 SaveNpy succeeds");
      llvm::consumeError(std::move(err));
    }
  }

  // BF16 (stored as <V2 void dtype in numpy)
  {
    // bf16 bit pattern for 1.0 = 0x3F80 (same as upper 16 bits of float32 1.0)
    std::vector<uint16_t> orig = {0x3F80, 0x4000, 0x0000};
    writeNpy("/tmp/rt_test_bf16.npy", "<V2", {3},
             orig.data(), orig.size() * 2);
    auto r = LoadNpy("/tmp/rt_test_bf16.npy");
    EXPECT((bool)r, "BF16 LoadNpy succeeds");
    if (r) {
      EXPECT(r->dtype == DType::BF16, "BF16 dtype");
      EXPECT(r->shape == std::vector<int64_t>{3}, "BF16 shape");
      uint16_t v; std::memcpy(&v, r->data, 2);
      EXPECT(v == 0x3F80, "BF16 data[0] = 0x3F80");
      auto err = SaveNpy("/tmp/rt_test_bf16_out.npy", *r);
      EXPECT(!err, "BF16 SaveNpy succeeds");
      llvm::consumeError(std::move(err));
      // Reload and verify round-trip
      auto r2 = LoadNpy("/tmp/rt_test_bf16_out.npy");
      EXPECT((bool)r2, "BF16 reload after SaveNpy");
      if (r2) {
        EXPECT(r2->dtype == DType::BF16, "BF16 reload dtype preserved");
        uint16_t v2; std::memcpy(&v2, r2->data, 2);
        EXPECT(v2 == 0x3F80, "BF16 reload data[0] preserved");
      }
    }
  }

  // F32
  {
    std::vector<float> orig = {1.0f, 2.0f, 3.0f, 4.0f};
    writeNpy("/tmp/rt_test_f32.npy", "<f4", {4},
             orig.data(), orig.size() * 4);
    auto r = LoadNpy("/tmp/rt_test_f32.npy");
    EXPECT((bool)r, "F32 LoadNpy succeeds");
    if (r) {
      EXPECT(r->dtype == DType::F32, "F32 dtype");
      float v; std::memcpy(&v, r->data, 4);
      EXPECT(std::fabs(v - 1.0f) < 1e-6f, "F32 data[0] = 1.0");
    }
  }

  // INT8
  {
    std::vector<int8_t> orig = {1, -2, 127, -128};
    writeNpy("/tmp/rt_test_i8.npy", "|i1", {4},
             orig.data(), orig.size() * 1);
    auto r = LoadNpy("/tmp/rt_test_i8.npy");
    EXPECT((bool)r, "INT8 LoadNpy succeeds");
    if (r) {
      EXPECT(r->dtype == DType::INT8, "INT8 dtype");
      EXPECT(r->nbytes() == 4, "INT8 nbytes = 4");
      int8_t v; std::memcpy(&v, r->data, 1);
      EXPECT(v == 1, "INT8 data[0] = 1");
      auto err = SaveNpy("/tmp/rt_test_i8_out.npy", *r);
      EXPECT(!err, "INT8 SaveNpy succeeds");
      llvm::consumeError(std::move(err));
    }
  }

  // INT32
  {
    std::vector<int32_t> orig = {100, -200, 0, 2147483647};
    writeNpy("/tmp/rt_test_i32.npy", "<i4", {4},
             orig.data(), orig.size() * 4);
    auto r = LoadNpy("/tmp/rt_test_i32.npy");
    EXPECT((bool)r, "INT32 LoadNpy succeeds");
    if (r) {
      EXPECT(r->dtype == DType::INT32, "INT32 dtype");
      int32_t v; std::memcpy(&v, r->data, 4);
      EXPECT(v == 100, "INT32 data[0] = 100");
    }
  }

  // INT64
  {
    std::vector<int64_t> orig = {1000000LL, -2000000LL, 0LL};
    writeNpy("/tmp/rt_test_i64.npy", "<i8", {3},
             orig.data(), orig.size() * 8);
    auto r = LoadNpy("/tmp/rt_test_i64.npy");
    EXPECT((bool)r, "INT64 LoadNpy succeeds");
    if (r) {
      EXPECT(r->dtype == DType::INT64, "INT64 dtype");
      EXPECT(r->nbytes() == 24, "INT64 nbytes = 24");
      int64_t v; std::memcpy(&v, r->data, 8);
      EXPECT(v == 1000000LL, "INT64 data[0] = 1000000");
      auto err = SaveNpy("/tmp/rt_test_i64_out.npy", *r);
      EXPECT(!err, "INT64 SaveNpy succeeds");
      llvm::consumeError(std::move(err));
    }
  }
}

static void testNpyIOErrors() {
  llvm::outs() << "\n[NpyIO error handling]\n";

  // Unsupported dtype (float64 = <f8)
  {
    std::vector<double> orig = {1.0, 2.0};
    writeNpy("/tmp/rt_test_f64.npy", "<f8", {2},
             orig.data(), orig.size() * 8);
    auto r = LoadNpy("/tmp/rt_test_f64.npy");
    EXPECT(!r, "float64 (<f8) LoadNpy returns error");
    if (!r) llvm::consumeError(r.takeError());
  }

  // File not found
  {
    auto r = LoadNpy("/tmp/rt_nonexistent_42.npy");
    EXPECT(!r, "missing file LoadNpy returns error");
    if (!r) llvm::consumeError(r.takeError());
  }

  // Truncated file: write valid header but short data
  {
    std::vector<float> orig = {1.0f, 2.0f, 3.0f};  // 12 bytes expected
    // Only write 4 bytes of data instead of 12
    std::ofstream f("/tmp/rt_test_trunc.npy", std::ios::binary);
    std::string descr = "<f4";
    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }";
    size_t total = 10 + dict.size() + 1;
    size_t pad   = (64 - total % 64) % 64;
    dict.append(pad, ' '); dict += '\n';
    uint16_t hlen = static_cast<uint16_t>(dict.size());
    f.write("\x93NUMPY", 6); f.put(1); f.put(0);
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(dict.data(), dict.size());
    f.write(reinterpret_cast<const char*>(orig.data()), 4);  // only 4 of 12 bytes
    f.close();
    auto r = LoadNpy("/tmp/rt_test_trunc.npy");
    EXPECT(!r, "truncated file LoadNpy returns error");
    if (!r) llvm::consumeError(r.takeError());
  }
}

static void testRuntimeNativePackedMixErrors() {
  llvm::outs() << "\n[Runtime-native packed mix error path]\n";

#ifdef _WIN32
  EXPECT(true, "runtime-native packed mix error path is covered on xvm");
#else
  const std::string ascendHome = findAscendHome();
  const std::string socVersion = findSocVersion();
  if (ascendHome.empty() || socVersion.empty()) {
    EXPECT(true,
           "runtime-native packed mix error path is skipped when Ascend runtime env is unavailable");
    return;
  }

  std::filesystem::path buildDir = std::filesystem::temp_directory_path() /
                                   "test_runtime_native_runner";
  std::filesystem::remove_all(buildDir);
  std::filesystem::create_directories(buildDir);
  std::filesystem::path missingSoPath = buildDir / "missing-packed.so";

  const std::filesystem::path errorPath = buildDir / "error.txt";
  pid_t pid = fork();
  EXPECT(pid >= 0, "fork for runtime-native packed mix error path succeeds");
  if (pid < 0) {
    std::filesystem::remove_all(buildDir);
    return;
  }

  if (pid == 0) {
    std::ofstream os(errorPath);
    NativeExecutionRunner runner(ExecutionRunnerMode::Simulation);
    auto initErr = runner.initialize(0);
    if (initErr) {
      os << llvm::toString(std::move(initErr));
      os.flush();
      std::_Exit(0);
    }

    RunArgs args;
    args.block_dim = 1;
    PackedMixExecutionLaunch launch{missingSoPath.string(), "fc_relu"};
    auto err = runner.runPackedMixFile(launch, args);
    if (err)
      os << llvm::toString(std::move(err));
    else
      os << "unexpected-success";
    os.flush();
    std::_Exit(0);
  }

  int status = 0;
  EXPECT(waitpid(pid, &status, 0) == pid,
         "waitpid for runtime-native packed mix error path succeeds");
  EXPECT(WIFEXITED(status),
         "runtime-native packed mix error child exits cleanly");

  std::ifstream is(errorPath);
  std::string message((std::istreambuf_iterator<char>(is)),
                      std::istreambuf_iterator<char>());
  EXPECT(!message.empty(),
         "runtime-native packed mix path returns an error for missing shared library");
  EXPECT(message.find("dlopen") != std::string::npos,
         "runtime-native packed mix error keeps a dlopen failure diagnostic");

  std::filesystem::remove_all(buildDir);
#endif
}

static void testRuntimePathUtils() {
  llvm::outs() << "\n[Runtime path utils]\n";
  const std::string testSoc = "TestSoc";

  {
    auto archDir = getHostCannArchDir("x86_64");
    EXPECT(archDir == "x86_64-linux",
           "x86_64 host arch maps to x86_64-linux");
  }

  {
    auto archDir = getHostCannArchDir("amd64");
    EXPECT(archDir == "x86_64-linux",
           "amd64 host arch maps to x86_64-linux");
  }

  {
    auto archDir = getHostCannArchDir("aarch64");
    EXPECT(archDir == "aarch64-linux",
           "aarch64 host arch maps to aarch64-linux");
  }

  {
    auto archDir = getHostCannArchDir("arm64");
    EXPECT(archDir == "aarch64-linux",
           "arm64 host arch maps to aarch64-linux");
  }

  {
    auto archDir = getHostCannArchDir("mips64");
    EXPECT(archDir.empty(),
           "unknown host arch does not silently default to ARM");
  }

  {
    auto home = resolveAscendHomeForTest(
        "/custom/ascend", "");
    EXPECT(home == "/custom/ascend",
           "resolver prefers ASCEND_HOME_PATH when it exists");
  }

  {
    auto home = resolveAscendHomeForTest(
        "", "/custom/toolkit");
    EXPECT(home == "/custom/toolkit",
           "resolver falls back to ASCEND_TOOLKIT_HOME");
  }

  {
    auto home = resolveAscendHomeForTest(
        "", "");
    EXPECT(home.empty(),
           "resolver requires environment variables");
  }

  {
    auto soc = resolveSocVersionForTest("ExplicitSoc", "EnvSoc", "FallbackSoc");
    EXPECT(soc == "ExplicitSoc",
           "soc resolver prefers explicit value when it exists");
  }

  {
    auto soc = resolveSocVersionForTest("", "EnvSoc", "FallbackSoc");
    EXPECT(soc == "EnvSoc",
           "soc resolver falls back to SOC_VERSION environment value");
  }

  {
    auto soc = resolveSocVersionForTest("", "", "FallbackSoc");
    EXPECT(soc == "FallbackSoc",
           "soc resolver uses fallback when explicit and environment values are absent");
  }

  {
    const std::filesystem::path root = "/tmp/rt_path_utils_arch";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "x86_64-linux/lib64");
    std::filesystem::create_directories(
        root / ("x86_64-linux/simulator/" + testSoc + "/lib"));
    std::filesystem::create_directories(
        root / "x86_64-linux/lib64/device/lib64");
    std::filesystem::create_directories(
        root / "x86_64-linux/simulator/custom_dav_variant/lib");
    std::ofstream(root / "x86_64-linux/lib64/libascendcl.so").put('\n');
    std::ofstream(root / ("x86_64-linux/simulator/" + testSoc +
                           "/lib/libruntime_camodel.so"))
        .put('\n');
    std::ofstream(root / "x86_64-linux/lib64/device/lib64/libascend_hal.so")
        .put('\n');
    std::ofstream(root / "x86_64-linux/simulator/custom_dav_variant/lib/libmodel_top.so")
        .put('\n');
    const char *savedDavVersion = std::getenv("ASCEND_DAV_SIM_VERSION");
    const std::string savedDavVersionValue =
        savedDavVersion ? std::string(savedDavVersion) : std::string();
    ::unsetenv("ASCEND_DAV_SIM_VERSION");

    EXPECT(findAscendAclLibPath(root.string(), "x86_64") ==
               (root / "x86_64-linux/lib64/libascendcl.so").string(),
           "path utils resolve x86_64 acl library path");
    EXPECT(findAscendRuntimeCamodelPath(root.string(), testSoc, "x86_64") ==
               (root / ("x86_64-linux/simulator/" + testSoc +
                         "/lib/libruntime_camodel.so"))
                   .string(),
           "path utils resolve x86_64 simulator runtime path");
    std::ofstream(root / "x86_64-linux/lib64/libruntime.so").put('\n');
    EXPECT(findAscendRuntimeLibPath(root.string(), "x86_64") ==
               (root / "x86_64-linux/lib64/libruntime.so").string(),
           "path utils resolve x86_64 real-device runtime path");
    EXPECT(findAscendDeviceLibDir(root.string(), "x86_64") ==
               (root / "x86_64-linux/lib64/device/lib64").string(),
           "path utils resolve x86_64 device lib directory when it is the only candidate");
    EXPECT(findAscendDavSimulatorLibDir(root.string(), "x86_64") ==
               (root / "x86_64-linux/simulator/custom_dav_variant/lib").string(),
           "path utils discover DAV simulator directory without hardcoded product id");
    auto requiredDav = requireAscendDavSimulatorLibDir(root.string(), "x86_64");
    EXPECT(static_cast<bool>(requiredDav),
           "path utils require a single DAV simulator directory");
    if (requiredDav) {
      EXPECT(*requiredDav ==
                 (root / "x86_64-linux/simulator/custom_dav_variant/lib").string(),
             "required DAV simulator directory matches discovered path");
    } else {
      llvm::consumeError(requiredDav.takeError());
    }
    if (savedDavVersion)
      ::setenv("ASCEND_DAV_SIM_VERSION", savedDavVersionValue.c_str(), 1);
  }

  {
    const std::filesystem::path root = "/tmp/rt_path_utils_generic";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "lib64");
    std::ofstream(root / "lib64/libascendcl.so").put('\n');
    std::ofstream(root / "lib64/libruntime.so").put('\n');
    EXPECT(findAscendAclLibPath(root.string()) ==
               (root / "lib64/libascendcl.so").string(),
           "path utils fall back to generic lib64 when arch dir is absent");
    EXPECT(findAscendRuntimeLibPath(root.string()) ==
               (root / "lib64/libruntime.so").string(),
           "path utils fall back to generic lib64 runtime when arch dir is absent");
  }

  {
    const std::filesystem::path root = "/tmp/rt_path_utils_devlib";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "devlib/linux/x86_64");
    std::ofstream(root / "devlib/linux/x86_64/libascend_hal.so").put('\n');
    EXPECT(findAscendDeviceLibDir(root.string(), "x86_64") ==
               (root / "devlib/linux/x86_64").string(),
           "path utils fall back to devlib linux x86_64 for device libs");
  }

  {
    const std::filesystem::path root = "/tmp/rt_path_utils_devlib_preferred";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "x86_64-linux/lib64/device/lib64");
    std::filesystem::create_directories(root / "devlib/linux/x86_64");
    std::ofstream(root / "x86_64-linux/lib64/device/lib64/libascend_hal.so").put('\n');
    std::ofstream(root / "devlib/linux/x86_64/libascend_hal.so").put('\n');
    EXPECT(findAscendDeviceLibDir(root.string(), "x86_64") ==
               (root / "devlib/linux/x86_64").string(),
           "path utils prefer host devlib linux x86_64 over device lib64 when both exist");
  }

  {
    const std::filesystem::path root = "/tmp/rt_path_utils_devlib_aarch64";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "aarch64-linux/lib64/device/lib64");
    std::filesystem::create_directories(root / "devlib/linux/aarch64");
    std::ofstream(root / "aarch64-linux/lib64/device/lib64/libascend_hal.so").put('\n');
    std::ofstream(root / "devlib/linux/aarch64/libascend_hal.so").put('\n');
    EXPECT(findAscendDeviceLibDir(root.string(), "aarch64") ==
               (root / "aarch64-linux/lib64/device/lib64").string(),
           "path utils keep device lib64 precedence for aarch64 when both candidates exist");
  }

  {
    const std::filesystem::path root = "/tmp/rt_path_utils_dav_multi";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "x86_64-linux/simulator/dav_a/lib");
    std::filesystem::create_directories(root / "x86_64-linux/simulator/dav_b/lib");
    std::ofstream(root / "x86_64-linux/simulator/dav_a/lib/libmodel_top.so")
        .put('\n');
    std::ofstream(root / "x86_64-linux/simulator/dav_b/lib/libmodel_top.so")
        .put('\n');
    const char *savedDavVersion = std::getenv("ASCEND_DAV_SIM_VERSION");
    const std::string savedDavVersionValue =
        savedDavVersion ? std::string(savedDavVersion) : std::string();
    ::unsetenv("ASCEND_DAV_SIM_VERSION");
    auto requiredDav = requireAscendDavSimulatorLibDir(root.string(), "x86_64");
    EXPECT(!requiredDav,
           "path utils reject ambiguous DAV simulator directories without override");
    if (!requiredDav)
      llvm::consumeError(requiredDav.takeError());

    ::setenv("ASCEND_DAV_SIM_VERSION", "dav_b", 1);
    auto configuredDav = requireAscendDavSimulatorLibDir(root.string(), "x86_64");
    EXPECT(static_cast<bool>(configuredDav),
           "path utils accept ASCEND_DAV_SIM_VERSION override");
    if (configuredDav) {
      EXPECT(*configuredDav ==
                 (root / "x86_64-linux/simulator/dav_b/lib").string(),
             "configured DAV simulator override wins");
    } else {
      llvm::consumeError(configuredDav.takeError());
    }
    if (savedDavVersion)
      ::setenv("ASCEND_DAV_SIM_VERSION", savedDavVersionValue.c_str(), 1);
    else
      ::unsetenv("ASCEND_DAV_SIM_VERSION");
  }
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  testDtypeSizes();
  testNDArrayRAII();
  testNpyIORoundTrip();
  testNpyIOErrors();
  testRuntimeNativePackedMixErrors();
  testRuntimePathUtils();

  llvm::outs() << "\n========================================\n"
               << "Results: " << g_pass << " passed, " << g_fail << " failed\n"
               << "========================================\n";
  return g_fail > 0 ? 1 : 0;
}
