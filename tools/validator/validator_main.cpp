// tools/validator/validator_main.cpp
#include "Runtime/Executor.h"
#include "Runtime/NpyIO.h"
#include "Runtime/SimValidator.h"
#include "Runtime/Types.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace mlir::runtime;
using namespace llvm;

static cl::opt<std::string> BinFile("bin",
    cl::desc("Pre-compiled kernel .bin file"), cl::Required);
static cl::opt<std::string> KernelName("name",
    cl::desc("Kernel function name"), cl::Required);
static cl::opt<std::string> Inputs("inputs",
    cl::desc("Comma-separated input .npy files"), cl::Required);
static cl::opt<std::string> ExpectedFile("expected",
    cl::desc("Expected output .npy file"), cl::Required);
static cl::opt<std::string> TilingParams("tiling-params",
    cl::desc("Comma-separated KEY=VALUE tiling params: TB_M=16,TB_N=4,..."),
    cl::init(""));
static cl::opt<std::string> TilingLayout("tiling-layout",
    cl::desc("Comma-separated tiling param types: int64,int64,..."), cl::init(""));
static cl::opt<int> BlockDim("block-dim",
    cl::desc("Number of AiCore blocks"), cl::init(1));
static cl::opt<std::string> KernelType("kernel-type",
    cl::desc("Kernel type: vec | cube | mix (default: vec)"), cl::init("vec"));
static cl::opt<bool> SimMode("sim",
    cl::desc("Use CPU simulator backend (default)"), cl::init(false));
static cl::opt<bool> NpuMode("npu",
    cl::desc("Use NPU backend (not implemented)"), cl::init(false));
static cl::opt<double> Atol("atol",
    cl::desc("Absolute tolerance"), cl::init(1.0));
static cl::opt<double> Rtol("rtol",
    cl::desc("Relative tolerance"), cl::init(1e-2));
static cl::opt<std::string> DumpActual("dump-actual",
    cl::desc("Write actual output values to text file"), cl::init(""));
static cl::opt<std::string> DumpExpected("dump-expected",
    cl::desc("Write expected output values to text file"), cl::init(""));
static cl::opt<int> Precision("precision",
    cl::desc("Decimal places for --dump-actual / --dump-expected (default: 4)"),
    cl::init(4));

static float toFloat(const void* base, size_t idx, DType dtype) {
  switch (dtype) {
    case DType::F16: {
      uint16_t h;
      std::memcpy(&h, static_cast<const uint8_t*>(base) + idx * 2, 2);
      uint32_t sign = (h >> 15) & 1;
      uint32_t exp  = (h >> 10) & 0x1f;
      uint32_t frac = h & 0x3ff;
      uint32_t f;
      if (exp == 0)       f = (sign << 31) | (frac << 13);
      else if (exp == 31) f = (sign << 31) | 0x7f800000u | (frac << 13);
      else                f = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
      float result; std::memcpy(&result, &f, 4); return result;
    }
    case DType::BF16: {
      uint16_t h;
      std::memcpy(&h, static_cast<const uint8_t*>(base) + idx * 2, 2);
      uint32_t f = static_cast<uint32_t>(h) << 16;
      float result; std::memcpy(&result, &f, 4); return result;
    }
    case DType::F32: { float v; std::memcpy(&v, static_cast<const uint8_t*>(base) + idx * 4, 4); return v; }
    case DType::INT8: { int8_t v; std::memcpy(&v, static_cast<const uint8_t*>(base) + idx, 1); return static_cast<float>(v); }
    case DType::INT32: { int32_t v; std::memcpy(&v, static_cast<const uint8_t*>(base) + idx * 4, 4); return static_cast<float>(v); }
    case DType::INT64: { int64_t v; std::memcpy(&v, static_cast<const uint8_t*>(base) + idx * 8, 8); return static_cast<float>(v); }
  }
  return 0.f;
}

static void dumpArray(const NDArray& arr, const std::string& path, int prec) {
  std::error_code ec;
  llvm::raw_fd_ostream f(path, ec);
  if (ec) { llvm::errs() << "Warning: cannot write " << path << ": " << ec.message() << "\n"; return; }
  size_t n = arr.numElements();
  // Header: shape
  f << "# shape:";
  for (auto d : arr.shape) f << " " << d;
  f << "\n";
  // Values: one per line, fixed decimal
  std::ostringstream buf;
  buf << std::fixed << std::setprecision(prec);
  for (size_t i = 0; i < n; ++i) {
    buf.str(""); buf.clear();
    buf << std::fixed << std::setprecision(prec) << toFloat(arr.data, i, arr.dtype);
    f << buf.str() << "\n";
  }
}

static std::vector<std::string> splitComma(const std::string& s) {
  std::vector<std::string> parts;
  std::istringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) parts.push_back(tok);
  return parts;
}

static bool buildTiling(const std::string& params, const std::string& layout,
                        std::vector<uint8_t>& bytes) {
  auto pvec = splitComma(params);
  auto lvec = splitComma(layout);
  for (size_t i = 0; i < pvec.size(); ++i) {
    auto eq = pvec[i].find('=');
    if (eq == std::string::npos) {
      llvm::errs() << "Error: --tiling-params token missing '=': " << pvec[i] << "\n";
      return false;
    }
    int64_t val = std::stoll(pvec[i].substr(eq + 1));
    std::string type = i < lvec.size() ? lvec[i] : "int64";
    if (type == "int64" || type == "int64_t") {
      uint8_t buf[8]; std::memcpy(buf, &val, 8);
      bytes.insert(bytes.end(), buf, buf + 8);
    } else if (type == "int32" || type == "int32_t") {
      int32_t v = static_cast<int32_t>(val);
      uint8_t buf[4]; std::memcpy(buf, &v, 4);
      bytes.insert(bytes.end(), buf, buf + 4);
    } else {
      llvm::errs() << "Error: unknown tiling type '" << type << "'\n";
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv, "AscendC Kernel Validator\n");

  // SimMode is accepted for symmetry with --npu; only simulation is implemented
  (void)SimMode;

  // --npu is reserved but not implemented
  if (NpuMode) {
    llvm::errs() << "Error: --npu not implemented in this version\n";
    _Exit(4);
  }

  // Build tiling bytes
  std::vector<uint8_t> tiling;
  if (!TilingParams.empty()) {
    if (!buildTiling(TilingParams, TilingLayout, tiling)) _Exit(4);
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
      _Exit(4);
    }
    args.inputs.push_back(std::move(*arr_or));
  }

  // Load expected output
  auto exp_or = LoadNpy(ExpectedFile);
  if (!exp_or) {
    llvm::errs() << "Error loading expected: "
                 << llvm::toString(exp_or.takeError()) << "\n";
    _Exit(4);
  }
  NDArray exp_arr = std::move(*exp_or);
  // Deep-copy expected data: simulator may overwrite the original buffer
  // during Initialize/Run (simulator maps host memory as device memory).
  NDArray exp_snapshot;
  exp_snapshot.shape = exp_arr.shape;
  exp_snapshot.dtype = exp_arr.dtype;
  exp_snapshot.allocate();
  std::memcpy(exp_snapshot.data, exp_arr.data, exp_arr.nbytes());
  std::vector<NDArray> expected_arrs;
  expected_arrs.push_back(std::move(exp_snapshot));

  // Pre-alloc output buffer (same shape/dtype as expected)
  NDArray out_buf;
  out_buf.shape = exp_arr.shape;
  out_buf.dtype = exp_arr.dtype;
  out_buf.allocate();
  args.outputs.push_back(std::move(out_buf));

  // Dump expected immediately after loading, before simulator touches memory
  if (!DumpExpected.empty()) dumpArray(exp_arr, DumpExpected, Precision);

  // Map kernel_type → magic
  // "mix" uses MAGIC_ELF_AIVEC as conservative default (same as "vec")
  uint32_t magic = Executor::MAGIC_ELF_AIVEC;
  if (KernelType.getValue() == "cube") magic = Executor::MAGIC_ELF_AICUBE;

  // Initialize executor
  Executor executor(BackendMode::Simulation);
  if (auto err = executor.Initialize()) {
    llvm::errs() << "Error: executor init failed: "
                 << llvm::toString(std::move(err)) << "\n";
    _Exit(3);
  }

  // Register binary once
  auto handle_or = executor.RegisterBinary(BinFile, KernelName, magic);
  if (!handle_or) {
    llvm::errs() << "Error: RegisterBinary failed: "
                 << llvm::toString(handle_or.takeError()) << "\n";
    _Exit(3);
  }

  // Run and compare
  SimValidator validator;
  auto result = validator.ValidateBinary(*handle_or, executor, args,
                                         expected_arrs, Atol, Rtol);

  // Dump actual immediately after run, before buffers go out of scope
  if (!DumpActual.empty()) dumpArray(args.outputs[0], DumpActual, Precision);

  // NDArray RAII: owned_data freed automatically when args/exp_arr go out of scope.

  // Determine exit code.
  // error_msg non-empty → runtime error → exit 3 (no PASS/FAIL printed).
  // Otherwise → accuracy result → exit 0 (PASS) or 1 (FAIL).
  int exit_code;
  if (!result.error_msg.empty()) {
    llvm::errs() << "Error: " << result.error_msg << "\n";
    exit_code = 3;
  } else {
    llvm::outs() << "max_abs_diff:  " << result.max_abs_diff  << "\n"
                 << "mean_abs_diff: " << result.mean_abs_diff << "\n";
    llvm::outs() << (result.passed ? "PASS\n" : "FAIL\n");
    exit_code = result.passed ? 0 : 1;
  }

  llvm::outs().flush();
  llvm::errs().flush();
  _Exit(exit_code);
}
