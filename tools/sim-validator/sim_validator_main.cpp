// tools/sim-validator/sim_validator_main.cpp
#include "Runtime/NpyIO.h"
#include "Runtime/SimValidator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
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
static cl::opt<std::string> ExpectedFile("expected",       cl::desc("Expected output .npy file"), cl::Required);
static cl::opt<int>         BlockDim("block-dim",      cl::desc("Number of AiCore blocks"), cl::init(1));
static cl::opt<std::string> SocVersion("soc",          cl::desc("SoC version (default: Ascend910B1)"), cl::init("Ascend910B1"));

static std::vector<std::string> splitComma(const std::string& s) {
  std::vector<std::string> parts;
  std::istringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) parts.push_back(tok);
  return parts;
}

// Returns false and prints error on bad input; caller exits.
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
      llvm::errs() << "Error: unknown tiling type '" << type
                   << "' (expected int64 or int32)\n";
      return false;
    }
  }
  return true;
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
    if (!buildTiling(TilingParams, TilingLayout, tiling)) return 1;
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
      // Free already-loaded inputs before exiting
      for (auto& inp : args.inputs) delete[] static_cast<uint8_t*>(inp.data);
      llvm::errs() << "Error loading input " << path << ": "
                   << llvm::toString(arr_or.takeError()) << "\n";
      return 1;
    }
    args.inputs.push_back(*arr_or);
  }

  // Load expected
  auto exp_or = LoadNpy(ExpectedFile);
  if (!exp_or) {
    for (auto& inp : args.inputs) delete[] static_cast<uint8_t*>(inp.data);
    llvm::errs() << "Error loading expected: " << llvm::toString(exp_or.takeError()) << "\n";
    return 1;
  }
  NDArray exp_arr = *exp_or;
  std::vector<NDArray> expected_arrs;
  expected_arrs.push_back(exp_arr);

  // Pre-alloc output (same shape/dtype as expected); ownership stays in args.outputs
  NDArray out_buf;
  out_buf.shape = expected_arrs[0].shape;
  out_buf.dtype = expected_arrs[0].dtype;
  out_buf.data  = new uint8_t[out_buf.nbytes()]();
  args.outputs.push_back(out_buf);

  Compiler::Config cc;
  cc.soc_version = SocVersion;

  SimValidator validator;
  auto result = validator.Validate(KernelSrc, KernelName, args, expected_arrs,
                                   /*atol=*/1.0, /*rtol=*/1e-2, cc);

  // Free allocations — free through args.outputs[0] (authoritative owner), not out_buf alias
  delete[] static_cast<uint8_t*>(args.outputs[0].data);
  for (auto& inp : args.inputs) delete[] static_cast<uint8_t*>(inp.data);
  delete[] static_cast<uint8_t*>(expected_arrs[0].data);

  llvm::outs() << "max_abs_diff:  " << result.max_abs_diff  << "\n"
               << "mean_abs_diff: " << result.mean_abs_diff << "\n";
  if (!result.error_msg.empty())
    llvm::errs() << "Error: " << result.error_msg << "\n";

  int exit_code = result.passed ? 0 : 1;
  llvm::outs() << (result.passed ? "PASS\n" : "FAIL\n");
  llvm::outs().flush();
  llvm::errs().flush();
  // Use _exit to skip C++ destructors and atexit handlers:
  // libruntime_camodel simulator leaves background threads running after
  // rtDeviceSynchronize; normal exit() races those threads and segfaults.
  _Exit(exit_code);
}
