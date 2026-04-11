// tools/validator/validator_main.cpp
#include "Runtime/CompatRuntime.h"
#include "Runtime/ExecutionSession.h"
#include "Runtime/NpyIO.h"
#include "Runtime/SimValidator.h"
#include "Runtime/TaskGraph.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <filesystem>
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
static cl::opt<std::string> TilingSchemaFile("tiling-schema",
    cl::desc("Path to tiling_space.json; validates and packs --tiling-params by name"),
    cl::init(""));
static cl::opt<std::string> TilingBinFile("tiling-bin",
    cl::desc("Path to raw packed tiling bytes (overrides --tiling-params / --tiling-schema)"),
    cl::init(""));
static cl::opt<int> BlockDim("block-dim",
    cl::desc("Number of AiCore blocks"), cl::init(1));
static cl::opt<int> WorkspaceSize("workspace-size",
    cl::desc("Workspace bytes allocated for packed/raw runtime paths"), cl::init(16 * 1024 * 1024));
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

static llvm::Expected<std::string>
makeTemporaryPath(llvm::StringRef prefix, llvm::StringRef fileName) {
  std::error_code ec;
  const std::filesystem::path tempRoot = std::filesystem::temp_directory_path(ec);
  if (ec) {
    return llvm::createStringError(ec,
                                   "cannot determine temp directory for validator");
  }

  llvm::SmallString<256> directoryPrefix(tempRoot.string());
  llvm::sys::path::append(directoryPrefix, prefix);
  llvm::SmallString<256> directory;
  if (auto createDirError =
          llvm::sys::fs::createUniqueDirectory(directoryPrefix, directory)) {
    return llvm::createStringError(createDirError,
                                   "cannot create temporary validator directory");
  }

  llvm::SmallString<256> filePath(directory);
  llvm::sys::path::append(filePath, fileName);
  return filePath.str().str();
}

static llvm::Expected<TaskGraph>
buildTaskGraphFromManifest(const KernelArtifact &artifact,
                           const RunManifestSpec &manifest) {
  TaskGraph graph;
  for (const RunTaskSpec &taskSpec : manifest.tasks) {
    RuntimeTask task;
    task.taskId = taskSpec.taskId;
    task.artifact = artifact;
    task.dependencies = taskSpec.dependencies;
    task.invocation = taskSpec.invocation;
    if (auto err = graph.addTask(task))
      return std::move(err);
  }
  return graph;
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

  auto tilingBinaryPathOr = prepareValidatorTilingBinaryPath(
      TilingBinFile, TilingSchemaFile, TilingParams, TilingLayout,
      &llvm::errs());
  if (!tilingBinaryPathOr) {
    llvm::errs() << "Error: " << llvm::toString(tilingBinaryPathOr.takeError())
                 << "\n";
    _Exit(4);
  }
  std::string tilingBinaryPath = *tilingBinaryPathOr;

  // Load inputs
  std::vector<std::string> inputPaths;
  for (auto& path : splitComma(Inputs)) {
    auto arr_or = LoadNpy(path);
    if (!arr_or) {
      llvm::errs() << "Error loading input " << path << ": "
                   << llvm::toString(arr_or.takeError()) << "\n";
      _Exit(4);
    }
    inputPaths.push_back(path);
  }

  // Load expected output
  auto exp_or = LoadNpy(ExpectedFile);
  if (!exp_or) {
    llvm::errs() << "Error loading expected: "
                 << llvm::toString(exp_or.takeError()) << "\n";
    _Exit(4);
  }
  NDArray exp_arr = std::move(*exp_or);
  NDArray exp_snapshot;
  exp_snapshot.shape = exp_arr.shape;
  exp_snapshot.dtype = exp_arr.dtype;
  exp_snapshot.allocate();
  std::memcpy(exp_snapshot.data, exp_arr.data, exp_arr.nbytes());
  std::vector<NDArray> expected_arrs;
  expected_arrs.push_back(std::move(exp_snapshot));

  // Dump expected immediately after loading, before simulator touches memory
  if (!DumpExpected.empty()) dumpArray(exp_arr, DumpExpected, Precision);

  auto actualOutputPathOr = makeTemporaryPath("ascendc-validator-actual", "actual.npy");
  if (!actualOutputPathOr) {
    llvm::errs() << "Error: " << llvm::toString(actualOutputPathOr.takeError()) << "\n";
    _Exit(3);
  }

  CompatValidateOptions compatOptions;
  compatOptions.artifactRoot = std::filesystem::absolute(
      std::filesystem::path(BinFile.getValue())).parent_path().string();
  compatOptions.inputPaths = inputPaths;
  compatOptions.expectedOutputPath = "";
  compatOptions.actualOutputPath = *actualOutputPathOr;
  compatOptions.tilingSchemaPath = TilingSchemaFile;
  compatOptions.tilingParams = TilingParams;
  compatOptions.tilingBinaryPath = tilingBinaryPath;
  compatOptions.actualOutputShape = exp_arr.shape;
  compatOptions.actualOutputDType = exp_arr.dtype;
  compatOptions.blockDim = BlockDim;
  compatOptions.atol = Atol;
  compatOptions.rtol = Rtol;

  auto manifestOr = buildCompatSingleTaskRunManifest(compatOptions);
  if (!manifestOr) {
    llvm::errs() << "Error: " << llvm::toString(manifestOr.takeError()) << "\n";
    _Exit(4);
  }

  RunManifestSpec manifest = *manifestOr;
  if (manifest.tasks.empty()) {
    llvm::errs() << "Error: runtime manifest did not contain any tasks\n";
    _Exit(4);
  }

  manifest.tasks[0].invocation.workspaceSize =
      static_cast<size_t>(WorkspaceSize);

  KernelArtifact artifact;
  artifact.kernelName = KernelName;
  artifact.kernelKind = KernelType.getValue() == "mix"
                             ? KernelKind::Mix
                             : KernelType.getValue() == "cube"
                                   ? KernelKind::Cube
                                   : KernelKind::Vec;
  artifact.artifactRoot = compatOptions.artifactRoot;
  if (artifact.kernelKind == KernelKind::Mix)
    artifact.packedSharedObjectPath = BinFile;
  else
    artifact.deviceBinaryPath = BinFile;

  auto graphOr = buildTaskGraphFromManifest(artifact, manifest);
  if (!graphOr) {
    llvm::errs() << "Error: " << llvm::toString(graphOr.takeError()) << "\n";
    _Exit(4);
  }

  ExecutionSession session(ExecutionBackendKind::Simulation);
  auto traceOr = session.run(*graphOr);
  if (!traceOr) {
    llvm::errs() << "Error: " << llvm::toString(traceOr.takeError()) << "\n";
    _Exit(3);
  }

  auto actualOr = LoadNpy(*actualOutputPathOr);
  if (!actualOr) {
    llvm::errs() << "Error loading actual output: "
                 << llvm::toString(actualOr.takeError()) << "\n";
    _Exit(3);
  }

  NDArray actual_arr = std::move(*actualOr);
  if (!DumpActual.empty()) dumpArray(actual_arr, DumpActual, Precision);

  RunArgs args;
  args.outputs.push_back(std::move(actual_arr));

  SimValidator validator;
  SimValidator::Result result = validator.CompareOnly(args, expected_arrs, Atol, Rtol);

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
