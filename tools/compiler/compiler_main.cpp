// tools/compiler/compiler_main.cpp
#include "Runtime/ArtifactCompiler.h"
#include "Runtime/CompatRuntime.h"
#include "Runtime/HostRunnerGen.h"
#include "Runtime/PathUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>
#include <string>
#include <vector>

using namespace mlir::runtime;
using namespace llvm;

static cl::opt<std::string> KernelFile("kernel",
    cl::desc("Kernel .cpp source file"), cl::Required);
static cl::opt<std::string> OutputDir("output",
    cl::desc("Output directory for .o, .bin, runner"), cl::init("./build"));
static cl::opt<std::string> KernelName("name",
    cl::desc("Kernel name (default: stem of --kernel filename)"), cl::init(""));
static cl::opt<std::string> SocVersion("soc",
    cl::desc("SoC version (defaults to SOC_VERSION or Ascend910B1)"), cl::init(""));
static cl::opt<std::string> Arch("arch",
    cl::desc("bisheng arch (default: dav-c220-vec)"), cl::init("dav-c220-vec"));
static cl::opt<int> NumInputs("num-inputs",
    cl::desc("Number of kernel inputs (for runner generation)"), cl::init(1));
static cl::opt<int> NumOutputs("num-outputs",
    cl::desc("Number of kernel outputs (for runner generation; only 1 supported)"),
    cl::init(1));
static cl::opt<std::string> TilingLayout("tiling-layout",
    cl::desc("Comma-separated tiling param types: int64,int64,int32,..."), cl::init(""));
static cl::opt<std::string> KernelType("kernel-type",
    cl::desc("Kernel type: vec | cube | mix (default: vec)"), cl::init("vec"));
static cl::opt<bool> Verbose("verbose",
    cl::desc("Print compilation commands"), cl::init(false));

static std::vector<std::string> splitComma(const std::string& s) {
  std::vector<std::string> parts;
  std::istringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) parts.push_back(tok);
  return parts;
}

static void printArtifactSummary(const KernelArtifact &artifact) {
  llvm::outs() << "artifact.kernel_name=" << artifact.kernelName << "\n";
  llvm::outs() << "artifact.root=" << artifact.artifactRoot << "\n";
  llvm::outs() << "artifact.manifest=" << artifact.manifestPath << "\n";
  if (!artifact.socVersion.empty())
    llvm::outs() << "artifact.soc=" << artifact.socVersion << "\n";
  if (!artifact.packedSharedObjectPath.empty()) {
    llvm::outs() << "artifact.binary=" << artifact.packedSharedObjectPath
                 << "\n";
    llvm::outs() << "artifact.packed_shared_object="
                 << artifact.packedSharedObjectPath << "\n";
  } else if (!artifact.deviceBinaryPath.empty()) {
    llvm::outs() << "artifact.binary=" << artifact.deviceBinaryPath << "\n";
  }
  if (!artifact.deviceBinaryPath.empty())
    llvm::outs() << "artifact.device_binary=" << artifact.deviceBinaryPath
                 << "\n";
}

static bool needsLegacyRunnerCompatibility(const std::string &kernelType) {
  return kernelType == "mix";
}

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv, "AscendC Kernel Compiler\n");

  // Derive kernel name from filename if not given
  std::string kernel_name = KernelName.getValue();
  if (kernel_name.empty())
    kernel_name = llvm::sys::path::stem(KernelFile.getValue()).str();

  if (kernel_name.empty()) {
    llvm::errs() << "Error: cannot derive kernel name from --kernel; use --name\n";
    return 4;
  }

  // Validate numeric args
  if (NumInputs <= 0) {
    llvm::errs() << "Error: --num-inputs must be >= 1\n"; return 4;
  }
  if (NumOutputs <= 0) {
    llvm::errs() << "Error: --num-outputs must be >= 1\n"; return 4;
  }
  if (NumOutputs != 1) {
    llvm::errs() << "Error: --num-outputs " << NumOutputs
                 << " not supported (only 1 is implemented)\n";
    return 4;
  }

  // Validate input file exists (exit 4 = input error, per spec)
  if (!llvm::sys::fs::exists(KernelFile.getValue())) {
    llvm::errs() << "Error: kernel file not found: " << KernelFile << "\n";
    return 4;
  }

  std::string resolvedSocVersion = resolveSocVersion(SocVersion, "Ascend910B1");

  CompatCompileOptions compileOptions;
  compileOptions.kernelSourcePath = KernelFile.getValue();
  compileOptions.outputRoot = OutputDir.getValue();
  compileOptions.requestedKernelName = kernel_name;
  compileOptions.socVersion = resolvedSocVersion;
  compileOptions.arch = Arch.getValue();
  compileOptions.kernelType = KernelType.getValue();
  compileOptions.verbose = Verbose.getValue();

  auto requestOr = buildCompatCompileRequest(compileOptions);
  if (!requestOr) {
    llvm::errs() << "Error: " << llvm::toString(requestOr.takeError())
                 << "\n";
    return 4;
  }

  ArtifactCompiler compiler;
  auto artifactOr = compiler.compile(*requestOr);
  if (!artifactOr) {
    llvm::errs() << "Compilation error: "
                 << llvm::toString(artifactOr.takeError()) << "\n";
    return 2;
  }
  printArtifactSummary(*artifactOr);

  if (!needsLegacyRunnerCompatibility(KernelType.getValue()))
    return 0;

  HostRunnerGen::Config hcfg;
  hcfg.kernel_name = kernel_name;
  hcfg.kernel_type = KernelType.getValue();
  hcfg.soc_version = resolvedSocVersion;
  hcfg.num_inputs = NumInputs;
  hcfg.num_outputs = NumOutputs;
  if (!TilingLayout.getValue().empty())
    hcfg.tiling_layout = splitComma(TilingLayout);
  hcfg.verbose = Verbose;

  HostRunnerGen gen;
  auto runner_or = gen.Generate(hcfg, OutputDir);
  if (!runner_or) {
    llvm::errs() << "Warning: runner generation unavailable: "
                 << llvm::toString(runner_or.takeError()) << "\n";
    return 0;
  }
  llvm::outs() << "runner=" << *runner_or << "\n";

  return 0;
}
