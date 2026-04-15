#include "Runtime/ArtifactCompiler.h"
#include "Runtime/MixAbiExtractor.h"
#include "Runtime/ToolDiscovery.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string> KernelFile("kernel", llvm::cl::Required);
static llvm::cl::opt<std::string> OutputDir("output", llvm::cl::init("./build/mix"));
static llvm::cl::opt<std::string> KernelName("name", llvm::cl::init(""));
static llvm::cl::opt<std::string> CannMlir(
    "cann-mlir",
    llvm::cl::desc("Path to step7_cann.mlir used to derive mix ABI"),
    llvm::cl::init(""));
static llvm::cl::opt<std::string> NpyDir(
    "npy-dir",
    llvm::cl::desc("Directory containing input/output .npy files used to resolve dynamic ABI shapes"),
    llvm::cl::init(""));
static llvm::cl::opt<std::string> SocVersion("soc", llvm::cl::init("Ascend910B1"));

int main(int argc, char** argv) {
  mlir::runtime::configureSiblingToolPathEnv("AFIR_MIX_TILING_HELPER", argv[0],
                                             "mix-tiling-helper");
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "RuntimeMix artifact compiler for mix kernels\n");

  if (CannMlir.empty()) {
    llvm::errs() << "Error: --cann-mlir is required so RuntimeMix can derive "
                    "kernel ABI and canonical IO metadata\n";
    return 4;
  }

  std::string kernelName = KernelName;
  if (kernelName.empty()) {
    auto abiOr = mlir::runtime::extractMixAbiFromCannMlir(CannMlir);
    if (!abiOr) {
      llvm::errs() << "Error: cannot derive kernel name from --cann-mlir: "
                   << llvm::toString(abiOr.takeError()) << "\n";
      return 4;
    }
    kernelName = abiOr->logicalKernelName;
  }
  if (kernelName.empty())
    kernelName = llvm::sys::path::stem(KernelFile).str();
  if (kernelName.empty()) {
    llvm::errs() << "Error: cannot derive kernel name\n";
    return 4;
  }
  if (!llvm::sys::fs::exists(KernelFile)) {
    llvm::errs() << "Error: kernel file not found: " << KernelFile << "\n";
    return 4;
  }

  mlir::runtime::ArtifactCompileRequest req;
  req.kernelSource = KernelFile;
  req.kernelName = kernelName;
  req.kernelKind = mlir::runtime::KernelKind::Mix;
  req.socVersion = SocVersion;
  req.outputDir = OutputDir;
  if (!CannMlir.empty())
    req.cannMlirPath = CannMlir;
  if (!NpyDir.empty())
    req.npyDir = NpyDir;

  mlir::runtime::ArtifactCompiler compiler;
  auto artifactOr = compiler.compile(req);
  if (!artifactOr) {
    llvm::errs() << "Compilation error: "
                 << llvm::toString(artifactOr.takeError()) << "\n";
    return 2;
  }
  const mlir::runtime::KernelArtifact &artifact = *artifactOr;

  llvm::outs() << "kernel_name=" << artifact.kernelName << "\n";
  llvm::outs() << "artifact_root=" << artifact.artifactRoot << "\n";
  llvm::outs() << "manifest_path=" << artifact.manifestPath << "\n";
  llvm::outs() << "soc_version=" << artifact.socVersion << "\n";
  llvm::outs() << "kernel_kind=mix\n";
  if (!artifact.deviceBinaryPath.empty())
    llvm::outs() << "device_binary=" << artifact.deviceBinaryPath << "\n";
  if (!artifact.packedSharedObjectPath.empty())
    llvm::outs() << "packed_shared_object="
                 << artifact.packedSharedObjectPath << "\n";
  return 0;
}
