#include "Runtime/MixDirectBackend.h"
#include "Runtime/MixArtifact.h"
#include "Runtime/MixAbiExtractor.h"
#include "llvm/Support/CommandLine.h"
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
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "RuntimeMix AscendC mix compiler\n");

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

  mlir::runtime::MixDirectCompileConfig cfg;
  cfg.kernelSrc = KernelFile;
  cfg.kernelName = kernelName;
  cfg.socVersion = SocVersion;
  cfg.outputDir = OutputDir;
  if (!CannMlir.empty())
    cfg.cannMlirPath = CannMlir;
  if (!NpyDir.empty())
    cfg.npyDir = NpyDir;

  mlir::runtime::MixDirectBackend backend;
  auto artifact = backend.compile(cfg);
  if (!artifact) {
    llvm::errs() << "Compilation error: " << llvm::toString(artifact.takeError())
                 << "\n";
    return 2;
  }

  llvm::outs() << "kernel_name=" << artifact->kernel_name << "\n";
  llvm::outs() << "kernel_so=" << artifact->kernel_so_path << "\n";
  llvm::outs() << "launcher_header_dir=" << artifact->launcher_header_dir << "\n";
  llvm::outs() << "install_dir=" << artifact->install_dir << "\n";
  llvm::outs() << "host_runner=" << artifact->host_runner_path << "\n";
  return 0;
}
