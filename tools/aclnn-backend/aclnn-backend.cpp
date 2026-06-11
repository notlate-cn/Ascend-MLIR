//===- aclnn-backend.cpp - AclnnBackend driver ------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
// Reads a network.mlir (output of afir-opt --aclnn-finalize-decl) and emits
// network_host.cpp — the host-side C++ orchestrator that dispatches aclnn ops
// and AscendC kernel launches.
//
// Usage:
//   aclnn-backend --input network.mlir --output network_host.cpp
//
//===----------------------------------------------------------------------===//

#include "Runtime/AclnnBackend.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> InputFile(
    "input", cl::Required,
    cl::desc("Path to network.mlir (afir-opt output)"));

static cl::opt<std::string> OutputFile(
    "output", cl::Required,
    cl::desc("Path to write network_host.cpp"));

static cl::opt<std::string> TilingsFile(
    "tilings", cl::init(""),
    cl::desc("Path to per-kernel tilings JSON (best-config map)"));

static cl::opt<std::string> KernelBinariesDir(
    "kernel-binaries", cl::init(""),
    cl::desc("Directory containing compiled AscendC kernel artifacts"));

int main(int argc, char **argv) {
  InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv,
                              "aclnn-backend: network.mlir → network_host.cpp\n");

  mlir::runtime::AclnnBackendConfig cfg;
  cfg.networkMlirPath   = InputFile;
  cfg.outputCppPath     = OutputFile;
  cfg.tilingsPath       = TilingsFile;
  cfg.kernelBinariesDir = KernelBinariesDir;

  mlir::runtime::AclnnBackend backend;
  if (auto err = backend.generate(cfg)) {
    errs() << "aclnn-backend: " << err << "\n";
    return 1;
  }

  outs() << "aclnn-backend: wrote " << OutputFile << "\n";
  return 0;
}
