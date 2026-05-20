#include "Runtime/Artifact/RuntimeSessionRequestBuilder.h"

#include "Runtime/ArtifactCompiler.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

namespace mlir::runtime {
namespace {

std::string defaultKernelName(llvm::StringRef kernelSource) {
  if (!kernelSource.empty())
    return llvm::sys::path::stem(kernelSource).str();
  return "kernel";
}

} // namespace

llvm::Expected<KernelArtifact>
prepareRuntimeSessionArtifact(const RuntimeSessionArtifactRequest &request) {
  const bool hasArtifactRoot = !request.artifactRoot.empty();
  const bool hasKernelSource = !request.kernelSource.empty();
  if (hasArtifactRoot == hasKernelSource) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "provide exactly one of --artifact-root or --kernel");
  }

  if (hasArtifactRoot)
    return loadRuntimeSessionArtifactFromRoot(request.artifactRoot);

  KernelKind kernelKind = KernelKind::Vec;
  switch (request.kernelKind) {
  case KernelKind::Vec:
  case KernelKind::Cube:
  case KernelKind::Mix:
    kernelKind = request.kernelKind;
    break;
  default:
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "unsupported kernel kind: %d",
        static_cast<int>(request.kernelKind));
  }

  std::string effectiveKernelName = request.kernelName;
  if (effectiveKernelName.empty())
    effectiveKernelName = defaultKernelName(request.kernelSource);

  ArtifactCompileRequest compileRequest;
  compileRequest.kernelSource = request.kernelSource;
  compileRequest.kernelName = effectiveKernelName;
  compileRequest.kernelKind = kernelKind;
  compileRequest.socVersion = request.socVersion;
  compileRequest.outputDir = request.outputDir;
  if (request.cannMlirPath)
    compileRequest.cannMlirPath = request.cannMlirPath;
  if (request.npyDir)
    compileRequest.npyDir = request.npyDir;

  ArtifactCompiler compiler;
  return compiler.compile(compileRequest);
}

} // namespace mlir::runtime
