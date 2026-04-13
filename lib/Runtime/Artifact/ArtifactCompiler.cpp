#include "Runtime/ArtifactCompiler.h"

#include "Runtime/Compiler.h"
#include "Runtime/MixDirectBackend.h"
#include "Runtime/PathUtils.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::runtime {

namespace {

static std::string defaultCompilerArch(KernelKind kind) {
  switch (kind) {
  case KernelKind::Cube:
    return "dav-c220-cube";
  case KernelKind::Vec:
  case KernelKind::Mix:
    return "dav-c220-vec";
  }
  return "dav-c220-vec";
}

static llvm::StringRef kernelKindName(KernelKind kind) {
  switch (kind) {
  case KernelKind::Vec:
    return "vec";
  case KernelKind::Cube:
    return "cube";
  case KernelKind::Mix:
    return "mix";
  }
  return "vec";
}

static llvm::Expected<std::string>
writeArtifactManifest(llvm::StringRef artifactRoot, llvm::StringRef kernelName,
                      KernelKind kernelKind, llvm::StringRef socVersion,
                      llvm::StringRef deviceBinaryPath) {
  llvm::SmallString<256> manifestDir(artifactRoot);
  llvm::sys::path::append(manifestDir, "out");
  if (auto ec = llvm::sys::fs::create_directories(manifestDir))
    return llvm::createStringError(ec,
                                   "cannot create artifact manifest directory");

  llvm::SmallString<256> manifestPath(manifestDir);
  llvm::sys::path::append(manifestPath, "manifest.txt");
  std::error_code ec;
  llvm::raw_fd_ostream os(manifestPath, ec);
  if (ec)
    return llvm::createStringError(ec, "cannot write artifact manifest");

  llvm::SmallString<256> relativeBinary(deviceBinaryPath);
  llvm::sys::path::remove_dots(relativeBinary, /*remove_dot_dot=*/true);
  if (llvm::sys::path::is_absolute(relativeBinary)) {
    llvm::StringRef relativeToRoot = relativeBinary;
    if (relativeToRoot.consume_front(artifactRoot))
      relativeBinary = relativeToRoot.ltrim("/").str();
  }

  os << "kernel_name=" << kernelName << "\n";
  os << "soc_version=" << socVersion << "\n";
  os << "kernel_kind=" << kernelKindName(kernelKind) << "\n";
  os << "device_binary_path=" << relativeBinary << "\n";
  os << "manifest_path=out/manifest.txt\n";
  os.flush();

  return manifestPath.str().str();
}

} // namespace

MixResourceType inferMixResourceTypeFromKernelKind(KernelKind kind) {
  switch (kind) {
  case KernelKind::Mix:
    return MixResourceType::Mix1C1V;
  case KernelKind::Vec:
  case KernelKind::Cube:
    return MixResourceType::Unknown;
  }
  return MixResourceType::Unknown;
}

llvm::Expected<KernelArtifact>
ArtifactCompiler::compile(const ArtifactCompileRequest &req) const {
  if (req.kernelSource.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "kernel source is required");
  if (req.kernelName.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "kernel name is required");
  if (req.outputDir.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "output directory is required");

  const std::string resolvedSoc =
      resolveSocVersion(req.socVersion, "Ascend910B1");

  if (req.kernelKind == KernelKind::Mix) {
    // The mix backend compiles both cube and vec variants internally, so the
    // compatibility arch flag does not change this path.
    MixDirectCompileConfig cfg;
    cfg.kernelSrc = req.kernelSource;
    cfg.kernelName = req.kernelName;
    cfg.socVersion = resolvedSoc;
    cfg.outputDir = req.outputDir;
    cfg.cannMlirPath = req.cannMlirPath;
    cfg.npyDir = req.npyDir;

    MixDirectBackend backend;
    auto mixOr = backend.compile(cfg);
    if (!mixOr)
      return mixOr.takeError();
    KernelArtifact artifact = normalizeMixArtifact(*mixOr, req.kernelKind,
                                inferMixResourceTypeFromKernelKind(
                                    req.kernelKind));
    artifact.socVersion = resolvedSoc;
    return artifact;
  }

  CompilerConfig cfg;
  cfg.soc_version = resolvedSoc;
  cfg.arch = req.arch.empty() ? defaultCompilerArch(req.kernelKind)
                              : req.arch;
  cfg.opt_level = req.optLevel;
  cfg.kernel_type = req.kernelKind == KernelKind::Cube ? "cube" : "vec";
  cfg.verbose = req.verbose;

  Compiler compiler(cfg);
  auto binaryOr = compiler.Compile(req.kernelSource, req.outputDir,
                                   req.kernelName);
  if (!binaryOr)
    return binaryOr.takeError();
  KernelArtifact artifact = normalizeCompiledArtifact(
      *binaryOr, req.kernelName, req.kernelKind, resolvedSoc, req.outputDir);
  auto manifestPathOr = writeArtifactManifest(req.outputDir, req.kernelName,
                                              req.kernelKind, resolvedSoc,
                                              artifact.deviceBinaryPath);
  if (!manifestPathOr)
    return manifestPathOr.takeError();
  artifact.manifestPath = *manifestPathOr;
  return artifact;
}

} // namespace mlir::runtime
