#include "Runtime/Artifact/VecCubeArtifactBackend.h"

#include "Runtime/PathUtils.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace mlir::runtime {

namespace {

static std::string defaultCompilerArch(KernelKind kind) {
  switch (kind) {
  case KernelKind::Cube:
    return "dav-c220-cube";
  case KernelKind::Vec:
    return "dav-c220-vec";
  case KernelKind::Mix:
    break;
  }
  return "dav-c220-vec";
}

static llvm::Error prepareVecCubeCompileOutputDir(llvm::StringRef outputDir) {
  if (outputDir.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "output directory is required");
  if (auto ec = llvm::sys::fs::create_directories(outputDir))
    return llvm::createStringError(ec, "cannot create output directory: %s",
                                   outputDir.str().c_str());
  return llvm::Error::success();
}

static llvm::Error runProcess(const std::vector<std::string> &args) {
  std::vector<llvm::StringRef> argv;
  argv.reserve(args.size());
  for (const std::string &arg : args)
    argv.push_back(arg);

  std::string errMsg;
  std::optional<llvm::StringRef> redirects[3];
  int ret = llvm::sys::ExecuteAndWait(argv[0], argv, std::nullopt, redirects,
                                      300, 0, &errMsg);
  if (ret != 0) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Process failed (exit %d): %s", ret,
                                   errMsg.c_str());
  }
  return llvm::Error::success();
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

  auto kernelKindName = [](KernelKind kind) -> llvm::StringRef {
    switch (kind) {
    case KernelKind::Vec:
      return "vec";
    case KernelKind::Cube:
      return "cube";
    case KernelKind::Mix:
      return "mix";
    }
    return "vec";
  };

  os << "kernel_name=" << kernelName << "\n";
  os << "soc_version=" << socVersion << "\n";
  os << "kernel_kind=" << kernelKindName(kernelKind) << "\n";
  os << "device_binary_path=" << relativeBinary << "\n";
  os << "manifest_path=out/manifest.txt\n";
  os.flush();

  return manifestPath.str().str();
}

static KernelArtifact normalizeVecCubeArtifact(llvm::StringRef binaryPath,
                                               llvm::StringRef kernelName,
                                               KernelKind kind,
                                               llvm::StringRef socVersion,
                                               llvm::StringRef artifactRoot) {
  KernelArtifact artifact;
  artifact.kernelName = kernelName.str();
  artifact.kernelKind = kind;
  artifact.mixResourceType = MixResourceType::Unknown;
  artifact.socVersion = socVersion.str();
  artifact.artifactRoot = artifactRoot.str();
  artifact.deviceBinaryPath = binaryPath.str();
  return artifact;
}

static llvm::Expected<std::string>
compileObjectAndLink(llvm::StringRef srcFile, llvm::StringRef outputDir,
                    llvm::StringRef kernelName, llvm::StringRef arch,
                    int optLevel, bool verbose) {
  auto ascendHomeOr = requireAscendHome();
  if (!ascendHomeOr)
    return ascendHomeOr.takeError();
  std::string ascendHome = *ascendHomeOr;
  std::string bisheng = ascendHome + "/toolkit/tools/ccec_compiler/bin/bisheng";
  std::string lld = ascendHome + "/toolkit/tools/ccec_compiler/bin/ld.lld";
  std::string tikcpp = findAscendTikcppDir(ascendHome);
  std::string binFile = (outputDir.str() + "/" + kernelName.str() + ".bin");

  llvm::SmallString<256> srcPath(srcFile);
  llvm::sys::fs::make_absolute(srcPath);
  if (!llvm::sys::fs::exists(srcPath)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Source file not found: %s",
                                   srcPath.c_str());
  }
  std::string srcDir = llvm::sys::path::parent_path(srcPath).str();
  std::string srcName = llvm::sys::path::filename(srcPath).str();

  auto makeCompileCmd = [&](llvm::StringRef targetArch,
                            llvm::StringRef objFile) -> std::string {
    std::string cmd = bisheng + " -c -x cce";
    cmd += " -O" + std::to_string(optLevel);
    cmd += " " + srcName;
    cmd += " --cce-aicore-arch=" + targetArch.str();
    cmd += " --cce-aicore-only -std=c++17";
    cmd += " --cce-disable-kernel-global-attr-check";
    cmd += " -mllvm -cce-aicore-stack-size=0x8000";
    cmd += " -mllvm -cce-aicore-function-stack-size=0x8000";
    cmd += " -mllvm -cce-aicore-dcci-insert-for-scalar=false";
    cmd += " -I " + tikcpp + "/tikcfw";
    cmd += " -I " + tikcpp + "/tikcfw/impl";
    cmd += " -I " + tikcpp + "/tikcfw/include";
    cmd += " -I " + tikcpp + "/tikcfw/interface";
    cmd += " -DASCENDC_DUMP=0 -D__NPU_TILING__ -DTILING_KEY_VAR=0";
    cmd += " -o ";
    cmd += objFile;
    return cmd;
  };

  auto runCmd = [&](const std::string &cmd) -> llvm::Error {
    std::string wrapped = "cd " + srcDir + " && " + cmd;
    if (verbose)
      llvm::errs() << "[compiler] " << wrapped << "\n";
    return runProcess({"/bin/sh", "-c", wrapped});
  };

  std::string objFile = (outputDir.str() + "/" + kernelName.str() + ".o");
  if (auto err = runCmd(makeCompileCmd(arch, objFile)))
    return err;

  std::string linkCmd = lld + " -m aicorelinux -Ttext=0 " + objFile +
                        " -static -o " + binFile;
  if (verbose)
    llvm::errs() << "[compiler] link: " << linkCmd << "\n";
  if (auto err = runProcess({"/bin/sh", "-c", linkCmd}))
    return err;

  return binFile;
}

} // namespace

llvm::Expected<KernelArtifact>
VecCubeArtifactBackend::compile(const ArtifactCompileRequest &req,
                                llvm::StringRef resolvedSoc) const {
  if (req.kernelSource.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "kernel source is required");
  if (req.kernelName.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "kernel name is required");
  if (req.outputDir.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "output directory is required");

  if (req.kernelKind != KernelKind::Vec && req.kernelKind != KernelKind::Cube) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported kernel kind for vec/cube backend");
  }

  const std::string socVersion = resolvedSoc.str();
  ::setenv("SOC_VERSION", socVersion.c_str(), 1);

  if (auto err = prepareVecCubeCompileOutputDir(req.outputDir))
    return std::move(err);

  const std::string resolvedArch =
      req.arch.empty() ? defaultCompilerArch(req.kernelKind) : req.arch;
  auto binaryOr = compileObjectAndLink(req.kernelSource, req.outputDir,
                                       req.kernelName, resolvedArch,
                                       req.optLevel, req.verbose);
  if (!binaryOr)
    return binaryOr.takeError();

  KernelArtifact artifact = normalizeVecCubeArtifact(
      *binaryOr, req.kernelName, req.kernelKind, resolvedSoc, req.outputDir);
  auto manifestPathOr = writeArtifactManifest(req.outputDir, req.kernelName,
                                              req.kernelKind, resolvedSoc,
                                              artifact.deviceBinaryPath);
  if (!manifestPathOr)
    return manifestPathOr.takeError();

  artifact.manifestPath = *manifestPathOr;
  artifact.mixResourceType = MixResourceType::Unknown;
  return artifact;
}

} // namespace mlir::runtime
