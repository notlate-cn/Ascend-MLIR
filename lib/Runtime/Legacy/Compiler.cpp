// lib/Runtime/Compiler.cpp
#include "Runtime/Compiler.h"
#include "Runtime/PathUtils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <optional>

namespace mlir::runtime {

Compiler::Compiler(const Config& cfg) : cfg_(cfg) {}

llvm::Error prepareCompileOutputDir(llvm::StringRef outputDir) {
  if (outputDir.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Output directory is required");
  if (auto ec = llvm::sys::fs::create_directories(outputDir))
    return llvm::createStringError(ec, "Cannot create output directory: %s",
                                   outputDir.str().c_str());
  return llvm::Error::success();
}

llvm::Error Compiler::RunProcess(const std::vector<std::string>& args) {
  std::vector<llvm::StringRef> argv;
  argv.reserve(args.size());
  for (auto& a : args) argv.push_back(a);

  std::string err_msg;
  std::optional<llvm::StringRef> redirects[3];

  int ret = llvm::sys::ExecuteAndWait(argv[0], argv, std::nullopt, redirects,
                                      300, 0, &err_msg);
  if (ret != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Process failed (exit %d): %s", ret,
                                   err_msg.c_str());
  return llvm::Error::success();
}

llvm::Expected<std::string> Compiler::Compile(const std::string& src_file,
                                              const std::string& output_dir,
                                              const std::string& kernel_name) {
  const std::string socVersion = resolveSocVersion(cfg_.soc_version, "Ascend910B1");
  ::setenv("SOC_VERSION", socVersion.c_str(), 1);

  if (auto err = prepareCompileOutputDir(output_dir))
    return std::move(err);

  auto ascendHomeOr = requireAscendHome();
  if (!ascendHomeOr)
    return ascendHomeOr.takeError();
  std::string ascend_home = *ascendHomeOr;
  std::string bisheng     = ascend_home + "/toolkit/tools/ccec_compiler/bin/bisheng";
  std::string lld         = ascend_home + "/toolkit/tools/ccec_compiler/bin/ld.lld";
  std::string tikcpp      = findAscendTikcppDir(ascend_home);
  std::string bin_file    = output_dir + "/" + kernel_name + ".bin";

  llvm::SmallString<256> src_path(src_file);
  llvm::sys::fs::make_absolute(src_path);
  if (!llvm::sys::fs::exists(src_path))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Source file not found: %s",
                                   src_path.c_str());
  std::string src_dir  = llvm::sys::path::parent_path(src_path).str();
  std::string src_name = llvm::sys::path::filename(src_path).str();

  auto makeCompileCmd = [&](const std::string& arch, const std::string& obj_file,
                            const std::string& extra_defs) -> std::string {
    return bisheng + " -c -x cce" + " -O" + std::to_string(cfg_.opt_level) +
           " " + src_name + " --cce-aicore-arch=" + arch +
           " --cce-aicore-only" + " -std=c++17" +
           " --cce-disable-kernel-global-attr-check" +
           " -mllvm -cce-aicore-stack-size=0x8000" +
           " -mllvm -cce-aicore-function-stack-size=0x8000" +
           " -mllvm -cce-aicore-dcci-insert-for-scalar=false" + " -I " +
           tikcpp + "/tikcfw" + " -I " + tikcpp + "/tikcfw/impl" + " -I " +
           tikcpp + "/tikcfw/include" + " -I " + tikcpp +
           "/tikcfw/interface" + " -DASCENDC_DUMP=0" + " -D__NPU_TILING__" +
           " -DTILING_KEY_VAR=0" + extra_defs + " -o " + obj_file;
  };

  auto runCmd = [&](const std::string& cmd) -> llvm::Error {
    std::string wrapped = "cd " + src_dir + " && " + cmd;
    if (cfg_.verbose)
      llvm::errs() << "[compiler] " << wrapped << "\n";
    return RunProcess({"/bin/sh", "-c", wrapped});
  };

  std::string obj_files;
  if (cfg_.kernel_type == "mix") {
    std::string aic_obj = output_dir + "/" + kernel_name + "_aic.o";
    std::string aiv_obj = output_dir + "/" + kernel_name + "_aiv.o";

    std::string mix_base = " -D__MIX_CORE_MACRO__=1";
    std::string aic_def =
        mix_base + " -Dauto_gen_" + kernel_name + "_kernel=" + kernel_name +
        "_0_mix_aic";
    std::string aiv_def =
        mix_base + " -Dauto_gen_" + kernel_name + "_kernel=" + kernel_name +
        "_0_mix_aiv";

    if (auto err = runCmd(makeCompileCmd("dav-c220-cube", aic_obj, aic_def)))
      return std::move(err);
    if (auto err = runCmd(makeCompileCmd("dav-c220-vec", aiv_obj, aiv_def)))
      return std::move(err);

    obj_files = aic_obj + " " + aiv_obj;
  } else {
    std::string obj_file = output_dir + "/" + kernel_name + ".o";
    if (auto err = runCmd(makeCompileCmd(cfg_.arch, obj_file, "")))
      return std::move(err);
    obj_files = obj_file;
  }

  std::string link_cmd =
      lld + " -m aicorelinux -Ttext=0 " + obj_files + " -static -o " + bin_file;
  if (cfg_.verbose)
    llvm::errs() << "[compiler] link: " << link_cmd << "\n";
  if (auto err = RunProcess({"/bin/sh", "-c", link_cmd}))
    return std::move(err);

  return bin_file;
}

KernelArtifact normalizeCompiledArtifact(llvm::StringRef binaryPath,
                                         llvm::StringRef kernelName,
                                         KernelKind kind,
                                         llvm::StringRef socVersion,
                                         llvm::StringRef artifactRoot) {
  KernelArtifact artifact;
  artifact.kernelName = kernelName.str();
  artifact.kernelKind = kind;
  artifact.mixResourceType =
      kind == KernelKind::Mix ? MixResourceType::Mix1C1V
                              : MixResourceType::Unknown;
  artifact.socVersion = socVersion.str();
  artifact.artifactRoot = artifactRoot.str();
  artifact.deviceBinaryPath = binaryPath.str();
  return artifact;
}

} // namespace mlir::runtime
