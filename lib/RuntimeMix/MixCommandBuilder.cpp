#include "RuntimeMix/MixCommandBuilder.h"
#include <cstdlib>
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace mlir::runtime {

namespace {

static std::string getAscendHome() {
  if (const char *home = std::getenv("ASCEND_HOME_PATH"))
    return home;
  if (const char *home = std::getenv("ASCEND_TOOLKIT_HOME"))
    return home;
  if (const char *userHome = std::getenv("HOME")) {
    std::string latest = std::string(userHome) + "/Ascend/latest";
    if (llvm::sys::fs::exists(latest))
      return latest;
    std::string toolkitLatest =
        std::string(userHome) + "/Ascend/ascend-toolkit/latest";
    if (llvm::sys::fs::exists(toolkitLatest))
      return toolkitLatest;
  }
  return "/usr/local/Ascend/ascend-toolkit/latest";
}

struct MixToolkitPaths {
  std::string root;
  std::string bisheng;
  std::string lld;
  std::string packScript;
  std::string packTool;
  std::string tikcpp;
  std::string includeDir;
};

static MixToolkitPaths getToolkitPaths() {
  std::string ascendHome = getAscendHome();
  std::string root = ascendHome + "/toolkit/tools";
  std::string tikcpp = ascendHome + "/aarch64-linux/tikcpp";
  if (!llvm::sys::fs::exists(tikcpp))
    tikcpp = root + "/tikcpp";
  return {
      root,
      root + "/ccec_compiler/bin/bisheng",
      root + "/ccec_compiler/bin/ld.lld",
      ascendHome + "/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/ascendc_pack_kernel.sh",
      ascendHome + "/bin/ascendc_pack_kernel",
      tikcpp,
      ascendHome + "/aarch64-linux/include",
  };
}

static std::string getBishengPath() {
  return getToolkitPaths().bisheng;
}

static std::string getLldPath() {
  return getToolkitPaths().lld;
}

static std::string getTikcppRoot() {
  return getToolkitPaths().tikcpp;
}

static std::string getPackScriptPath() {
  return getToolkitPaths().packScript;
}

static std::string getPackToolPath() {
  return getToolkitPaths().packTool;
}

static std::string getMergeObjScriptPath() {
  return getAscendHome() +
         "/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/merge_obj.sh";
}

static std::string getMergeMixObjScriptPath() {
  return getAscendHome() +
         "/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/merge_mix_obj.sh";
}

static std::string getHostCxxPath() {
  const char *candidates[] = {
      "/usr/bin/c++",
      "/usr/bin/clang++",
      "/bin/c++",
      "/bin/clang++",
  };
  for (const char *candidate : candidates) {
    if (llvm::sys::fs::exists(candidate))
      return candidate;
  }
  return "/usr/bin/c++";
}

static std::string getAclIncludeDir() {
  auto paths = getToolkitPaths();
  if (llvm::sys::fs::exists(paths.includeDir))
    return paths.includeDir;
  std::string alt = getAscendHome() + "/include";
  if (llvm::sys::fs::exists(alt))
    return alt;
  std::string armAlt = getAscendHome() + "/arm64-linux/include";
  if (llvm::sys::fs::exists(armAlt))
    return armAlt;
  return paths.includeDir;
}

static std::string getAscRoot() {
  return getAscendHome() + "/aarch64-linux/asc";
}

static std::string getVersionHeader() {
  return getAscendHome() + "/include/version/asc_devkit_version.h";
}

static std::string shellQuote(llvm::StringRef value) {
  std::string quoted = "'";
  for (char c : value) {
    if (c == '\'')
      quoted += "'\\''";
    else
      quoted.push_back(c);
  }
  quoted.push_back('\'');
  return quoted;
}

static std::string getArchForCore(MixCoreType coreType) {
  switch (coreType) {
  case MixCoreType::AIC:
    return "dav-c220-cube";
  case MixCoreType::AIV:
    return "dav-c220-vec";
  }
  return "dav-c220-vec";
}

static llvm::ArrayRef<std::string>
getDefinesForCore(const MixAnalyzedKernel &info, MixCoreType coreType) {
  switch (coreType) {
  case MixCoreType::AIC:
    return info.aicDefines;
  case MixCoreType::AIV:
    return info.aivDefines;
  }
  return info.aivDefines;
}

static void appendDefines(std::vector<std::string> &args,
                          llvm::ArrayRef<std::string> defs) {
  for (const std::string &def : defs)
    args.push_back("-D" + def);
}

static void appendTikcppIncludes(std::vector<std::string> &args) {
  std::string tikcpp = getTikcppRoot();
  args.push_back("-I");
  args.push_back(tikcpp + "/tikcfw");
  args.push_back("-I");
  args.push_back(tikcpp + "/tikcfw/impl");
  args.push_back("-I");
  args.push_back(tikcpp + "/tikcfw/include");
  args.push_back("-I");
  args.push_back(tikcpp + "/tikcfw/interface");
}

static void appendAscIncludes(std::vector<std::string> &args) {
  const std::string asc = getAscRoot();
  const char *suffixes[] = {
      "/impl/adv_api",          "/impl/basic_api",
      "/impl/c_api",            "/impl/basic_api/reg_compute",
      "/impl/simt_api",         "/impl/utils",
      "",                       "/include",
      "/include/adv_api",       "/include/basic_api",
      "/include/aicpu_api",     "/include/c_api",
      "/include/basic_api/reg_compute",
      "/include/simt_api",      "/include/utils",
  };
  for (const char *suffix : suffixes) {
    args.push_back("-I");
    args.push_back(asc + suffix);
  }
}

} // namespace

std::string renderCommandForDebug(llvm::ArrayRef<std::string> args) {
  std::string out;
  for (size_t i = 0; i < args.size(); ++i) {
    if (i)
      out.push_back(' ');
    out += shellQuote(args[i]);
  }
  return out;
}

std::string renderCommandForCompileCommands(llvm::ArrayRef<std::string> args) {
  std::string out;
  for (size_t i = 0; i < args.size(); ++i) {
    if (i)
      out.push_back(' ');
    const llvm::StringRef arg = args[i];
    if (arg.contains(' ') || arg.contains('\t') || arg.contains('\'') ||
        arg.contains('"'))
      out += shellQuote(arg);
    else
      out += arg;
  }
  return out;
}

std::vector<std::string>
buildBishengCommand(const MixAnalyzedKernel &info, llvm::StringRef src,
                    llvm::StringRef obj, MixCoreType coreType) {
  std::vector<std::string> args;
  args.push_back(getBishengPath());
  args.insert(args.end(), info.commonFlags.begin(), info.commonFlags.end());
  args.push_back("--cce-aicore-arch=" + getArchForCore(coreType));
  appendTikcppIncludes(args);
  appendDefines(args, getDefinesForCore(info, coreType));
  args.push_back(src.str());
  args.push_back("-o");
  args.push_back(obj.str());
  return args;
}

std::vector<std::string>
buildPreprocessedDeviceCompileCommand(llvm::StringRef src,
                                      llvm::StringRef obj,
                                      MixCoreType coreType) {
  std::vector<std::string> args;
  args.push_back(getBishengPath());
  args.push_back("-DHAVE_TILING");
  args.push_back("-DHAVE_WORKSPACE");
  args.push_back("-DTILING_KEY_VAR=0");
  appendAscIncludes(args);
  appendTikcppIncludes(args);
  args.push_back("-g");
  args.push_back("--cce-disable-kernel-global-attr-check");
  args.push_back("--cce-aicore-arch=" + getArchForCore(coreType));
  args.push_back("--cce-aicore-only");
  args.push_back("--cce-auto-sync");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-stack-size=0x8000");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-function-stack-size=0x8000");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-record-overflow=true");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-addr-transform");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-dcci-insert-for-scalar=false");
  args.push_back("-O3");
  args.push_back("-std=c++17");
  args.push_back("--cce-aicore-lang");
  args.push_back("-include");
  args.push_back(getVersionHeader());
  args.push_back("-o");
  args.push_back(obj.str());
  args.push_back("-c");
  args.push_back(src.str());
  return args;
}

std::vector<std::string> buildLldRelocCommand(llvm::StringRef inputObj,
                                              llvm::StringRef outputObj) {
  return {getLldPath(), "-r", "-m", "aicorelinux", "-Ttext=0",
          inputObj.str(), "-o", outputObj.str()};
}

std::vector<std::string> buildLldMergeCommand(llvm::StringRef aicObj,
                                              llvm::StringRef aivObj,
                                              llvm::StringRef outputObj) {
  return {getLldPath(), "-r", "-m", "aicorelinux", "-Ttext=0",
          aicObj.str(), aivObj.str(), "-o", outputObj.str()};
}

std::vector<std::string> buildDeviceMergeCommand(llvm::StringRef inputObj,
                                                 llvm::StringRef outputDir,
                                                 llvm::StringRef outputName,
                                                 llvm::StringRef flagPath,
                                                 llvm::StringRef buildType) {
  return {"/bin/bash", getMergeObjScriptPath(), "-l", getLldPath(), "-o",
          outputDir.str(), "-t", buildType.str(), "-n", outputName.str(), "-f",
          flagPath.str(), inputObj.str()};
}

std::vector<std::string> buildMixFinalMergeCommand(llvm::StringRef aicDir,
                                                   llvm::StringRef aivDir,
                                                   llvm::StringRef outputDir,
                                                   llvm::StringRef buildType) {
  return {"/bin/bash", getMergeMixObjScriptPath(), "-l", getLldPath(), "-o",
          outputDir.str(), "--aic-dir", aicDir.str(), "--aiv-dir", aivDir.str(),
          "--build-type", buildType.str()};
}

std::vector<std::string>
buildPreprocessCommand(llvm::StringRef src, llvm::StringRef outputPath) {
  std::vector<std::string> args;
  args.push_back(getBishengPath());
  args.push_back("-DHAVE_TILING");
  args.push_back("-DHAVE_WORKSPACE");
  args.push_back("-DTILING_KEY_VAR=0");
  args.push_back("-D__CHECK_FEATURE_AT_PRECOMPILE");
  appendAscIncludes(args);
  appendTikcppIncludes(args);
  args.push_back("-g");
  args.push_back("-E");
  args.push_back("-includestdio.h");
  args.push_back("--cce-aicore-arch=dav-c220-vec");
  args.push_back("--cce-aicore-only");
  args.push_back("--cce-auto-sync");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-stack-size=0x8000");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-function-stack-size=0x8000");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-record-overflow=true");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-addr-transform");
  args.push_back("-mllvm");
  args.push_back("-cce-aicore-dcci-insert-for-scalar=false");
  args.push_back("-O3");
  args.push_back("-std=c++17");
  args.push_back("--cce-aicore-lang");
  args.push_back("-include");
  args.push_back(getVersionHeader());
  args.push_back("--cce-disable-kernel-global-attr-check");
  args.push_back("-o");
  args.push_back(outputPath.str());
  args.push_back("-c");
  args.push_back(src.str());
  return args;
}

std::vector<std::string>
buildExtractHostStubCommand(llvm::StringRef preprocessedPath,
                            llvm::StringRef dstDir,
                            llvm::StringRef headerDir,
                            llvm::ArrayRef<std::string> aivObjects,
                            llvm::ArrayRef<std::string> aicObjects,
                            llvm::StringRef compileCommandsPath,
                            llvm::StringRef buildMode,
                            llvm::StringRef runMode) {
  std::vector<std::string> args = {
      getAscendHome() +
          "/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/"
          "extract_host_stub.py",
      preprocessedPath.str(),
      "--dynamic-mode",
      "-d",
      dstDir.str(),
      "-hd",
      headerDir.str(),
      "--compile-commands",
      compileCommandsPath.str(),
      "--generate-definition",
      "--build-mode",
      buildMode.str(),
      "--run-mode",
      runMode.str(),
  };
  args.push_back("--aiv-o");
  args.insert(args.end(), aivObjects.begin(), aivObjects.end());
  args.push_back("--aic-o");
  args.insert(args.end(), aicObjects.begin(), aicObjects.end());
  return args;
}

std::vector<std::string>
buildUpdateHostStubCommand(llvm::StringRef codeDir, llvm::StringRef objDir,
                           llvm::StringRef lowerSocVersion,
                           llvm::StringRef targetName) {
  return {getAscendHome() +
              "/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/"
              "update_host_stub.py",
          codeDir.str(), objDir.str(), lowerSocVersion.str(), targetName.str()};
}

std::vector<std::string>
buildHostBishengCommand(llvm::StringRef src, llvm::StringRef obj,
                        llvm::StringRef tripleChevronHeader) {
  std::vector<std::string> args;
  args.push_back(getBishengPath());
  args.push_back("-DTILING_KEY_VAR=0");
  appendAscIncludes(args);
  args.push_back("-I");
  args.push_back(getTikcppRoot() + "/tikcfw");
  args.push_back("-I");
  args.push_back(getTikcppRoot() + "/tikcfw/interface");
  args.push_back("-I");
  args.push_back(getTikcppRoot() + "/tikcfw/impl");
  args.push_back("-g");
  args.push_back("-include");
  args.push_back(tripleChevronHeader.str());
  args.push_back("-O3");
  args.push_back("-std=c++17");
  args.push_back("--cce-aicore-lang");
  args.push_back("-include");
  args.push_back(getVersionHeader());
  args.push_back("-fPIC");
  args.push_back("--cce-host-only");
  args.push_back("-fcce-kernel-launch-custom");
  args.push_back("-DONE_CORE_DUMP_SIZE=1048576");
  args.push_back("-o");
  args.push_back(obj.str());
  args.push_back("-c");
  args.push_back(src.str());
  return args;
}

std::vector<std::string>
buildHostStubCompileCommand(llvm::StringRef source, llvm::StringRef object,
                            llvm::StringRef headerDir) {
  return {getHostCxxPath(), "-std=c++17", "-fPIC", "-c", source.str(), "-o",
          object.str(), "-I", headerDir.str(), "-I", getAclIncludeDir()};
}

std::vector<std::string> buildPackCommand(llvm::StringRef hostStubObject,
                                          llvm::StringRef addDir) {
  return {"/bin/bash", getPackScriptPath(), "--pack_tool", getPackToolPath(),
          "--elf_in", hostStubObject.str(), "--add_dir", addDir.str()};
}

std::vector<std::string>
buildRecompileBinaryCommand(llvm::StringRef rootDir, llvm::StringRef targetName,
                            llvm::StringRef addDir) {
  return {
      "python3",
      getAscendHome() +
          "/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/"
          "recompile_binary.py",
      "--root-dir",
      rootDir.str(),
      "--target-name",
      targetName.str(),
      "--add-dir",
      addDir.str(),
  };
}

std::vector<std::string> buildHostSharedLinkCommand(llvm::StringRef hostStubObject,
                                                    llvm::StringRef outputSo,
                                                    llvm::StringRef socVersion) {
  std::string ascendHome = getAscendHome();
  return {getHostCxxPath(),
          "-fPIC",
          "-shared",
          "-Wl,-rpath-link," + ascendHome + "/lib64",
          "-Wl,-rpath-link," + ascendHome + "/tools/simulator/" + socVersion.str() + "/lib",
          "-Wl,-rpath-link," + ascendHome + "/tools/simulator/dav_3002/lib",
          "-o",
          outputSo.str(),
          hostStubObject.str(),
          "-L" + ascendHome + "/tools/simulator/" + socVersion.str() + "/lib",
          "-L" + ascendHome + "/tools/simulator/dav_3002/lib",
          "-L" + ascendHome + "/lib64",
          ascendHome + "/lib64/libascendc_runtime.a",
          "-lascendcl",
          "-ltiling_api",
          "-lregister",
          "-lplatform",
          "-lascendalog",
          "-lunified_dlog",
          "-ldl",
          "-lruntime_camodel",
          "-lnpu_drv",
          "-lstars",
          "-lmodel_top",
          "-lerror_manager",
          "-lprofapi",
          "-lge_common_base",
          "-lmmpa",
          "-lascend_dump",
          "-lc_sec"};
}

std::vector<std::string>
buildHostRunnerCompileCommand(llvm::StringRef workDir,
                              llvm::StringRef launcherDir,
                              llvm::StringRef outIncludeDir,
                              llvm::StringRef runnerMainPath,
                              llvm::StringRef runnerTilingPath,
                              llvm::StringRef runnerBinaryPath,
                              llvm::StringRef kernelSoPath,
                              llvm::StringRef runnerLib64,
                              llvm::StringRef runnerSimLibDir,
                              llvm::StringRef davSimLibDir,
                              llvm::StringRef socVersion) {
  const std::string ascendHome = getAscendHome();

  return {
      getHostCxxPath(),
      "-g",
      "-O2",
      "-std=c++17",
      "-DSOC_VERSION=\"" + socVersion.str() + "\"",
      "-D_GLIBCXX_USE_CXX11_ABI=0",
      "-Wall",
      "-Werror",
      "-fPIC",
      "-O0",
      "-fvisibility-inlines-hidden",
      "-fstack-protector-all",
      "-I" + workDir.str(),
      "-I" + launcherDir.str(),
      "-I" + outIncludeDir.str(),
      "-I" + ascendHome + "/include",
      "-I" + ascendHome + "/aarch64-linux/tikcpp/tikcfw",
      runnerMainPath.str(),
      runnerTilingPath.str(),
      "-Wl,-rpath-link," + runnerLib64.str(),
      "-Wl,-rpath-link," + runnerSimLibDir.str(),
      "-Wl,-rpath-link," + davSimLibDir.str(),
      "-pie",
      "-Wl,-z,relro",
      "-Wl,-z,now",
      "-Wl,-z,noexecstack",
      "-L" + runnerSimLibDir.str(),
      "-L" + davSimLibDir.str(),
      "-L" + runnerLib64.str(),
      "-o",
      runnerBinaryPath.str(),
      kernelSoPath.str(),
      "-ltiling_api",
      "-lregister",
      "-lplatform",
      "-lascendalog",
      "-lunified_dlog",
      "-ldl",
      "-lruntime_camodel",
      "-lnpu_drv",
      "-lstars",
      "-lmodel_top",
      "-lascendcl",
      "-lerror_manager",
      "-lprofapi",
      "-lge_common_base",
      "-lmmpa",
      "-lascend_dump",
      "-lc_sec",
      "-lunified_dlog",
      "-ldl",
      "-lmmpa",
      "-ldl",
      "-lascend_dump",
      "-lc_sec",
  };
}

} // namespace mlir::runtime
