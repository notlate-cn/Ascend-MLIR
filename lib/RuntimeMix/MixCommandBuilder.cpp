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

} // namespace

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
  return {"/bin/bash", "-lc",
          shellQuote(getBishengPath()) + " -E -includestdio.h -x cce -O3 "
          "--cce-aicore-lang -std=c++17 -DTILING_KEY_VAR=0 "
          "-I " + shellQuote(getTikcppRoot() + "/tikcfw") + " "
          "-I " + shellQuote(getTikcppRoot() + "/tikcfw/interface") + " "
          "-I " + shellQuote(getTikcppRoot() + "/tikcfw/impl") + " "
          "-D__CHECK_FEATURE_AT_PRECOMPILE " + shellQuote(src.str()) + " > " +
          shellQuote(outputPath.str())};
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

} // namespace mlir::runtime
