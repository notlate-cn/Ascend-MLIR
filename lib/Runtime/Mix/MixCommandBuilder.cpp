#include "Runtime/MixCommandBuilder.h"
#include "Runtime/PathUtils.h"
#include <cstdlib>
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::runtime {

namespace {

struct MixToolkitPaths {
  std::string root;
  std::string bisheng;
  std::string lld;
  std::string packScript;
  std::string packTool;
  std::string tikcpp;
  std::string ascIncludeDir;
  std::string includeDir;
};

static MixToolkitPaths getToolkitPaths() {
  std::string ascendHome = findAscendHome();
  std::string root = ascendHome + "/toolkit/tools";
  std::string tikcpp = findAscendTikcppDir(ascendHome);
  return {
      root,
      root + "/ccec_compiler/bin/bisheng",
      root + "/ccec_compiler/bin/ld.lld",
      ascendHome + "/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/ascendc_pack_kernel.sh",
      ascendHome + "/bin/ascendc_pack_kernel",
      tikcpp,
      findAscendAscDir(ascendHome) + "/include",
      findAscendIncludeDir(ascendHome),
  };
}

static std::string getBishengPath() {
  return getToolkitPaths().bisheng;
}

static std::string getLldPath() {
  return getToolkitPaths().lld;
}

static std::string getPackScriptPath() {
  return getToolkitPaths().packScript;
}

static std::string getPackToolPath() {
  return getToolkitPaths().packTool;
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
  return getToolkitPaths().includeDir;
}

static std::string getMixTilingHelperPath() {
  if (const char *configured = std::getenv("AFIR_MIX_TILING_HELPER"))
    if (*configured)
      return configured;
  return "mix-tiling-helper";
}

static std::string getVersionHeader() {
  return findAscendIncludeDir(findAscendHome()) + "/version/asc_devkit_version.h";
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

static llvm::StringRef getTilingHelperDTypeName(DType dtype) {
  switch (dtype) {
  case DType::F16:
    return "f16";
  case DType::BF16:
    return "bf16";
  case DType::F32:
    return "f32";
  default:
    return "unsupported";
  }
}

static std::string joinShape(llvm::ArrayRef<int64_t> shape) {
  std::string out;
  llvm::raw_string_ostream os(out);
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i)
      os << ",";
    os << shape[i];
  }
  os.flush();
  return out;
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

static void appendDefines(std::vector<std::string> &args,
                          llvm::ArrayRef<std::string> defs) {
  for (const std::string &def : defs)
    args.push_back("-D" + def);
}

static void appendTikcppIncludes(std::vector<std::string> &args) {
  MixToolkitPaths paths = getToolkitPaths();
  std::string tikcpp = paths.tikcpp;
  args.push_back("-I");
  args.push_back(tikcpp + "/tikcfw");
  args.push_back("-I");
  args.push_back(tikcpp + "/tikcfw/impl");
  args.push_back("-I");
  args.push_back(tikcpp + "/tikcfw/interface");
  args.push_back("-I");
  args.push_back(paths.ascIncludeDir);
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

std::vector<std::string>
buildPreprocessedDeviceCompileCommand(llvm::StringRef src,
                                      llvm::StringRef obj,
                                      MixCoreType coreType,
                                      llvm::ArrayRef<std::string> defs) {
  std::vector<std::string> args;
  args.push_back(getBishengPath());
  args.push_back("-c");
  args.push_back("-x");
  args.push_back("cce");
  args.push_back("-O3");
  args.push_back(src.str());
  args.push_back("--cce-aicore-arch=" + getArchForCore(coreType));
  args.push_back("--cce-aicore-only");
  args.push_back("-o");
  args.push_back(obj.str());
  args.push_back("-DHAVE_TILING");
  args.push_back("-DHAVE_WORKSPACE");
  args.push_back("-DTILING_KEY_VAR=0");
  appendDefines(args, defs);
  appendTikcppIncludes(args);
  args.push_back("--cce-disable-kernel-global-attr-check");
  args.push_back("--cce-auto-sync");
  args.push_back("-mllvm");
  args.push_back("-api-deps-filter");
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
  args.push_back("-std=c++17");
  args.push_back("-include");
  args.push_back(getVersionHeader());
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
                                                    llvm::StringRef socVersion,
                                                    llvm::StringRef deviceLibDir) {
  (void)socVersion;
  std::string ascendHome = findAscendHome();
  std::string runnerLib64 = findAscendLib64Dir(ascendHome);
  std::vector<std::string> cmd = {getHostCxxPath(),
                                  "-fPIC",
                                  "-shared",
                                  "-Wl,-rpath-link," + runnerLib64};
  if (!deviceLibDir.empty())
    cmd.push_back("-Wl,-rpath-link," + deviceLibDir.str());
  cmd.push_back("-o");
  cmd.push_back(outputSo.str());
  cmd.push_back(hostStubObject.str());
  cmd.push_back("-L" + runnerLib64);
  if (!deviceLibDir.empty())
    cmd.push_back("-L" + deviceLibDir.str());
  cmd.push_back(runnerLib64 + "/libascendc_runtime.a");
  cmd.push_back("-lascendcl");
  cmd.push_back("-ltiling_api");
  cmd.push_back("-lregister");
  cmd.push_back("-lplatform");
  cmd.push_back("-lascendalog");
  cmd.push_back("-lunified_dlog");
  cmd.push_back("-ldl");
  cmd.push_back("-lerror_manager");
  cmd.push_back("-lprofapi");
  cmd.push_back("-lge_common_base");
  cmd.push_back("-lmmpa");
  cmd.push_back("-lascend_dump");
  cmd.push_back("-lc_sec");
  return cmd;
}

std::vector<std::string>
buildMixTilingHelperCommand(llvm::StringRef kernelName,
                            llvm::StringRef socVersion,
                            llvm::ArrayRef<int64_t> inputAShape,
                            DType inputADType,
                            llvm::ArrayRef<int64_t> inputBShape,
                            DType inputBDType,
                            llvm::ArrayRef<int64_t> outputShape,
                            DType outputDType,
                            const std::optional<DType> &biasDType,
                            llvm::ArrayRef<int64_t> biasShape,
                            llvm::StringRef tilingOutputPath,
                            llvm::StringRef launchInfoOutputPath) {
  std::vector<std::string> cmd = {
      getMixTilingHelperPath(),
      "--name", kernelName.str(),
      "--soc", socVersion.str(),
      "--a-shape", joinShape(inputAShape),
      "--a-dtype", getTilingHelperDTypeName(inputADType).str(),
      "--b-shape", joinShape(inputBShape),
      "--b-dtype", getTilingHelperDTypeName(inputBDType).str(),
      "--c-shape", joinShape(outputShape),
      "--c-dtype", getTilingHelperDTypeName(outputDType).str(),
      "--tiling-out", tilingOutputPath.str(),
      "--launch-info-out", launchInfoOutputPath.str(),
  };
  if (biasDType) {
    cmd.push_back("--bias-dtype");
    cmd.push_back(getTilingHelperDTypeName(*biasDType).str());
    if (!biasShape.empty()) {
      cmd.push_back("--bias-shape");
      cmd.push_back(joinShape(biasShape));
    }
  }
  return cmd;
}

} // namespace mlir::runtime
