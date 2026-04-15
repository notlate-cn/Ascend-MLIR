#include "MixDirectCompileInternal.h"

#include "Runtime/MixCommandBuilder.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstring>
#include <utility>

namespace mlir::runtime {
namespace {

static llvm::json::Array toJsonStringArray(llvm::ArrayRef<std::string> values) {
  llvm::json::Array array;
  for (const std::string &value : values)
    array.push_back(value);
  return array;
}

static llvm::StringRef contractModeName(MixDirectContractMode mode) {
  switch (mode) {
  case MixDirectContractMode::DirectSource:
    return "direct-source";
  }
  return "unknown";
}

static std::string makeDirectWrapperGuard(llvm::StringRef sourcePath) {
  std::string guard = llvm::sys::path::filename(sourcePath).str();
  for (char &ch : guard) {
    unsigned char c = static_cast<unsigned char>(ch);
    ch = std::isalnum(c) ? static_cast<char>(std::toupper(c)) : '_';
  }
  return "__" + guard + "__KERNEL_FUN_H__";
}

static llvm::Expected<std::vector<std::string>>
parseDirectWrapperArgNames(llvm::StringRef sourcePath, llvm::StringRef kernelName) {
  auto sourceOr = readTextFileOrErr(sourcePath);
  if (!sourceOr)
    return sourceOr.takeError();
  llvm::StringRef source(*sourceOr);
  const std::string needle = ("void " + kernelName).str();
  size_t namePos = source.find(needle);
  if (namePos == llvm::StringRef::npos)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "direct-source wrapper cannot find kernel signature for %s in %s",
        kernelName.str().c_str(), sourcePath.str().c_str());
  size_t openParen = source.find('(', namePos + needle.size());
  if (openParen == llvm::StringRef::npos)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "direct-source wrapper cannot find kernel argument list for %s in %s",
        kernelName.str().c_str(), sourcePath.str().c_str());
  int depth = 0;
  size_t closeParen = llvm::StringRef::npos;
  for (size_t i = openParen; i < source.size(); ++i) {
    if (source[i] == '(')
      ++depth;
    if (source[i] == ')' && --depth == 0) {
      closeParen = i;
      break;
    }
  }
  if (closeParen == llvm::StringRef::npos)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "direct-source wrapper cannot close kernel argument list for %s in %s",
        kernelName.str().c_str(), sourcePath.str().c_str());

  llvm::StringRef argsText = source.slice(openParen + 1, closeParen);
  llvm::SmallVector<llvm::StringRef> args;
  argsText.split(args, ',');
  std::vector<std::string> names;
  for (llvm::StringRef arg : args) {
    arg = arg.trim();
    if (arg.empty() || arg == "void")
      continue;
    while (!arg.empty() && (arg.back() == '&' || arg.back() == '*'))
      arg = arg.drop_back().rtrim();
    size_t nameStart = arg.find_last_of(" \t*&");
    llvm::StringRef name =
        nameStart == llvm::StringRef::npos ? arg : arg.drop_front(nameStart + 1);
    name = name.trim();
    if (name.empty())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "direct-source wrapper found an unnamed argument in %s",
          sourcePath.str().c_str());
    names.push_back(name.str());
  }
  return names;
}

static llvm::Expected<std::string>
writeDirectSourceWrapper(const MixCompileLayout &layout,
                         llvm::StringRef sourcePath,
                         llvm::StringRef kernelName) {
  auto argNamesOr = parseDirectWrapperArgNames(sourcePath, kernelName);
  if (!argNamesOr)
    return argNamesOr.takeError();

  llvm::SmallString<256> absoluteSource(sourcePath);
  if (auto ec = llvm::sys::fs::make_absolute(absoluteSource))
    return llvm::createStringError(ec, "Cannot make source path absolute: %s",
                                   sourcePath.str().c_str());

  const std::string generatedDir = joinPath(layout.workDir, "generated");
  if (auto err = ensureDirectory(generatedDir))
    return std::move(err);
  const std::string generatedPath =
      joinPath(generatedDir,
               "auto_gen_" + llvm::sys::path::filename(sourcePath).str());
  const std::string guard = makeDirectWrapperGuard(sourcePath);

  std::string wrapper;
  llvm::raw_string_ostream os(wrapper);
  os << "#ifndef " << guard << "\n";
  os << "#define " << guard << "\n\n";
  os << "#undef __global__\n";
  os << "#define __global__ inline\n";
  os << "#define " << kernelName << " " << kernelName << "_origin\n";
  os << "#include \"" << absoluteSource.str() << "\"\n\n";
  os << "#undef " << kernelName << "\n";
  os << "#undef __global__\n";
  os << "#if ASCENDC_CPU_DEBUG\n";
  os << "#define __global__\n";
  os << "#else\n";
  os << "#define __global__ __attribute__((cce_kernel))\n";
  os << "#endif\n\n";
  os << "#ifndef ONE_CORE_DUMP_SIZE\n";
  os << "#define ONE_CORE_DUMP_SIZE 1048576 * 1\n";
  os << "#endif\n\n";
  os << "extern \"C\" __global__ [aicore] void auto_gen_" << kernelName
     << "_kernel(\n";
  os << "GM_ADDR ffts_addr";
  for (const std::string &name : *argNamesOr)
    os << ", __attribute__((cce_global)) uint8_t* " << name;
  os << ", GM_ADDR overflow_status) {\n";
  os << "    icache_preload(1);\n";
  os << "    if (ffts_addr != nullptr) {\n";
  os << "        set_ffts_base_addr((uint64_t)ffts_addr);\n";
  os << "    }\n";
  os << "#ifdef ASCENDC_TIME_STAMP_ON\n";
  os << "    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_FFTS_ADDR));\n";
  os << "#endif\n";
  os << "#if defined(HAVE_WORKSPACE)\n";
  os << "    GM_ADDR workspace_param;\n";
  os << "    GM_ADDR workspace_usr;\n";
  os << "#if defined(HAVE_TILING)\n";
  os << "    workspace_param = workspace;\n";
  os << "#else\n";
  os << "    workspace_param = tilingGm;\n";
  os << "#endif\n";
  os << "    if (workspace_param == nullptr) {\n";
  os << "        return;\n";
  os << "    }\n";
  os << "    AscendC::SetSysWorkspaceForce(workspace_param);\n";
  os << "    workspace_usr = AscendC::GetUserWorkspace(workspace_param);\n";
  os << "#if defined(REGIST_MATMUL_OBJ) || defined(__MIX_CORE_MACRO__)\n";
  os << "    if constexpr (g_coreType == AscendC::AIC) {\n";
  os << "        matmul::clearWorkspace(workspace_param);\n";
  os << "#ifdef ASCENDC_TIME_STAMP_ON\n";
  os << "        AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_CLEAR_WK_SPAC));\n";
  os << "#endif\n";
  os << "    }\n";
  os << "#endif\n";
  os << "#if defined(HAVE_TILING)\n";
  os << "    workspace = workspace_usr;\n";
  os << "#else\n";
  os << "    tilingGm = workspace_usr;\n";
  os << "#endif\n";
  os << "#endif\n";
  os << "    " << kernelName << "_origin(";
  for (size_t i = 0; i < argNamesOr->size(); ++i) {
    if (i)
      os << ", ";
    os << (*argNamesOr)[i];
  }
  os << ");\n";
  os << "#if !(defined(ASCENDC_DUMP) && ASCENDC_DUMP == 0) && defined(ASCENDC_DEBUG)\n";
  os << "    AscendC::WriteBackOverflow(overflow_status);\n";
  os << "#endif\n";
  os << "#if defined(__DAV_C310__)\n";
  os << "    pipe_barrier(PIPE_ALL);\n";
  os << "    dsb(mem_dsb_t::DSB_ALL);\n";
  os << "    dci();\n";
  os << "#endif\n";
  os << "}\n\n";
  os << "#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__)\n";
  os << "static const struct FunLevelMixCoreType " << kernelName
     << "_mix_aic_section __attribute__ ((used, section (\".ascend.meta."
     << kernelName
     << "_0_mix_aic\"))) = { { {F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN}, {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2} };\n";
  os << "#endif\n";
  os << "#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)\n";
  os << "static const struct FunLevelMixCoreType " << kernelName
     << "_mix_aiv_section __attribute__ ((used, section (\".ascend.meta."
     << kernelName
     << "_0_mix_aiv\"))) = { { {F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN}, {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2} };\n";
  os << "#endif\n";
  os << "#endif\n";
  os.flush();

  if (auto err = writeTextFile(generatedPath, wrapper))
    return std::move(err);
  return generatedPath;
}

} // namespace

llvm::Expected<MixDirectCompileContract> buildMixDirectSourceCompileContract(
    const MixCompileLayout &layout, llvm::StringRef sourcePath,
    llvm::StringRef kernelName, const MixAnalyzedKernel &analyzed) {
  if (sourcePath.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "direct-source mix contract requires source path");
  if (kernelName.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "direct-source mix contract requires kernel name");

  MixDirectCompileContract contract;
  contract.mode = MixDirectContractMode::DirectSource;
  contract.layout = layout;
  auto generatedSourceOr =
      writeDirectSourceWrapper(layout, sourcePath, kernelName);
  if (!generatedSourceOr)
    return generatedSourceOr.takeError();
  contract.generatedSourceName =
      llvm::sys::path::filename(*generatedSourceOr).str();
  contract.generatedSourcePath = *generatedSourceOr;
  contract.runtimeKernelName = kernelName.str();
  contract.aicDefinitions = {
      "auto_gen_" + kernelName.str() + "_kernel=" + analyzed.aicEntry,
      "ONE_CORE_DUMP_SIZE=1048576",
      "__MIX_CORE_MACRO__=1",
  };
  contract.aivDefinitions = {
      "auto_gen_" + kernelName.str() + "_kernel=" + analyzed.aivEntry,
      "ONE_CORE_DUMP_SIZE=1048576",
      "__MIX_CORE_MACRO__=1",
  };

  contract.preprocess.hostStubPath = joinPath(layout.stubDir, "host_stub.cpp");
  contract.preprocess.includeDir = layout.outIncludeDir;
  contract.preprocess.actualLauncherKernelName = kernelName.str();
  contract.preprocess.preprocessCommand = "direct-source";
  contract.preprocess.generatedDir = layout.workDir;
  return contract;
}

llvm::Expected<MixDirectCompileContract> loadMixDirectCompileContract(
    const MixCompileLayout &layout, llvm::StringRef sourcePath,
    llvm::StringRef kernelName, llvm::StringRef socVersion,
    const MixAnalyzedKernel &analyzed) {
  (void)socVersion;
  return buildMixDirectSourceCompileContract(layout, sourcePath, kernelName,
                                             analyzed);
}

llvm::Expected<std::string> buildMixDirectSourceContractSummaryForTest(
    llvm::StringRef outputRoot, llvm::StringRef sourcePath,
    llvm::StringRef kernelName, llvm::StringRef socVersion) {
  auto layoutOr = buildMixDirectCompileLayout(outputRoot, kernelName);
  if (!layoutOr)
    return layoutOr.takeError();
  auto analyzedOr = analyzeMixKernel(sourcePath, kernelName, socVersion);
  if (!analyzedOr)
    return analyzedOr.takeError();
  auto contractOr = buildMixDirectSourceCompileContract(
      *layoutOr, sourcePath, kernelName, *analyzedOr);
  if (!contractOr)
    return contractOr.takeError();

  llvm::json::Object root;
  root["contract_mode"] = "direct-source";
  root["runtime_kernel_name"] = contractOr->runtimeKernelName;
  root["generated_source_path"] = contractOr->generatedSourcePath;
  root["host_stub_source_path"] = contractOr->preprocess.hostStubPath;
  root["host_stub_include_dir"] = contractOr->preprocess.includeDir;
  root["aic_definitions"] = toJsonStringArray(contractOr->aicDefinitions);
  root["aiv_definitions"] = toJsonStringArray(contractOr->aivDefinitions);

  std::string out;
  llvm::raw_string_ostream os(out);
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(root)));
  os.flush();
  out.push_back('\n');
  return out;
}

llvm::Expected<std::string> buildMixDirectDefaultContractSummaryForTest(
    llvm::StringRef outputRoot, llvm::StringRef sourcePath,
    llvm::StringRef kernelName, llvm::StringRef socVersion) {
  auto layoutOr = buildMixDirectCompileLayout(outputRoot, kernelName);
  if (!layoutOr)
    return layoutOr.takeError();
  auto analyzedOr = analyzeMixKernel(sourcePath, kernelName, socVersion);
  if (!analyzedOr)
    return analyzedOr.takeError();
  auto contractOr = loadMixDirectCompileContract(
      *layoutOr, sourcePath, kernelName, socVersion, *analyzedOr);
  if (!contractOr)
    return contractOr.takeError();

  llvm::json::Object root;
  root["contract_mode"] = contractModeName(contractOr->mode).str();
  root["preprocess_command"] = contractOr->preprocess.preprocessCommand;
  root["generated_source_path"] = contractOr->generatedSourcePath;

  std::string out;
  llvm::raw_string_ostream os(out);
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(root)));
  os.flush();
  out.push_back('\n');
  return out;
}

} // namespace mlir::runtime
