#include "MixDirectCompileInternal.h"

#include "Runtime/MixCommandBuilder.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace mlir::runtime {
namespace {

static constexpr const char *kStageAicPreprocessProbe = "AIC preprocess probe";
static constexpr const char *kStageAivPreprocessProbe = "AIV preprocess probe";
static constexpr const char *kStagePreprocessSource = "preprocess source";
static constexpr const char *kStageExtractHostStub = "extract host stub";
static constexpr const char *kStageFinalizeHostStub = "finalize host stub";

static std::vector<std::string> splitDefinitions(llvm::StringRef raw) {
  std::vector<std::string> out;
  llvm::SmallVector<llvm::StringRef> pieces;
  raw.split(pieces, ';');
  for (llvm::StringRef piece : pieces) {
    piece = piece.trim();
    if (!piece.empty())
      out.push_back(piece.str());
  }
  return out;
}

static std::vector<std::string>
collectConfigSources(const MixGeneratedConfig &config) {
  std::vector<std::string> sources;
  for (const std::string &source : config.mixSources) {
    if (!source.empty() &&
        std::find(sources.begin(), sources.end(), source) == sources.end())
      sources.push_back(source);
  }
  if (sources.empty()) {
    for (const auto &entry : config.definitionsBySource) {
      const std::string source = entry.getKey().str();
      if (!source.empty() &&
          std::find(sources.begin(), sources.end(), source) == sources.end())
        sources.push_back(source);
    }
  }
  return sources;
}

static llvm::Expected<std::string>
findOnlyMixSourceOrErr(const MixGeneratedConfig &aicConfig,
                       const MixGeneratedConfig &aivConfig) {
  std::vector<std::string> sources = collectConfigSources(aicConfig);
  for (const std::string &source : collectConfigSources(aivConfig)) {
    if (std::find(sources.begin(), sources.end(), source) == sources.end())
      sources.push_back(source);
  }
  if (sources.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Generated baremix config did not list any MIX_SOURCES entry");
  if (sources.size() != 1)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix direct backend only supports a single generated source in "
        "this stage, but found %zu",
        sources.size());
  return sources.front();
}

static std::vector<std::string>
definitionsForSource(const MixGeneratedConfig &config,
                     llvm::StringRef sourcePath) {
  const std::string exactKey = sourcePath.str();
  if (auto it = config.definitionsBySource.find(exactKey);
      it != config.definitionsBySource.end())
    return it->second;

  const std::string fileName = llvm::sys::path::filename(sourcePath).str();
  if (auto it = config.definitionsBySource.find(fileName);
      it != config.definitionsBySource.end())
    return it->second;

  return {};
}

static std::string resolveGeneratedSourcePath(llvm::StringRef generatedDir,
                                              llvm::StringRef sourceName) {
  if (llvm::sys::path::is_absolute(sourceName))
    return sourceName.str();
  return joinPath(generatedDir, sourceName);
}

static void appendDefineIfMissing(std::vector<std::string> &defs,
                                  llvm::StringRef needle) {
  if (llvm::find(defs, needle.str()) == defs.end())
    defs.push_back(needle.str());
}

static llvm::json::Array toJsonStringArray(llvm::ArrayRef<std::string> values) {
  llvm::json::Array array;
  for (const std::string &value : values)
    array.push_back(value);
  return array;
}

static llvm::StringRef contractModeName(MixDirectContractMode mode) {
  switch (mode) {
  case MixDirectContractMode::LegacyPreprocess:
    return "legacy-preprocess";
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

static llvm::Error runMixDirectProbeStage(const MixCompileLayout &layout,
                                          llvm::StringRef sourcePath,
                                          llvm::StringRef kernelName,
                                          const MixAnalyzedKernel &analyzed) {
  const std::vector<std::string> aicProbeCmd = buildBishengCommand(
      analyzed, sourcePath, layout.aicProbeObject, MixCoreType::AIC);
  const std::vector<std::string> aivProbeCmd = buildBishengCommand(
      analyzed, sourcePath, layout.aivProbeObject, MixCoreType::AIV);
  const std::string aicProbeContext = makeStageContext({
      {"kernel", kernelName},
      {"source", sourcePath},
      {"output", layout.aicProbeObject},
  });
  const std::string aivProbeContext = makeStageContext({
      {"kernel", kernelName},
      {"source", sourcePath},
      {"output", layout.aivProbeObject},
  });
  if (auto err =
          runProcess(aicProbeCmd, kStageAicPreprocessProbe, aicProbeContext))
    return err;
  if (auto err = ensureFileExists(layout.aicProbeObject,
                                  kStageAicPreprocessProbe, aicProbeContext))
    return err;
  if (auto err =
          runProcess(aivProbeCmd, kStageAivPreprocessProbe, aivProbeContext))
    return err;
  return ensureFileExists(layout.aivProbeObject, kStageAivPreprocessProbe,
                          aivProbeContext);
}

static llvm::Error writeCompileCommandsJson(llvm::StringRef path,
                                            llvm::StringRef directory,
                                            llvm::StringRef command,
                                            llvm::StringRef file) {
  llvm::json::Object entry;
  entry["directory"] = directory.str();
  entry["command"] = command.str();
  entry["file"] = file.str();
  llvm::json::Array entries;
  entries.push_back(std::move(entry));
  std::string json;
  llvm::raw_string_ostream os(json);
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(entries)));
  os.flush();
  json.push_back('\n');
  return writeTextFile(path, json);
}

static llvm::Expected<std::string>
discoverLauncherHeaderPath(llvm::StringRef includeDir) {
  std::error_code ec;
  llvm::sys::fs::directory_iterator it(includeDir, ec), end;
  if (ec)
    return llvm::createStringError(ec, "Cannot iterate include dir: %s",
                                   includeDir.str().c_str());

  std::vector<std::string> matches;
  for (; it != end && !ec; it.increment(ec)) {
    if (!llvm::sys::fs::is_regular_file(it->path()))
      continue;
    llvm::StringRef fileName = llvm::sys::path::filename(it->path());
    if (!fileName.starts_with("aclrtlaunch_") || !fileName.ends_with(".h"))
      continue;
    if (fileName == "aclrtlaunch_triple_chevrons_func.h")
      continue;
    matches.push_back(it->path());
  }
  if (ec)
    return llvm::createStringError(ec, "Cannot iterate include dir: %s",
                                   includeDir.str().c_str());
  if (matches.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No launcher header found under: %s",
                                   includeDir.str().c_str());
  if (matches.size() != 1)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Expected exactly 1 launcher header under %s, found %zu",
        includeDir.str().c_str(), matches.size());
  return matches.front();
}

static llvm::Expected<std::string>
deriveLauncherKernelName(llvm::StringRef launcherHeaderPath) {
  llvm::StringRef fileName = llvm::sys::path::filename(launcherHeaderPath);
  if (!fileName.starts_with("aclrtlaunch_") || !fileName.ends_with(".h"))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot derive launcher kernel name from: %s",
                                   launcherHeaderPath.str().c_str());
  fileName = fileName.drop_front(strlen("aclrtlaunch_"));
  fileName = fileName.drop_back(strlen(".h"));
  if (fileName.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Derived empty launcher kernel name from: %s",
                                   launcherHeaderPath.str().c_str());
  return fileName.str();
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

llvm::Expected<MixGeneratedConfig>
parseMixGeneratedConfig(llvm::StringRef path) {
  auto bufferOr = llvm::MemoryBuffer::getFile(path, /*IsText=*/true);
  if (!bufferOr)
    return llvm::createStringError(bufferOr.getError(),
                                   "Cannot read generated config: %s",
                                   path.str().c_str());

  MixGeneratedConfig config;
  llvm::StringRef content = (*bufferOr)->getBuffer();
  llvm::SmallVector<llvm::StringRef> lines;
  content.split(lines, '\n');

  enum class BlockKind { None, SetSources, SourceProps };
  BlockKind block = BlockKind::None;
  std::string currentSource;
  std::string sourcePropsBlock;

  for (llvm::StringRef rawLine : lines) {
    llvm::StringRef line = rawLine.trim();
    if (line.empty())
      continue;

    if (block == BlockKind::SetSources) {
      if (line == ")") {
        block = BlockKind::None;
        continue;
      }
      config.mixSources.push_back(line.str());
      continue;
    }

    if (block == BlockKind::SourceProps) {
      if (currentSource.empty() && line != ")" &&
          !line.starts_with("PROPERTIES")) {
        llvm::StringRef sourceLine = line;
        size_t propertiesPos = sourceLine.find("PROPERTIES");
        if (propertiesPos != llvm::StringRef::npos)
          sourceLine = sourceLine.take_front(propertiesPos);
        size_t closeParenPos = sourceLine.rfind(')');
        if (closeParenPos != llvm::StringRef::npos)
          sourceLine = sourceLine.take_front(closeParenPos);
        sourceLine = sourceLine.trim();
        if (!sourceLine.empty())
          currentSource = sourceLine.str();
      }
      if (!sourcePropsBlock.empty())
        sourcePropsBlock.push_back('\n');
      sourcePropsBlock += line.str();
      if (line == ")") {
        llvm::StringRef blockRef(sourcePropsBlock);
        size_t defsPos = blockRef.find("COMPILE_DEFINITIONS");
        if (defsPos != llvm::StringRef::npos) {
          llvm::StringRef defsTail = blockRef.drop_front(defsPos);
          size_t firstQuote = defsTail.find('"');
          size_t lastQuote = defsTail.rfind('"');
          if (firstQuote != llvm::StringRef::npos &&
              lastQuote != llvm::StringRef::npos && lastQuote > firstQuote) {
            llvm::StringRef defs = defsTail.slice(firstQuote + 1, lastQuote);
            config.definitionsBySource[currentSource] = splitDefinitions(defs);
          }
        }
        block = BlockKind::None;
        currentSource.clear();
        sourcePropsBlock.clear();
      }
      continue;
    }

    if (line.starts_with("set(MIX_SOURCES")) {
      block = BlockKind::SetSources;
      continue;
    }

    if (line.starts_with("set(AIC_SOURCES") ||
        line.starts_with("set(AIV_SOURCES")) {
      block = BlockKind::SetSources;
      continue;
    }

    if (line.starts_with("set_source_files_properties(")) {
      llvm::StringRef rest =
          line.drop_front(strlen("set_source_files_properties("));
      size_t propertiesPos = rest.find("PROPERTIES");
      if (propertiesPos != llvm::StringRef::npos)
        rest = rest.take_front(propertiesPos);
      size_t closeParenPos = rest.rfind(')');
      if (closeParenPos != llvm::StringRef::npos)
        rest = rest.take_front(closeParenPos);
      currentSource = rest.trim().str();
      block = BlockKind::SourceProps;
      sourcePropsBlock = line.str();
      continue;
    }
  }

  return config;
}

llvm::Expected<MixPreprocessOutputs>
runMixDirectPreprocessStage(llvm::StringRef workDir, llvm::StringRef sourcePath,
                            llvm::StringRef kernelName,
                            llvm::StringRef socVersion,
                            llvm::StringRef aivProbeObject,
                            llvm::StringRef aicProbeObject) {
  llvm::SmallString<256> preprocessedDir(workDir);
  llvm::sys::path::append(preprocessedDir, "preprocessed");
  llvm::SmallString<256> generatedDir(workDir);
  llvm::sys::path::append(generatedDir, "generated");
  llvm::SmallString<256> includeDir(generatedDir);
  llvm::sys::path::append(includeDir, "include");

  if (auto err = ensureDirectory(preprocessedDir))
    return std::move(err);
  if (auto err = ensureDirectory(generatedDir))
    return std::move(err);
  if (auto err = ensureDirectory(includeDir))
    return std::move(err);

  const std::string preprocessedPath =
      joinPath(preprocessedDir, kernelName.str() + ".cpp.o");
  const std::vector<std::string> preprocessCmd =
      buildPreprocessCommand(sourcePath, preprocessedPath);
  const std::string preprocessContext = makeStageContext({
      {"kernel", kernelName},
      {"source", sourcePath},
      {"output", preprocessedPath},
      {"generated_dir", generatedDir},
      {"include_dir", includeDir},
      {"aic_probe_object", aicProbeObject},
      {"aiv_probe_object", aivProbeObject},
  });
  if (auto err =
          runProcess(preprocessCmd, kStagePreprocessSource, preprocessContext))
    return std::move(err);
  if (auto err = ensureFileExists(preprocessedPath, kStagePreprocessSource,
                                  preprocessContext))
    return std::move(err);

  const std::string compileCommandsPath =
      joinPath(preprocessedDir, "compile_commands.json");
  llvm::SmallString<256> compileDir;
  if (auto ec = llvm::sys::fs::current_path(compileDir))
    return llvm::createStringError(ec, "Cannot resolve compile directory");
  const std::string fakeCommand =
      renderCommandForCompileCommands(preprocessCmd);
  if (auto err = writeCompileCommandsJson(compileCommandsPath, compileDir,
                                          fakeCommand, sourcePath))
    return std::move(err);

  const std::vector<std::string> extractCmd = buildExtractHostStubCommand(
      preprocessedPath, generatedDir, includeDir, {aivProbeObject.str()},
      {aicProbeObject.str()}, compileCommandsPath, "c220", "sim");
  const std::string extractContext = makeStageContext({
      {"preprocessed", preprocessedPath},
      {"generated_dir", generatedDir},
      {"include_dir", includeDir},
      {"compile_commands", compileCommandsPath},
      {"aic_probe_object", aicProbeObject},
      {"aiv_probe_object", aivProbeObject},
  });
  if (auto err = runProcess(extractCmd, kStageExtractHostStub, extractContext))
    return std::move(err);

  const std::string lowerSoc = llvm::StringRef(socVersion).lower();
  const std::vector<std::string> updateCmd = buildUpdateHostStubCommand(
      generatedDir, preprocessedDir, lowerSoc, "ascendc_kernels_sim");
  const std::string finalizeContext = makeStageContext({
      {"generated_dir", generatedDir},
      {"preprocessed_dir", preprocessedDir},
      {"soc_version", lowerSoc},
      {"target", "ascendc_kernels_sim"},
  });
  if (auto err =
          runProcess(updateCmd, kStageFinalizeHostStub, finalizeContext))
    return std::move(err);

  MixPreprocessOutputs outputs;
  outputs.preprocessedSourcePath = preprocessedPath;
  outputs.compileCommandsPath = compileCommandsPath;
  outputs.preprocessCommand = fakeCommand;
  outputs.generatedDir = generatedDir.str().str();
  outputs.includeDir = includeDir.str().str();
  outputs.hostStubPath = joinPath(generatedDir, "host_stub.cpp");
  outputs.aicConfigPath = joinPath(generatedDir, "aic_config.cmake");
  outputs.aivConfigPath = joinPath(generatedDir, "aiv_config.cmake");
  if (auto err = ensureFileExists(outputs.hostStubPath, kStageExtractHostStub,
                                  extractContext))
    return std::move(err);
  if (auto err = ensureFileExists(outputs.aicConfigPath, kStageExtractHostStub,
                                  extractContext))
    return std::move(err);
  if (auto err = ensureFileExists(outputs.aivConfigPath, kStageExtractHostStub,
                                  extractContext))
    return std::move(err);

  if (auto launcherHeaderOr = discoverLauncherHeaderPath(includeDir)) {
    outputs.launcherHeaderPath = *launcherHeaderOr;
    auto launcherKernelNameOr =
        deriveLauncherKernelName(outputs.launcherHeaderPath);
    if (!launcherKernelNameOr)
      return launcherKernelNameOr.takeError();
    outputs.actualLauncherKernelName = *launcherKernelNameOr;
    if (auto err = ensureFileExists(outputs.launcherHeaderPath,
                                    kStageExtractHostStub, extractContext))
      return std::move(err);
  } else {
    llvm::consumeError(launcherHeaderOr.takeError());
  }

  return outputs;
}

static llvm::Expected<MixDirectCompileContract>
loadMixDirectLegacyPreprocessCompileContract(
    const MixCompileLayout &layout, llvm::StringRef sourcePath,
    llvm::StringRef kernelName, llvm::StringRef socVersion,
    const MixAnalyzedKernel &analyzed) {
  if (auto err = runMixDirectProbeStage(layout, sourcePath, kernelName, analyzed))
    return std::move(err);
  auto preprocessOr =
      runMixDirectPreprocessStage(layout.workDir, sourcePath, kernelName,
                                  socVersion, layout.aivProbeObject,
                                  layout.aicProbeObject);
  if (!preprocessOr)
    return preprocessOr.takeError();

  auto aicConfigOr = parseMixGeneratedConfig(preprocessOr->aicConfigPath);
  if (!aicConfigOr)
    return aicConfigOr.takeError();
  auto aivConfigOr = parseMixGeneratedConfig(preprocessOr->aivConfigPath);
  if (!aivConfigOr)
    return aivConfigOr.takeError();

  auto generatedSourceOr = findOnlyMixSourceOrErr(*aicConfigOr, *aivConfigOr);
  if (!generatedSourceOr)
    return generatedSourceOr.takeError();

  MixDirectCompileContract contract;
  contract.mode = MixDirectContractMode::LegacyPreprocess;
  contract.preprocess = *preprocessOr;
  contract.layout = layout;
  contract.generatedSourceName = *generatedSourceOr;
  contract.generatedSourcePath =
      resolveGeneratedSourcePath(preprocessOr->generatedDir, *generatedSourceOr);
  if (!llvm::sys::fs::exists(contract.generatedSourcePath))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] generated mix source file not found: %s (kernel=%s)",
        kStagePreprocessSource, contract.generatedSourcePath.c_str(),
        kernelName.str().c_str());

  contract.aicDefinitions =
      definitionsForSource(*aicConfigOr, *generatedSourceOr);
  contract.aivDefinitions =
      definitionsForSource(*aivConfigOr, *generatedSourceOr);
  if (contract.aicDefinitions.empty() && !contract.aivDefinitions.empty()) {
    contract.synthesizedAicFromAiv = true;
    std::string coreDumpSize;
    for (const std::string &def : contract.aivDefinitions) {
      if (llvm::StringRef(def).starts_with("ONE_CORE_DUMP_SIZE="))
        coreDumpSize = def;
    }
    contract.aivDefinitions = analyzed.aivDefines;
    appendDefineIfMissing(contract.aivDefinitions, "HAVE_WORKSPACE");
    appendDefineIfMissing(contract.aivDefinitions, "HAVE_TILING");
    if (!coreDumpSize.empty())
      appendDefineIfMissing(contract.aivDefinitions, coreDumpSize);
    for (const std::string &def : contract.aivDefinitions) {
      std::string aicDef = def;
      size_t pos = aicDef.find("_0_mix_aiv");
      if (pos != std::string::npos)
        aicDef.replace(pos, 10, "_0_mix_aic");
      pos = aicDef.find("__DAV_C220_VEC__");
      if (pos != std::string::npos)
        aicDef.replace(pos, 16, "__DAV_C220_CUBE__");
      contract.aicDefinitions.push_back(std::move(aicDef));
    }
  }
  if (contract.aicDefinitions.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] generated AIC config did not provide compile definitions for %s",
        kStagePreprocessSource, contract.generatedSourceName.c_str());
  if (contract.aivDefinitions.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] generated AIV config did not provide compile definitions for %s",
        kStagePreprocessSource, contract.generatedSourceName.c_str());
  contract.runtimeKernelName =
      preprocessOr->actualLauncherKernelName.empty()
          ? kernelName.str()
          : preprocessOr->actualLauncherKernelName;
  return contract;
}

llvm::Expected<MixDirectCompileContract> loadMixDirectCompileContract(
    const MixCompileLayout &layout, llvm::StringRef sourcePath,
    llvm::StringRef kernelName, llvm::StringRef socVersion,
    const MixAnalyzedKernel &analyzed) {
  const char *modeEnv = std::getenv("ASCEND_MIX_CONTRACT_MODE");
  const llvm::StringRef mode = modeEnv ? llvm::StringRef(modeEnv) : "auto";
  if (mode == "legacy-preprocess")
    return loadMixDirectLegacyPreprocessCompileContract(
        layout, sourcePath, kernelName, socVersion, analyzed);

  auto directOr =
      buildMixDirectSourceCompileContract(layout, sourcePath, kernelName, analyzed);
  if (directOr)
    return std::move(*directOr);
  if (mode == "direct-source")
    return directOr.takeError();

  llvm::consumeError(directOr.takeError());
  return loadMixDirectLegacyPreprocessCompileContract(
      layout, sourcePath, kernelName, socVersion, analyzed);
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
