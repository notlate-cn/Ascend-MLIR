#include "Runtime/Mix/MixLegacyCompileCompat.h"

#include "Runtime/Mix/MixCompileMetadata.h"
#include "Runtime/MixCommandBuilder.h"
#include "Runtime/Mix/MixStubTemplate.h"
#include "Runtime/Support/PathUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <utility>

namespace mlir::runtime {

namespace {

static constexpr const char *kStageAicPreprocessProbe = "AIC preprocess probe";
static constexpr const char *kStageAivPreprocessProbe = "AIV preprocess probe";
static constexpr const char *kStagePreprocessSource = "preprocess source";
static constexpr const char *kStageExtractHostStub = "extract host stub";
static constexpr const char *kStageFinalizeHostStub = "finalize host stub";
static constexpr const char *kStageCompileAic = "compile AIC object";
static constexpr const char *kStageCompileAiv = "compile AIV object";
static constexpr const char *kStageMergeAic = "merge AIC object";
static constexpr const char *kStageMergeAiv = "merge AIV object";
static constexpr const char *kStageMergeDevice = "merge device objects";
static constexpr const char *kStageCompileHostStub = "compile host stub";
static constexpr const char *kStageCompileHostBisheng = "compile host bisheng";
static constexpr const char *kStagePack = "pack mix kernel";
static constexpr const char *kStageLinkHostStub = "link host runner library";
static constexpr const char *kStageRecompile = "recompile packed binary";

static llvm::Error writeTextFile(llvm::StringRef path,
                                 llvm::StringRef content) {
  std::ofstream os(path.str(), std::ios::binary);
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot write file: %s",
                                   path.str().c_str());
  os << content.str();
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to write file: %s",
                                   path.str().c_str());
  return llvm::Error::success();
}

static llvm::Error ensureDirectory(llvm::StringRef path) {
  if (auto ec = llvm::sys::fs::create_directories(path))
    return llvm::createStringError(ec, "Cannot create directory: %s",
                                   path.str().c_str());
  return llvm::Error::success();
}

static llvm::Error ensureFileExists(llvm::StringRef path, llvm::StringRef stage,
                                    llvm::StringRef context = {}) {
  if (llvm::sys::fs::exists(path))
    return llvm::Error::success();
  const std::string contextText = context.str();
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "[%s] completed but did not create expected file: %s%s%s",
      stage.str().c_str(), path.str().c_str(), context.empty() ? "" : " (",
      context.empty() ? "" : contextText.c_str(),
      context.empty() ? "" : ")");
}

static std::string joinPath(llvm::StringRef base, llvm::StringRef leaf) {
  llvm::SmallString<256> joined(base);
  llvm::sys::path::append(joined, leaf);
  return joined.str().str();
}

static std::string makeStageContext(
    std::initializer_list<std::pair<llvm::StringRef, llvm::StringRef>> fields) {
  std::string out;
  llvm::raw_string_ostream os(out);
  bool first = true;
  for (const auto &field : fields) {
    if (field.second.empty())
      continue;
    if (!first)
      os << ", ";
    first = false;
    os << field.first << "=" << field.second;
  }
  os.flush();
  return out;
}

static llvm::Error runProcess(const std::vector<std::string> &args,
                              llvm::StringRef stage,
                              llvm::StringRef context = {}) {
  if (args.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "[%s] received an empty command",
                                   stage.str().c_str());

  std::vector<llvm::StringRef> argv;
  argv.reserve(args.size());
  for (const auto &arg : args)
    argv.push_back(arg);

  std::string errMsg;
  std::optional<llvm::StringRef> redirects[3];
  int ret = llvm::sys::ExecuteAndWait(argv[0], argv, std::nullopt, redirects,
                                      300, 0, &errMsg);
  if (ret == 0)
    return llvm::Error::success();

  const std::string program = argv[0].str();
  const std::string contextText = context.str();
  std::string renderedCommand = renderCommandForDebug(args);
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "[%s] program=%s%s%s failed (exit %d): %s\n  command: %s",
      stage.str().c_str(), program.c_str(), context.empty() ? "" : " inputs: ",
      context.empty() ? "" : contextText.c_str(), ret, errMsg.c_str(),
      renderedCommand.c_str());
}

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

static std::vector<std::string>
buildFinalMergeCommand(llvm::StringRef aicObj, llvm::StringRef aivObj,
                       llvm::StringRef outputObj) {
  std::vector<std::string> cmd =
      buildLldMergeCommand(aicObj, aivObj, outputObj);
  cmd.insert(cmd.begin() + 1, "--allow-multiple-definition");
  if (cmd.size() > 2 && cmd[2] == "-r")
    cmd.erase(cmd.begin() + 2);
  return cmd;
}

static llvm::Error runLegacyMixProbeStage(const MixCompileLayout &layout,
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
  if (auto err = ensureFileExists(layout.aicProbeObject, kStageAicPreprocessProbe,
                                  aicProbeContext))
    return err;
  if (auto err =
          runProcess(aivProbeCmd, kStageAivPreprocessProbe, aivProbeContext))
    return err;
  return ensureFileExists(layout.aivProbeObject, kStageAivPreprocessProbe,
                          aivProbeContext);
}

static llvm::Error copyFileOrErr(llvm::StringRef from, llvm::StringRef to) {
  if (auto ec = llvm::sys::fs::copy_file(from, to))
    return llvm::createStringError(ec, "Cannot copy %s -> %s",
                                   from.str().c_str(), to.str().c_str());
  return llvm::Error::success();
}

static llvm::Expected<uint64_t> getFileSizeOrErr(llvm::StringRef path) {
  uint64_t size = 0;
  if (auto ec = llvm::sys::fs::file_size(path, size))
    return llvm::createStringError(ec, "Cannot stat file: %s",
                                   path.str().c_str());
  return size;
}

static uint64_t alignTo4(uint64_t size) {
  return (size + 3ULL) & ~3ULL;
}

static llvm::StringRef getDTypeName(DType dtype) {
  switch (dtype) {
  case DType::F16:
    return "f16";
  case DType::BF16:
    return "bf16";
  case DType::F32:
    return "f32";
  case DType::INT8:
    return "int8";
  case DType::INT32:
    return "int32";
  case DType::INT64:
    return "int64";
  default:
    return "unknown";
  }
}

static MixCompileMetadataTensorDesc
makeMetadataTensorDesc(const MixAbiTensorDesc &tensor) {
  MixCompileMetadataTensorDesc out;
  out.name = tensor.name;
  out.dtype = getDTypeName(tensor.dtype).str();
  out.shape = tensor.shape;
  out.runtimeFile = tensor.runtimeFile;
  out.goldenFile = tensor.goldenFile;
  return out;
}

static llvm::Error writeRecompileLinkFile(llvm::StringRef rootDir,
                                          llvm::StringRef targetName,
                                          llvm::StringRef linkCommand) {
  llvm::SmallString<256> linkDir(rootDir);
  llvm::sys::path::append(linkDir, "CMakeFiles", targetName.str() + ".dir");
  if (auto err = ensureDirectory(linkDir))
    return err;
  llvm::SmallString<256> linkPath(linkDir);
  llvm::sys::path::append(linkPath, "link.txt");
  return writeTextFile(linkPath, linkCommand.str() + "\n");
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
runLegacyMixPreprocessStage(llvm::StringRef workDir, llvm::StringRef sourcePath,
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
  if (auto err =
          ensureFileExists(outputs.hostStubPath, kStageExtractHostStub,
                           extractContext))
    return std::move(err);
  if (auto err =
          ensureFileExists(outputs.aicConfigPath, kStageExtractHostStub,
                           extractContext))
    return std::move(err);
  if (auto err =
          ensureFileExists(outputs.aivConfigPath, kStageExtractHostStub,
                           extractContext))
    return std::move(err);

  if (auto launcherHeaderOr = discoverLauncherHeaderPath(includeDir)) {
    outputs.launcherHeaderPath = *launcherHeaderOr;
    auto launcherKernelNameOr =
        deriveLauncherKernelName(outputs.launcherHeaderPath);
    if (!launcherKernelNameOr)
      return launcherKernelNameOr.takeError();
    outputs.actualLauncherKernelName = *launcherKernelNameOr;
    if (auto err =
            ensureFileExists(outputs.launcherHeaderPath, kStageExtractHostStub,
                             extractContext))
      return std::move(err);
  } else {
    llvm::consumeError(launcherHeaderOr.takeError());
  }

  return outputs;
}

llvm::Expected<MixLegacyCompileContract> loadLegacyMixCompileContract(
    const MixCompileLayout &layout, llvm::StringRef sourcePath,
    llvm::StringRef kernelName, llvm::StringRef socVersion,
    const MixAnalyzedKernel &analyzed) {
  if (auto err = runLegacyMixProbeStage(layout, sourcePath, kernelName, analyzed))
    return std::move(err);
  auto preprocessOr =
      runLegacyMixPreprocessStage(layout.workDir, sourcePath, kernelName,
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

  MixLegacyCompileContract contract;
  contract.preprocess = *preprocessOr;
  contract.layout = layout;
  contract.generatedSourceName = *generatedSourceOr;
  contract.generatedSourcePath = resolveGeneratedSourcePath(
      preprocessOr->generatedDir, *generatedSourceOr);
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

llvm::Expected<MixLegacyBuildOutputs>
executeLegacyMixBinaryBuild(const MixLegacyCompileContract &contract,
                            llvm::StringRef sourcePath,
                            llvm::StringRef requestedKernelName,
                            llvm::StringRef socVersion) {
  const MixCompileLayout &layout = contract.layout;
  MixLegacyBuildOutputs outputs;
  outputs.runtimeKernelName = contract.runtimeKernelName;
  outputs.generatedSourcePath = contract.generatedSourcePath;
  outputs.hostSourcePath = sourcePath.str();
  outputs.hostStubSourcePath = contract.preprocess.hostStubPath;
  outputs.hostStubIncludeDir = contract.preprocess.includeDir;
  outputs.preprocessIncludeDir = contract.preprocess.includeDir;
  outputs.preprocessCompileCommandsPath = contract.preprocess.compileCommandsPath;
  outputs.preprocessCommand = contract.preprocess.preprocessCommand;
  outputs.preprocessGeneratedDir = contract.preprocess.generatedDir;

  const std::string runnerLauncherCopyPath =
      joinPath(layout.outIncludeDir,
               "aclrtlaunch_" + outputs.runtimeKernelName + ".h");
  const bool needsManualStubTemplate =
      contract.preprocess.launcherHeaderPath.empty() ||
      contract.synthesizedAicFromAiv;
  std::string launcherHeaderPath = contract.preprocess.launcherHeaderPath;
  if (needsManualStubTemplate) {
    outputs.hostStubSourcePath = joinPath(layout.stubDir, "host_stub.cpp");
    outputs.hostStubIncludeDir = layout.outIncludeDir;
    launcherHeaderPath = runnerLauncherCopyPath;
  }

  const std::vector<std::string> aicCmd =
      buildPreprocessedDeviceCompileCommand(outputs.generatedSourcePath,
                                            layout.aicObj, MixCoreType::AIC,
                                            contract.aicDefinitions);
  const std::vector<std::string> aivCmd =
      buildPreprocessedDeviceCompileCommand(outputs.generatedSourcePath,
                                            layout.aivObj, MixCoreType::AIV,
                                            contract.aivDefinitions);
  const std::vector<std::string> aicRelocCmd =
      buildLldRelocCommand(layout.aicObj, layout.aicRelocObj);
  const std::vector<std::string> aivRelocCmd =
      buildLldRelocCommand(layout.aivObj, layout.aivRelocObj);
  const std::vector<std::string> mergeCmd = buildFinalMergeCommand(
      layout.aicRelocObj, layout.aivRelocObj, layout.mergeDeviceObj);
  const std::vector<std::string> hostCompileCmd =
      buildHostStubCompileCommand(outputs.hostStubSourcePath,
                                  layout.hostStubObjectPath,
                                  outputs.hostStubIncludeDir);
  const std::vector<std::string> packCmd =
      buildPackCommand(layout.hostStubObjectPath, layout.mergeDir);
  const std::vector<std::string> hostLinkCmd =
      buildHostSharedLinkCommand(layout.hostStubObjectPath, layout.kernelSoPath,
                                 socVersion,
                                 findAscendDeviceLibDir(findAscendHome()));

  const std::string aicCompileContext = makeStageContext({
      {"kernel", requestedKernelName},
      {"source", outputs.generatedSourcePath},
      {"output", layout.aicObj},
      {"generated_dir", outputs.preprocessGeneratedDir},
  });
  const std::string aivCompileContext = makeStageContext({
      {"kernel", requestedKernelName},
      {"source", outputs.generatedSourcePath},
      {"output", layout.aivObj},
      {"generated_dir", outputs.preprocessGeneratedDir},
  });
  const std::string aicMergeContext = makeStageContext({
      {"input", layout.aicObj},
      {"output", layout.aicRelocObj},
      {"kernel", requestedKernelName},
      {"generated_dir", outputs.preprocessGeneratedDir},
  });
  const std::string aivMergeContext = makeStageContext({
      {"input", layout.aivObj},
      {"output", layout.aivRelocObj},
      {"kernel", requestedKernelName},
      {"generated_dir", outputs.preprocessGeneratedDir},
  });
  const std::string mergeContext = makeStageContext({
      {"aic_input", layout.aicRelocObj},
      {"aiv_input", layout.aivRelocObj},
      {"output", layout.mergeDeviceObj},
      {"kernel", requestedKernelName},
  });
  const std::string finalizeContext = makeStageContext({
      {"generated_dir", outputs.preprocessGeneratedDir},
      {"merge_dir", layout.mergeDir},
      {"soc_version", socVersion},
      {"target", "ascendc_kernels_sim"},
  });
  const std::string hostCompileContext = makeStageContext({
      {"source", outputs.hostStubSourcePath},
      {"output", layout.hostStubObjectPath},
      {"include_dir", outputs.hostStubIncludeDir},
      {"kernel", requestedKernelName},
  });
  const std::string packContext = makeStageContext({
      {"input", layout.hostStubObjectPath},
      {"add_dir", layout.mergeDir},
      {"kernel", requestedKernelName},
  });
  const std::string hostLinkContext = makeStageContext({
      {"input", layout.hostStubObjectPath},
      {"output", layout.kernelSoPath},
      {"soc_version", socVersion},
      {"kernel", requestedKernelName},
  });
  const std::string tripleChevronHeaderPath =
      joinPath(outputs.preprocessIncludeDir.empty() ? outputs.hostStubIncludeDir
                                                    : outputs.preprocessIncludeDir,
               "aclrtlaunch_triple_chevrons_func.h");
  outputs.hostBishengObjectPath =
      joinPath(layout.hostObjectsDir,
               llvm::sys::path::filename(outputs.hostSourcePath).str() + ".o");
  outputs.hostObjectDir = layout.hostDir;
  const std::vector<std::string> hostBishengCmd =
      buildHostBishengCommand(outputs.hostSourcePath, outputs.hostBishengObjectPath,
                              tripleChevronHeaderPath);
  const std::vector<std::string> recompileCmd =
      buildRecompileBinaryCommand(layout.outputRoot, "ascendc_kernels_sim",
                                  layout.hostDir);

  if (auto err = runProcess(aicCmd, kStageCompileAic, aicCompileContext))
    return std::move(err);
  if (auto err =
          ensureFileExists(layout.aicObj, kStageCompileAic, aicCompileContext))
    return std::move(err);
  if (auto err = runProcess(aivCmd, kStageCompileAiv, aivCompileContext))
    return std::move(err);
  if (auto err =
          ensureFileExists(layout.aivObj, kStageCompileAiv, aivCompileContext))
    return std::move(err);
  if (auto err = runProcess(aicRelocCmd, kStageMergeAic, aicMergeContext))
    return std::move(err);
  if (auto err =
          ensureFileExists(layout.aicRelocObj, kStageMergeAic, aicMergeContext))
    return std::move(err);
  if (auto err = runProcess(aivRelocCmd, kStageMergeAiv, aivMergeContext))
    return std::move(err);
  if (auto err =
          ensureFileExists(layout.aivRelocObj, kStageMergeAiv, aivMergeContext))
    return std::move(err);
  if (auto err = runProcess(mergeCmd, kStageMergeDevice, mergeContext))
    return std::move(err);
  if (auto err =
          ensureFileExists(layout.mergeDeviceObj, kStageMergeDevice, mergeContext))
    return std::move(err);
  if (auto err = copyFileOrErr(layout.mergeDeviceObj, layout.mergedDeviceObj))
    return std::move(err);
  auto mixLen = getFileSizeOrErr(layout.mergedDeviceObj);
  if (!mixLen)
    return mixLen.takeError();
  if (auto err = writeTextFile(layout.mixFlagPath, ""))
    return std::move(err);

  if (needsManualStubTemplate) {
    MixStubTemplateArgs stubArgs;
    stubArgs.kernelName = outputs.runtimeKernelName;
    stubArgs.targetName = "ascendc_kernels_sim";
    stubArgs.socVersion = socVersion.str();
    stubArgs.launcherSymbol = "aclrtlaunch_" + outputs.runtimeKernelName;
    stubArgs.launcherHeaderPath = runnerLauncherCopyPath;
    stubArgs.hostStubSourcePath = outputs.hostStubSourcePath;
    stubArgs.mixLen = alignTo4(*mixLen);
    stubArgs.mixFileLen = *mixLen;
    stubArgs.aivOnly = false;
    if (auto err = writeMixStubTemplate(stubArgs))
      return std::move(err);
    launcherHeaderPath = runnerLauncherCopyPath;
  } else {
    const std::string lowerSocVersion = llvm::StringRef(socVersion).lower();
    const std::vector<std::string> finalizeHostStubCmd =
        buildUpdateHostStubCommand(outputs.preprocessGeneratedDir, layout.mergeDir,
                                   lowerSocVersion, "ascendc_kernels_sim");
    if (auto err = runProcess(finalizeHostStubCmd, kStageFinalizeHostStub,
                              finalizeContext))
      return std::move(err);
    if (auto err = copyFileOrErr(launcherHeaderPath, runnerLauncherCopyPath))
      return std::move(err);
  }

  if (auto err = runProcess(hostCompileCmd, kStageCompileHostStub,
                            hostCompileContext))
    return std::move(err);
  if (auto err = ensureFileExists(layout.hostStubObjectPath,
                                  kStageCompileHostStub, hostCompileContext))
    return std::move(err);
  const std::string hostBishengContext = makeStageContext({
      {"source", outputs.hostSourcePath},
      {"output", outputs.hostBishengObjectPath},
      {"triple_chevron_header", tripleChevronHeaderPath},
      {"kernel", requestedKernelName},
  });
  if (auto err = ensureFileExists(tripleChevronHeaderPath,
                                  kStageCompileHostBisheng,
                                  hostBishengContext))
    return std::move(err);
  if (auto err = runProcess(hostBishengCmd, kStageCompileHostBisheng,
                            hostBishengContext))
    return std::move(err);
  if (auto err = ensureFileExists(outputs.hostBishengObjectPath,
                                  kStageCompileHostBisheng,
                                  hostBishengContext))
    return std::move(err);
  if (auto err = runProcess(packCmd, kStagePack, packContext))
    return std::move(err);
  if (auto err =
          ensureFileExists(layout.hostStubObjectPath, kStagePack, packContext))
    return std::move(err);
  if (auto err = runProcess(hostLinkCmd, kStageLinkHostStub, hostLinkContext))
    return std::move(err);
  if (auto err = ensureFileExists(layout.kernelSoPath, kStageLinkHostStub,
                                  hostLinkContext))
    return std::move(err);

  const std::string recompileHostStubObjectPath =
      joinPath(layout.stubDir, "host_stub.cpp.o");
  if (auto err =
          copyFileOrErr(layout.hostStubObjectPath, recompileHostStubObjectPath))
    return std::move(err);
  std::vector<std::string> recompileLinkArgs = hostLinkCmd;
  for (std::string &arg : recompileLinkArgs) {
    if (arg == layout.hostStubObjectPath)
      arg = recompileHostStubObjectPath;
  }
  const std::string recompileLinkCmd =
      renderCommandForCompileCommands(recompileLinkArgs);
  if (auto err = writeRecompileLinkFile(layout.outputRoot, "ascendc_kernels_sim",
                                        recompileLinkCmd))
    return std::move(err);
  const std::string recompileContext = makeStageContext({
      {"root_dir", layout.outputRoot},
      {"target_name", "ascendc_kernels_sim"},
      {"add_dir", layout.hostDir},
      {"kernel", requestedKernelName},
  });
  if (auto err = runProcess(recompileCmd, kStageRecompile, recompileContext))
    return std::move(err);
  const std::string recompiledKernelSoPath =
      joinPath(layout.outDir, "lib" + outputs.runtimeKernelName + "_packed.so");
  if (auto err = ensureFileExists(recompiledKernelSoPath, kStageRecompile,
                                  recompileContext))
    return std::move(err);

  outputs.aicCompileCommand = renderCommandForDebug(aicCmd);
  outputs.aivCompileCommand = renderCommandForDebug(aivCmd);
  outputs.aicRelocCommand = renderCommandForDebug(aicRelocCmd);
  outputs.aivRelocCommand = renderCommandForDebug(aivRelocCmd);
  outputs.mergeCommand = renderCommandForDebug(mergeCmd);
  outputs.hostCompileCommand = renderCommandForDebug(hostCompileCmd);
  outputs.hostBishengCommand = renderCommandForDebug(hostBishengCmd);
  outputs.packCommand = renderCommandForDebug(packCmd);
  outputs.hostLinkCommand = renderCommandForDebug(hostLinkCmd);
  outputs.recompileCommand = renderCommandForDebug(recompileCmd);
  return outputs;
}

llvm::Expected<std::string>
writeLegacyMixCompileMetadataFile(llvm::StringRef metadataPath,
                                  llvm::StringRef runtimeKernelName,
                                  llvm::StringRef socVersion,
                                  llvm::StringRef mixKernelType,
                                  llvm::StringRef generatedSourcePath,
                                  llvm::ArrayRef<std::string> aicDefinitions,
                                  llvm::ArrayRef<std::string> aivDefinitions,
                                  llvm::StringRef deviceObjectPath,
                                  llvm::StringRef packedSharedObjectPath,
                                  llvm::StringRef tilingFilePath,
                                  llvm::StringRef launchInfoFilePath,
                                  const MixAbiMetadata &abi,
                                  bool useLegacyRunner) {
  MixCompileMetadata metadata;
  metadata.schemaVersion = 1;
  metadata.kernelKind = "mix";
  metadata.kernelName = runtimeKernelName.str();
  metadata.runtimeKernelName = runtimeKernelName.str();
  metadata.socVersion = socVersion.str();
  metadata.mixKernelType = mixKernelType.str();
  metadata.launcherSymbol = abi.launcherSymbol;
  metadata.entries.aic = abi.aicEntry;
  metadata.entries.aiv = abi.aivEntry;
  metadata.generated.sourcePath = generatedSourcePath.str();
  metadata.deviceCompile.aicArch = "dav-c220-cube";
  metadata.deviceCompile.aivArch = "dav-c220-vec";
  metadata.deviceCompile.aicDefinitions.assign(aicDefinitions.begin(),
                                               aicDefinitions.end());
  metadata.deviceCompile.aivDefinitions.assign(aivDefinitions.begin(),
                                               aivDefinitions.end());
  metadata.artifacts.deviceObjectPath = deviceObjectPath.str();
  metadata.artifacts.packedSharedObjectPath = packedSharedObjectPath.str();
  metadata.artifacts.tilingFilePath = tilingFilePath.str();
  metadata.artifacts.launchInfoFilePath = launchInfoFilePath.str();
  metadata.abi.workspaceMode = abi.workspaceMode;
  metadata.abi.workspaceBytes = abi.workspaceBytes;
  metadata.abi.tilingMode = abi.tilingMode;
  metadata.abi.tilingSource = abi.tilingSource;
  if (abi.workspaceArgIndex) {
    metadata.abi.workspaceArgIndex = *abi.workspaceArgIndex;
    metadata.abi.hasWorkspaceArgIndex = true;
  }
  if (abi.tilingArgIndex) {
    metadata.abi.tilingArgIndex = *abi.tilingArgIndex;
    metadata.abi.hasTilingArgIndex = true;
  }
  metadata.abi.inputs.reserve(abi.inputs.size());
  for (const auto &tensor : abi.inputs)
    metadata.abi.inputs.push_back(makeMetadataTensorDesc(tensor));
  metadata.abi.outputs.reserve(abi.outputs.size());
  for (const auto &tensor : abi.outputs)
    metadata.abi.outputs.push_back(makeMetadataTensorDesc(tensor));
  metadata.hostLaunch.mode = useLegacyRunner ? "legacy_runner" : "helper";
  metadata.hostLaunch.helperKind =
      useLegacyRunner ? "mix_runner" : "mix-tiling-helper";
  metadata.hostLaunch.helperInputsJson = "{}";

  auto jsonOr = serializeMixCompileMetadataJson(metadata);
  if (!jsonOr)
    return jsonOr.takeError();
  if (auto err = writeTextFile(metadataPath, *jsonOr))
    return std::move(err);
  return metadataPath.str();
}

llvm::Expected<MixCompileLayout>
buildLegacyMixCompileLayout(llvm::StringRef outputDir,
                            llvm::StringRef kernelName) {
  llvm::SmallString<256> outputRoot(outputDir);
  llvm::sys::fs::make_absolute(outputRoot);
  llvm::SmallString<256> workDir(outputRoot);
  llvm::sys::path::append(workDir, "work");
  llvm::SmallString<256> objectDir(outputRoot);
  llvm::sys::path::append(objectDir, "objects");
  llvm::SmallString<256> outDir(outputRoot);
  llvm::sys::path::append(outDir, "out");
  llvm::SmallString<256> outBinDir(outDir);
  llvm::sys::path::append(outBinDir, "bin");
  llvm::SmallString<256> outIncludeDir(outDir);
  llvm::sys::path::append(outIncludeDir, "include", "ascendc_kernels_sim");
  llvm::SmallString<256> mergeDir(workDir);
  llvm::sys::path::append(mergeDir, "merge_obj");
  llvm::SmallString<256> launcherDir(workDir);
  llvm::sys::path::append(launcherDir, "launcher");
  llvm::SmallString<256> stubDir(workDir);
  llvm::sys::path::append(stubDir, "stub");
  llvm::SmallString<256> hostDir(outputRoot);
  llvm::sys::path::append(hostDir, "host_dir");
  llvm::SmallString<256> hostObjectsDir(hostDir);
  llvm::sys::path::append(hostObjectsDir, "objects-Debug", "host_bisheng_obj");
  llvm::SmallString<256> aicMergeDir(workDir);
  llvm::sys::path::append(aicMergeDir, "aic_merge");
  llvm::SmallString<256> aivMergeDir(workDir);
  llvm::sys::path::append(aivMergeDir, "aiv_merge");
  llvm::SmallString<256> preprocessProbeDir(workDir);
  llvm::sys::path::append(preprocessProbeDir, "preprocess_probe");

  for (llvm::StringRef dir : {outputRoot.str(), workDir.str(), objectDir.str(),
                              outDir.str(), outBinDir.str(),
                              outIncludeDir.str(), mergeDir.str(),
                              launcherDir.str(), stubDir.str(), hostDir.str(),
                              hostObjectsDir.str(), aicMergeDir.str(),
                              aivMergeDir.str(), preprocessProbeDir.str()}) {
    if (auto err = ensureDirectory(dir))
      return std::move(err);
  }

  MixCompileLayout layout;
  layout.outputRoot = outputRoot.str().str();
  layout.workDir = workDir.str().str();
  layout.objectDir = objectDir.str().str();
  layout.outDir = outDir.str().str();
  layout.outBinDir = outBinDir.str().str();
  layout.outIncludeDir = outIncludeDir.str().str();
  layout.mergeDir = mergeDir.str().str();
  layout.launcherDir = launcherDir.str().str();
  layout.stubDir = stubDir.str().str();
  layout.hostDir = hostDir.str().str();
  layout.hostObjectsDir = hostObjectsDir.str().str();
  layout.aicMergeDir = aicMergeDir.str().str();
  layout.aivMergeDir = aivMergeDir.str().str();
  layout.aicObj = joinPath(layout.objectDir, kernelName.str() + "_aic.o");
  layout.aivObj = joinPath(layout.objectDir, kernelName.str() + "_aiv.o");
  layout.aicRelocObj =
      joinPath(layout.objectDir, kernelName.str() + "_aic.reloc.o");
  layout.aivRelocObj =
      joinPath(layout.objectDir, kernelName.str() + "_aiv.reloc.o");
  layout.mergedDeviceObj = joinPath(layout.outDir, "device.o");
  layout.manifestPath = joinPath(layout.outDir, "manifest.txt");
  layout.metadataPath = joinPath(layout.outDir, "mix_metadata.json");
  layout.analysisPath = joinPath(layout.workDir, "analysis.txt");
  layout.mergeDeviceObj = joinPath(layout.mergeDir, "device.o");
  layout.hostStubObjectPath = joinPath(layout.stubDir, "host_stub.o");
  layout.kernelSoPath =
      joinPath(layout.outDir, "lib" + kernelName.str() + "_packed.so");
  layout.mixFlagPath = joinPath(layout.mergeDir, "mix_build.flag");
  layout.runnerMainPath = joinPath(layout.workDir, "main.cpp");
  layout.runnerTilingPath =
      joinPath(layout.workDir, kernelName.str() + "_tiling.cpp");
  layout.runnerDataUtilsPath = joinPath(layout.workDir, "data_utils.h");
  layout.runnerBinaryPath = joinPath(layout.outBinDir, "mix_runner");
  layout.tilingArtifactPath = joinPath(layout.outDir, "tiling.bin");
  layout.launchInfoPath = joinPath(layout.outDir, "launch_info.txt");
  layout.preprocessProbeDir = preprocessProbeDir.str().str();
  layout.aicProbeObject =
      joinPath(layout.preprocessProbeDir, kernelName.str() + "_aic_probe.o");
  layout.aivProbeObject =
      joinPath(layout.preprocessProbeDir, kernelName.str() + "_aiv_probe.o");
  return layout;
}

} // namespace mlir::runtime
