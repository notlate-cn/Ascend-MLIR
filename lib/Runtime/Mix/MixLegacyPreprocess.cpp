#include "MixLegacyCompileCompatInternal.h"

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

} // namespace mlir::runtime
