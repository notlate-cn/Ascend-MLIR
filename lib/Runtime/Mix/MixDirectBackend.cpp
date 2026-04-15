#include "Runtime/MixDirectBackend.h"
#include "Runtime/MixCommandBuilder.h"
#include "Runtime/MixAbi.h"
#include "Runtime/MixAbiExtractor.h"
#include "Runtime/MixCompileMetadata.h"
#include "Runtime/NpyIO.h"
#include "Runtime/PathUtils.h"
#include "Runtime/MixSourceAnalyzer.h"
#include "Runtime/MixStubTemplate.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <utility>

namespace mlir::runtime {

namespace {

static llvm::Error writeTextFile(llvm::StringRef path, llvm::StringRef content) {
  std::ofstream os(path.str(), std::ios::binary);
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot write file: %s", path.str().c_str());
  os << content.str();
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to write file: %s", path.str().c_str());
  return llvm::Error::success();
}

static llvm::Expected<std::string> readTextFileOrErr(llvm::StringRef path) {
  auto bufferOr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOr)
    return llvm::createStringError(bufferOr.getError(),
                                   "Cannot read file: %s", path.str().c_str());
  return (*bufferOr)->getBuffer().str();
}

static llvm::Expected<uint32_t> readBlockDimFromLaunchInfo(llvm::StringRef path) {
  auto textOr = readTextFileOrErr(path);
  if (!textOr)
    return textOr.takeError();

  llvm::SmallVector<llvm::StringRef> lines;
  llvm::StringRef(*textOr).split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (!line.starts_with("block_dim="))
      continue;
    uint64_t value = 0;
    if (line.drop_front(strlen("block_dim=")).getAsInteger(10, value))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Invalid block_dim in launch info: %s",
                                     path.str().c_str());
    return static_cast<uint32_t>(value);
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Missing block_dim in launch info: %s",
                                 path.str().c_str());
}

struct MixGeneratedConfig {
  std::vector<std::string> mixSources;
  llvm::StringMap<std::vector<std::string>> definitionsBySource;
};

struct MixPreprocessOutputs {
  std::string preprocessedSourcePath;
  std::string compileCommandsPath;
  std::string preprocessCommand;
  std::string generatedDir;
  std::string includeDir;
  std::string hostStubPath;
  std::string launcherHeaderPath;
  std::string actualLauncherKernelName;
  std::string aicConfigPath;
  std::string aivConfigPath;
};

static constexpr const char *kStageAnalyzeSource = "analyze source";
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
static constexpr const char *kStageBuildRunner = "build host runner";
static constexpr const char *kStageEmitTilingArtifact = "emit tiling artifact";

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

static llvm::Error ensureDirectory(llvm::StringRef path) {
  if (auto ec = llvm::sys::fs::create_directories(path))
    return llvm::createStringError(ec, "Cannot create directory: %s",
                                   path.str().c_str());
  return llvm::Error::success();
}

static llvm::Error copyFileOrErr(llvm::StringRef from, llvm::StringRef to) {
  if (auto ec = llvm::sys::fs::copy_file(from, to))
    return llvm::createStringError(ec, "Cannot copy %s -> %s", from.str().c_str(),
                                   to.str().c_str());
  return llvm::Error::success();
}

static llvm::Expected<uint64_t> getFileSizeOrErr(llvm::StringRef path) {
  uint64_t size = 0;
  if (auto ec = llvm::sys::fs::file_size(path, size))
    return llvm::createStringError(ec, "Cannot stat file: %s", path.str().c_str());
  return size;
}

static uint64_t alignTo4(uint64_t size) {
  return (size + 3ULL) & ~3ULL;
}

static llvm::Error ensureFileExists(llvm::StringRef path, llvm::StringRef stage,
                                    llvm::StringRef context = {}) {
  if (!llvm::sys::fs::exists(path))
  {
    const std::string contextText = context.str();
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] completed but did not create expected file: %s%s%s",
        stage.str().c_str(), path.str().c_str(), context.empty() ? "" : " (",
        context.empty() ? "" : contextText.c_str(),
        context.empty() ? "" : ")");
  }
  return llvm::Error::success();
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

static std::string joinPath(llvm::StringRef base, llvm::StringRef leaf);
static llvm::Error runProcess(const std::vector<std::string> &args,
                              llvm::StringRef stage,
                              llvm::StringRef context = {});
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

static std::string joinDefinitions(llvm::ArrayRef<std::string> defs) {
  std::string out;
  for (size_t i = 0; i < defs.size(); ++i) {
    if (i)
      out.push_back(';');
    out += defs[i];
  }
  return out;
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

static llvm::Expected<MixGeneratedConfig>
parseGeneratedConfig(llvm::StringRef path) {
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
      llvm::StringRef rest = line.drop_front(strlen("set_source_files_properties("));
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

static llvm::Expected<MixPreprocessOutputs>
runPreprocessStage(llvm::StringRef workDir, llvm::StringRef sourcePath,
                   llvm::StringRef kernelName, llvm::StringRef socVersion,
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
  if (auto err = runProcess(preprocessCmd, kStagePreprocessSource,
                            preprocessContext))
    return std::move(err);
  if (auto err =
          ensureFileExists(preprocessedPath, kStagePreprocessSource,
                           preprocessContext))
    return std::move(err);

  const std::string compileCommandsPath =
      joinPath(preprocessedDir, "compile_commands.json");
  llvm::SmallString<256> compileDir;
  if (auto ec = llvm::sys::fs::current_path(compileDir))
    return llvm::createStringError(ec, "Cannot resolve compile directory");
  const std::string fakeCommand = renderCommandForCompileCommands(preprocessCmd);
  if (auto err = writeCompileCommandsJson(compileCommandsPath, compileDir,
                                          fakeCommand, sourcePath))
    return std::move(err);

  const std::vector<std::string> extractCmd = buildExtractHostStubCommand(
      preprocessedPath, generatedDir, includeDir, {aivProbeObject.str()},
      {aicProbeObject.str()},
      compileCommandsPath, "c220", "sim");
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
  if (auto err = runProcess(updateCmd, kStageFinalizeHostStub, finalizeContext))
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

static bool useLegacyMixRunner() {
  if (const char *value = std::getenv("AFIR_MIX_USE_LEGACY_RUNNER"))
    return *value != '\0' && std::strcmp(value, "0") != 0;
  return false;
}

static std::string getRunnerToolkitHome() {
  return findAscendHome();
}

static std::string getRunnerLib64(const std::string &ascendHome) {
  return findAscendLib64Dir(ascendHome);
}

static std::string getRunnerSimLibDir(const std::string &ascendHome,
                                      llvm::StringRef socVersion) {
  return findAscendSimulatorLibDir(ascendHome, socVersion);
}

static std::string getRunnerDeviceLibDir(const std::string &ascendHome) {
  return findAscendDeviceLibDir(ascendHome);
}

static llvm::Error writeFileOrErr(llvm::StringRef path, llvm::StringRef content) {
  return writeTextFile(path, content);
}

static std::string emitRunnerDataUtilsHeader() {
  return R"runner(#pragma once
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "acl/acl.h"

#define CHECK_ACL(x) do { aclError __ret = (x); if (__ret != ACL_ERROR_NONE) { \
  std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
} } while (0)

static bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize) {
  struct stat sBuf;
  if (stat(filePath.data(), &sBuf) == -1) return false;
  if (S_ISREG(sBuf.st_mode) == 0) return false;
  std::ifstream file(filePath, std::ios::binary);
  if (!file.is_open()) return false;
  std::filebuf *buf = file.rdbuf();
  size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
  if (size == 0 || size > bufferSize) return false;
  buf->pubseekpos(0, std::ios::in);
  buf->sgetn(static_cast<char *>(buffer), size);
  fileSize = size;
  return true;
}

static bool WriteFile(const std::string &filePath, const void *buffer, size_t size) {
  if (buffer == nullptr) return false;
  int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (fd < 0) return false;
  size_t writeSize = write(fd, buffer, size);
  (void)close(fd);
  return writeSize == size;
}
)runner";
}

static std::string emitRunnerMainSource(llvm::StringRef kernelName,
                                        const MixAbiMetadata &abi);
static std::string emitRunnerTilingSource(const MixAbiMetadata &abi);

static llvm::Error runProcess(const std::vector<std::string> &args,
                              llvm::StringRef stage,
                              llvm::StringRef context) {
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
  if (ret != 0) {
    const std::string program = argv[0].str();
    const std::string contextText = context.str();
    std::string renderedCommand = renderCommandForDebug(args);
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] program=%s%s%s failed (exit %d): %s\n  command: %s",
        stage.str().c_str(), program.c_str(),
        context.empty() ? "" : " inputs: ",
        context.empty() ? "" : contextText.c_str(), ret, errMsg.c_str(),
        renderedCommand.c_str());
  }
  return llvm::Error::success();
}

static std::string joinPath(llvm::StringRef base, llvm::StringRef leaf) {
  llvm::SmallString<256> joined(base);
  llvm::sys::path::append(joined, leaf);
  return joined.str().str();
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

static uint64_t getElementBytes(DType dtype) {
  if (dtype == DType::F16)
    return sizeof(int16_t);
  if (dtype == DType::F32)
    return sizeof(float);
  return 0;
}

static uint64_t getTensorElementCount(llvm::ArrayRef<int64_t> shape) {
  uint64_t count = 1;
  for (int64_t dim : shape)
    count *= static_cast<uint64_t>(dim);
  return count;
}

static bool hasDynamicShape(llvm::ArrayRef<int64_t> shape) {
  for (int64_t dim : shape)
    if (dim < 0)
      return true;
  return false;
}

static llvm::Expected<size_t>
findFirstDynamicTensorIndex(llvm::ArrayRef<MixAbiTensorDesc> tensors) {
  for (size_t i = 0; i < tensors.size(); ++i)
    if (hasDynamicShape(tensors[i].shape))
      return i;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "no dynamic tensor shape found");
}

static uint64_t getTensorBytes(const MixAbiTensorDesc &tensor) {
  return getTensorElementCount(tensor.shape) * getElementBytes(tensor.dtype);
}

static llvm::StringRef getDTypeName(DType dtype) {
  if (dtype == DType::F16)
    return "f16";
  if (dtype == DType::BF16)
    return "bf16";
  if (dtype == DType::F32)
    return "f32";
  if (dtype == DType::INT8)
    return "int8";
  if (dtype == DType::INT32)
    return "int32";
  if (dtype == DType::INT64)
    return "int64";
  return "unknown";
}

static std::string buildOrdinalTensorNpyName(bool isOutput, size_t index) {
  return (llvm::Twine(isOutput ? "output" : "input") + llvm::Twine(index) +
          ".npy")
      .str();
}

static llvm::Expected<std::string>
resolveTensorNpyPath(llvm::StringRef npyDir, const MixAbiTensorDesc &tensor,
                     size_t index, bool isOutput) {
  const std::string namedPath = joinPath(npyDir, tensor.name + ".npy");
  if (llvm::sys::fs::exists(namedPath))
    return namedPath;
  const std::string ordinalPath =
      joinPath(npyDir, buildOrdinalTensorNpyName(isOutput, index));
  if (llvm::sys::fs::exists(ordinalPath))
    return ordinalPath;
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "RuntimeMix cannot resolve concrete %s tensor shape for ABI tensor '%s': "
      "expected either %s or %s",
      isOutput ? "output" : "input", tensor.name.c_str(), namedPath.c_str(),
      ordinalPath.c_str());
}

static llvm::Error reconcileTensorWithNpy(MixAbiTensorDesc &tensor,
                                          llvm::StringRef npyPath) {
  auto arrayOr = LoadNpy(npyPath.str());
  if (!arrayOr)
    return arrayOr.takeError();
  if (arrayOr->dtype != tensor.dtype)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix ABI dtype mismatch for tensor '%s': MLIR expects %s but %s "
        "contains %s",
        tensor.name.c_str(), getDTypeName(tensor.dtype).str().c_str(),
        npyPath.str().c_str(), getDTypeName(arrayOr->dtype).str().c_str());
  if (!tensor.shape.empty() && tensor.shape.size() != arrayOr->shape.size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix ABI rank mismatch for tensor '%s': MLIR rank=%zu but %s "
        "rank=%zu",
        tensor.name.c_str(), tensor.shape.size(), npyPath.str().c_str(),
        arrayOr->shape.size());
  if (tensor.shape.empty()) {
    tensor.shape = arrayOr->shape;
    return llvm::Error::success();
  }
  for (size_t i = 0; i < arrayOr->shape.size(); ++i) {
    if (tensor.shape[i] >= 0 && tensor.shape[i] != arrayOr->shape[i])
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "RuntimeMix ABI shape mismatch for tensor '%s' dim %zu: MLIR expects "
          "%lld but %s provides %lld",
          tensor.name.c_str(), i, static_cast<long long>(tensor.shape[i]),
          npyPath.str().c_str(), static_cast<long long>(arrayOr->shape[i]));
    tensor.shape[i] = arrayOr->shape[i];
  }
  return llvm::Error::success();
}

static llvm::Error resolveDynamicShapesFromNpyDir(MixAbiMetadata &abi,
                                                  llvm::StringRef npyDir) {
  for (size_t i = 0; i < abi.inputs.size(); ++i) {
    if (!hasDynamicShape(abi.inputs[i].shape))
      continue;
    auto npyPathOr = resolveTensorNpyPath(npyDir, abi.inputs[i], i, false);
    if (!npyPathOr)
      return npyPathOr.takeError();
    if (auto err = reconcileTensorWithNpy(abi.inputs[i], *npyPathOr))
      return err;
  }
  for (size_t i = 0; i < abi.outputs.size(); ++i) {
    if (!hasDynamicShape(abi.outputs[i].shape))
      continue;
    auto npyPathOr = resolveTensorNpyPath(npyDir, abi.outputs[i], i, true);
    if (!npyPathOr)
      return npyPathOr.takeError();
    if (auto err = reconcileTensorWithNpy(abi.outputs[i], *npyPathOr))
      return err;
  }
  return llvm::Error::success();
}

static llvm::StringRef getAclDataType(DType dtype) {
  if (dtype == DType::F16)
    return "DataType::DT_FLOAT16";
  if (dtype == DType::F32)
    return "DataType::DT_FLOAT";
  return "DataType::DT_UNDEFINED";
}

static std::string emitRunnerMainSource(llvm::StringRef kernelName,
                                        const MixAbiMetadata &abi) {
  const auto &inputA = abi.inputs[0];
  const auto &inputB = abi.inputs[1];
  const auto &output = abi.outputs[0];
  const MixAbiTensorDesc *bias =
      abi.inputs.size() > 2 ? &abi.inputs[2] : nullptr;
  std::ostringstream os;
  os << "#include \"data_utils.h\"\n"
     << "#include \"kernel_tiling/kernel_tiling.h\"\n"
     << "#include \"tiling/platform/platform_ascendc.h\"\n"
     << "#include \"acl/acl.h\"\n"
     << "#include \"aclrtlaunch_" << kernelName.str() << ".h\"\n"
     << "#include <cstdint>\n"
     << "#include <cstdlib>\n"
     << "#include <cstring>\n"
     << "#include <string>\n\n"
     << "extern \"C\" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf);\n"
     << "extern \"C\" uint32_t GetBlockDim(const char *socVersion);\n\n"
     << "int main(int argc, char *argv[]) {\n"
     << "  std::string inputDir = \"./input\";\n"
     << "  std::string outputFile = \"./output/" << output.runtimeFile << "\";\n"
     << "  std::string emitTilingFile;\n"
     << "  std::string emitLaunchInfoFile;\n"
     << "  for (int i = 1; i < argc; ++i) {\n"
     << "    std::string arg = argv[i];\n"
     << "    if (arg == \"--input-dir\" && i + 1 < argc) inputDir = argv[++i];\n"
     << "    else if (arg == \"--output-file\" && i + 1 < argc) outputFile = argv[++i];\n"
     << "    else if (arg == \"--emit-tiling-file\" && i + 1 < argc) emitTilingFile = argv[++i];\n"
     << "    else if (arg == \"--emit-launch-info\" && i + 1 < argc) emitLaunchInfoFile = argv[++i];\n"
     << "  }\n\n"
     << "  const char *socVersion = SOC_VERSION;\n"
     << "  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);\n"
     << "  size_t aFileSize = " << getTensorBytes(inputA) << ";\n"
     << "  size_t bFileSize = " << getTensorBytes(inputB) << ";\n"
     << "  size_t cFileSize = " << getTensorBytes(output) << ";\n";
  if (bias)
    os << "  size_t biasFileSize = " << getTensorBytes(*bias) << ";\n";
  os << "  size_t userWorkspaceSize = "
     << static_cast<unsigned long long>(abi.workspaceBytes) << ";\n"
     << "  size_t systemWorkspaceSize = ascendcPlatform ? static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize()) : 0;\n"
     << "  size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;\n"
     << "  size_t tilingFileSize = sizeof(TCubeTiling);\n"
     << "  uint8_t *tilingBuf = static_cast<uint8_t *>(malloc(tilingFileSize));\n"
     << "  GenerateTiling(socVersion, tilingBuf);\n"
     << "  uint32_t blockDim = GetBlockDim(socVersion);\n"
     << "  auto ensureParentDir = [&](const std::string& path) {\n"
     << "    size_t lastSlash = path.find_last_of('/');\n"
     << "    if (lastSlash != std::string::npos) {\n"
     << "      std::string outDir = path.substr(0, lastSlash);\n"
     << "      std::string mkdirCmd = \"mkdir -p \" + outDir;\n"
     << "      (void)std::system(mkdirCmd.c_str());\n"
     << "    }\n"
     << "  };\n"
     << "  if (!emitTilingFile.empty()) {\n"
     << "    ensureParentDir(emitTilingFile);\n"
     << "    if (!WriteFile(emitTilingFile, tilingBuf, tilingFileSize)) {\n"
     << "      free(tilingBuf);\n"
     << "      return 5;\n"
     << "    }\n"
     << "  }\n"
     << "  if (!emitLaunchInfoFile.empty()) {\n"
     << "    ensureParentDir(emitLaunchInfoFile);\n"
     << "    std::string payload = std::string(\"block_dim=\") + std::to_string(blockDim) + \"\\n\";\n"
     << "    if (!WriteFile(emitLaunchInfoFile, payload.data(), payload.size())) {\n"
     << "      free(tilingBuf);\n"
     << "      return 5;\n"
     << "    }\n"
     << "  }\n"
     << "  if (!emitTilingFile.empty() || !emitLaunchInfoFile.empty()) {\n"
     << "    free(tilingBuf);\n"
     << "    return 0;\n"
     << "  }\n\n"
     << "  CHECK_ACL(aclInit(nullptr));\n"
     << "  int32_t deviceId = 0;\n"
     << "  CHECK_ACL(aclrtSetDevice(deviceId));\n"
     << "  aclrtStream stream = nullptr;\n"
     << "  CHECK_ACL(aclrtCreateStream(&stream));\n\n"
     << "  auto readHostToDevice = [&](const std::string& path, size_t bytes, uint8_t** host, uint8_t** device) {\n"
     << "    size_t fileSize = 0;\n"
     << "    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(host), bytes));\n"
     << "    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(device), bytes, ACL_MEM_MALLOC_HUGE_FIRST));\n"
     << "    if (!ReadFile(path, fileSize, *host, bytes)) return false;\n"
     << "    CHECK_ACL(aclrtMemcpy(*device, bytes, *host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));\n"
     << "    return true;\n"
     << "  };\n\n"
     << "  uint8_t *inputAHost = nullptr, *inputADevice = nullptr;\n"
     << "  uint8_t *inputBHost = nullptr, *inputBDevice = nullptr;\n"
     << "  uint8_t *inputBiasHost = nullptr, *inputBiasDevice = nullptr;\n"
     << "  uint8_t *outputCHost = nullptr, *outputCDevice = nullptr;\n"
     << "  uint8_t *tilingHost = nullptr, *tilingDevice = nullptr;\n"
     << "  uint8_t *workspaceDevice = nullptr;\n\n"
     << "  if (!readHostToDevice(inputDir + \"/" << inputA.runtimeFile
     << "\", aFileSize, &inputAHost, &inputADevice)) return 2;\n"
     << "  if (!readHostToDevice(inputDir + \"/" << inputB.runtimeFile
     << "\", bFileSize, &inputBHost, &inputBDevice)) return 2;\n";
  if (bias) {
    os << "  if (!readHostToDevice(inputDir + \"/" << bias->runtimeFile
       << "\", biasFileSize, &inputBiasHost, &inputBiasDevice)) return 2;\n\n";
  } else {
    os << "\n";
  }
  os << "  CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&outputCHost), cFileSize));\n"
     << "  CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&outputCDevice), cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));\n"
     << "  CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&tilingHost), tilingFileSize));\n"
     << "  CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&tilingDevice), tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));\n"
     << "  CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, tilingBuf, tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));\n"
     << "  CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));\n"
     << "  CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&workspaceDevice), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));\n\n"
     << "  ACLRT_LAUNCH_KERNEL(" << kernelName.str() << ")(blockDim, stream, inputADevice, inputBDevice, inputBiasDevice, outputCDevice, workspaceDevice, tilingDevice);\n"
     << "  CHECK_ACL(aclrtSynchronizeStream(stream));\n"
     << "  CHECK_ACL(aclrtMemcpy(outputCHost, cFileSize, outputCDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));\n\n"
     << "  size_t lastSlash = outputFile.find_last_of('/');\n"
     << "  if (lastSlash != std::string::npos) {\n"
     << "    std::string outDir = outputFile.substr(0, lastSlash);\n"
     << "    std::string mkdirCmd = \"mkdir -p \" + outDir;\n"
     << "    (void)std::system(mkdirCmd.c_str());\n"
     << "  }\n"
     << "  if (!WriteFile(outputFile, outputCHost, cFileSize)) return 3;\n\n"
     << "  CHECK_ACL(aclrtFree(inputADevice));\n"
     << "  CHECK_ACL(aclrtFreeHost(inputAHost));\n"
     << "  CHECK_ACL(aclrtFree(inputBDevice));\n"
     << "  CHECK_ACL(aclrtFreeHost(inputBHost));\n"
     << "  CHECK_ACL(aclrtFree(outputCDevice));\n"
     << "  CHECK_ACL(aclrtFreeHost(outputCHost));\n"
     << "  CHECK_ACL(aclrtFree(inputBiasDevice));\n"
     << "  CHECK_ACL(aclrtFreeHost(inputBiasHost));\n"
     << "  CHECK_ACL(aclrtFree(tilingDevice));\n"
     << "  CHECK_ACL(aclrtFreeHost(tilingHost));\n"
     << "  CHECK_ACL(aclrtFree(workspaceDevice));\n"
     << "  CHECK_ACL(aclrtDestroyStream(stream));\n"
     << "  CHECK_ACL(aclrtResetDevice(deviceId));\n"
     << "  CHECK_ACL(aclFinalize());\n"
     << "  free(tilingBuf);\n"
     << "  return 0;\n"
     << "}\n";
  return os.str();
}

static std::string emitRunnerTilingSource(const MixAbiMetadata &abi) {
  const auto &inputA = abi.inputs[0];
  const auto &inputB = abi.inputs[1];
  const auto &output = abi.outputs[0];
  const bool hasBias = abi.inputs.size() > 2;
  const int64_t m = output.shape[0];
  const int64_t n = output.shape[1];
  const int64_t k = inputA.shape[1];
  std::ostringstream os;
  os << "#include <cstdint>\n"
     << "#include \"tiling/tiling_api.h\"\n"
     << "#include \"tiling/platform/platform_ascendc.h\"\n\n"
     << "using namespace matmul_tiling;\n\n"
     << "extern \"C\" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf) {\n"
     << "  int M = " << m << ";\n"
     << "  int N = " << n << ";\n"
     << "  int K = " << k << ";\n"
     << "  optiling::TCubeTiling tilingData;\n"
     << "  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);\n"
     << "  MatmulApiTiling tilingApi(*ascendcPlatform);\n\n"
     << "  tilingApi.SetAType(TPosition::GM, CubeFormat::ND, "
     << getAclDataType(inputA.dtype).str() << ", false);\n"
     << "  tilingApi.SetBType(TPosition::GM, CubeFormat::ND, "
     << getAclDataType(inputB.dtype).str() << ", false);\n"
     << "  tilingApi.SetCType(TPosition::GM, CubeFormat::ND, "
     << getAclDataType(output.dtype).str() << ");\n";
  if (hasBias)
    os << "  tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND, "
       << getAclDataType(abi.inputs[2].dtype).str() << ");\n";
  os << "  tilingApi.SetOrgShape(M, N, K);\n"
     << "  tilingApi.SetShape(M, N, K);\n"
     << "  tilingApi.SetBias(" << (hasBias ? "true" : "false") << ");\n"
     << "  tilingApi.SetTraverse(MatrixTraverse::FIRSTM);\n"
     << "  tilingApi.SetFixSplit(M, N, -1);\n"
     << "  tilingApi.SetBufferSpace(-1, -1, -1);\n"
     << "  (void)tilingApi.GetTiling(tilingData);\n"
     << "  tilingData.SaveToBuffer(tilingBuf, tilingData.GetDataSize());\n"
     << "}\n\n"
     << "extern \"C\" uint32_t GetBlockDim(const char *socVersion) {\n"
     << "  int M = " << m << ";\n"
     << "  int N = " << n << ";\n"
     << "  int K = " << k << ";\n"
     << "  optiling::TCubeTiling tilingData;\n"
     << "  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);\n"
     << "  MatmulApiTiling tilingApi(*ascendcPlatform);\n\n"
     << "  tilingApi.SetAType(TPosition::GM, CubeFormat::ND, "
     << getAclDataType(inputA.dtype).str() << ", false);\n"
     << "  tilingApi.SetBType(TPosition::GM, CubeFormat::ND, "
     << getAclDataType(inputB.dtype).str() << ", false);\n"
     << "  tilingApi.SetCType(TPosition::GM, CubeFormat::ND, "
     << getAclDataType(output.dtype).str() << ");\n";
  if (hasBias)
    os << "  tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND, "
       << getAclDataType(abi.inputs[2].dtype).str() << ");\n";
  os << "  tilingApi.SetOrgShape(M, N, K);\n"
     << "  tilingApi.SetShape(M, N, K);\n"
     << "  tilingApi.SetBias(" << (hasBias ? "true" : "false") << ");\n"
     << "  tilingApi.SetTraverse(MatrixTraverse::FIRSTM);\n"
     << "  tilingApi.SetFixSplit(M, N, -1);\n"
     << "  tilingApi.SetBufferSpace(-1, -1, -1);\n"
     << "  (void)tilingApi.GetTiling(tilingData);\n"
     << "  return static_cast<uint32_t>(tilingData.get_usedCoreNum());\n"
     << "}\n";
  return os.str();
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

static llvm::Error writeDebugManifest(const MixAnalyzedKernel &analyzed,
                                      llvm::StringRef runtimeKernelName,
                                      const MixAbiMetadata &abi,
                                      llvm::StringRef sourcePath,
                                      llvm::StringRef hostSourcePath,
                                      llvm::StringRef preprocessCompileCommandsPath,
                                      llvm::StringRef preprocessCommand,
                                      llvm::StringRef preprocessGeneratedDir,
                                      llvm::StringRef generatedSourcePath,
                                      llvm::StringRef aicDefinitions,
                                      llvm::StringRef aivDefinitions,
                                      llvm::StringRef workDir,
                                      llvm::StringRef objectDir,
                                      llvm::StringRef outDir,
                                      llvm::StringRef mergeDir,
                                      llvm::StringRef launcherHeaderDir,
                                      llvm::StringRef hostStubSourcePath,
                                      llvm::StringRef hostStubObjectPath,
                                      llvm::StringRef kernelSoPath,
                                      llvm::StringRef mixFlagPath,
                                      llvm::StringRef runnerSourcePath,
                                      llvm::StringRef runnerBinaryPath,
                                      llvm::StringRef aicObj,
                                      llvm::StringRef aivObj,
                                      llvm::StringRef aicRelocObj,
                                      llvm::StringRef aivRelocObj,
                                      llvm::StringRef mergedDeviceObj,
                                      llvm::StringRef aicCompileCmd,
                                      llvm::StringRef aivCompileCmd,
                                      llvm::StringRef aicRelocCmd,
                                      llvm::StringRef aivRelocCmd,
                                      llvm::StringRef mergeCmd,
                                      llvm::StringRef hostCompileCmd,
                                      llvm::StringRef hostBishengObjectPath,
                                      llvm::StringRef hostBishengCmd,
                                      llvm::StringRef hostObjectDir,
                                      llvm::StringRef packCmd,
                                      llvm::StringRef linkCmd,
                                      llvm::StringRef recompileCmd,
                                      llvm::StringRef runnerCompileCmd,
                                      llvm::StringRef metadataPath,
                                      llvm::StringRef manifestPath) {
  std::string manifest;
  manifest += std::string("kernel_name=") + runtimeKernelName.str() + "\n";
  manifest += std::string("requested_kernel_name=") + analyzed.kernelName + "\n";
  manifest += std::string("soc_version=") + analyzed.socVersion + "\n";
  manifest += std::string("kernel_kind=mix\n");
  manifest += std::string("mix_resource_type=mix_1c1v\n");
  manifest += std::string("source_path=") + sourcePath.str() + "\n";
  if (!hostSourcePath.empty())
    manifest += std::string("host_source_path=") + hostSourcePath.str() + "\n";
  manifest += std::string("preprocess_compile_commands=") +
              preprocessCompileCommandsPath.str() + "\n";
  manifest += std::string("preprocess_command=") + preprocessCommand.str() +
              "\n";
  manifest += std::string("preprocess_generated_dir=") +
              preprocessGeneratedDir.str() + "\n";
  manifest += std::string("generated_source_path=") + generatedSourcePath.str() +
              "\n";
  manifest += std::string("aic_definitions=") + aicDefinitions.str() + "\n";
  manifest += std::string("aiv_definitions=") + aivDefinitions.str() + "\n";
  manifest += std::string("work_dir=") + workDir.str() + "\n";
  manifest += std::string("build_dir=") + objectDir.str() + "\n";
  manifest += std::string("install_dir=") + outDir.str() + "\n";
  manifest += std::string("object_dir=") + objectDir.str() + "\n";
  manifest += std::string("out_dir=") + outDir.str() + "\n";
  manifest += std::string("abi_kind=mix_gm_workspace_tiling\n");
  manifest += std::string("abi_metadata_path=") + manifestPath.str() + "\n";
  auto abiManifestOr = serializeMixAbiManifest(abi);
  if (!abiManifestOr)
    return abiManifestOr.takeError();
  manifest += *abiManifestOr;
  manifest += std::string("merge_obj_dir=") + mergeDir.str() + "\n";
  manifest += std::string("launcher_header_dir=") +
              launcherHeaderDir.str() + "\n";
  manifest += std::string("host_runner_path=") + runnerBinaryPath.str() + "\n";
  manifest += std::string("manifest_path=") + manifestPath.str() + "\n";
  if (!metadataPath.empty())
    manifest += std::string("metadata_path=") + metadataPath.str() + "\n";
  if (!hostStubSourcePath.empty())
    manifest += std::string("host_stub_source_path=") +
                hostStubSourcePath.str() + "\n";
  manifest += std::string("host_stub_object_path=") + hostStubObjectPath.str() +
              "\n";
  if (!hostBishengObjectPath.empty())
    manifest += std::string("host_bisheng_object=") +
                hostBishengObjectPath.str() + "\n";
  if (!hostObjectDir.empty())
    manifest += std::string("host_object_dir=") + hostObjectDir.str() + "\n";
  manifest += std::string("kernel_so_path=") + kernelSoPath.str() + "\n";
  manifest += std::string("mix_build_flag=") + mixFlagPath.str() + "\n";
  manifest += std::string("host_runner_source_path=") +
              runnerSourcePath.str() + "\n";
  manifest += std::string("launcher_symbol=aclrtlaunch_") +
              runtimeKernelName.str() + "\n";
  manifest += std::string("aic_entry=") + analyzed.aicEntry + "\n";
  manifest += std::string("aiv_entry=") + analyzed.aivEntry + "\n";
  manifest += std::string("aic_object=") + aicObj.str() + "\n";
  manifest += std::string("aiv_object=") + aivObj.str() + "\n";
  manifest += std::string("aic_reloc_object=") + aicRelocObj.str() + "\n";
  manifest += std::string("aiv_reloc_object=") + aivRelocObj.str() + "\n";
  manifest += std::string("bisheng_aic=") + aicCompileCmd.str() + "\n";
  manifest += std::string("bisheng_aiv=") + aivCompileCmd.str() + "\n";
  manifest += std::string("lld_reloc_aic=") + aicRelocCmd.str() + "\n";
  manifest += std::string("lld_reloc_aiv=") + aivRelocCmd.str() + "\n";
  manifest += std::string("lld_merge=") + mergeCmd.str() + "\n";
  manifest += std::string("host_compile_cmd=") + hostCompileCmd.str() + "\n";
  if (!hostBishengCmd.empty())
    manifest += std::string("host_bisheng_cmd=") + hostBishengCmd.str() + "\n";
  manifest += std::string("pack_cmd=") + packCmd.str() + "\n";
  manifest += std::string("host_link_cmd=") + linkCmd.str() + "\n";
  if (!recompileCmd.empty())
    manifest += std::string("recompile_cmd=") + recompileCmd.str() + "\n";
  manifest += std::string("host_runner_compile_cmd=") + runnerCompileCmd.str() +
              "\n";
  if (!mergedDeviceObj.empty())
    manifest += std::string("device_object_path=") + mergedDeviceObj.str() +
                "\n";
  return writeTextFile(manifestPath, manifest);
}

static MixCompileMetadataTensorDesc
makeMetadataTensorDesc(const MixAbiTensorDesc &tensor) {
  MixCompileMetadataTensorDesc out;
  out.name = tensor.name;
  out.dtype = getDTypeName(tensor.dtype).str();
  out.shape = tensor.shape;
  out.runtimeFile = tensor.runtimeFile;
  return out;
}

static llvm::Expected<std::string>
writeMixCompileMetadataFile(llvm::StringRef metadataPath,
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

static const char *mixResourceTypeToMetadataString(MixResourceType type) {
  switch (type) {
  case MixResourceType::Unknown:
    return "unknown";
  case MixResourceType::AIVOnly:
    return "aiv_only";
  case MixResourceType::AICOnly:
    return "aic_only";
  case MixResourceType::Mix1C1V:
    return "mix_1c1v";
  case MixResourceType::Mix1C2V:
    return "mix_1c2v";
  }
  return "unknown";
}

} // namespace

llvm::Expected<MixArtifact>
MixDirectBackend::compile(const MixDirectCompileConfig &cfg) {
  if (cfg.outputDir.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "RuntimeMix direct backend requires an output directory");
  if (cfg.kernelSrc.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "RuntimeMix direct backend requires a kernel source path");
  if (cfg.kernelName.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "RuntimeMix direct backend requires a kernel name");
  if (auto ascendHomeOr = requireAscendHome(); !ascendHomeOr)
    return ascendHomeOr.takeError();

  llvm::SmallString<256> outputRoot(cfg.outputDir);
  if (auto ec = llvm::sys::fs::make_absolute(outputRoot))
    return llvm::createStringError(ec, "Cannot resolve output directory: %s",
                                   cfg.outputDir.c_str());

  llvm::SmallString<256> workDir(outputRoot);
  llvm::sys::path::append(workDir, "work");
  llvm::SmallString<256> objectDir(outputRoot);
  llvm::sys::path::append(objectDir, "objects");
  llvm::SmallString<256> outDir(outputRoot);
  llvm::sys::path::append(outDir, "out");
  llvm::SmallString<256> outBinDir(outDir);
  llvm::sys::path::append(outBinDir, "bin");
  llvm::SmallString<256> outIncludeDir(outDir);
  llvm::sys::path::append(outIncludeDir, "include");
  llvm::sys::path::append(outIncludeDir, "ascendc_kernels_sim");
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

  if (auto err = ensureDirectory(outputRoot))
    return err;
  if (auto err = ensureDirectory(workDir))
    return err;
  if (auto err = ensureDirectory(objectDir))
    return err;
  if (auto err = ensureDirectory(outDir))
    return err;
  if (auto err = ensureDirectory(outBinDir))
    return err;
  if (auto err = ensureDirectory(outIncludeDir))
    return err;
  if (auto err = ensureDirectory(mergeDir))
    return err;
  if (auto err = ensureDirectory(launcherDir))
    return err;
  if (auto err = ensureDirectory(stubDir))
    return err;
  if (auto err = ensureDirectory(hostObjectsDir))
    return err;
  if (auto err = ensureDirectory(aicMergeDir))
    return err;
  if (auto err = ensureDirectory(aivMergeDir))
    return err;

  const std::string analyzeContext = makeStageContext({
      {"kernel", cfg.kernelName},
      {"source", cfg.kernelSrc},
      {"soc_version", cfg.socVersion},
  });
  llvm::SmallString<256> sourcePath(cfg.kernelSrc);
  if (auto ec = llvm::sys::fs::make_absolute(sourcePath))
    return llvm::createStringError(
        ec, "[%s] cannot resolve kernel source path: %s (inputs: %s)",
        kStageAnalyzeSource, cfg.kernelSrc.c_str(),
        analyzeContext.c_str());
  if (!llvm::sys::fs::exists(sourcePath))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] kernel source file not found: %s (inputs: %s)",
        kStageAnalyzeSource, sourcePath.c_str(),
        analyzeContext.c_str());

  auto analyzed = analyzeMixKernel(sourcePath, cfg.kernelName, cfg.socVersion);
  if (!analyzed)
  {
    const std::string analysisError = llvm::toString(analyzed.takeError());
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] inputs: %s: %s", kStageAnalyzeSource,
        analyzeContext.c_str(), analysisError.c_str());
  }

  const std::string aicObj = joinPath(objectDir, cfg.kernelName + "_aic.o");
  const std::string aivObj = joinPath(objectDir, cfg.kernelName + "_aiv.o");
  const std::string aicRelocObj =
      joinPath(objectDir, cfg.kernelName + "_aic.reloc.o");
  const std::string aivRelocObj =
      joinPath(objectDir, cfg.kernelName + "_aiv.reloc.o");
  const std::string mergedDeviceObj = joinPath(outDir, "device.o");
  const std::string manifestPath = joinPath(outDir, "manifest.txt");
  const std::string metadataPath = joinPath(outDir, "mix_metadata.json");
  const std::string analysisPath = joinPath(workDir, "analysis.txt");
  const std::string mergeDeviceObj = joinPath(mergeDir, "device.o");
  const std::string hostStubObjectPath = joinPath(stubDir, "host_stub.o");
  const std::string kernelSoPath =
      joinPath(outDir, "lib" + cfg.kernelName + "_packed.so");
  const std::string mixFlagPath = joinPath(mergeDir, "mix_build.flag");
  const std::string runnerMainPath = joinPath(workDir, "main.cpp");
  const std::string runnerTilingPath =
      joinPath(workDir, cfg.kernelName + "_tiling.cpp");
  const std::string runnerDataUtilsPath = joinPath(workDir, "data_utils.h");
  const std::string runnerBinaryPath = joinPath(outBinDir, "mix_runner");
  const std::string tilingArtifactPath = joinPath(outDir, "tiling.bin");
  const std::string launchInfoPath = joinPath(outDir, "launch_info.txt");
  const std::string tilingArtifactSource = "out/tiling.bin";

  llvm::SmallString<256> preprocessProbeDir(workDir);
  llvm::sys::path::append(preprocessProbeDir, "preprocess_probe");
  if (auto err = ensureDirectory(preprocessProbeDir))
    return err;
  const std::string aicProbeObject =
      joinPath(preprocessProbeDir, cfg.kernelName + "_aic_probe.o");
  const std::string aivProbeObject =
      joinPath(preprocessProbeDir, cfg.kernelName + "_aiv_probe.o");

  const std::vector<std::string> aicProbeCmd =
      buildBishengCommand(*analyzed, sourcePath, aicProbeObject,
                          MixCoreType::AIC);
  const std::vector<std::string> aivProbeCmd =
      buildBishengCommand(*analyzed, sourcePath, aivProbeObject,
                          MixCoreType::AIV);
  const std::string aicProbeContext = makeStageContext({
      {"kernel", cfg.kernelName},
      {"source", sourcePath},
      {"output", aicProbeObject},
  });
  const std::string aivProbeContext = makeStageContext({
      {"kernel", cfg.kernelName},
      {"source", sourcePath},
      {"output", aivProbeObject},
  });
  if (auto err =
          runProcess(aicProbeCmd, kStageAicPreprocessProbe, aicProbeContext))
    return err;
  if (auto err = ensureFileExists(aicProbeObject, kStageAicPreprocessProbe,
                                  aicProbeContext))
    return err;
  if (auto err =
          runProcess(aivProbeCmd, kStageAivPreprocessProbe, aivProbeContext))
    return err;
  if (auto err = ensureFileExists(aivProbeObject, kStageAivPreprocessProbe,
                                  aivProbeContext))
    return err;

  MixAnalyzedKernel deviceAnalyzed = *analyzed;
  std::string generatedSourcePath;
  std::string launcherHeaderPath;
  std::string hostStubSourcePath;
  std::string hostStubIncludeDir;
  std::string preprocessIncludeDir;
  std::string preprocessCompileCommandsPath;
  std::string preprocessCommand;
  std::string preprocessGeneratedDir;
  std::string hostSourcePath = sourcePath.str().str();
  std::string runtimeKernelName = cfg.kernelName;
  bool aicWasSynthesizedFromAiv = false;
  auto preprocessOr = runPreprocessStage(workDir, sourcePath, cfg.kernelName,
                                         cfg.socVersion, aivProbeObject,
                                         aicProbeObject);
  if (!preprocessOr)
    return preprocessOr.takeError();
  auto aicConfigOr = parseGeneratedConfig(preprocessOr->aicConfigPath);
  if (!aicConfigOr)
    return aicConfigOr.takeError();
  auto aivConfigOr = parseGeneratedConfig(preprocessOr->aivConfigPath);
  if (!aivConfigOr)
    return aivConfigOr.takeError();
  auto generatedSourceOr = findOnlyMixSourceOrErr(*aicConfigOr, *aivConfigOr);
  if (!generatedSourceOr)
    return generatedSourceOr.takeError();

  generatedSourcePath =
      resolveGeneratedSourcePath(preprocessOr->generatedDir, *generatedSourceOr);
  if (!llvm::sys::fs::exists(generatedSourcePath))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] generated mix source file not found: %s (kernel=%s)",
        kStagePreprocessSource, generatedSourcePath.c_str(),
        cfg.kernelName.c_str());

  deviceAnalyzed.aicDefines =
      definitionsForSource(*aicConfigOr, *generatedSourceOr);
  deviceAnalyzed.aivDefines =
      definitionsForSource(*aivConfigOr, *generatedSourceOr);
  auto appendDefineIfMissing = [](std::vector<std::string> &defs,
                                  llvm::StringRef needle) {
    if (llvm::find(defs, needle.str()) == defs.end())
      defs.push_back(needle.str());
  };
  // For vector-only kernels, the AIC probe produces an empty config.
  // Use MixSourceAnalyzer defines (which carry the correct flags like
  // __MIX_CORE_MACRO__, __DAV_C220_VEC__, HAVE_WORKSPACE, HAVE_TILING) and
  // synthesize a matching AIC define set by swapping the architecture flag
  // and entry name suffix.
  if (deviceAnalyzed.aicDefines.empty() && !deviceAnalyzed.aivDefines.empty()) {
    aicWasSynthesizedFromAiv = true;
    // Replace toolkit-generated AIV defines with analyzer-generated ones that
    // include all required flags. Preserve ONE_CORE_DUMP_SIZE if present.
    std::string coreDumpSize;
    for (const std::string &def : deviceAnalyzed.aivDefines) {
      if (llvm::StringRef(def).starts_with("ONE_CORE_DUMP_SIZE="))
        coreDumpSize = def;
    }
    deviceAnalyzed.aivDefines = analyzed->aivDefines;
    appendDefineIfMissing(deviceAnalyzed.aivDefines, "HAVE_WORKSPACE");
    appendDefineIfMissing(deviceAnalyzed.aivDefines, "HAVE_TILING");
    if (!coreDumpSize.empty())
      appendDefineIfMissing(deviceAnalyzed.aivDefines, coreDumpSize);
    for (const std::string &def : deviceAnalyzed.aivDefines) {
      std::string aicDef = def;
      // Replace AIV entry suffix with AIC entry suffix in the macro define.
      size_t pos = aicDef.find("_0_mix_aiv");
      if (pos != std::string::npos)
        aicDef.replace(pos, 10, "_0_mix_aic");
      // Replace vector architecture flag with cube architecture flag.
      pos = aicDef.find("__DAV_C220_VEC__");
      if (pos != std::string::npos)
        aicDef.replace(pos, 16, "__DAV_C220_CUBE__");
      deviceAnalyzed.aicDefines.push_back(std::move(aicDef));
    }
  }
  if (deviceAnalyzed.aicDefines.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] generated AIC config did not provide compile definitions for %s",
        kStagePreprocessSource, generatedSourceOr->c_str());
  if (deviceAnalyzed.aivDefines.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] generated AIV config did not provide compile definitions for %s",
        kStagePreprocessSource, generatedSourceOr->c_str());

  launcherHeaderPath = preprocessOr->launcherHeaderPath;
  hostStubSourcePath = preprocessOr->hostStubPath;
  runtimeKernelName =
      preprocessOr->actualLauncherKernelName.empty()
          ? runtimeKernelName
          : preprocessOr->actualLauncherKernelName;
  hostStubIncludeDir = preprocessOr->includeDir;
  preprocessIncludeDir = preprocessOr->includeDir;
  preprocessCompileCommandsPath = preprocessOr->compileCommandsPath;
  preprocessCommand = preprocessOr->preprocessCommand;
  preprocessGeneratedDir = preprocessOr->generatedDir;

  const std::string runnerLauncherCopyPath =
      joinPath(outIncludeDir, "aclrtlaunch_" + runtimeKernelName + ".h");

  const bool needsManualStubTemplate =
      launcherHeaderPath.empty() || aicWasSynthesizedFromAiv;
  if (needsManualStubTemplate) {
    hostStubSourcePath = joinPath(stubDir, "host_stub.cpp");
    hostStubIncludeDir = outIncludeDir.str().str();
    launcherHeaderPath = runnerLauncherCopyPath;
  }
  const std::vector<std::string> aicCmd =
      buildPreprocessedDeviceCompileCommand(generatedSourcePath, aicObj,
                                            MixCoreType::AIC,
                                            deviceAnalyzed.aicDefines);
  const std::vector<std::string> aivCmd =
      buildPreprocessedDeviceCompileCommand(generatedSourcePath, aivObj,
                                            MixCoreType::AIV,
                                            deviceAnalyzed.aivDefines);
  const std::vector<std::string> aicRelocCmd =
      buildLldRelocCommand(aicObj, aicRelocObj);
  const std::vector<std::string> aivRelocCmd =
      buildLldRelocCommand(aivObj, aivRelocObj);
  const std::vector<std::string> mergeCmd =
      buildFinalMergeCommand(aicRelocObj, aivRelocObj, mergeDeviceObj);
  const std::vector<std::string> hostCompileCmd =
      buildHostStubCompileCommand(hostStubSourcePath, hostStubObjectPath,
                                  hostStubIncludeDir);
  const std::vector<std::string> packCmd =
      buildPackCommand(hostStubObjectPath, mergeDir);
  const std::vector<std::string> hostLinkCmd =
      buildHostSharedLinkCommand(hostStubObjectPath, kernelSoPath,
                                 cfg.socVersion,
                                 findAscendDeviceLibDir(findAscendHome()));

  const std::string aicCompileContext = makeStageContext({
      {"kernel", cfg.kernelName},
      {"source", generatedSourcePath},
      {"output", aicObj},
      {"generated_dir", preprocessGeneratedDir},
  });
  const std::string aivCompileContext = makeStageContext({
      {"kernel", cfg.kernelName},
      {"source", generatedSourcePath},
      {"output", aivObj},
      {"generated_dir", preprocessGeneratedDir},
  });
  const std::string aicMergeContext = makeStageContext({
      {"input", aicObj},
      {"output", aicRelocObj},
      {"kernel", cfg.kernelName},
      {"generated_dir", preprocessGeneratedDir},
  });
  const std::string aivMergeContext = makeStageContext({
      {"input", aivObj},
      {"output", aivRelocObj},
      {"kernel", cfg.kernelName},
      {"generated_dir", preprocessGeneratedDir},
  });
  const std::string mergeContext = makeStageContext({
      {"aic_input", aicRelocObj},
      {"aiv_input", aivRelocObj},
      {"output", mergeDeviceObj},
      {"kernel", cfg.kernelName},
  });
  const std::string finalizeContext = makeStageContext({
      {"generated_dir", preprocessGeneratedDir},
      {"merge_dir", mergeDir},
      {"soc_version", cfg.socVersion},
      {"target", "ascendc_kernels_sim"},
  });
  const std::string hostCompileContext = makeStageContext({
      {"source", hostStubSourcePath},
      {"output", hostStubObjectPath},
      {"include_dir", hostStubIncludeDir},
      {"kernel", cfg.kernelName},
  });
  const std::string packContext = makeStageContext({
      {"input", hostStubObjectPath},
      {"add_dir", mergeDir},
      {"kernel", cfg.kernelName},
  });
  const std::string hostLinkContext = makeStageContext({
      {"input", hostStubObjectPath},
      {"output", kernelSoPath},
      {"soc_version", cfg.socVersion},
      {"kernel", cfg.kernelName},
  });
  const std::string tripleChevronHeaderPath =
      joinPath(preprocessIncludeDir.empty() ? hostStubIncludeDir
                                            : preprocessIncludeDir,
               "aclrtlaunch_triple_chevrons_func.h");
  const std::string hostBishengObjectPath =
      joinPath(hostObjectsDir,
               llvm::sys::path::filename(hostSourcePath).str() + ".o");
  const std::string hostObjectDir = hostDir.str().str();
  const std::vector<std::string> hostBishengCmd =
      buildHostBishengCommand(hostSourcePath, hostBishengObjectPath,
                              tripleChevronHeaderPath);
  const std::vector<std::string> recompileCmd =
      buildRecompileBinaryCommand(outputRoot, "ascendc_kernels_sim", hostDir);

  if (auto err = runProcess(aicCmd, kStageCompileAic, aicCompileContext))
    return err;
  if (auto err =
          ensureFileExists(aicObj, kStageCompileAic, aicCompileContext))
    return err;

  if (auto err = runProcess(aivCmd, kStageCompileAiv, aivCompileContext))
    return err;
  if (auto err =
          ensureFileExists(aivObj, kStageCompileAiv, aivCompileContext))
    return err;

  if (auto err = runProcess(aicRelocCmd, kStageMergeAic, aicMergeContext))
    return err;
  if (auto err =
          ensureFileExists(aicRelocObj, kStageMergeAic, aicMergeContext))
    return err;

  if (auto err = runProcess(aivRelocCmd, kStageMergeAiv, aivMergeContext))
    return err;
  if (auto err =
          ensureFileExists(aivRelocObj, kStageMergeAiv, aivMergeContext))
    return err;

  if (auto err = runProcess(mergeCmd, kStageMergeDevice, mergeContext))
    return err;
  if (auto err =
          ensureFileExists(mergeDeviceObj, kStageMergeDevice, mergeContext))
    return err;

  if (auto err = copyFileOrErr(mergeDeviceObj, mergedDeviceObj))
    return err;
  auto mixLen = getFileSizeOrErr(mergedDeviceObj);
  if (!mixLen)
    return mixLen.takeError();
  if (auto err = writeTextFile(mixFlagPath, ""))
    return err;

  if (needsManualStubTemplate) {
    MixStubTemplateArgs stubArgs;
    stubArgs.kernelName = runtimeKernelName;
    stubArgs.targetName = "ascendc_kernels_sim";
    stubArgs.socVersion = cfg.socVersion;
    stubArgs.launcherSymbol = "aclrtlaunch_" + runtimeKernelName;
    stubArgs.launcherHeaderPath = runnerLauncherCopyPath;
    stubArgs.hostStubSourcePath = hostStubSourcePath;
    stubArgs.mixLen = alignTo4(*mixLen);
    stubArgs.mixFileLen = *mixLen;
    stubArgs.aivOnly = false;
    if (auto err = writeMixStubTemplate(stubArgs))
      return err;
    launcherHeaderPath = runnerLauncherCopyPath;
  } else {
    const std::string lowerSocVersion = llvm::StringRef(cfg.socVersion).lower();
    const std::vector<std::string> finalizeHostStubCmd =
        buildUpdateHostStubCommand(preprocessGeneratedDir, mergeDir,
                                   lowerSocVersion, "ascendc_kernels_sim");
    if (auto err = runProcess(finalizeHostStubCmd, kStageFinalizeHostStub,
                              finalizeContext))
      return err;

    if (auto err = copyFileOrErr(launcherHeaderPath, runnerLauncherCopyPath)) {
      return err;
    }
  }

  if (auto err = runProcess(hostCompileCmd, kStageCompileHostStub,
                            hostCompileContext))
    return err;
  if (auto err = ensureFileExists(hostStubObjectPath, kStageCompileHostStub,
                                  hostCompileContext))
    return err;

  const std::string hostBishengContext = makeStageContext({
      {"source", hostSourcePath},
      {"output", hostBishengObjectPath},
      {"triple_chevron_header", tripleChevronHeaderPath},
      {"kernel", cfg.kernelName},
  });
  if (auto err = ensureFileExists(tripleChevronHeaderPath,
                                  kStageCompileHostBisheng,
                                  hostBishengContext))
    return err;
  if (auto err = runProcess(hostBishengCmd, kStageCompileHostBisheng,
                            hostBishengContext))
    return err;
  if (auto err = ensureFileExists(hostBishengObjectPath,
                                  kStageCompileHostBisheng,
                                  hostBishengContext))
    return err;

  if (auto err = runProcess(packCmd, kStagePack, packContext))
    return err;
  if (auto err = ensureFileExists(hostStubObjectPath, kStagePack, packContext))
    return err;

  if (auto err = runProcess(hostLinkCmd, kStageLinkHostStub, hostLinkContext))
    return err;
  if (auto err = ensureFileExists(kernelSoPath, kStageLinkHostStub,
                                  hostLinkContext))
    return err;

  const std::string recompileHostStubObjectPath =
      joinPath(stubDir, "host_stub.cpp.o");
  if (auto err = copyFileOrErr(hostStubObjectPath, recompileHostStubObjectPath))
    return err;
  std::vector<std::string> recompileLinkArgs = hostLinkCmd;
  for (std::string &arg : recompileLinkArgs) {
    if (arg == hostStubObjectPath)
      arg = recompileHostStubObjectPath;
  }
  const std::string recompileLinkCmd =
      renderCommandForCompileCommands(recompileLinkArgs);
  if (auto err = writeRecompileLinkFile(outputRoot, "ascendc_kernels_sim",
                                        recompileLinkCmd))
    return err;
  const std::string recompileContext = makeStageContext({
      {"root_dir", outputRoot.str()},
      {"target_name", "ascendc_kernels_sim"},
      {"add_dir", hostDir.str()},
      {"kernel", cfg.kernelName},
  });
  if (auto err = runProcess(recompileCmd, kStageRecompile, recompileContext))
    return err;
  const std::string recompiledKernelSoPath =
      joinPath(outDir, "lib" + runtimeKernelName + "_packed.so");
  if (auto err = ensureFileExists(recompiledKernelSoPath, kStageRecompile,
                                  recompileContext))
    return err;

  MixAbiMetadata abi;
  if (cfg.cannMlirPath && !cfg.cannMlirPath->empty()) {
    auto abiOr = extractMixAbiFromCannMlir(*cfg.cannMlirPath);
    if (!abiOr)
      return abiOr.takeError();
    abi = std::move(*abiOr);
    if (cfg.npyDir && !cfg.npyDir->empty()) {
      if (auto err = resolveDynamicShapesFromNpyDir(abi, *cfg.npyDir))
        return err;
    }
    if (auto inputIndexOr = findFirstDynamicTensorIndex(abi.inputs))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "RuntimeMix MLIR ABI extraction from %s produced unresolved dynamic "
          "input shape for tensor '%s'; pass --npy-dir with concrete IO data "
          "to resolve dynamic extents",
          cfg.cannMlirPath->c_str(), abi.inputs[*inputIndexOr].name.c_str());
    else
      llvm::consumeError(inputIndexOr.takeError());
    if (auto outputIndexOr = findFirstDynamicTensorIndex(abi.outputs))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "RuntimeMix MLIR ABI extraction from %s produced unresolved dynamic "
          "output shape for tensor '%s'; pass --npy-dir with concrete IO data "
          "to resolve dynamic extents",
          cfg.cannMlirPath->c_str(), abi.outputs[*outputIndexOr].name.c_str());
    else
      llvm::consumeError(outputIndexOr.takeError());
  } else {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix direct backend requires --cann-mlir to derive ABI and "
        "canonical IO metadata for kernel '%s'",
        cfg.kernelName.c_str());
  }
  if (abi.logicalKernelName.empty())
    abi.logicalKernelName = runtimeKernelName;
  abi.runtimeKernelName = runtimeKernelName;
  if (abi.launcherSymbol.empty())
    abi.launcherSymbol = "aclrtlaunch_" + runtimeKernelName;
  if (abi.aicEntry.empty())
    abi.aicEntry = runtimeKernelName + "_0_mix_aic";
  if (abi.aivEntry.empty())
    abi.aivEntry = runtimeKernelName + "_0_mix_aiv";
  if (abi.workspaceBytes == 0)
    abi.workspaceBytes = 16777216ULL;
  abi.workspaceMode = "fixed";
  abi.tilingMode = "generated_file";
  abi.tilingSource = tilingArtifactSource;
  const bool useLegacyRunner = useLegacyMixRunner();
  std::vector<std::string> tilingEmitCmd;
  std::vector<std::string> runnerCompileCmd;
  std::string runnerMainSourcePath;
  std::string runnerBinaryOutputPath;
  if (useLegacyRunner) {
    if (auto err =
            writeFileOrErr(runnerDataUtilsPath, emitRunnerDataUtilsHeader()))
      return err;
    if (auto err = writeFileOrErr(
            runnerMainPath, emitRunnerMainSource(runtimeKernelName, abi)))
      return err;
    if (auto err =
            writeFileOrErr(runnerTilingPath, emitRunnerTilingSource(abi)))
      return err;

    const std::string ascendHome = getRunnerToolkitHome();
    auto davSimLibDirOr = requireAscendDavSimulatorLibDir(ascendHome);
    if (!davSimLibDirOr)
      return davSimLibDirOr.takeError();
    const std::string runnerLib64 = getRunnerLib64(ascendHome);
    const std::string hostCannArch = getHostCannArchDir();
    const std::string runnerAltLib64 =
        hostCannArch.empty() ? std::string()
                             : ascendHome + "/" + hostCannArch + "/lib64";
    const std::string runnerDeviceLibDir = getRunnerDeviceLibDir(ascendHome);
    const std::string runnerSimLibDir =
        getRunnerSimLibDir(ascendHome, cfg.socVersion);
    const std::string davSimLibDir = *davSimLibDirOr;
    runnerCompileCmd = buildHostRunnerCompileCommand(
        workDir, launcherDir, outIncludeDir, runnerMainPath, runnerTilingPath,
        runnerBinaryPath, kernelSoPath, runnerLib64, runnerSimLibDir,
        davSimLibDir, runnerDeviceLibDir, cfg.socVersion);
    const std::string runnerBuildContext = makeStageContext({
        {"main_source", runnerMainPath},
        {"tiling_source", runnerTilingPath},
        {"kernel_so", kernelSoPath},
        {"runner_binary", runnerBinaryPath},
        {"soc_version", cfg.socVersion},
    });
    std::string runnerLdLibraryPath = runnerLib64;
    if (runnerAltLib64 != runnerLib64)
      runnerLdLibraryPath += ":" + runnerAltLib64;
    if (!runnerDeviceLibDir.empty())
      runnerLdLibraryPath += ":" + runnerDeviceLibDir;
    runnerLdLibraryPath += ":" + runnerSimLibDir + ":" + davSimLibDir +
                           ":${LD_LIBRARY_PATH:-}";
    tilingEmitCmd = {
        "/bin/bash",
        "-lc",
        "LD_LIBRARY_PATH='" + runnerLdLibraryPath + "' " + runnerBinaryPath +
            " --emit-tiling-file " + tilingArtifactPath +
            " --emit-launch-info " + launchInfoPath,
    };
    if (auto err = runProcess(runnerCompileCmd, kStageBuildRunner,
                              runnerBuildContext))
      return err;
    if (auto err = ensureFileExists(runnerBinaryPath, kStageBuildRunner,
                                    runnerBuildContext))
      return err;
    runnerMainSourcePath = runnerMainPath;
    runnerBinaryOutputPath = runnerBinaryPath;
  } else {
    tilingEmitCmd = buildMixTilingHelperCommand(
        runtimeKernelName, cfg.socVersion, abi.inputs[0].shape,
        abi.inputs[0].dtype, abi.inputs[1].shape, abi.inputs[1].dtype,
        abi.outputs[0].shape, abi.outputs[0].dtype,
        abi.inputs.size() > 2 ? std::optional<DType>(abi.inputs[2].dtype)
                              : std::nullopt,
        tilingArtifactPath, launchInfoPath);
  }
  const std::string tilingArtifactContext = makeStageContext({
      {useLegacyRunner ? "runner_binary" : "helper", tilingEmitCmd.front()},
      {"tiling_artifact", tilingArtifactPath},
      {"launch_info", launchInfoPath},
      {"kernel", runtimeKernelName},
      {"soc_version", cfg.socVersion},
  });
  if (auto err =
          runProcess(tilingEmitCmd, kStageEmitTilingArtifact, tilingArtifactContext))
    return err;
  if (auto err = ensureFileExists(tilingArtifactPath, kStageEmitTilingArtifact,
                                  tilingArtifactContext))
    return err;
  if (auto err = ensureFileExists(launchInfoPath, kStageEmitTilingArtifact,
                                  tilingArtifactContext))
    return err;

  auto blockDimOr = readBlockDimFromLaunchInfo(launchInfoPath);
  if (!blockDimOr)
    return blockDimOr.takeError();
  abi.blockDim = *blockDimOr;

  auto metadataPathOr = writeMixCompileMetadataFile(
      metadataPath, runtimeKernelName, cfg.socVersion,
      mixResourceTypeToMetadataString(MixResourceType::Mix1C1V),
      generatedSourcePath, deviceAnalyzed.aicDefines, deviceAnalyzed.aivDefines,
      mergedDeviceObj, kernelSoPath, tilingArtifactPath, launchInfoPath, abi,
      useLegacyRunner);
  if (!metadataPathOr)
    return metadataPathOr.takeError();

  if (auto err = writeTextFile(
          analysisPath,
          std::string("kernel_name=") + runtimeKernelName + "\n" +
              std::string("requested_kernel_name=") + analyzed->kernelName +
              "\n" +
              std::string("soc_version=") + analyzed->socVersion + "\n" +
              std::string("source_path=") + sourcePath.str().str() + "\n" +
              (hostSourcePath.empty() ? std::string{}
                                      : std::string("host_source_path=") +
                                            hostSourcePath + "\n") +
              std::string("generated_source_path=") + generatedSourcePath + "\n" +
              std::string("aic_definitions=") +
              joinDefinitions(deviceAnalyzed.aicDefines) + "\n" +
              std::string("aiv_definitions=") +
              joinDefinitions(deviceAnalyzed.aivDefines) + "\n" +
              std::string("launcher_symbol=aclrtlaunch_") + runtimeKernelName +
              "\n" + std::string("aic_entry=") + analyzed->aicEntry + "\n" +
              std::string("aiv_entry=") + analyzed->aivEntry + "\n" +
              std::string("aic_object=") + aicObj + "\n" +
              std::string("aiv_object=") + aivObj + "\n" +
              std::string("aic_reloc_object=") + aicRelocObj + "\n" +
              std::string("aiv_reloc_object=") + aivRelocObj + "\n" +
              std::string("device_object=") + mergedDeviceObj + "\n"))
    return err;

  if (auto err = writeDebugManifest(*analyzed, runtimeKernelName, abi, sourcePath,
                                    hostSourcePath,
                                    preprocessCompileCommandsPath,
                                    preprocessCommand,
                                    preprocessGeneratedDir,
                                    generatedSourcePath,
                                    joinDefinitions(deviceAnalyzed.aicDefines),
                                    joinDefinitions(deviceAnalyzed.aivDefines),
                                    workDir, objectDir, outDir, mergeDir,
                                    outIncludeDir.str(),
                                    hostStubSourcePath, hostStubObjectPath,
                                    kernelSoPath, mixFlagPath,
                                    runnerMainSourcePath, runnerBinaryOutputPath,
                                    aicObj, aivObj,
                                    aicRelocObj, aivRelocObj, mergedDeviceObj,
                                    renderCommandForDebug(aicCmd),
                                    renderCommandForDebug(aivCmd),
                                    renderCommandForDebug(aicRelocCmd),
                                    renderCommandForDebug(aivRelocCmd),
                                    renderCommandForDebug(mergeCmd),
                                    renderCommandForDebug(hostCompileCmd),
                                    hostBishengObjectPath,
                                    renderCommandForDebug(hostBishengCmd),
                                    hostObjectDir,
                                    renderCommandForDebug(packCmd),
                                    renderCommandForDebug(hostLinkCmd),
                                    renderCommandForDebug(recompileCmd),
                                    renderCommandForDebug(useLegacyRunner
                                                              ? runnerCompileCmd
                                                              : tilingEmitCmd),
                                    *metadataPathOr,
                                    manifestPath))
    return err;

  MixArtifact artifact;
  artifact.kernel_name = runtimeKernelName;
  artifact.soc_version = analyzed->socVersion;
  artifact.work_dir = workDir.str().str();
  artifact.build_dir = objectDir.str().str();
  artifact.install_dir = outDir.str().str();
  artifact.kernel_so_path = kernelSoPath;
  artifact.launcher_header_dir = outIncludeDir.str().str();
  artifact.host_runner_path = runnerBinaryOutputPath;
  artifact.host_stub_source_path = hostStubSourcePath;
  artifact.device_object_path = mergedDeviceObj;
  artifact.manifest_path = manifestPath;
  artifact.metadata_path = *metadataPathOr;
  artifact.abi_metadata_path = manifestPath;
  return artifact;
}

KernelArtifact normalizeMixArtifact(const MixArtifact &artifact, KernelKind kind,
                                    MixResourceType mixResourceType) {
  KernelArtifact normalized;
  normalized.kernelName = artifact.kernel_name;
  normalized.kernelKind = kind;
  normalized.mixResourceType = mixResourceType;
  normalized.socVersion = artifact.soc_version;
  normalized.artifactRoot = llvm::sys::path::parent_path(artifact.work_dir).str();
  if (normalized.artifactRoot.empty())
    normalized.artifactRoot = artifact.install_dir;
  normalized.deviceBinaryPath = artifact.device_object_path.empty()
                                    ? artifact.kernel_so_path
                                    : artifact.device_object_path;
  normalized.packedSharedObjectPath = artifact.kernel_so_path;
  normalized.manifestPath = artifact.manifest_path;
  normalized.metadataPath = artifact.metadata_path;
  return normalized;
}

} // namespace mlir::runtime
