#include "Runtime/Mix/MixLegacyCompileCompat.h"

#include "Runtime/MixCommandBuilder.h"
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

#include <cstring>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <utility>

namespace mlir::runtime {

namespace {

static constexpr const char *kStagePreprocessSource = "preprocess source";
static constexpr const char *kStageExtractHostStub = "extract host stub";
static constexpr const char *kStageFinalizeHostStub = "finalize host stub";

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

} // namespace mlir::runtime
