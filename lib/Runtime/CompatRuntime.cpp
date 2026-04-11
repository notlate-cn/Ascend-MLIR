#include "Runtime/CompatRuntime.h"
#include "Runtime/NpyIO.h"
#include "Runtime/TilingSchema.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <fstream>
#include <cstring>
#include <iterator>
#include <map>
#include <sstream>
#include <utility>

namespace mlir::runtime {

namespace {

static std::vector<std::string> splitComma(const std::string &value) {
  std::vector<std::string> parts;
  std::istringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ','))
    parts.push_back(token);
  return parts;
}

static llvm::Expected<std::string>
makeTemporaryPath(llvm::StringRef prefix, llvm::StringRef fileName) {
  std::error_code ec;
  const std::filesystem::path tempRoot = std::filesystem::temp_directory_path(ec);
  if (ec) {
    return llvm::createStringError(ec,
                                   "cannot determine temp directory for validator");
  }

  llvm::SmallString<256> directoryPrefix(tempRoot.string());
  llvm::sys::path::append(directoryPrefix, prefix);
  llvm::SmallString<256> directory;
  if (auto createDirError =
          llvm::sys::fs::createUniqueDirectory(directoryPrefix, directory)) {
    return llvm::createStringError(createDirError,
                                   "cannot create temporary validator directory");
  }

  llvm::SmallString<256> filePath(directory);
  llvm::sys::path::append(filePath, fileName);
  return filePath.str().str();
}

static llvm::Expected<std::vector<uint8_t>>
readBinaryFile(llvm::StringRef path) {
  std::ifstream is(path.str(), std::ios::binary);
  if (!is) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot open %s", path.str().c_str());
  }
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(is)),
                             std::istreambuf_iterator<char>());
  if (!is.good() && !is.eof()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed reading %s", path.str().c_str());
  }
  return bytes;
}

static llvm::Error writeBinaryFile(llvm::StringRef path,
                                   llvm::ArrayRef<uint8_t> bytes) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    return llvm::createStringError(ec, "cannot open %s for writing",
                                   path.str().c_str());
  }
  os.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  os.close();
  if (ec) {
    return llvm::createStringError(ec, "cannot write %s", path.str().c_str());
  }
  return llvm::Error::success();
}

static llvm::Expected<std::vector<uint8_t>>
buildLegacyTilingBytes(llvm::StringRef params, llvm::StringRef layout) {
  std::vector<uint8_t> bytes;
  auto pvec = splitComma(params.str());
  auto lvec = splitComma(layout.str());
  for (size_t i = 0; i < pvec.size(); ++i) {
    auto eq = pvec[i].find('=');
    if (eq == std::string::npos) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "tiling params token missing '=': %s",
                                     pvec[i].c_str());
    }
    int64_t value = std::stoll(pvec[i].substr(eq + 1));
    std::string type = i < lvec.size() ? lvec[i] : "int64";
    if (type == "int64" || type == "int64_t") {
      uint8_t buf[8];
      std::memcpy(buf, &value, 8);
      bytes.insert(bytes.end(), buf, buf + 8);
    } else if (type == "int32" || type == "int32_t") {
      int32_t narrowed = static_cast<int32_t>(value);
      uint8_t buf[4];
      std::memcpy(buf, &narrowed, 4);
      bytes.insert(bytes.end(), buf, buf + 4);
    } else {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "unknown tiling type: %s",
                                     type.c_str());
    }
  }
  return bytes;
}

} // namespace

llvm::Expected<ArtifactCompileRequest>
buildCompatCompileRequest(const CompatCompileOptions &options) {
  if (options.kernelType != "vec" && options.kernelType != "cube" &&
      options.kernelType != "mix") {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported kernel type: %s",
                                   options.kernelType.c_str());
  }

  ArtifactCompileRequest request;
  request.kernelSource = options.kernelSourcePath;
  request.kernelName = options.requestedKernelName;
  if (options.kernelType == "cube")
    request.kernelKind = KernelKind::Cube;
  else if (options.kernelType == "mix")
    request.kernelKind = KernelKind::Mix;
  else
    request.kernelKind = KernelKind::Vec;
  request.socVersion = options.socVersion;
  request.outputDir = options.outputRoot;
  request.arch = options.arch;
  request.verbose = options.verbose;
  return request;
}

llvm::Expected<RunManifestSpec>
buildCompatSingleTaskRunManifest(const CompatValidateOptions &options) {
  const bool hasExpectedOutput = !options.expectedOutputPath.empty();
  const bool hasActualOutput = !options.actualOutputPath.empty();
  const bool hasExplicitOutputMetadata =
      options.actualOutputShape.has_value() && options.actualOutputDType.has_value();

  if (hasExpectedOutput && !hasActualOutput) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "expected output path requires an actual output path");
  }
  if (!hasExpectedOutput && hasActualOutput && !hasExplicitOutputMetadata) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "actual output path requires explicit output shape and dtype metadata");
  }

  RunManifestSpec manifest;
  manifest.backendKind = ExecutionBackendKind::Simulation;

  RunTaskSpec task;
  task.taskId = "main";
  task.artifactRoot = options.artifactRoot;

  for (size_t index = 0; index < options.inputPaths.size(); ++index) {
    TensorBinding binding;
    binding.name = "data" + std::to_string(index);
    binding.sourceKind = BindingSourceKind::ExternalFile;
    binding.path = options.inputPaths[index];
    task.invocation.inputs.push_back(std::move(binding));
  }

  if (hasExpectedOutput) {
    auto expectedArrOr = LoadNpy(options.expectedOutputPath);
    if (!expectedArrOr)
      return expectedArrOr.takeError();

    TensorBinding output;
    output.name = "out";
    output.sourceKind = BindingSourceKind::ExternalFile;
    output.path = options.actualOutputPath;
    output.shape = expectedArrOr->shape;
    output.dtype = expectedArrOr->dtype;
    task.invocation.outputs.push_back(std::move(output));

    TensorBinding expectedOutput;
    expectedOutput.name = "out";
    expectedOutput.sourceKind = BindingSourceKind::ExternalFile;
    expectedOutput.path = options.expectedOutputPath;
    task.invocation.expectedOutputs.push_back(std::move(expectedOutput));
  } else if (hasActualOutput) {
    TensorBinding output;
    output.name = "out";
    output.sourceKind = BindingSourceKind::ExternalFile;
    output.path = options.actualOutputPath;
    output.shape = options.actualOutputShape;
    output.dtype = options.actualOutputDType;
    task.invocation.outputs.push_back(std::move(output));
  }

  if (!options.tilingBinaryPath.empty()) {
    TilingBinding tiling;
    tiling.binaryPath = options.tilingBinaryPath;
    if (!options.tilingSchemaPath.empty())
      tiling.schemaPath = options.tilingSchemaPath;
    if (!options.tilingParams.empty())
      tiling.params = options.tilingParams;
    task.invocation.tiling = std::move(tiling);
  } else if (!options.tilingSchemaPath.empty() || !options.tilingParams.empty()) {
    TilingBinding tiling;
    tiling.schemaPath = options.tilingSchemaPath;
    tiling.params = options.tilingParams;
    task.invocation.tiling = std::move(tiling);
  }

  task.invocation.blockDim = options.blockDim;
  task.invocation.atol = options.atol;
  task.invocation.rtol = options.rtol;
  manifest.tasks.push_back(std::move(task));

  return manifest;
}

llvm::Expected<std::string>
prepareValidatorTilingBinaryPath(llvm::StringRef tilingBinFile,
                                 llvm::StringRef tilingSchemaFile,
                                 llvm::StringRef tilingParams,
                                 llvm::StringRef tilingLayout,
                                 llvm::raw_ostream *warningStream) {
  if (!tilingBinFile.empty()) {
    auto bytesOr = readBinaryFile(tilingBinFile);
    if (!bytesOr)
      return bytesOr.takeError();
    auto pathOr = makeTemporaryPath("ascendc-validator-tiling", "tiling.bin");
    if (!pathOr)
      return pathOr.takeError();
    if (auto err = writeBinaryFile(*pathOr, *bytesOr))
      return err;
    return *pathOr;
  }

  if (tilingParams.empty())
    return std::string{};

  if (!tilingSchemaFile.empty()) {
    auto schemaOrErr = TilingSchema::fromJson(tilingSchemaFile);
    if (!schemaOrErr)
      return schemaOrErr.takeError();

    auto pvec = splitComma(tilingParams.str());
    std::map<std::string, int64_t> pmap;
    for (const std::string &token : pvec) {
      auto eq = token.find('=');
      if (eq == std::string::npos) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "tiling params token missing '=': %s",
                                       token.c_str());
      }
      pmap[token.substr(0, eq)] = std::stoll(token.substr(eq + 1));
    }

    std::vector<std::pair<std::string, int64_t>> namedParams;
    for (const auto &field : schemaOrErr->fields()) {
      auto it = pmap.find(field.name);
      if (it == pmap.end()) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "tiling params missing field '%s' required by schema",
            field.name.c_str());
      }
      namedParams.push_back({field.name, it->second});
    }

    for (const auto &kv : pmap) {
      bool found = false;
      for (const auto &field : schemaOrErr->fields()) {
        if (field.name == kv.first) {
          found = true;
          break;
        }
      }
      if (!found && warningStream) {
        *warningStream << "Warning: --tiling-params field '" << kv.first
                       << "' not in schema (ignored)\n";
      }
    }

    auto bytesOrErr = schemaOrErr->pack(namedParams);
    if (!bytesOrErr)
      return bytesOrErr.takeError();

    auto pathOr = makeTemporaryPath("ascendc-validator-tiling", "tiling.bin");
    if (!pathOr)
      return pathOr.takeError();
    if (auto err = writeBinaryFile(*pathOr, *bytesOrErr))
      return err;
    return *pathOr;
  }

  auto bytesOrErr = buildLegacyTilingBytes(tilingParams, tilingLayout);
  if (!bytesOrErr)
    return bytesOrErr.takeError();
  auto pathOr = makeTemporaryPath("ascendc-validator-tiling", "tiling.bin");
  if (!pathOr)
    return pathOr.takeError();
  if (auto err = writeBinaryFile(*pathOr, *bytesOrErr))
    return err;
  return *pathOr;
}

} // namespace mlir::runtime
