#include "Runtime/Artifact/DebugCase.h"

#include "Runtime/NpyIO.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir::runtime {

namespace {

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Value *value, const char *fieldName) {
  if (!value)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "missing required object field: %s",
                                   fieldName);
  if (auto *object = value->getAsObject())
    return object;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "field must be an object: %s", fieldName);
}

llvm::Expected<std::string>
requireString(const llvm::json::Object &object, const char *fieldName) {
  if (auto value = object.getString(fieldName))
    return value->str();
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "missing required string field: %s",
                                 fieldName);
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, const char *fieldName) {
  if (auto *array = object.getArray(fieldName))
    return array;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "missing required array field: %s",
                                 fieldName);
}

llvm::Expected<ExecutionBackendKind> parseBackendKind(llvm::StringRef value) {
  if (value == "sim" || value == "simulation")
    return ExecutionBackendKind::Simulation;
  if (value == "npu")
    return ExecutionBackendKind::Npu;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported backend kind: %s",
                                 value.str().c_str());
}

llvm::Expected<DType> parseDType(llvm::StringRef value) {
  if (value == "f16")
    return DType::F16;
  if (value == "bf16")
    return DType::BF16;
  if (value == "f32")
    return DType::F32;
  if (value == "int8")
    return DType::INT8;
  if (value == "int32")
    return DType::INT32;
  if (value == "int64")
    return DType::INT64;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported tensor dtype: %s",
                                 value.str().c_str());
}

llvm::Expected<std::vector<int64_t>>
parseShape(const llvm::json::Array &array) {
  std::vector<int64_t> shape;
  shape.reserve(array.size());
  for (const llvm::json::Value &value : array) {
    auto integer = value.getAsInteger();
    if (!integer)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "shape entries must be integers");
    shape.push_back(static_cast<int64_t>(*integer));
  }
  return shape;
}

llvm::Error appendShapeArgField(ArtifactManifestPrepareRequest &prepare,
                                llvm::StringRef name,
                                const llvm::json::Value &value) {
  if (auto integer = value.getAsInteger()) {
    prepare.shapeArgs.emplace_back(name.str(), static_cast<int64_t>(*integer));
    return llvm::Error::success();
  }

  if (const llvm::json::Array *array = value.getAsArray()) {
    for (auto [index, element] : llvm::enumerate(*array)) {
      auto integer = element.getAsInteger();
      if (!integer) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "shape_args array entries must be integers: %s[%zu]",
            name.str().c_str(), index);
      }
      prepare.shapeArgs.emplace_back(
          (name + "_dim" + llvm::Twine(index)).str(),
          static_cast<int64_t>(*integer));
    }
    return llvm::Error::success();
  }

  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "shape_args values must be integers or integer arrays: %s",
      name.str().c_str());
}

std::string resolvePath(llvm::StringRef base, llvm::StringRef path) {
  if (path.empty() || llvm::sys::path::is_absolute(path))
    return path.str();
  llvm::SmallString<256> resolved(base);
  llvm::sys::path::append(resolved, path);
  return resolved.str().str();
}

bool hasNpyExtension(llvm::StringRef path) {
  return llvm::sys::path::extension(path).equals_insensitive(".npy");
}

llvm::Error enrichBindingFromNpy(ArtifactManifestBindingPath &binding) {
  if (!hasNpyExtension(binding.path)) {
    if (!binding.shape || !binding.dtype) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "case tensor binding requires .npy data or explicit shape/dtype: %s",
          binding.bindingName.c_str());
    }
    return llvm::Error::success();
  }

  if (binding.shape && binding.dtype)
    return llvm::Error::success();

  auto arrayOr = LoadNpy(binding.path);
  if (!arrayOr)
    return arrayOr.takeError();
  if (!binding.shape)
    binding.shape = arrayOr->shape;
  if (!binding.dtype)
    binding.dtype = arrayOr->dtype;
  return llvm::Error::success();
}

llvm::Expected<ArtifactManifestBindingPath>
parseCaseBinding(const llvm::json::Object &object, llvm::StringRef baseDir,
                 llvm::StringRef fieldName, bool inferFromExistingData) {
  auto nameOr = requireString(object, "name");
  if (!nameOr)
    return nameOr.takeError();
  auto pathOr = requireString(object, "path");
  if (!pathOr)
    return pathOr.takeError();

  ArtifactManifestBindingPath binding;
  auto [taskId, bindingName] = llvm::StringRef(*nameOr).split('.');
  if (bindingName.empty()) {
    binding.bindingName = *nameOr;
  } else {
    binding.taskId = taskId.str();
    binding.bindingName = bindingName.str();
  }
  binding.path = resolvePath(baseDir, *pathOr);

  if (auto *shape = object.getArray("shape")) {
    auto shapeOr = parseShape(*shape);
    if (!shapeOr)
      return shapeOr.takeError();
    binding.shape = std::move(*shapeOr);
  }
  if (auto dtype = object.getString("dtype")) {
    auto dtypeOr = parseDType(*dtype);
    if (!dtypeOr)
      return dtypeOr.takeError();
    binding.dtype = *dtypeOr;
  }

  if (inferFromExistingData) {
    if (auto err = enrichBindingFromNpy(binding)) {
      std::string message = llvm::toString(std::move(err));
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "%s binding %s: %s",
                                     fieldName.str().c_str(),
                                     binding.bindingName.c_str(),
                                     message.c_str());
    }
  }
  return binding;
}

llvm::Expected<std::vector<ArtifactManifestBindingPath>>
parseCaseBindings(const llvm::json::Object &root, llvm::StringRef baseDir,
                  const char *fieldName, bool required,
                  bool inferFromExistingData) {
  std::vector<ArtifactManifestBindingPath> bindings;
  const llvm::json::Array *array = root.getArray(fieldName);
  if (!array) {
    if (required)
      return requireArray(root, fieldName).takeError();
    return bindings;
  }
  bindings.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    auto objectOr = requireObject(&value, fieldName);
    if (!objectOr)
      return objectOr.takeError();
    auto bindingOr = parseCaseBinding(**objectOr, baseDir, fieldName,
                                      inferFromExistingData);
    if (!bindingOr)
      return bindingOr.takeError();
    bindings.push_back(std::move(*bindingOr));
  }
  return bindings;
}

llvm::Error rejectGeneratedFields(const llvm::json::Object &root) {
  static constexpr const char *forbidden[] = {"tiling", "block_dim",
                                              "workspace_size"};
  for (const char *field : forbidden) {
    if (root.get(field)) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "case.json field is generated by runtime prepare and must not be "
          "written by users: %s",
          field);
    }
  }
  return llvm::Error::success();
}

} // namespace

llvm::Error emitRunManifestFromDebugCase(
    const DebugCasePrepareRequest &request) {
  if (request.casePath.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "case path is required");
  if (request.outputRunManifestPath.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "output run manifest path is required");
  }

  auto bufferOr = llvm::MemoryBuffer::getFile(request.casePath, /*IsText=*/true);
  if (!bufferOr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot read case.json: %s",
                                   request.casePath.c_str());
  }

  auto jsonOr = llvm::json::parse((*bufferOr)->getBuffer());
  if (!jsonOr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid JSON in case.json: %s",
                                   request.casePath.c_str());
  }

  const llvm::json::Object *root = jsonOr->getAsObject();
  if (!root) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "case.json root must be an object: %s",
                                   request.casePath.c_str());
  }
  if (auto err = rejectGeneratedFields(*root))
    return err;

  auto schemaVersion = root->getInteger("schema_version");
  if (!schemaVersion || *schemaVersion != 1) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "case.json requires schema_version: 1");
  }

  llvm::SmallString<256> caseDir(request.casePath);
  llvm::sys::path::remove_filename(caseDir);
  if (caseDir.empty())
    caseDir = ".";

  const llvm::json::Object *artifact =
      root->getObject("artifact");
  if (!artifact) {
    if (root->getObject("source")) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "source-mode case.json is not supported by runtime-session yet; "
          "prepare artifacts first and use artifact.root/artifact.manifest");
    }
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "missing required object field: artifact");
  }

  auto artifactRootOr = requireString(*artifact, "root");
  if (!artifactRootOr)
    return artifactRootOr.takeError();
  auto artifactManifestOr = requireString(*artifact, "manifest");
  if (!artifactManifestOr)
    return artifactManifestOr.takeError();

  ArtifactManifestPrepareRequest prepare;
  prepare.artifactRoot = resolvePath(caseDir, *artifactRootOr);
  prepare.artifactManifestPath = resolvePath(caseDir, *artifactManifestOr);
  prepare.outputRunManifestPath = request.outputRunManifestPath;
  prepare.defaultOutputDirectory = request.defaultOutputDirectory;

  auto backendOr =
      requireObject(root->get("backend"), "backend");
  if (!backendOr)
    return backendOr.takeError();
  auto backendKindOr = requireString(**backendOr, "kind");
  if (!backendKindOr)
    return backendKindOr.takeError();
  auto parsedBackendOr = parseBackendKind(*backendKindOr);
  if (!parsedBackendOr)
    return parsedBackendOr.takeError();
  prepare.backendKind = *parsedBackendOr;

  if (const llvm::json::Object *shapeArgs = root->getObject("shape_args")) {
    for (const auto &field : *shapeArgs) {
      if (auto err = appendShapeArgField(prepare, field.getFirst(),
                                         field.getSecond()))
        return err;
    }
  }

  auto inputsOr = parseCaseBindings(*root, caseDir, "inputs", true,
                                    /*inferFromExistingData=*/true);
  if (!inputsOr)
    return inputsOr.takeError();
  prepare.inputPaths = std::move(*inputsOr);

  auto outputsOr = parseCaseBindings(*root, caseDir, "outputs", false,
                                     /*inferFromExistingData=*/false);
  if (!outputsOr)
    return outputsOr.takeError();
  prepare.outputPaths = std::move(*outputsOr);

  auto expectedOr =
      parseCaseBindings(*root, caseDir, "expected_outputs", false,
                        /*inferFromExistingData=*/true);
  if (!expectedOr)
    return expectedOr.takeError();
  prepare.expectedOutputPaths = std::move(*expectedOr);

  if (const llvm::json::Object *validation = root->getObject("validation")) {
    if (auto atol = validation->getNumber("atol"))
      prepare.atol = *atol;
    if (auto rtol = validation->getNumber("rtol"))
      prepare.rtol = *rtol;
  }
  if (auto profiling = root->getBoolean("profiling"))
    prepare.enableProfiling = *profiling;

  return emitRunManifestFromArtifactManifest(prepare);
}

} // namespace mlir::runtime
