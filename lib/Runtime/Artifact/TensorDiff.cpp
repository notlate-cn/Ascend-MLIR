#include "Runtime/Artifact/TensorDiff.h"

#include "Runtime/NpyIO.h"
#include "Runtime/OutputComparator.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>
#include <vector>

namespace mlir::runtime {

namespace {

struct TensorComparisonSpec {
  std::string id;
  std::string lhs;
  std::string rhs;
  double atol = 1.0e-5;
  double rtol = 1.0e-5;
  std::string kernelId;
  std::string taskId;
  std::string stage;
  std::string semanticBoundary;
};

llvm::Expected<std::string> requireString(const llvm::json::Object &object,
                                          llvm::StringRef fieldName) {
  if (auto value = object.getString(fieldName))
    return value->str();
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "missing required string field: %s",
                                 fieldName.str().c_str());
}

llvm::Expected<std::string> optionalString(const llvm::json::Object &object,
                                           llvm::StringRef fieldName) {
  if (!object.get(fieldName))
    return std::string();
  if (auto value = object.getString(fieldName))
    return value->str();
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "field must be a string: %s",
                                 fieldName.str().c_str());
}

llvm::Expected<double> optionalNumber(const llvm::json::Object &object,
                                      llvm::StringRef fieldName,
                                      double fallback) {
  if (!object.get(fieldName))
    return fallback;
  if (auto value = object.getNumber(fieldName))
    return *value;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "field must be a number: %s",
                                 fieldName.str().c_str());
}

llvm::Expected<std::vector<TensorComparisonSpec>>
parseTensorComparisons(const llvm::json::Object &root) {
  const llvm::json::Array *comparisons = root.getArray("comparisons");
  if (!comparisons) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "tensor manifest comparisons must be an array");
  }

  std::vector<TensorComparisonSpec> specs;
  specs.reserve(comparisons->size());
  for (size_t index = 0; index < comparisons->size(); ++index) {
    const llvm::json::Value &value = (*comparisons)[index];
    const llvm::json::Object *object = value.getAsObject();
    if (!object) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "tensor comparison %zu must be an object",
                                     index);
    }

    TensorComparisonSpec spec;
    if (auto id = object->getString("id"))
      spec.id = id->str();
    else
      spec.id = ("comparison_" + llvm::Twine(index)).str();
    if (spec.id.empty()) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "tensor comparison %zu id must not be empty",
                                     index);
    }

    auto lhsOr = requireString(*object, "lhs");
    if (!lhsOr)
      return lhsOr.takeError();
    spec.lhs = std::move(*lhsOr);
    auto rhsOr = requireString(*object, "rhs");
    if (!rhsOr)
      return rhsOr.takeError();
    spec.rhs = std::move(*rhsOr);

    auto atolOr = optionalNumber(*object, "atol", spec.atol);
    if (!atolOr)
      return atolOr.takeError();
    spec.atol = *atolOr;
    auto rtolOr = optionalNumber(*object, "rtol", spec.rtol);
    if (!rtolOr)
      return rtolOr.takeError();
    spec.rtol = *rtolOr;
    if (spec.atol < 0.0 || spec.rtol < 0.0) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "tensor comparison %zu tolerance must be non-negative",
                                     index);
    }

    auto kernelIdOr = optionalString(*object, "kernel_id");
    if (!kernelIdOr)
      return kernelIdOr.takeError();
    spec.kernelId = std::move(*kernelIdOr);
    auto taskIdOr = optionalString(*object, "task_id");
    if (!taskIdOr)
      return taskIdOr.takeError();
    spec.taskId = std::move(*taskIdOr);
    auto stageOr = optionalString(*object, "stage");
    if (!stageOr)
      return stageOr.takeError();
    spec.stage = std::move(*stageOr);
    auto semanticBoundaryOr = optionalString(*object, "semantic_boundary");
    if (!semanticBoundaryOr)
      return semanticBoundaryOr.takeError();
    spec.semanticBoundary = std::move(*semanticBoundaryOr);

    specs.push_back(std::move(spec));
  }
  return specs;
}

bool pathContainsParentReference(llvm::StringRef path) {
  for (auto it = llvm::sys::path::begin(path), end = llvm::sys::path::end(path);
       it != end; ++it) {
    if (*it == "..")
      return true;
  }
  return false;
}

llvm::Expected<std::string> resolveTensorPath(llvm::StringRef runDir,
                                              llvm::StringRef path) {
  if (path.empty() || llvm::sys::path::is_absolute(path) ||
      pathContainsParentReference(path)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "tensor comparison path must be relative and stay inside run dir: %s",
        path.str().c_str());
  }
  llvm::SmallString<256> resolved(runDir);
  llvm::sys::path::append(resolved, path);
  return resolved.str().str();
}

std::string inferRunDirFromManifestPath(llvm::StringRef manifestPath) {
  llvm::SmallString<256> dir(manifestPath);
  llvm::sys::path::remove_filename(dir);
  if (llvm::sys::path::filename(dir) == "tensors")
    llvm::sys::path::remove_filename(dir);
  if (dir.empty())
    return ".";
  return dir.str().str();
}

const char *dtypeName(DType dtype) {
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
  }
  return "unknown";
}

llvm::json::Array shapeToJson(llvm::ArrayRef<int64_t> shape) {
  llvm::json::Array array;
  for (int64_t dim : shape)
    array.push_back(dim);
  return array;
}

llvm::json::Object structuralFailureObject(const TensorComparisonSpec &spec,
                                           llvm::StringRef reason) {
  llvm::json::Object object;
  object["id"] = spec.id;
  object["status"] = "fail";
  object["reason"] = reason.str();
  object["lhs"] = spec.lhs;
  object["rhs"] = spec.rhs;
  object["atol"] = spec.atol;
  object["rtol"] = spec.rtol;
  if (!spec.kernelId.empty())
    object["kernel_id"] = spec.kernelId;
  if (!spec.taskId.empty())
    object["task_id"] = spec.taskId;
  if (!spec.stage.empty())
    object["stage"] = spec.stage;
  if (!spec.semanticBoundary.empty())
    object["semantic_boundary"] = spec.semanticBoundary;
  return object;
}

llvm::json::Object comparisonObject(const TensorComparisonSpec &spec,
                                    const NDArray &lhs, const NDArray &rhs,
                                    const OutputComparisonResult &result) {
  llvm::json::Object object;
  object["id"] = spec.id;
  object["status"] = result.passed ? "pass" : "fail";
  object["lhs"] = spec.lhs;
  object["rhs"] = spec.rhs;
  object["shape"] = shapeToJson(lhs.shape);
  object["lhs_dtype"] = dtypeName(lhs.dtype);
  object["rhs_dtype"] = dtypeName(rhs.dtype);
  object["element_count"] = static_cast<int64_t>(lhs.numElements());
  object["atol"] = spec.atol;
  object["rtol"] = spec.rtol;
  object["max_abs_error"] = result.maxAbsDiff;
  object["max_rel_error"] = result.maxRelDiff;
  object["mean_abs_error"] = result.meanAbsDiff;
  if (!spec.kernelId.empty())
    object["kernel_id"] = spec.kernelId;
  if (!spec.taskId.empty())
    object["task_id"] = spec.taskId;
  if (!spec.stage.empty())
    object["stage"] = spec.stage;
  if (!spec.semanticBoundary.empty())
    object["semantic_boundary"] = spec.semanticBoundary;
  return object;
}

llvm::Error writeJsonFile(llvm::StringRef path, llvm::json::Object object) {
  llvm::SmallString<256> directory(path);
  llvm::sys::path::remove_filename(directory);
  if (!directory.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(directory))
      return llvm::createStringError(ec, "cannot create summary directory: %s",
                                     directory.c_str());
  }

  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return llvm::createStringError(ec, "cannot write validation summary: %s",
                                   path.str().c_str());
  llvm::json::OStream json(os, /*IndentSize=*/2);
  json.value(llvm::json::Value(std::move(object)));
  os << "\n";
  return llvm::Error::success();
}

} // namespace

llvm::Expected<TensorDiffRunResult>
emitTensorDiffSummaryFromManifest(const TensorDiffRequest &request) {
  if (request.manifestPath.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "tensor manifest path is required");
  if (request.outputSummaryPath.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "validation summary path is required");

  auto bufferOr =
      llvm::MemoryBuffer::getFile(request.manifestPath, /*IsText=*/true);
  if (!bufferOr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot read tensor manifest: %s",
                                   request.manifestPath.c_str());
  }
  auto jsonOr = llvm::json::parse((*bufferOr)->getBuffer());
  if (!jsonOr) {
    std::string reason = llvm::toString(jsonOr.takeError());
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid JSON in tensor manifest: %s: %s",
                                   request.manifestPath.c_str(),
                                   reason.c_str());
  }
  const llvm::json::Object *root = jsonOr->getAsObject();
  if (!root) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "tensor manifest root must be an object");
  }
  auto schemaVersion = root->getInteger("schema_version");
  if (!schemaVersion || *schemaVersion != 1) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "tensor manifest requires schema_version: 1");
  }
  auto specsOr = parseTensorComparisons(*root);
  if (!specsOr)
    return specsOr.takeError();

  const std::string runDir = inferRunDirFromManifestPath(request.manifestPath);
  llvm::json::Array comparisonJson;
  size_t failedCount = 0;

  for (const TensorComparisonSpec &spec : *specsOr) {
    auto lhsPathOr = resolveTensorPath(runDir, spec.lhs);
    if (!lhsPathOr)
      return lhsPathOr.takeError();
    auto rhsPathOr = resolveTensorPath(runDir, spec.rhs);
    if (!rhsPathOr)
      return rhsPathOr.takeError();

    auto lhsOr = LoadNpy(*lhsPathOr);
    if (!lhsOr)
      return lhsOr.takeError();
    auto rhsOr = LoadNpy(*rhsPathOr);
    if (!rhsOr)
      return rhsOr.takeError();

    std::vector<NDArray> lhsValues;
    lhsValues.push_back(std::move(*lhsOr));
    std::vector<NDArray> rhsValues;
    rhsValues.push_back(std::move(*rhsOr));
    auto comparisonOr =
        compareRuntimeOutputs(rhsValues, lhsValues, spec.atol, spec.rtol);
    if (!comparisonOr) {
      ++failedCount;
      comparisonJson.push_back(
          structuralFailureObject(spec, llvm::toString(comparisonOr.takeError())));
      continue;
    }

    if (!comparisonOr->passed)
      ++failedCount;
    comparisonJson.push_back(
        comparisonObject(spec, lhsValues[0], rhsValues[0], *comparisonOr));
  }

  llvm::json::Object summary;
  summary["schema_version"] = 1;
  summary["tool"] = "runtime-session";
  summary["status"] = failedCount == 0 ? "pass" : "fail";
  summary["comparison_count"] = static_cast<int64_t>(specsOr->size());
  summary["failed_count"] = static_cast<int64_t>(failedCount);
  summary["comparisons"] = std::move(comparisonJson);

  if (auto err = writeJsonFile(request.outputSummaryPath, std::move(summary)))
    return std::move(err);

  TensorDiffRunResult result;
  result.passed = failedCount == 0;
  result.comparisonCount = specsOr->size();
  result.failedCount = failedCount;
  return result;
}

} // namespace mlir::runtime
