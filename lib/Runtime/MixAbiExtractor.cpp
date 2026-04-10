#include "Runtime/MixAbiExtractor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstring>

namespace mlir::runtime {

namespace {

struct ParsedFuncArg {
  std::string name;
  std::string type;
};

static std::string stripLeadingPercent(llvm::StringRef value) {
  value = value.trim();
  if (value.starts_with("%"))
    value = value.drop_front();
  return value.str();
}

static llvm::Expected<uint64_t> parseUnsigned(llvm::StringRef value,
                                              llvm::StringRef field,
                                              llvm::StringRef path) {
  uint64_t parsed = 0;
  if (value.trim().getAsInteger(10, parsed))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot extract mix ABI from %s: invalid %s '%s'",
                                   path.str().c_str(), field.str().c_str(),
                                   value.str().c_str());
  return parsed;
}

static llvm::Expected<DType> parseElementDType(llvm::StringRef elementType,
                                               llvm::StringRef path,
                                               llvm::StringRef argName) {
  if (elementType == "f16")
    return DType::F16;
  if (elementType == "f32")
    return DType::F32;
  if (elementType == "bf16")
    return DType::BF16;
  if (elementType == "i8")
    return DType::INT8;
  if (elementType == "i32")
    return DType::INT32;
  if (elementType == "i64")
    return DType::INT64;
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Cannot extract mix ABI from %s: unsupported element type '%s' for arg %s",
      path.str().c_str(), elementType.str().c_str(), argName.str().c_str());
}

static llvm::SmallVector<llvm::StringRef>
splitTopLevelList(llvm::StringRef text) {
  llvm::SmallVector<llvm::StringRef> out;
  size_t start = 0;
  int angleDepth = 0;
  int squareDepth = 0;
  int parenDepth = 0;
  bool inString = false;
  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    if (c == '"' && (i == 0 || text[i - 1] != '\\')) {
      inString = !inString;
      continue;
    }
    if (inString)
      continue;
    switch (c) {
    case '<':
      ++angleDepth;
      break;
    case '>':
      --angleDepth;
      break;
    case '[':
      ++squareDepth;
      break;
    case ']':
      --squareDepth;
      break;
    case '(':
      ++parenDepth;
      break;
    case ')':
      --parenDepth;
      break;
    case ',':
      if (angleDepth == 0 && squareDepth == 0 && parenDepth == 0) {
        out.push_back(text.slice(start, i).trim());
        start = i + 1;
      }
      break;
    default:
      break;
    }
  }
  if (start < text.size())
    out.push_back(text.drop_front(start).trim());
  return out;
}

static llvm::Expected<std::vector<int64_t>>
parseMemrefShape(llvm::StringRef shapeSpec, llvm::StringRef path,
                 llvm::StringRef argName) {
  std::vector<int64_t> shape;
  llvm::SmallVector<llvm::StringRef> dims;
  shapeSpec.split(dims, 'x', -1, false);
  if (dims.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: missing memref shape tokens for arg %s",
        path.str().c_str(), argName.str().c_str());
  if (dims.size() == 1)
    return shape;
  for (size_t i = 0; i + 1 < dims.size(); ++i) {
    llvm::StringRef dim = dims[i].trim();
    if (dim == "?") {
      shape.push_back(-1);
      continue;
    }
    auto parsed = parseUnsigned(dim, "memref dim", path);
    if (!parsed)
      return parsed.takeError();
    shape.push_back(static_cast<int64_t>(*parsed));
  }
  return shape;
}

static llvm::Expected<MixAbiTensorDesc>
parseTensorArg(const ParsedFuncArg &arg, llvm::StringRef kernelName,
               llvm::StringRef path, bool isOutput) {
  llvm::StringRef type = llvm::StringRef(arg.type).trim();
  if (!type.starts_with("memref<") || !type.ends_with(">"))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: unsupported tensor arg type '%s' for %s",
        path.str().c_str(), type.str().c_str(), arg.name.c_str());

  llvm::StringRef inner = type.drop_front(strlen("memref<")).drop_back();
  llvm::SmallVector<llvm::StringRef> parts = splitTopLevelList(inner);
  if (parts.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: empty memref payload for %s",
        path.str().c_str(), arg.name.c_str());

  llvm::StringRef shapeAndElement = parts.front().trim();
  llvm::SmallVector<llvm::StringRef> dimsAndType;
  shapeAndElement.split(dimsAndType, 'x', -1, false);
  if (dimsAndType.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: cannot parse memref payload '%s' for %s",
        path.str().c_str(), shapeAndElement.str().c_str(), arg.name.c_str());

  llvm::StringRef elementType = dimsAndType.back().trim();
  auto dtype = parseElementDType(elementType, path, arg.name);
  if (!dtype)
    return dtype.takeError();
  auto shape = parseMemrefShape(shapeAndElement, path, arg.name);
  if (!shape)
    return shape.takeError();

  MixAbiTensorDesc tensor;
  tensor.name = stripLeadingPercent(arg.name);
  tensor.dtype = *dtype;
  tensor.shape = std::move(*shape);
  tensor.runtimeFile = isOutput
                           ? buildCanonicalOutputFileName(kernelName, tensor.name)
                           : buildCanonicalInputFileName(kernelName, tensor.name);
  if (isOutput)
    tensor.goldenFile = buildCanonicalGoldenFileName(kernelName, tensor.name);
  return tensor;
}

static llvm::Expected<std::string>
readFileText(llvm::StringRef path) {
  auto bufferOr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOr)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: failed to read file",
        path.str().c_str());
  return bufferOr.get()->getBuffer().str();
}

static bool isTensorArgReferencedInBody(llvm::StringRef bodyText,
                                        llvm::StringRef argName) {
  const std::string needle = argName.str();
  size_t pos = bodyText.find(needle);
  while (pos != llvm::StringRef::npos) {
    const bool startsToken = pos == 0 || !llvm::isAlnum(bodyText[pos - 1]);
    const size_t end = pos + needle.size();
    const bool endsToken =
        end >= bodyText.size() || !llvm::isAlnum(bodyText[end]);
    if (startsToken && endsToken)
      return true;
    pos = bodyText.find(needle, pos + needle.size());
  }
  return false;
}

} // namespace

llvm::Expected<MixAbiMetadata>
extractMixAbiFromCannMlir(llvm::StringRef cannMlirPath) {
  if (cannMlirPath.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot extract mix ABI: empty MLIR path");
  if (!llvm::sys::fs::exists(cannMlirPath))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot extract mix ABI from %s: file not found",
                                   cannMlirPath.str().c_str());

  auto textOr = readFileText(cannMlirPath);
  if (!textOr)
    return textOr.takeError();
  llvm::StringRef text = *textOr;

  size_t funcPos = text.find("func.func @");
  if (funcPos == llvm::StringRef::npos)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: missing func.func declaration",
        cannMlirPath.str().c_str());

  text = text.drop_front(funcPos + strlen("func.func @"));
  size_t nameEnd = text.find('(');
  if (nameEnd == llvm::StringRef::npos)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: missing function argument list",
        cannMlirPath.str().c_str());
  llvm::StringRef kernelName = text.take_front(nameEnd).trim();
  if (kernelName.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: empty function name",
        cannMlirPath.str().c_str());

  llvm::StringRef afterName = text.drop_front(nameEnd + 1);
  int parenDepth = 1;
  size_t argListEnd = llvm::StringRef::npos;
  for (size_t i = 0; i < afterName.size(); ++i) {
    char c = afterName[i];
    if (c == '(')
      ++parenDepth;
    else if (c == ')') {
      --parenDepth;
      if (parenDepth == 0) {
        argListEnd = i;
        break;
      }
    }
  }
  if (argListEnd == llvm::StringRef::npos)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: unterminated function argument list",
        cannMlirPath.str().c_str());

  llvm::StringRef argList = afterName.take_front(argListEnd);
  llvm::StringRef tail = afterName.drop_front(argListEnd + 1);

  size_t numInputsPos = tail.find("cann.num_inputs");
  if (numInputsPos == llvm::StringRef::npos)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: missing cann.num_inputs attribute",
        cannMlirPath.str().c_str());
  llvm::StringRef numInputsTail = tail.drop_front(numInputsPos);
  size_t equalPos = numInputsTail.find('=');
  size_t colonPos = numInputsTail.find(':');
  if (equalPos == llvm::StringRef::npos || colonPos == llvm::StringRef::npos ||
      colonPos <= equalPos)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: malformed cann.num_inputs attribute",
        cannMlirPath.str().c_str());
  auto numInputsOr =
      parseUnsigned(numInputsTail.slice(equalPos + 1, colonPos).trim(),
                    "cann.num_inputs", cannMlirPath);
  if (!numInputsOr)
    return numInputsOr.takeError();
  size_t numInputs = static_cast<size_t>(*numInputsOr);

  llvm::SmallVector<ParsedFuncArg> parsedArgs;
  for (llvm::StringRef argSpec : splitTopLevelList(argList)) {
    if (argSpec.empty())
      continue;
    size_t colon = argSpec.find(':');
    if (colon == llvm::StringRef::npos)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Cannot extract mix ABI from %s: malformed argument '%s'",
          cannMlirPath.str().c_str(), argSpec.str().c_str());
    parsedArgs.push_back(
        {argSpec.take_front(colon).trim().str(),
         argSpec.drop_front(colon + 1).trim().str()});
  }

  MixAbiMetadata abi;
  abi.logicalKernelName = kernelName.str();
  abi.runtimeKernelName = abi.logicalKernelName;
  abi.workspaceBytes = 16777216ULL;
  abi.blockDim = 1;
  abi.workspaceMode = "fixed";
  abi.tilingMode = "generated_file";
  abi.tilingSource = "out/tiling.bin";

  if (parsedArgs.size() < 2)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: missing workspace/tiling arguments",
        cannMlirPath.str().c_str());

  const size_t workspaceIndex = parsedArgs.size() - 2;
  const size_t tilingIndex = parsedArgs.size() - 1;
  if (llvm::StringRef(parsedArgs[workspaceIndex].type).trim() != "memref<ui8>")
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: second-to-last arg must be memref<ui8>, got '%s'",
        cannMlirPath.str().c_str(), parsedArgs[workspaceIndex].type.c_str());
  if (!llvm::StringRef(parsedArgs[tilingIndex].type).trim().starts_with(
          "!emitasc.py_struct<"))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: last arg must be !emitasc.py_struct<...>, got '%s'",
        cannMlirPath.str().c_str(), parsedArgs[tilingIndex].type.c_str());

  abi.workspaceArgIndex = workspaceIndex;
  abi.tilingArgIndex = tilingIndex;

  for (size_t i = 0; i < workspaceIndex; ++i) {
    llvm::StringRef type = llvm::StringRef(parsedArgs[i].type).trim();
    if (!type.starts_with("memref<"))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Cannot extract mix ABI from %s: unsupported function arg type '%s' for %s",
          cannMlirPath.str().c_str(), type.str().c_str(),
          parsedArgs[i].name.c_str());
    auto tensorOr =
        parseTensorArg(parsedArgs[i], kernelName, cannMlirPath, i >= numInputs);
    if (!tensorOr)
      return tensorOr.takeError();
    if (!isTensorArgReferencedInBody(tail, parsedArgs[i].name))
      continue;
    if (i < numInputs)
      abi.inputs.push_back(std::move(*tensorOr));
    else
      abi.outputs.push_back(std::move(*tensorOr));
  }

  if (abi.inputs.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: no live input tensors remain after "
        "filtering unused kernel arguments",
        cannMlirPath.str().c_str());
  if (abi.outputs.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot extract mix ABI from %s: no live output tensors remain after "
        "filtering unused kernel arguments",
        cannMlirPath.str().c_str());
  return abi;
}

} // namespace mlir::runtime
