#include "Runtime/TilingPack.h"

#include "Runtime/TilingSchema.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <map>
#include <sstream>
#include <string>

namespace mlir::runtime {

namespace {

std::vector<std::string> splitComma(const std::string &value) {
  std::vector<std::string> parts;
  std::istringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ','))
    parts.push_back(token);
  return parts;
}

llvm::Expected<std::map<std::string, int64_t>>
parseNamedTilingParams(const std::string &params) {
  std::map<std::string, int64_t> parsed;
  for (const std::string &token : splitComma(params)) {
    if (token.empty())
      continue;
    const size_t split = token.find('=');
    if (split == std::string::npos) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "tiling params token missing '=': %s",
                                     token.c_str());
    }
    parsed[token.substr(0, split)] = std::stoll(token.substr(split + 1));
  }
  return parsed;
}

} // namespace

llvm::Expected<std::vector<uint8_t>>
packTilingBytes(const std::optional<TilingBinding> &tiling) {
  if (!tiling)
    return std::vector<uint8_t>{};

  if (!tiling->binaryPath.empty()) {
    auto bufferOr = llvm::MemoryBuffer::getFile(tiling->binaryPath,
                                                /*IsText=*/false);
    if (!bufferOr) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cannot read tiling binary: %s",
                                     tiling->binaryPath.c_str());
    }
    const llvm::StringRef buffer = (*bufferOr)->getBuffer();
    return std::vector<uint8_t>(buffer.bytes_begin(), buffer.bytes_end());
  }

  if (!tiling->schemaPath.empty()) {
    auto schemaOr = TilingSchema::fromJson(tiling->schemaPath);
    if (!schemaOr)
      return schemaOr.takeError();

    auto parsedParamsOr = parseNamedTilingParams(tiling->params);
    if (!parsedParamsOr)
      return parsedParamsOr.takeError();

    std::vector<std::pair<std::string, int64_t>> orderedParams;
    for (const auto &field : schemaOr->fields()) {
      auto it = parsedParamsOr->find(field.name);
      if (it == parsedParamsOr->end()) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "tiling params missing field: %s",
                                       field.name.c_str());
      }
      orderedParams.push_back(*it);
    }
    return schemaOr->pack(orderedParams);
  }

  if (!tiling->params.empty()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "tiling params require either a schema path or a binary path");
  }

  return std::vector<uint8_t>{};
}

} // namespace mlir::runtime
