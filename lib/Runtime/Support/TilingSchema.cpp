//===- TilingSchema.cpp ---------------------------------------------------===//
#include "Runtime/TilingSchema.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <string>

namespace mlir {
namespace runtime {

llvm::Expected<TilingSchema> TilingSchema::fromJson(llvm::StringRef path) {
  auto bufOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufOrErr)
    return llvm::createStringError(bufOrErr.getError(),
                                   "cannot open '%s'", path.data());

  auto parsed = llvm::json::parse((*bufOrErr)->getBuffer());
  if (!parsed)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "JSON parse error in '%s': %s",
                                   path.data(),
                                   llvm::toString(parsed.takeError()).c_str());

  auto *root = parsed->getAsObject();
  if (!root)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "'%s': root must be a JSON object",
                                   path.data());

  auto *paramsArr = root->getArray("tiling_params");
  if (!paramsArr)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "'%s': missing 'tiling_params' array",
                                   path.data());

  TilingSchema schema;
  for (auto &elem : *paramsArr) {
    auto *obj = elem.getAsObject();
    if (!obj)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "'%s': tiling_params entry is not an object",
                                     path.data());
    auto nameStr = obj->getString("name");
    if (!nameStr)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "'%s': tiling_params entry missing 'name'",
                                     path.data());
    TilingField f;
    f.name = nameStr->str();
    if (auto typeStr = obj->getString("type"))
      f.type = typeStr->str();
    else
      f.type = "int64";
    // Shape-derived (dynamic) params carry a `shape_key` ("arg<N>_dim<D>") so
    // the launcher can resolve them from the runtime input shapes.
    if (auto shapeKey = obj->getString("shape_key"))
      f.shapeKey = shapeKey->str();
    schema.fields_.push_back(std::move(f));
  }
  return schema;
}

llvm::Expected<std::vector<uint8_t>>
TilingSchema::pack(
    llvm::ArrayRef<std::pair<std::string, int64_t>> params) const {
  if (params.size() != fields_.size()) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "tiling param count mismatch: got " << params.size()
       << ", expected " << fields_.size() << " (";
    for (size_t i = 0; i < fields_.size(); ++i) {
      if (i) os << ", ";
      os << fields_[i].name;
    }
    os << ")";
    return llvm::createStringError(llvm::inconvertibleErrorCode(), os.str());
  }

  std::vector<uint8_t> bytes;
  for (size_t i = 0; i < fields_.size(); ++i) {
    if (params[i].first != fields_[i].name) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "tiling param name mismatch at position " << i
         << ": got '" << params[i].first
         << "', expected '" << fields_[i].name << "'";
      return llvm::createStringError(llvm::inconvertibleErrorCode(), os.str());
    }
    int64_t val = params[i].second;
    if (fields_[i].type == "int32" || fields_[i].type == "int32_t") {
      int32_t v = static_cast<int32_t>(val);
      uint8_t buf[4];
      std::memcpy(buf, &v, 4);
      bytes.insert(bytes.end(), buf, buf + 4);
    } else {
      uint8_t buf[8];
      std::memcpy(buf, &val, 8);
      bytes.insert(bytes.end(), buf, buf + 8);
    }
  }
  return bytes;
}

} // namespace runtime
} // namespace mlir
