//===- TilingSchema.h - Validated tiling parameter packing ------*- C++ -*-===//
//
// Reads the `tiling_params` array from a tiling_space.json file and packs
// named parameters into a little-endian byte vector suitable for RunArgs::tiling.
//
// Only `name` and `type` fields are read; autotuner fields (fixed, shape_key,
// min, max, step, values) are ignored.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>
#include <utility>
#include <vector>

namespace mlir {
namespace runtime {

struct TilingField {
  std::string name;
  std::string type; // "int32" | "int64" (unknown defaults to int64)
};

class TilingSchema {
public:
  /// Load schema from a tiling_space.json file.
  /// Returns error if the file cannot be read or is malformed.
  static llvm::Expected<TilingSchema> fromJson(llvm::StringRef path);

  /// Pack named parameters into a little-endian byte vector.
  ///
  /// params must be supplied in field-declaration order and each name must
  /// match the corresponding schema field name positionally.
  ///
  /// Returns error if:
  ///   - params.size() != fields_.size()
  ///   - params[i].first != fields_[i].name for any i
  llvm::Expected<std::vector<uint8_t>>
  pack(llvm::ArrayRef<std::pair<std::string, int64_t>> params) const;

  llvm::ArrayRef<TilingField> fields() const { return fields_; }
  size_t size() const { return fields_.size(); }

private:
  std::vector<TilingField> fields_;
};

} // namespace runtime
} // namespace mlir
