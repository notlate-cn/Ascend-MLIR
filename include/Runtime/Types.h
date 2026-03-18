// include/Runtime/Types.h
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlir::runtime {

enum class DType { F16, F32, INT32 };

inline size_t dtypeBytes(DType d) {
  switch (d) {
    case DType::F16:   return 2;
    case DType::F32:   return 4;
    case DType::INT32: return 4;
  }
  return 0;
}

struct NDArray {
  // caller-owned; must be non-null when nbytes() > 0
  void*                data   = nullptr;
  std::vector<int64_t> shape;
  DType                dtype  = DType::F16;

  size_t numElements() const {
    // empty shape → scalar (1 element)
    size_t n = 1;
    for (auto s : shape) n *= static_cast<size_t>(s);
    return n;
  }
  size_t nbytes() const { return numElements() * dtypeBytes(dtype); }
};

struct RunArgs {
  std::vector<NDArray> inputs;
  std::vector<NDArray> outputs;      // pre-allocated, filled after Run()
  std::vector<uint8_t> tiling;       // packed TilingData bytes (little-endian)
  int                  block_dim     = 1;
  size_t               workspace_size = 8192;
};

} // namespace mlir::runtime
