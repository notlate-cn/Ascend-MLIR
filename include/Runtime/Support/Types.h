// include/Runtime/Types.h
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace mlir::runtime {

enum class DType { F16, BF16, F32, INT8, INT32, INT64 };

inline size_t dtypeBytes(DType d) {
  switch (d) {
    case DType::F16:   return 2;
    case DType::BF16:  return 2;
    case DType::F32:   return 4;
    case DType::INT8:  return 1;
    case DType::INT32: return 4;
    case DType::INT64: return 8;
  }
  return 0;
}

struct NDArray {
  std::unique_ptr<uint8_t[]> owned_data;  // RAII owner (null if externally managed)
  void*                data   = nullptr;  // active pointer (may point into owned_data or external)
  std::vector<int64_t> shape;
  DType                dtype  = DType::F16;

  NDArray() = default;

  // Move-only (unique_ptr is not copyable).
  // After move, the source NDArray is fully reset (data = nullptr).
  NDArray(NDArray&& o) noexcept
      : owned_data(std::move(o.owned_data)),
        data(o.data),
        shape(std::move(o.shape)),
        dtype(o.dtype) {
    o.data = nullptr;
  }
  NDArray& operator=(NDArray&& o) noexcept {
    if (this != &o) {
      owned_data = std::move(o.owned_data);
      data       = o.data;  o.data = nullptr;
      shape      = std::move(o.shape);
      dtype      = o.dtype;
    }
    return *this;
  }
  NDArray(const NDArray&) = delete;
  NDArray& operator=(const NDArray&) = delete;

  // Allocate internal storage of nbytes(), sets data pointer.
  void allocate() {
    size_t n = nbytes();
    owned_data = std::make_unique<uint8_t[]>(n);
    data = owned_data.get();
  }

  // Wrap external pointer (no ownership transfer, caller ensures lifetime).
  void setExternal(void* ptr) {
    owned_data.reset();
    data = ptr;
  }

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
