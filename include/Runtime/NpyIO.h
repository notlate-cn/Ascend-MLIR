// include/Runtime/NpyIO.h
#pragma once
#include "Runtime/Types.h"
#include "llvm/Support/Error.h"
#include <string>

namespace mlir::runtime {

// Load a .npy file. Allocates NDArray::data with new uint8_t[].
// Caller must delete[] the data pointer.
llvm::Expected<NDArray> LoadNpy(const std::string& path);

// Save NDArray to .npy file.
llvm::Error SaveNpy(const std::string& path, const NDArray& arr);

} // namespace mlir::runtime
