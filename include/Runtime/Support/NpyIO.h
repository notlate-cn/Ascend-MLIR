// include/Runtime/NpyIO.h
#pragma once
#include "Runtime/Support/Types.h"
#include "llvm/Support/Error.h"
#include <string>

namespace mlir::runtime {

// Load a .npy file. Returns NDArray with owned_data (RAII).
// Supported dtypes: f16, bf16, f32, i1 (INT8), i4 (INT32), i8 (INT64).
llvm::Expected<NDArray> LoadNpy(const std::string& path);

// Save NDArray to .npy file.
llvm::Error SaveNpy(const std::string& path, const NDArray& arr);

} // namespace mlir::runtime
