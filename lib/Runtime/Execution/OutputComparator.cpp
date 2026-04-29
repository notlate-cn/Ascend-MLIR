#include "Runtime/OutputComparator.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

namespace mlir::runtime {

namespace {

float toFloat(const void *base, size_t idx, DType dtype) {
  switch (dtype) {
  case DType::F16: {
    uint16_t h;
    std::memcpy(&h, static_cast<const uint8_t *>(base) + idx * 2, 2);
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;
    uint32_t f;
    if (exp == 0)
      f = (sign << 31) | (frac << 13);
    else if (exp == 31)
      f = (sign << 31) | 0x7f800000u | (frac << 13);
    else
      f = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
    float result;
    std::memcpy(&result, &f, 4);
    return result;
  }
  case DType::BF16: {
    uint16_t h;
    std::memcpy(&h, static_cast<const uint8_t *>(base) + idx * 2, 2);
    uint32_t f = static_cast<uint32_t>(h) << 16;
    float result;
    std::memcpy(&result, &f, 4);
    return result;
  }
  case DType::F32: {
    float v;
    std::memcpy(&v, static_cast<const uint8_t *>(base) + idx * 4, 4);
    return v;
  }
  case DType::INT8: {
    int8_t v;
    std::memcpy(&v, static_cast<const uint8_t *>(base) + idx, 1);
    return static_cast<float>(v);
  }
  case DType::INT32: {
    int32_t v;
    std::memcpy(&v, static_cast<const uint8_t *>(base) + idx * 4, 4);
    return static_cast<float>(v);
  }
  case DType::INT64: {
    int64_t v;
    std::memcpy(&v, static_cast<const uint8_t *>(base) + idx * 8, 8);
    return static_cast<float>(v);
  }
  }
  return 0.0f;
}

llvm::Error structuralMismatch(llvm::StringRef detail) {
  std::string message = detail.str();
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                 message.c_str());
}

} // namespace

llvm::Expected<OutputComparisonResult>
compareRuntimeOutputs(llvm::ArrayRef<NDArray> actual,
                      llvm::ArrayRef<NDArray> expected, double atol,
                      double rtol) {
  if (actual.size() != expected.size()) {
    return structuralMismatch(
        llvm::formatv("output count mismatch: got {0}, expected {1}",
                      actual.size(), expected.size())
            .str());
  }

  OutputComparisonResult result;
  result.passed = true;
  double totalDiff = 0.0;
  size_t totalElements = 0;

  for (size_t outputIndex = 0; outputIndex < actual.size(); ++outputIndex) {
    const NDArray &actualArray = actual[outputIndex];
    const NDArray &expectedArray = expected[outputIndex];

    if (actualArray.shape != expectedArray.shape) {
      return structuralMismatch(
          llvm::formatv("output {0} shape mismatch", outputIndex).str());
    }
    if (actualArray.dtype != expectedArray.dtype) {
      return structuralMismatch(
          llvm::formatv("output {0} dtype mismatch", outputIndex).str());
    }
    if (actualArray.nbytes() != expectedArray.nbytes()) {
      return structuralMismatch(
          llvm::formatv("output {0} byte-size mismatch", outputIndex).str());
    }
    if (!actualArray.data || !expectedArray.data) {
      return structuralMismatch(
          llvm::formatv("output {0} has null data pointer", outputIndex).str());
    }

    const size_t elements = actualArray.numElements();
    totalElements += elements;
    for (size_t elementIndex = 0; elementIndex < elements; ++elementIndex) {
      const double actualValue =
          static_cast<double>(toFloat(actualArray.data, elementIndex,
                                      actualArray.dtype));
      const double expectedValue =
          static_cast<double>(toFloat(expectedArray.data, elementIndex,
                                      expectedArray.dtype));
      const double diff = std::abs(actualValue - expectedValue);
      totalDiff += diff;
      result.maxAbsDiff = std::max(result.maxAbsDiff, diff);
      if (diff > atol + rtol * std::abs(expectedValue))
        result.passed = false;
    }
  }

  result.meanAbsDiff =
      totalElements == 0 ? 0.0 : totalDiff / static_cast<double>(totalElements);
  if (!result.passed) {
    result.errorMessage =
        llvm::formatv("output comparison failed: max_abs_diff={0}, mean_abs_diff={1}",
                      result.maxAbsDiff, result.meanAbsDiff)
            .str();
  }
  return result;
}

} // namespace mlir::runtime
