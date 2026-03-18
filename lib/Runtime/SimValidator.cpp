// lib/Runtime/SimValidator.cpp
#include "Runtime/SimValidator.h"
#include "llvm/Support/FileSystem.h"
#include <cmath>
#include <cstdint>
#include <cstring>

namespace mlir::runtime {

// Convert a single element to float for comparison.
static float toFloat(const void* base, size_t idx, DType dtype) {
  switch (dtype) {
    case DType::F16: {
      uint16_t h;
      std::memcpy(&h, static_cast<const uint8_t*>(base) + idx * 2, 2);
      uint32_t sign = (h >> 15) & 1;
      uint32_t exp  = (h >> 10) & 0x1f;
      uint32_t frac = h & 0x3ff;
      uint32_t f;
      if (exp == 0)        f = (sign << 31) | (frac << 13);
      else if (exp == 31)  f = (sign << 31) | 0x7f800000u | (frac << 13);
      else                 f = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
      float result;
      std::memcpy(&result, &f, 4);
      return result;
    }
    case DType::F32: {
      float v;
      std::memcpy(&v, static_cast<const uint8_t*>(base) + idx * 4, 4);
      return v;
    }
    case DType::INT32: {
      int32_t v;
      std::memcpy(&v, static_cast<const uint8_t*>(base) + idx * 4, 4);
      return static_cast<float>(v);
    }
  }
  return 0.f;
}

SimValidator::Result SimValidator::Validate(
    const std::string& kernel_src,
    const std::string& kernel_name,
    RunArgs& args,
    const std::vector<NDArray>& expected,
    double atol, double rtol,
    const Compiler::Config& compiler_cfg) {

  Result r;

  // Create temp build dir
  llvm::SmallString<256> build_dir;
  if (llvm::sys::fs::createUniqueDirectory("sim_validator_build", build_dir)) {
    r.error_msg = "Cannot create temp build dir";
    return r;
  }

  // Compile kernel
  Compiler compiler(compiler_cfg);
  auto bin_or = compiler.Compile(kernel_src, build_dir.str().str(), kernel_name);
  if (!bin_or) {
    r.error_msg = "Compile failed: " + llvm::toString(bin_or.takeError());
    return r;
  }

  // Execute kernel
  Executor executor;
  if (auto err = executor.Initialize()) {
    r.error_msg = "Executor init failed: " + llvm::toString(std::move(err));
    return r;
  }
  if (auto err = executor.RunFile(*bin_or, kernel_name, args)) {
    r.error_msg = "Kernel run failed: " + llvm::toString(std::move(err));
    return r;
  }

  // Compare outputs
  if (args.outputs.size() != expected.size()) {
    r.error_msg = "Output count mismatch: got " +
                  std::to_string(args.outputs.size()) + ", expected " +
                  std::to_string(expected.size());
    return r;
  }

  double sum_diff = 0.0, max_diff = 0.0;
  size_t total_elements = 0;
  bool all_close = true;

  for (size_t oi = 0; oi < args.outputs.size(); ++oi) {
    const NDArray& act = args.outputs[oi];
    const NDArray& exp = expected[oi];
    size_t n = act.numElements();
    total_elements += n;
    for (size_t i = 0; i < n; ++i) {
      double a = static_cast<double>(toFloat(act.data, i, act.dtype));
      double e = static_cast<double>(toFloat(exp.data, i, exp.dtype));
      double diff = std::abs(a - e);
      sum_diff += diff;
      if (diff > max_diff) max_diff = diff;
      if (diff > atol + rtol * std::abs(e)) all_close = false;
    }
  }

  r.max_abs_diff  = max_diff;
  r.mean_abs_diff = total_elements > 0 ? sum_diff / static_cast<double>(total_elements) : 0.0;
  r.passed        = all_close;
  return r;
}

} // namespace mlir::runtime
