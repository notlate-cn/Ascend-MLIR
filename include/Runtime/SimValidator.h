// include/Runtime/SimValidator.h
#pragma once
#include "Runtime/Compiler.h"
#include "Runtime/Executor.h"
#include "Runtime/Types.h"
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace mlir::runtime {

class SimValidator {
public:
  struct Result {
    bool        passed        = false;
    double      max_abs_diff  = 0.0;
    double      mean_abs_diff = 0.0;
    std::string error_msg;
  };

  Result Validate(const std::string&          kernel_src,
                  const std::string&          kernel_name,
                  RunArgs&                    args,
                  const std::vector<NDArray>& expected,
                  double                      atol = 1.0,
                  double                      rtol = 1e-2,
                  const Compiler::Config&     compiler_cfg = Compiler::Config{});
};

} // namespace mlir::runtime
