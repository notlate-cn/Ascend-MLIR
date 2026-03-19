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
    int64_t     cycle_count   = -1;   // -1 = not available (real hw or log missing)
    std::string error_msg;
  };

  // Compile kernel_src, then run and compare against expected.
  Result Validate(const std::string&          kernel_src,
                  const std::string&          kernel_name,
                  RunArgs&                    args,
                  const std::vector<NDArray>& expected,
                  double                      atol = 1.0,
                  double                      rtol = 1e-2,
                  const Compiler::Config&     compiler_cfg = Compiler::Config{});

  // Run a pre-registered kernel (skip compilation and re-registration).
  // Use Executor::RegisterBinary() once to get func_handle, then call this N
  // times with different tiling configs using the same Executor.
  Result ValidateBinary(void*                       func_handle,
                        Executor&                   executor,
                        RunArgs&                    args,
                        const std::vector<NDArray>& expected,
                        double                      atol,
                        double                      rtol);
};

} // namespace mlir::runtime
