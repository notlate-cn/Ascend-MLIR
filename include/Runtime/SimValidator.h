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

  // Run a pre-registered kernel (skip compilation and re-registration).
  // Use Executor::RegisterBinary() once to get func_handle, then call this N
  // times with different tiling configs using the same Executor.
  Result ValidateBinary(void*                       func_handle,
                        Executor&                   executor,
                        RunArgs&                    args,
                        const std::vector<NDArray>& expected,
                        double                      atol,
                        double                      rtol);

  // Compare outputs after they have already been produced by a simulator-backed
  // external run path, while still reusing validator diff and cycle-count
  // reporting from the current working directory.
  Result CompareOnly(RunArgs&                    args,
                     const std::vector<NDArray>& expected,
                     double                      atol,
                     double                      rtol);
};

} // namespace mlir::runtime
